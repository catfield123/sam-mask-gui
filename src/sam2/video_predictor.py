"""SAM2 mask propagation across an ordered sequence of images."""

import contextlib
import gc
import os
import tempfile
from pathlib import Path
from typing import Callable, Dict, Generator, List, Optional, Set, Tuple

import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor  # type: ignore[import-not-found]

from src.logging_config import get_logger
from src.sam2.config import cfg_for_ckpt

logger = get_logger(__name__)


def _cleanup_gpu():
    """Force-release GPU memory held by the video predictor."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def propagate_masks_in_video(
    ckpt_path: str,
    image_paths: List[Path],
    conditioning_masks: Dict[int, np.ndarray],
    device: str = "cuda",
    offload_to_cpu: bool = True,
    cancel_check: Optional[Callable[[], bool]] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Generator[Tuple[int, np.ndarray], None, None]:
    """Run SAM2 video propagation and yield masks for unlabelled frames.

    The function creates a temporary directory of numbered JPEG symlinks
    (the format SAM2 expects), builds a ``SAM2VideoPredictor``, adds the
    user-provided conditioning masks, and propagates **forward then
    backward** so that all frames receive a mask.

    Args:
        - ckpt_path (str): Path to the SAM2 checkpoint file.
        - image_paths (list[Path]): Ordered list of image file paths
          (chronological / filename order).
        - conditioning_masks (dict[int, np.ndarray]): Mapping of frame
          index → binary mask (uint8, 0/255) at **original** image
          resolution.  These are the manually annotated "anchor" frames.
        - device (str): Torch device string (``"cuda"`` or ``"cpu"``).
        - offload_to_cpu (bool): Keep video frames on CPU to save VRAM.
        - cancel_check (callable | None): Called between frames; if it
          returns ``True`` the generator stops early.
        - progress_callback (callable | None): ``(current, total, msg)``
          called during long-running phases for UI updates.

    Yields:
        - tuple[int, np.ndarray]: ``(frame_idx, mask)`` for every
          non-conditioning frame that received a propagated mask.
          *mask* is uint8 with values 0 or 255 at the original video
          resolution.

    Raises:
        - ValueError: If inputs are empty or invalid.
    """
    logger.info("propagate_masks_in_video: %s frames, %s conditioning masks", len(image_paths), len(conditioning_masks))
    if not image_paths:
        raise ValueError("No image paths provided")
    if not conditioning_masks:
        raise ValueError("No conditioning masks provided")

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        logger.info("CUDA not available, using CPU for video propagation")

    cfg = cfg_for_ckpt(ckpt_path)
    num_frames = len(image_paths)
    logger.debug("propagate_masks_in_video: cfg=%s", cfg)
    conditioning_indices: Set[int] = set(conditioning_masks.keys())

    # Estimate the actual amount of work more closely than ``num_frames * 2``.
    # SAM2 propagates forward from the earliest conditioning frame and backward
    # from the latest one, so one of the passes can be almost empty.
    min_conditioning_idx = min(conditioning_indices)
    max_conditioning_idx = max(conditioning_indices)
    total_steps = (num_frames - min_conditioning_idx) + (max_conditioning_idx + 1)

    # ---- 1. Temporary directory with numbered JPEG symlinks -------------
    with tempfile.TemporaryDirectory(prefix="sam2_video_") as tmp_dir:
        if progress_callback:
            # total=0 → indeterminate / busy mode
            progress_callback(0, 0, "Preparing video frames…")

        for idx, img_path in enumerate(image_paths):
            link_name = os.path.join(tmp_dir, f"{idx:05d}.jpg")
            os.symlink(str(img_path.resolve()), link_name)

        # ---- 2. Build the video predictor --------------------------------
        if progress_callback:
            progress_callback(0, 0, "Loading SAM2 video model…")

        logger.debug("Building SAM2 video predictor (ckpt=%s, device=%s)", ckpt_path, device)
        predictor = build_sam2_video_predictor(
            cfg,
            ckpt_path,
            device=device,
        )
        logger.info("SAM2 video predictor loaded")

        try:
            # ---- 3. Initialise inference state ----------------------------
            if progress_callback:
                progress_callback(0, 0, "Loading video frames…")

            inference_state = predictor.init_state(
                video_path=tmp_dir,
                offload_video_to_cpu=offload_to_cpu,
            )

            # ---- 4. Add conditioning masks --------------------------------
            if progress_callback:
                progress_callback(0, 0, "Adding conditioning masks…")

            for frame_idx, mask in conditioning_masks.items():
                mask_bool = mask > 127 if mask.dtype == np.uint8 else mask > 0
                predictor.add_new_mask(
                    inference_state,
                    frame_idx=frame_idx,
                    obj_id=1,
                    mask=mask_bool,
                )

            # ---- 5. Forward propagation ----------------------------------
            step = 0
            if progress_callback:
                progress_callback(0, total_steps, "Propagating forward…")

            for frame_idx, _obj_ids, video_res_masks in predictor.propagate_in_video(inference_state):
                if cancel_check and cancel_check():
                    break

                step += 1
                if progress_callback:
                    progress_callback(step, total_steps, "Propagating forward…")

                if frame_idx not in conditioning_indices:
                    mask_np = (video_res_masks[0] > 0.0).cpu().numpy().squeeze().astype(np.uint8) * 255
                    yield frame_idx, mask_np

            # ---- 6. Backward propagation ---------------------------------
            if not (cancel_check and cancel_check()):
                if progress_callback:
                    progress_callback(step, total_steps, "Propagating backward…")

                for frame_idx, _obj_ids, video_res_masks in predictor.propagate_in_video(
                    inference_state,
                    reverse=True,
                ):
                    if cancel_check and cancel_check():
                        break

                    step += 1
                    if progress_callback:
                        progress_callback(step, total_steps, "Propagating backward…")

                    if frame_idx not in conditioning_indices:
                        mask_np = (video_res_masks[0] > 0.0).cpu().numpy().squeeze().astype(np.uint8) * 255
                        yield frame_idx, mask_np

        finally:
            # ---- 7. Cleanup — runs even on cancellation / error ----------
            del predictor
            with contextlib.suppress(UnboundLocalError):
                del inference_state
            _cleanup_gpu()

"""Wrapper around the SAM3 image predictor with image scaling support."""

import gc
import importlib
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from src.logging_config import get_logger
from src.models.predictor_base import BasePredictor
from src.utils.check_packages import check_sam3_installed

logger = get_logger(__name__)


class SAM3PredictorWrapper(BasePredictor):
    """High-level wrapper for SAM3 that handles image scaling transparently.

    Args:
        - checkpoint_path (str | None): Path to local checkpoint file.
        - bpe_path (str | None): Optional path to BPE tokenizer.
        - device (str): Torch device string ("cuda" or "cpu").
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        bpe_path: Optional[str] = None,
        device: str = "cuda",
    ):
        """Initialise the SAM3 predictor.

        Args:
            - checkpoint_path (str | None): Path to local checkpoint file.
            - bpe_path (str | None): Path to BPE tokenizer file.
            - device (str): Torch device ("cuda" or "cpu").
              Falls back to CPU if CUDA is unavailable.
        """
        logger.debug("SAM3PredictorWrapper.__init__(checkpoint_path=%s, bpe_path=%s, device=%s)", checkpoint_path, bpe_path, device)
        # Check if SAM3 is installed
        is_installed, error_msg = check_sam3_installed()
        if not is_installed:
            logger.error("SAM3 package not installed: %s", error_msg)
            raise ImportError(f"SAM3 package not installed. {error_msg}")

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
            logger.info("CUDA not available, using CPU for SAM3")

        self.device = device
        self.checkpoint_path = checkpoint_path
        self.bpe_path = bpe_path
        self._load_from_hf = not bool(checkpoint_path)

        logger.info("Loading SAM3 model (checkpoint=%s, from_hf=%s)", checkpoint_path, self._load_from_hf)

        if bpe_path:
            bpe_file = Path(bpe_path)
            if not bpe_file.exists():
                raise FileNotFoundError(f"SAM3 BPE file not found: {bpe_path}")
            if not bpe_file.name.endswith(".txt.gz"):
                raise ValueError(
                    "Invalid SAM3 BPE file. Expected a gzip vocabulary file "
                    "like 'bpe_simple_vocab_16e6.txt.gz'. "
                    "Leave BPE path empty to use SAM3 built-in defaults."
                )
            logger.info("BPE tokenizer: %s", bpe_path)
        else:
            logger.debug("No BPE path; using SAM3 built-in defaults")

        logger.debug("Calling _build_sam3_runtime(device=%s)", device)
        self._build_sam3_runtime(device)
        logger.info("SAM3 runtime built successfully")

        self.current_image: Optional[np.ndarray] = None
        self.original_image: Optional[np.ndarray] = None
        self.original_size: Optional[Tuple[int, int]] = None
        self.scaled_size: Optional[Tuple[int, int]] = None
        self.scale_factor: float = 1.0
        self.max_side: int = 0
        self.inference_state: Optional[dict] = None
        self._image_state_ready: bool = False

    def _build_sam3_runtime(self, device: str) -> None:
        """Build/rebuild the SAM3 model and processors on the requested device."""
        logger.debug("_build_sam3_runtime: importing sam3.model_builder")
        try:
            sam3_model_builder = importlib.import_module("sam3.model_builder")
            build_sam3_image_model = sam3_model_builder.build_sam3_image_model
        except Exception as e:
            logger.error("Failed to import sam3.model_builder: %s", e, exc_info=True)
            raise

        self.device = device
        logger.debug("_build_sam3_runtime: calling build_sam3_image_model(...)")
        try:
            self.model = build_sam3_image_model(
                bpe_path=self.bpe_path,
                device=device,
                eval_mode=True,
                checkpoint_path=self.checkpoint_path,
                load_from_HF=self._load_from_hf,
                enable_segmentation=True,
                # Prompt-only mode matches the lightweight image example from the
                # official SAM3 README and avoids loading the extra interactive path.
                enable_inst_interactivity=False,
                compile=False,
            )
        except Exception as e:
            logger.error("build_sam3_image_model failed: %s", e, exc_info=True)
            raise

        logger.debug("_build_sam3_runtime: loading Sam3Processor")
        # Different sam3 package revisions expose this class in different modules.
        try:
            from sam3.model.sam3_image_processor import Sam3Processor  # type: ignore

            logger.debug("_build_sam3_runtime: using sam3.model.sam3_image_processor.Sam3Processor")
        except ImportError as e1:
            logger.debug("Sam3Processor not in sam3.model.sam3_image_processor: %s, trying sam3.processor", e1)
            try:
                from sam3.processor import SAM3Processor as Sam3Processor  # type: ignore

                logger.debug("_build_sam3_runtime: using sam3.processor.SAM3Processor")
            except ImportError as e2:
                logger.error("Could not import Sam3Processor from sam3.model or sam3.processor: %s; %s", e1, e2, exc_info=True)
                raise
        try:
            self.processor = Sam3Processor(self.model, device=device)
        except Exception as e:
            logger.error("Sam3Processor(model, device=%s) failed: %s", device, e, exc_info=True)
            raise
        self.inference_state = None
        self._image_state_ready = False
        logger.debug("_build_sam3_runtime: done")

    def _fallback_to_cpu(self) -> None:
        """Rebuild SAM3 on CPU after a CUDA OOM."""
        if self.device == "cpu":
            raise RuntimeError("SAM3 is already running on CPU and still failed to allocate memory.")
        logger.warning("SAM3 ran out of CUDA memory. Rebuilding on CPU.")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._build_sam3_runtime("cpu")

    def set_max_side(self, max_side: int) -> None:
        """Set the maximum side length used when loading images.

        Args:
            - max_side (int): Max pixels on the longest side (0 = no limit).
        """
        self.max_side = max_side

    def load_image(self, image_path: str) -> Tuple[np.ndarray, Tuple[int, int], float]:
        """Load an image from disk, optionally down-scale, and set it on the predictor.

        Args:
            - image_path (str): Filesystem path to the image file.

        Returns:
            - tuple: ``(scaled_image, original_size, scale_factor)`` where
              *original_size* is ``(h, w)`` and *scale_factor* <= 1.0.

        Raises:
            - ValueError: If the image cannot be read.
        """
        logger.debug("load_image: reading %s", image_path)
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            logger.error("Could not load image: %s", image_path)
            raise ValueError(f"Could not load image: {image_path}")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        logger.debug("load_image: set_image_from_array (max_side=%s)", self.max_side)
        self.set_image_from_array(img_rgb, self.max_side)
        # Return tuple for compatibility with load_image interface
        if self.current_image is None:
            raise RuntimeError("Failed to set image")
        return (
            self.current_image,
            self.original_size if self.original_size else (0, 0),
            self.scale_factor,
        )

    def set_image_from_array(self, image: np.ndarray, max_side: int = 0) -> None:
        """Set an image directly from a numpy array.

        Args:
            - image (np.ndarray): RGB image array of shape ``(H, W, 3)``.
            - max_side (int): Max pixels on the longest side (0 = no limit).
        """
        self.max_side = max_side
        h0, w0 = image.shape[:2]
        self.original_size = (h0, w0)

        img_scaled = image
        scale = 1.0

        if max_side > 0:
            long_side = max(h0, w0)
            if long_side > max_side:
                scale = max_side / float(long_side)
                new_w = int(round(w0 * scale))
                new_h = int(round(h0 * scale))
                img_scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        h, w = img_scaled.shape[:2]
        self.scaled_size = (h, w)
        self.scale_factor = scale
        self.original_image = image
        self.current_image = img_scaled

        # Defer SAM3 embedding computation until a specific workflow is used
        # (text prompt or interactive prompts). Keeping both active at once can
        # double VRAM usage and trigger OOM on medium-sized GPUs.
        self.inference_state = None
        self._image_state_ready = False

    def _ensure_image_state_ready(self) -> None:
        """Prepare SAM3 shared image inference state on demand."""
        if self.current_image is None:
            raise RuntimeError("No image set. Call load_image() or set_image_from_array() first.")
        if self._image_state_ready and self.inference_state is not None:
            return

        pil_image = Image.fromarray(self.current_image)
        try:
            self.inference_state = self.processor.set_image(pil_image)
        except torch.OutOfMemoryError:
            self._fallback_to_cpu()
            pil_image = Image.fromarray(self.current_image)
            self.inference_state = self.processor.set_image(pil_image)
        self._image_state_ready = True

    def predict_mask(
        self,
        point_coords: List[Tuple[int, int]],
        point_labels: List[int],
        multimask_output: bool = True,
        mask_input: Optional[np.ndarray] = None,
    ) -> Tuple[List[np.ndarray], np.ndarray, Optional[np.ndarray]]:
        """Predict segmentation masks from keypoint prompts.

        Args:
            - point_coords (list[tuple[int, int]]): Keypoint ``(x, y)`` positions
              in the scaled image coordinate space.
            - point_labels (list[int]): Labels per point (1 = positive, 0 = negative).
            - multimask_output (bool): If True, return multiple candidate masks.
            - mask_input (np.ndarray | None): Not used in SAM3 (kept for compatibility).

        Returns:
            - tuple: ``(binary_masks, scores, logits)`` where *binary_masks* is a
              list of ``uint8`` arrays (0/255), *scores* is a 1-D float array,
              and *logits* is None (SAM3 doesn't return logits in the same format).
        """
        if self.current_image is None:
            raise RuntimeError("No image set. Call load_image() or set_image_from_array() first.")

        interactive_predictor = getattr(self.model, "inst_interactive_predictor", None)
        if interactive_predictor is None:
            raise RuntimeError(
                "SAM3 interactive point prompts are disabled in this app build. "
                "Use text prompts or SAM2 clicks instead."
            )

        self._ensure_image_state_ready()

        if not point_coords:
            h, w = self.scaled_size if self.scaled_size else (0, 0)
            empty_mask = np.zeros((h, w), dtype=np.uint8)
            return [empty_mask], np.array([0.0]), None

        h, w = self.scaled_size if self.scaled_size else (0, 0)
        if h == 0 or w == 0:
            h, w = self.current_image.shape[:2]

        points = np.array(point_coords, dtype=np.float32)
        labels = np.array(point_labels, dtype=np.int32)
        if points.ndim == 1:
            points = points.reshape(1, -1)

        try:
            masks, scores, logits = self.model.predict_inst(
                self.inference_state,
                point_coords=points,
                point_labels=labels,
                mask_input=mask_input,
                multimask_output=multimask_output,
                return_logits=True,
                normalize_coords=True,
            )
        except torch.OutOfMemoryError:
            self._fallback_to_cpu()
            self._ensure_image_state_ready()
            masks, scores, logits = self.model.predict_inst(
                self.inference_state,
                point_coords=points,
                point_labels=labels,
                mask_input=mask_input,
                multimask_output=multimask_output,
                return_logits=True,
                normalize_coords=True,
            )

        mask_binaries: List[np.ndarray] = []
        for mask in masks:
            mask_binary = (mask > 0).astype(np.uint8) * 255
            mask_binaries.append(mask_binary)

        return mask_binaries, scores.astype(np.float32), logits

    def predict_mask_from_text(self, prompt: str) -> Tuple[List[np.ndarray], np.ndarray]:
        """Predict segmentation masks from text prompt (SAM3-specific method).

        Args:
            - prompt (str): Text prompt describing what to segment.

        Returns:
            - tuple: ``(binary_masks, scores)`` where *binary_masks* is a list
              of ``uint8`` arrays (0/255) and *scores* is a 1-D float array.
        """
        if self.current_image is None:
            raise RuntimeError("No image set. Call load_image() or set_image_from_array() first.")

        self._ensure_image_state_ready()

        if not prompt or not prompt.strip():
            h, w = self.scaled_size if self.scaled_size else (0, 0)
            empty_mask = np.zeros((h, w), dtype=np.uint8)
            return [empty_mask], np.array([0.0])

        # Use SAM3 text prompt API
        try:
            output = self.processor.set_text_prompt(state=self.inference_state, prompt=prompt)
        except torch.OutOfMemoryError:
            self._fallback_to_cpu()
            self._ensure_image_state_ready()
            output = self.processor.set_text_prompt(state=self.inference_state, prompt=prompt)

        if output is None:
            h, w = self.scaled_size if self.scaled_size else (0, 0)
            empty_mask = np.zeros((h, w), dtype=np.uint8)
            return [empty_mask], np.array([0.0])

        masks_raw = output.get("masks", []) if isinstance(output, dict) else []
        scores_raw = output.get("scores", []) if isinstance(output, dict) else []

        # Convert masks to binary format (uint8, 0/255)
        binary_masks = []
        mask_items: List[np.ndarray | torch.Tensor] = []
        if masks_raw is None:
            mask_items = []
        elif isinstance(masks_raw, torch.Tensor):
            # SAM3 may return a single tensor mask or a batched tensor mask.
            masks_np = masks_raw.detach().cpu().numpy()
            if masks_np.size == 0:
                mask_items = []
            elif masks_np.ndim == 2:
                mask_items = [masks_np]
            elif masks_np.ndim == 3:
                mask_items = [masks_np[i] for i in range(masks_np.shape[0])]
            else:
                # Flatten leading dims while keeping spatial HW dimensions.
                h, w = masks_np.shape[-2], masks_np.shape[-1]
                mask_items = list(masks_np.reshape(-1, h, w))
        elif isinstance(masks_raw, (list, tuple)):
            mask_items = list(masks_raw)
        else:
            mask_items = [masks_raw]

        if len(mask_items) == 0:
            h, w = self.scaled_size if self.scaled_size else (0, 0)
            if h == 0 or w == 0:
                h, w = 256, 256  # Fallback size
            empty_mask = np.zeros((h, w), dtype=np.uint8)
            return [empty_mask], np.array([0.0])

        for mask in mask_items:
            if mask is None:
                continue
            mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else np.array(mask)

            # Ensure mask is 2D
            if mask_np.ndim > 2:
                mask_np = mask_np.squeeze()

            # Convert to binary (0/255)
            binary_mask = (mask_np > 0.5).astype(np.uint8) * 255
            binary_masks.append(binary_mask)

        # Convert scores to numpy array
        if isinstance(scores_raw, (list, tuple)):
            scores_array = np.array(scores_raw, dtype=np.float32).reshape(-1)
        elif isinstance(scores_raw, torch.Tensor):
            scores_array = scores_raw.detach().cpu().numpy().reshape(-1)
        else:
            scores_array = np.array([1.0] * len(binary_masks), dtype=np.float32)

        if scores_array.size == 0:
            scores_array = np.array([1.0] * len(binary_masks), dtype=np.float32)
        elif scores_array.size < len(binary_masks):
            pad = np.array([1.0] * (len(binary_masks) - scores_array.size), dtype=np.float32)
            scores_array = np.concatenate([scores_array, pad], axis=0)
        elif scores_array.size > len(binary_masks):
            scores_array = scores_array[: len(binary_masks)]

        if len(binary_masks) == 0:
            h, w = self.scaled_size if self.scaled_size else (0, 0)
            empty_mask = np.zeros((h, w), dtype=np.uint8)
            return [empty_mask], np.array([0.0])

        return binary_masks, scores_array

    def upscale_mask(self, mask: np.ndarray) -> np.ndarray:
        """Scale a mask up to the original image resolution.

        Args:
            - mask (np.ndarray): Binary mask at the scaled size.

        Returns:
            - np.ndarray: Binary mask at the original size.
        """
        if self.original_size is None:
            return mask

        h0, w0 = self.original_size
        h, w = mask.shape[:2]

        if h == h0 and w == w0:
            return mask

        return cv2.resize(mask, (w0, h0), interpolation=cv2.INTER_NEAREST)

    def downscale_mask(self, mask: np.ndarray) -> np.ndarray:
        """Scale a mask down from original resolution to the scaled size.

        Args:
            - mask (np.ndarray): Binary mask at the original size.

        Returns:
            - np.ndarray: Binary mask at the scaled size.
        """
        if self.scaled_size is None:
            return mask

        h, w = self.scaled_size
        h0, w0 = mask.shape[:2]

        if h == h0 and w == w0:
            return mask

        return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    def get_scaled_size(self) -> Tuple[int, int]:
        """Return the current scaled image dimensions as ``(h, w)``."""
        return self.scaled_size if self.scaled_size else (0, 0)

    def get_original_size(self) -> Tuple[int, int]:
        """Return the original image dimensions as ``(h, w)``."""
        return self.original_size if self.original_size else (0, 0)

    def get_scale_factor(self) -> float:
        """Return the current scale factor (original -> scaled)."""
        return self.scale_factor

    def mask_to_logits(self, mask: np.ndarray) -> Optional[np.ndarray]:
        """Convert a binary mask to low-resolution logits for use as ``mask_input``.

        Note: SAM3 doesn't use logits in the same way as SAM2, but we provide
        this method for compatibility with the base interface.

        Args:
            - mask (np.ndarray): Binary mask ``(H, W)`` with values 0 or 255.

        Returns:
            - np.ndarray | None: Logits of shape ``(1, 256, 256)``, or ``None``
              on failure.
        """
        if mask is None:
            return None

        try:
            mask_resized = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
            mask_float = (mask_resized.astype(np.float32) / 255.0) * 12.0 - 6.0
            logits = mask_float[np.newaxis, :, :]
            return logits
        except Exception as e:
            logger.error("Error converting mask to logits: %s", e)
            return None

    def release(self) -> None:
        """Free model references and clear cached CUDA memory."""
        self.current_image = None
        self.original_image = None
        self.original_size = None
        self.scaled_size = None
        self.inference_state = None
        self.processor = None
        self.interactive_predictor = None
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

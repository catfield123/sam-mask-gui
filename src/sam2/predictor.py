"""Wrapper around the SAM2 image predictor with image scaling support."""

import gc
import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2  # type: ignore[import-not-found]
from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore[import-not-found]

from src.models.predictor_base import BasePredictor
from src.sam2.config import cfg_for_ckpt

logger = logging.getLogger(__name__)


class SAM2PredictorWrapper(BasePredictor):
    """High-level wrapper for SAM2 that handles image scaling transparently.

    Args:
        - ckpt_path (str): Path to the SAM2 ``.pt`` checkpoint file.
        - device (str): Torch device string (``"cuda"`` or ``"cpu"``).
    """

    def __init__(self, ckpt_path: str, device: str = "cuda"):
        """Initialise the SAM2 predictor.

        Args:
            - ckpt_path (str): Path to the SAM2 checkpoint file.
            - device (str): Torch device (``"cuda"`` or ``"cpu"``).
              Falls back to CPU if CUDA is unavailable.
        """
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
            logger.info("CUDA not available, using CPU")

        self.device = device
        self.ckpt_path = ckpt_path
        self.cfg = cfg_for_ckpt(ckpt_path)

        logger.info("Loading SAM2 model (checkpoint=%s, config=%s)", ckpt_path, self.cfg)
        sam2_model = build_sam2(self.cfg, ckpt_path, device=device)
        self.predictor = SAM2ImagePredictor(sam2_model)

        self.current_image: Optional[np.ndarray] = None
        self.original_image: Optional[np.ndarray] = None
        self.original_size: Optional[Tuple[int, int]] = None
        self.scaled_size: Optional[Tuple[int, int]] = None
        self.scale_factor: float = 1.0
        self.max_side: int = 0

    def set_max_side(self, max_side: int):
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
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise ValueError(f"Could not load image: {image_path}")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h0, w0 = img_rgb.shape[:2]
        self.original_size = (h0, w0)

        img_scaled = img_rgb
        scale = 1.0

        if self.max_side > 0:
            long_side = max(h0, w0)
            if long_side > self.max_side:
                scale = self.max_side / float(long_side)
                new_w = int(round(w0 * scale))
                new_h = int(round(h0 * scale))
                img_scaled = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

        h, w = img_scaled.shape[:2]
        self.scaled_size = (h, w)
        self.scale_factor = scale
        self.original_image = img_rgb
        self.current_image = img_scaled

        self.predictor.set_image(img_scaled)
        return img_scaled, (h0, w0), scale

    def set_image_from_array(self, image: np.ndarray, max_side: int = 0):
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

        self.predictor.set_image(img_scaled)

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
            - mask_input (np.ndarray | None): Low-resolution logits ``(1, H, W)``
              from a previous prediction for iterative refinement.

        Returns:
            - tuple: ``(binary_masks, scores, logits)`` where *binary_masks* is a
              list of ``uint8`` arrays (0/255), *scores* is a 1-D float array,
              and *logits* is the raw logit tensor or ``None``.

        Raises:
            - RuntimeError: If no image has been set yet.
        """
        if self.current_image is None:
            raise RuntimeError("No image set. Call load_image() or set_image_from_array() first.")

        if not point_coords:
            if self.scaled_size is None:
                raise RuntimeError("No image set. Call load_image() or set_image_from_array() first.")
            h, w = self.scaled_size
            empty_mask = np.zeros((h, w), dtype=np.uint8)
            return [empty_mask], np.array([0.0]), None

        points = np.array(point_coords, dtype=np.float32)
        labels = np.array(point_labels, dtype=np.int32)

        if points.ndim == 1:
            points = points.reshape(1, -1)

        h, w = self.scaled_size if self.scaled_size else (0, 0)

        if len(points) > 0:
            logger.debug(
                "SAM2 input: points shape=%s, labels=%s, image_size=(%s, %s)",
                points.shape,
                labels,
                w,
                h,
            )
            if points[0, 0] < 0 or points[0, 0] >= w or points[0, 1] < 0 or points[0, 1] >= h:
                logger.warning("Point out of bounds: point=%s, image_size=(%s, %s)", points[0], w, h)

        logger.debug("Using mask_input: %s", mask_input is not None)
        if mask_input is not None:
            logger.debug("mask_input shape: %s", mask_input.shape)

        masks, scores, logits = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            mask_input=mask_input,
            multimask_output=multimask_output,
            normalize_coords=True,
        )
        logger.debug("SAM2 output: %s masks, scores=%s", len(masks), scores)

        mask_binaries = []
        for mask in masks:
            mask_binary = (mask > 0).astype(np.uint8) * 255
            mask_binaries.append(mask_binary)

        return mask_binaries, scores, logits

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

    def scale_point_to_image(self, x: int, y: int) -> Tuple[int, int]:
        """Convert display-space coordinates to scaled-image-space coordinates.

        Args:
            - x (int): X coordinate in display space.
            - y (int): Y coordinate in display space.

        Returns:
            - tuple[int, int]: ``(x, y)`` in scaled image space.
        """
        return (x, y)

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

        The mask is resized to 256×256 and converted into a soft prior rather
        than a hard 0/1 bitmap. Using linear interpolation and log-odds keeps
        boundaries smoother, which makes SAM2 refinement behave much better
        when the starting mask came from SAM3 text prompting.

        Args:
            - mask (np.ndarray): Binary mask ``(H, W)`` with values 0 or 255.

        Returns:
            - np.ndarray | None: Logits of shape ``(1, 256, 256)``, or ``None``
              on failure.
        """
        if mask is None:
            return None

        try:
            mask_float = mask.astype(np.float32)
            if mask_float.max() > 1.0:
                mask_float /= 255.0

            mask_resized = cv2.resize(mask_float, (256, 256), interpolation=cv2.INTER_LINEAR)
            mask_resized = cv2.GaussianBlur(mask_resized, (0, 0), sigmaX=1.0, sigmaY=1.0)
            mask_prob = np.clip(mask_resized, 1e-4, 1.0 - 1e-4)
            logits = np.log(mask_prob / (1.0 - mask_prob)).astype(np.float32)[np.newaxis, :, :]
            logits = np.clip(logits, -8.0, 8.0)
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
        self.predictor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

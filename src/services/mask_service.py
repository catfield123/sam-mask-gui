"""Service for loading, saving, and rescaling segmentation masks."""

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from src.logging_config import get_logger
from src.models.predictor_base import BasePredictor
from src.services.image_service import ImageService

logger = get_logger(__name__)


def _scaled_size_for_image(original_height: int, original_width: int, max_side: int) -> Tuple[int, int]:
    """Compute (height, width) after scaling by max_side (0 = no limit)."""
    if max_side <= 0:
        return (original_height, original_width)
    long_side = max(original_height, original_width)
    if long_side <= max_side:
        return (original_height, original_width)
    scale = max_side / float(long_side)
    new_w = int(round(original_width * scale))
    new_h = int(round(original_height * scale))
    return (new_h, new_w)


class MaskService:
    """Handles mask I/O and rescaling per image (original size on disk, scaled in memory).

    Args:
        - predictor (BasePredictor): Active predictor used for in-memory scaling helpers.
    """

    def __init__(self, predictor: BasePredictor):
        """Initialise the mask service.

        Args:
            - predictor (BasePredictor): Predictor used for in-memory scaling (e.g. downscale_mask).
        """
        self.predictor = predictor
        logger.debug("MaskService initialised with predictor=%s", type(predictor).__name__)

    def load_mask(
        self,
        mask_path: Path,
        image_path: Optional[Path] = None,
        max_side: int = 0,
    ) -> Optional[np.ndarray]:
        """Load a grayscale mask from disk, optionally downscaled to match image scaling.

        On disk masks are stored at original image resolution. When ``image_path`` and
        ``max_side`` are provided, the mask is resized down to the same dimensions used
        for that image in the app (so it matches the scaled image / predictor).

        Args:
            - mask_path (Path): Path to the PNG mask file.
            - image_path (Path | None): Source image path; used to get original size and
              compute scaled size when ``max_side`` > 0.
            - max_side (int): Max length of the longer side (0 = no downscaling).

        Returns:
            - np.ndarray | None: Grayscale mask (at original or scaled size), or ``None`` on failure.
        """
        logger.debug("load_mask: %s", mask_path)
        try:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                logger.debug("load_mask: failed to read %s", mask_path)
                return None
            if image_path is not None and max_side > 0:
                size = ImageService.get_image_size(image_path)
                if size is not None:
                    h0, w0 = size
                    target_h, target_w = _scaled_size_for_image(h0, w0, max_side)
                    if (mask.shape[0], mask.shape[1]) != (target_h, target_w):
                        mask = cv2.resize(
                            mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST
                        )
            return mask
        except Exception as e:
            logger.debug("load_mask: exception for %s: %s", mask_path, e)
            return None

    def save_mask(
        self,
        mask: np.ndarray,
        mask_path: Path,
        image_path: Path,
    ) -> bool:
        """Save a mask to disk at the same resolution as the source image.

        The mask (at scaled size in memory) is upscaled to the original image dimensions
        so that the saved file has the same width and height in pixels as the source image.

        Args:
            - mask (np.ndarray): Binary mask at the scaled size.
            - mask_path (Path): Destination file path.
            - image_path (Path): Source image path; its dimensions define the save size.

        Returns:
            - bool: ``True`` on success, ``False`` on failure.
        """
        logger.debug("save_mask: %s (shape=%s)", mask_path, mask.shape if mask is not None else None)
        try:
            size = ImageService.get_image_size(image_path)
            if size is None:
                logger.error("save_mask: could not read image size for %s", image_path)
                return False
            h0, w0 = size
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            if mask.shape[0] != h0 or mask.shape[1] != w0:
                mask = cv2.resize(mask, (w0, h0), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(mask_path), mask)
            logger.info("Mask saved: %s", mask_path)
            return True
        except Exception as e:
            logger.error("Error saving mask: %s", e)
            return False

    @staticmethod
    def grow_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
        """Grow (dilate) or shrink (erode) a binary mask by a given pixel radius.

        Uses a circular (elliptical) structuring element so the boundary
        expands/contracts uniformly along the contour rather than in a
        square pattern.

        Args:
            - mask (np.ndarray): Binary mask (uint8, 0/255).
            - pixels (int): Number of pixels to grow (positive) or
              shrink (negative).  Zero returns the mask unchanged.

        Returns:
            - np.ndarray: Modified mask with the same dtype and shape.
        """
        if pixels == 0 or mask is None:
            return mask
        radius = abs(pixels)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * radius + 1, 2 * radius + 1),
        )
        if pixels > 0:
            return cv2.dilate(mask, kernel, iterations=1)
        return cv2.erode(mask, kernel, iterations=1)

    def downscale_mask(self, mask: np.ndarray) -> np.ndarray:
        """Down-scale a mask from original to the current scaled image size.

        Args:
            - mask (np.ndarray): Mask at original resolution.

        Returns:
            - np.ndarray: Mask at the scaled resolution.
        """
        return self.predictor.downscale_mask(mask)

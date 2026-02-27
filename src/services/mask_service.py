"""Service for loading, saving, and rescaling segmentation masks."""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.models.predictor_base import BasePredictor

logger = logging.getLogger(__name__)


class MaskService:
    """Handles mask I/O and delegates rescaling to the SAM2 predictor wrapper.

    Args:
        - predictor (BasePredictor): Active predictor used for up/down-scaling.
    """

    def __init__(self, predictor: BasePredictor):
        """Initialise the mask service.

        Args:
            - predictor (BasePredictor): Predictor used for mask scaling.
        """
        self.predictor = predictor

    def load_mask(self, mask_path: Path) -> Optional[np.ndarray]:
        """Load a grayscale mask from disk.

        Args:
            - mask_path (Path): Path to the PNG mask file.

        Returns:
            - np.ndarray | None: Grayscale mask array, or ``None`` on failure.
        """
        try:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            return mask if mask is not None else None
        except Exception:
            return None

    def save_mask(self, mask: np.ndarray, mask_path: Path) -> bool:
        """Save a mask to disk after up-scaling to original resolution.

        Args:
            - mask (np.ndarray): Binary mask at the scaled size.
            - mask_path (Path): Destination file path.

        Returns:
            - bool: ``True`` on success, ``False`` on failure.
        """
        try:
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            mask_upscaled = self.predictor.upscale_mask(mask)
            cv2.imwrite(str(mask_path), mask_upscaled)
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

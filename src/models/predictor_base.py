"""Base abstract class for predictor wrappers (SAM2, SAM3, etc.)."""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np


class BasePredictor(ABC):
    """Abstract base class for all predictor wrappers.

    This class defines the common interface that all predictors (SAM2, SAM3, etc.)
    must implement to work with the application's mask service and controllers.
    """

    @abstractmethod
    def set_max_side(self, max_side: int) -> None:
        """Set the maximum side length used when loading images.

        Args:
            - max_side (int): Max pixels on the longest side (0 = no limit).
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def set_image_from_array(self, image: np.ndarray, max_side: int = 0) -> None:
        """Set an image directly from a numpy array.

        Args:
            - image (np.ndarray): RGB image array of shape ``(H, W, 3)``.
            - max_side (int): Max pixels on the longest side (0 = no limit).
        """
        pass

    @abstractmethod
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
        """
        pass

    @abstractmethod
    def upscale_mask(self, mask: np.ndarray) -> np.ndarray:
        """Scale a mask up to the original image resolution.

        Args:
            - mask (np.ndarray): Binary mask at the scaled size.

        Returns:
            - np.ndarray: Binary mask at the original size.
        """
        pass

    @abstractmethod
    def downscale_mask(self, mask: np.ndarray) -> np.ndarray:
        """Scale a mask down from original resolution to the scaled size.

        Args:
            - mask (np.ndarray): Binary mask at the original size.

        Returns:
            - np.ndarray: Binary mask at the scaled size.
        """
        pass

    @abstractmethod
    def get_scaled_size(self) -> Tuple[int, int]:
        """Return the current scaled image dimensions as ``(h, w)``."""
        pass

    @abstractmethod
    def get_original_size(self) -> Tuple[int, int]:
        """Return the original image dimensions as ``(h, w)``."""
        pass

    @abstractmethod
    def get_scale_factor(self) -> float:
        """Return the current scale factor (original -> scaled)."""
        pass

    @abstractmethod
    def mask_to_logits(self, mask: np.ndarray) -> Optional[np.ndarray]:
        """Convert a binary mask to low-resolution logits for use as ``mask_input``.

        Args:
            - mask (np.ndarray): Binary mask ``(H, W)`` with values 0 or 255.

        Returns:
            - np.ndarray | None: Logits of shape ``(1, 256, 256)``, or ``None``
              on failure.
        """
        pass

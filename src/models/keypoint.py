"""Data models for keypoints used in image segmentation."""

from dataclasses import dataclass
from enum import Enum


class KeypointType(Enum):
    """Type of a keypoint (positive or negative prompt for SAM2)."""

    POSITIVE = 1
    NEGATIVE = 0


@dataclass
class Keypoint:
    """Single keypoint placed on an image.

    Args:
        - x (int): X coordinate in image pixel space.
        - y (int): Y coordinate in image pixel space.
        - type (KeypointType): Whether the keypoint is positive or negative.
    """

    x: int
    y: int
    type: KeypointType

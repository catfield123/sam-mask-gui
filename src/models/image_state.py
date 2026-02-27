"""Data model representing the per-image editing state."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from src.models.keypoint import Keypoint


@dataclass
class ImageState:
    """Mutable state for a single image in the segmentation workflow.

    Tracks keypoints, mask data, undo/redo history, and version tokens
    used to determine whether the current state has been saved to disk.

    Args:
        - path (Path): Filesystem path to the source image.
    """

    path: Path
    keypoints: List[Keypoint] = field(default_factory=list)
    mask: Optional[np.ndarray] = None
    mask_candidates: List[np.ndarray] = field(default_factory=list)
    mask_scores: Optional[np.ndarray] = None
    mask_logits_all: Optional[np.ndarray] = None
    selected_mask_idx: int = 0
    mask_logits: Optional[np.ndarray] = None
    mask_saved: bool = False
    has_unsaved_changes: bool = True
    original_size: Optional[Tuple[int, int]] = None
    scaled_size: Optional[Tuple[int, int]] = None
    scale_factor: float = 1.0

    # Version tokens: state_version is bumped on every mutation;
    # saved_version records state_version at last Ctrl+S (-1 = never saved).
    state_version: int = 0
    saved_version: int = -1

    # Undo/redo stacks store tuples:
    # (keypoints, mask_logits, has_loaded_mask, mask, mask_candidates,
    #  mask_scores, selected_mask_idx, state_version)
    undo_stack: List[tuple] = field(default_factory=list)
    redo_stack: List[tuple] = field(default_factory=list)

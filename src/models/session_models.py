"""Data models for tracking batch operation sessions and per-frame undo state."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


@dataclass
class FrameBackup:
    """Snapshot of a single frame's state taken before a batch operation.

    Stores everything needed to fully restore the frame on undo.

    Args:
        - image_path (Path): Path to the source image file.
        - frame_idx (int): Index of this frame in the ordered image list.
        - old_mask (np.ndarray | None): ``state.mask`` before the operation.
        - old_mask_saved (bool): ``state.mask_saved`` before the operation.
        - old_mask_logits (np.ndarray | None): ``state.mask_logits`` before
          the operation.
        - new_mask (np.ndarray | None): Resulting mask (uint8, 0/255).
    """

    image_path: Path
    frame_idx: int
    old_mask: Optional[np.ndarray] = None
    old_mask_saved: bool = False
    old_mask_logits: Optional[np.ndarray] = None
    new_mask: Optional[np.ndarray] = None


@dataclass
class BatchSession:
    """Tracks the last batch operation (mask propagation, mask grow, etc.)
    for unified save-all / revert-all support.

    Args:
        - operation_type (str): Identifier for the operation that created
          this session (e.g. ``"video_propagation"`` or ``"grow_mask"``).
        - frame_backups (list[FrameBackup]): Per-frame backup entries for
          every frame that was modified by the operation.
        - saved_to_disk (bool): Whether *Save All* has been executed.
        - undone (bool): Whether *Revert* has been executed.
        - saved_mask_paths (list[Path]): Mask file paths written by
          *Save All* (used for disk-level revert).
        - old_disk_masks (dict[Path, bytes | None]): For each affected mask
          path, the raw PNG bytes that existed on disk before *Save All*
          (``None`` means the file did not exist).
    """

    operation_type: str = "unknown"
    frame_backups: List[FrameBackup] = field(default_factory=list)
    saved_to_disk: bool = False
    undone: bool = False
    saved_mask_paths: List[Path] = field(default_factory=list)
    old_disk_masks: Dict[Path, Optional[bytes]] = field(default_factory=dict)

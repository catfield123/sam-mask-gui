"""Background thread for SAM2 mask propagation along an image sequence."""

from pathlib import Path
from typing import Dict, List

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal


class MaskPropagationWorker(QThread):
    """Run SAM2 mask propagation in a background thread.

    Signals:
        - progress(int, int, str): ``(current, total, message)``
        - frame_completed(int, object): ``(frame_idx, mask_ndarray)``
        - finished_all(): Emitted when propagation completes (or is cancelled).
        - error(str): Emitted on unrecoverable failure.
    """

    progress = pyqtSignal(int, int, str)
    frame_completed = pyqtSignal(int, object)
    finished_all = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(
        self,
        ckpt_path: str,
        image_paths: List[Path],
        conditioning_masks: Dict[int, np.ndarray],
        skip_indices: set,
        device: str = "cuda",
        parent=None,
    ):
        """Initialise the mask propagation worker.

        Args:
            - ckpt_path (str): Path to the SAM2 checkpoint.
            - image_paths (list[Path]): Ordered image file paths.
            - conditioning_masks (dict[int, np.ndarray]): Frame index →
              binary mask (uint8 0/255) at original resolution.
            - skip_indices (set[int]): Frame indices to skip (already have
              masks, non-conditioning).
            - device (str): Torch device.
            - parent (QObject | None): Parent object.
        """
        super().__init__(parent)
        self._ckpt_path = ckpt_path
        self._image_paths = image_paths
        self._conditioning_masks = conditioning_masks
        self._skip_indices = skip_indices
        self._device = device
        self._cancelled = False

    def cancel(self) -> None:
        """Request graceful cancellation of the propagation loop."""
        self._cancelled = True

    def _is_cancelled(self) -> bool:
        """Return whether cancellation has been requested.

        Returns:
            bool: True if cancel() was called.
        """
        return self._cancelled

    def run(self) -> None:
        """Execute the propagation (runs in the worker thread)."""
        try:
            from src.sam2.video_predictor import propagate_masks_in_video

            for frame_idx, mask in propagate_masks_in_video(
                ckpt_path=self._ckpt_path,
                image_paths=self._image_paths,
                conditioning_masks=self._conditioning_masks,
                device=self._device,
                offload_to_cpu=True,
                cancel_check=self._is_cancelled,
                progress_callback=self._on_progress,
            ):
                if self._cancelled:
                    break

                # Skip frames that already have masks (skip policy)
                if frame_idx in self._skip_indices:
                    continue

                self.frame_completed.emit(frame_idx, mask)

        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            self.finished_all.emit()

    def _on_progress(self, current: int, total: int, message: str) -> None:
        """Forward progress information to the UI thread.

        Args:
            - current (int): Current step index.
            - total (int): Total number of steps.
            - message (str): Progress message string.
        """
        self.progress.emit(current, total, message)

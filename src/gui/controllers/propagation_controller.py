"""Mask propagation session management: start propagation, delegate save/revert."""

from pathlib import Path
from typing import Dict, List, Optional, Set

import cv2
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMessageBox, QProgressDialog

from src.gui.controllers.batch_session_controller import BatchSessionController
from src.gui.workers.propagation_worker import MaskPropagationWorker
from src.models.image_state import ImageState
from src.models.session_models import BatchSession, FrameBackup


class PropagationController:
    """Manages mask propagation: validation, worker thread, and session
    creation.  Session save/revert is delegated to
    :class:`BatchSessionController`.

    Args:
        - get_state (callable): Returns ``(sam2_checkpoint_path, device,
          images_dir, save_dir, image_states, sort_index,
          image_list_widget, current_image_path)`` from the main window.
        - update_ui_cb (callable): Called after bulk state changes to
          refresh list labels, counter and the displayed image.
        - batch_session_ctrl (BatchSessionController): Controller that
          owns the unified batch session.
    """

    def __init__(
        self,
        get_state,
        update_ui_cb,
        batch_session_ctrl: BatchSessionController,
    ):
        """Initialise the propagation controller.

        Args:
            - get_state (callable): State accessor callback.
            - update_ui_cb (callable): UI refresh callback.
            - batch_session_ctrl (BatchSessionController): Shared session
              controller.
        """
        self._get_state = get_state
        self._update_ui = update_ui_cb
        self._batch_ctrl = batch_session_ctrl

        self._worker: Optional[MaskPropagationWorker] = None
        self._progress: Optional[QProgressDialog] = None
        self._starting_propagation: bool = False

        # Ordered image paths used in the last / current propagation
        self._ordered_paths: List[Path] = []
        # Frame indices that are conditioning (user-annotated anchors)
        self._conditioning_indices: Set[int] = set()
        # Pending backups being filled as frames complete
        self._pending_backups: Dict[int, FrameBackup] = {}
        # Session being built during propagation
        self._building_session: Optional[BatchSession] = None

    # ------------------------------------------------------------------
    # Public queries
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """Return whether propagation is starting or the worker thread is active.

        Returns:
            bool: True if propagation is in progress or starting.
        """
        return self._starting_propagation or (
            self._worker is not None and self._worker.isRunning()
        )

    # ------------------------------------------------------------------
    # Start propagation
    # ------------------------------------------------------------------

    def start_propagation(self, parent_widget) -> None:
        """Validate inputs, back up state, and launch the worker thread.

        Shows warning/information dialogs on validation failure. Does nothing
        if propagation is already running or starting.

        Args:
            - parent_widget (QWidget): Parent for dialogs and progress bar.
        """
        if self._starting_propagation or (
            self._worker is not None and self._worker.isRunning()
        ):
            return
        self._starting_propagation = True
        self._update_ui()

        state_tuple = self._get_state()
        (
            ckpt_path,
            device,
            _images_dir,
            save_dir,
            image_states,
            sort_index,
            image_list,
            _current_path,
            _predictor,
        ) = state_tuple

        # ---- validation --------------------------------------------------
        if sort_index != 0:
            self._starting_propagation = False
            self._update_ui()
            QMessageBox.warning(
                parent_widget,
                "Warning",
                'Propagate Masks requires images sorted "By name".',
            )
            return

        if not ckpt_path or not Path(ckpt_path).exists():
            self._starting_propagation = False
            self._update_ui()
            QMessageBox.warning(
                parent_widget,
                "Warning",
                "Please set a valid SAM2 checkpoint in Settings.",
            )
            return

        if not save_dir:
            self._starting_propagation = False
            self._update_ui()
            QMessageBox.warning(
                parent_widget,
                "Warning",
                "Please set a save folder first (File → Set Save Folder).",
            )
            return

        # Build ordered path list from the list widget (already sorted)
        ordered_paths: List[Path] = []
        for i in range(image_list.count()):
            item = image_list.item(i)
            if item is None:
                continue
            p = item.data(Qt.ItemDataRole.UserRole)
            if p is not None:
                ordered_paths.append(p)

        if not ordered_paths:
            self._starting_propagation = False
            self._update_ui()
            return

        # Identify conditioning frames (have mask) and target frames (no mask)
        conditioning_masks: Dict[int, np.ndarray] = {}
        skip_indices: Set[int] = set()
        target_count = 0

        for idx, img_path in enumerate(ordered_paths):
            state = image_states.get(img_path)
            if state is None:
                continue
            has_mask = state.mask is not None or state.mask_saved
            if has_mask and state.mask is not None:
                mask_full = self._upscale_mask_for_conditioning(state)
                conditioning_masks[idx] = mask_full
                skip_indices.add(idx)
            elif has_mask:
                skip_indices.add(idx)
            else:
                target_count += 1

        if not conditioning_masks:
            self._starting_propagation = False
            self._update_ui()
            QMessageBox.information(
                parent_widget,
                "No Conditioning Frames",
                "Please annotate at least one image with a mask before running Propagate Masks.",
            )
            return

        if target_count == 0:
            self._starting_propagation = False
            self._update_ui()
            QMessageBox.information(
                parent_widget,
                "Nothing to Propagate",
                "All images already have masks. There are no frames to propagate to.",
            )
            return

        # ---- back up pre-propagation state --------------------------------
        self._ordered_paths = ordered_paths
        self._conditioning_indices = set(conditioning_masks.keys())
        self._building_session = BatchSession(operation_type="video_propagation")

        self._pending_backups = {}
        for idx, img_path in enumerate(ordered_paths):
            if idx in skip_indices:
                continue
            state = image_states.get(img_path)
            if state is None:
                continue
            backup = FrameBackup(
                image_path=img_path,
                frame_idx=idx,
                old_mask=state.mask.copy() if state.mask is not None else None,
                old_mask_saved=state.mask_saved,
                old_mask_logits=(state.mask_logits.copy() if state.mask_logits is not None else None),
            )
            self._pending_backups[idx] = backup

        # ---- progress dialog ---------------------------------------------
        self._progress = QProgressDialog(
            "Preparing mask propagation…",
            "Cancel",
            0,
            0,
            parent_widget,
        )
        self._progress.setWindowTitle("Propagate Masks")
        self._progress.setWindowModality(Qt.WindowModality.WindowModal)
        self._progress.setMinimumDuration(0)

        # ---- launch worker -----------------------------------------------
        self._worker = MaskPropagationWorker(
            ckpt_path=ckpt_path,
            image_paths=ordered_paths,
            conditioning_masks=conditioning_masks,
            skip_indices=skip_indices,
            device=device,
            parent=None,
        )
        self._worker.progress.connect(self._on_worker_progress)
        self._worker.frame_completed.connect(self._on_frame_completed)
        self._worker.finished_all.connect(self._on_worker_finished)
        self._worker.error.connect(self._on_worker_error)
        self._progress.canceled.connect(self._on_cancel_requested)

        self._worker.start()
        self._starting_propagation = False

    # ------------------------------------------------------------------
    # Worker signal handlers
    # ------------------------------------------------------------------

    def _on_worker_progress(self, current: int, total: int, message: str) -> None:
        """Update the progress dialog with current step and message.

        Args:
            - current (int): Current step index.
            - total (int): Total number of steps (0 for indeterminate).
            - message (str): Label text for the progress dialog.
        """
        if self._progress is None:
            return
        if total == 0:
            if self._progress.maximum() != 0:
                self._progress.setRange(0, 0)
        else:
            if self._progress.maximum() != total:
                self._progress.setRange(0, total)
            self._progress.setValue(current)
        self._progress.setLabelText(message)

    def _on_frame_completed(self, frame_idx: int, mask: object) -> None:
        """Store the propagated mask in the image state and session.

        Args:
            - frame_idx (int): Index of the frame in the ordered path list.
            - mask (object): Propagated mask as numpy array (uint8, 0/255).
        """
        mask_np: np.ndarray = mask  # type: ignore[assignment]

        state_tuple = self._get_state()
        (
            _ckpt,
            _dev,
            _idir,
            _sdir,
            image_states,
            _si,
            _il,
            _cur,
            _predictor,
        ) = state_tuple

        if frame_idx >= len(self._ordered_paths):
            return

        img_path = self._ordered_paths[frame_idx]
        state = image_states.get(img_path)
        if state is None:
            return

        state.mask = mask_np
        state.keypoints = []
        state.mask_logits = None
        state.mask_candidates = []
        state.mask_scores = None
        state.selected_mask_idx = 0

        backup = self._pending_backups.get(frame_idx)
        if backup is not None and self._building_session is not None:
            backup.new_mask = mask_np
            self._building_session.frame_backups.append(backup)
            self._pending_backups.pop(frame_idx, None)

    def _on_worker_finished(self) -> None:
        """Clean up after the worker completes and register the session."""
        self._starting_propagation = False
        if self._progress is not None:
            if self._progress.maximum() > 0:
                self._progress.setValue(self._progress.maximum())
            self._progress.close()
            self._progress = None

        self._worker = None
        self._pending_backups.clear()

        # Register the completed session with the batch controller
        if self._building_session is not None and self._building_session.frame_backups:
            self._batch_ctrl.set_session(self._building_session)
        self._building_session = None

        self._update_ui()

    def _on_worker_error(self, message: str) -> None:
        """Handle worker errors, keeping any partially completed results.

        Args:
            - message (str): Error message from the worker.
        """
        self._starting_propagation = False
        if self._progress is not None:
            if self._progress.maximum() > 0 and self._progress.value() >= self._progress.maximum():
                self._progress.setValue(self._progress.maximum())
            self._progress.close()
            self._progress = None

        self._worker = None
        self._pending_backups.clear()

        # Register whatever was completed so far
        if self._building_session is not None and self._building_session.frame_backups:
            self._batch_ctrl.set_session(self._building_session)
        self._building_session = None

        self._update_ui()

        QMessageBox.critical(
            None,
            "Propagate Masks Error",
            f"An error occurred during mask propagation:\n\n{message}",
        )

    def _on_cancel_requested(self) -> None:
        """Forward cancellation to the worker thread."""
        if self._worker is not None:
            self._worker.cancel()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _upscale_mask_for_conditioning(state: ImageState) -> np.ndarray:
        """Return the mask at original image resolution for use as
        a conditioning input to SAM2 mask propagation.

        Args:
            - state (ImageState): Image state with a non-None mask.

        Returns:
            np.ndarray: Binary mask (uint8, 0/255) at original size.

        Raises:
            ValueError: If ``state.mask`` is None.
        """
        mask = state.mask
        if mask is None:
            raise ValueError("state.mask is None")

        if state.original_size is not None:
            h0, w0 = state.original_size
            mh, mw = mask.shape[:2]
            if mh != h0 or mw != w0:
                mask = cv2.resize(
                    mask,
                    (w0, h0),
                    interpolation=cv2.INTER_NEAREST,
                )
        return mask

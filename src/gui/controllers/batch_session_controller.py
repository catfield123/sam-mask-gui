"""Unified batch-session management: save-all and revert-all for any bulk operation."""

from typing import Optional

import cv2
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QMessageBox, QProgressDialog

from src.models.session_models import BatchSession
from src.services.image_service import ImageService


class BatchSessionController:
    """Manages a single :class:`BatchSession` that tracks the most recent
    bulk mask operation (mask propagation, batch mask grow, etc.) and
    provides unified *Save All* and *Revert* functionality.

    Args:
        - get_state (callable): Returns ``(save_dir, image_states)`` from
          the main window.
        - update_ui_cb (callable): Called after bulk state changes to
          refresh list labels, counter and the displayed image.
    """

    def __init__(self, get_state, update_ui_cb):
        """Initialise the batch-session controller.

        Args:
            - get_state (callable): Returns ``(save_dir, image_states)``.
            - update_ui_cb (callable): UI refresh callback.
        """
        self._get_state = get_state
        self._update_ui = update_ui_cb
        self._session: Optional[BatchSession] = None

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def has_session(self) -> bool:
        """Return whether a batch session exists and has not been reverted.

        Returns:
            bool: True if there is an active (non-reverted) session.
        """
        return self._session is not None and not self._session.undone

    @property
    def is_saved(self) -> bool:
        """Return whether the current session has been saved to disk.

        Returns:
            bool: True if the session masks have been written to disk.
        """
        return self._session is not None and self._session.saved_to_disk

    @property
    def session(self) -> Optional[BatchSession]:
        """Direct access to the current session (read-only convenience).

        Returns:
            BatchSession | None: The current batch session, or None.
        """
        return self._session

    # ------------------------------------------------------------------
    # Session registration
    # ------------------------------------------------------------------

    def set_session(self, session: BatchSession) -> None:
        """Register a new batch session, replacing any previous one.

        Args:
            - session (BatchSession): The session to track.
        """
        self._session = session

    def clear_session(self) -> None:
        """Discard the current session entirely."""
        self._session = None

    # ------------------------------------------------------------------
    # Save All
    # ------------------------------------------------------------------

    def save_all(self, parent_widget) -> None:
        """Save all masks from the current session to disk.

        Shows a confirmation dialog and a progress bar.

        Args:
            - parent_widget (QWidget): Parent for dialogs.
        """
        if self._session is None or self._session.undone:
            return

        n = len(self._session.frame_backups)
        if n == 0:
            QMessageBox.information(
                parent_widget,
                "Nothing to Save",
                "No modified masks to save.",
            )
            return

        answer = QMessageBox.question(
            parent_widget,
            "Save All Modified Masks",
            f"Save masks for {n} image(s) to disk?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        save_dir, image_states = self._get_state()

        if save_dir is None:
            QMessageBox.warning(
                parent_widget,
                "Warning",
                "Please set a save folder first.",
            )
            return

        # Progress dialog
        progress = QProgressDialog(
            "Saving masks…",
            None,
            0,
            n,
            parent_widget,
        )
        progress.setWindowTitle("Saving Masks")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        for i, backup in enumerate(self._session.frame_backups):
            mask_path = ImageService.get_mask_path(backup.image_path, save_dir)

            # Remember what was on disk before (for potential revert)
            if mask_path not in self._session.old_disk_masks:
                if mask_path.exists():
                    self._session.old_disk_masks[mask_path] = mask_path.read_bytes()
                else:
                    self._session.old_disk_masks[mask_path] = None

            # Write new mask
            if backup.new_mask is not None:
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(mask_path), backup.new_mask)
                if mask_path not in self._session.saved_mask_paths:
                    self._session.saved_mask_paths.append(mask_path)

            # Update in-memory state
            state = image_states.get(backup.image_path)
            if state is not None:
                state.mask_saved = True
                state.saved_version = state.state_version

            progress.setValue(i + 1)
            QApplication.processEvents()

        progress.close()
        self._session.saved_to_disk = True
        self._update_ui()

    # ------------------------------------------------------------------
    # Revert All
    # ------------------------------------------------------------------

    def revert_all(self, parent_widget) -> None:
        """Revert all frames affected by the current batch session.

        If masks were saved to disk, disk files are also reverted.

        Args:
            - parent_widget (QWidget): Parent for dialogs.
        """
        if self._session is None or self._session.undone:
            return

        n = len(self._session.frame_backups)
        if n == 0:
            self._session.undone = True
            self._update_ui()
            return

        op = self._session.operation_type
        label = {
            "video_propagation": "mask propagation",
            "grow_mask": "mask grow/shrink",
            "prompt_batch": "prompt batch segmentation",
        }.get(op, "batch operation")

        # Confirmation message
        if self._session.saved_to_disk:
            msg = f"Revert {label} for {n} image(s)?\n\nThis will also revert saved mask files on disk."
        else:
            msg = f"Revert {label} for {n} image(s)?"

        answer = QMessageBox.question(
            parent_widget,
            "Revert Batch Operation",
            msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        save_dir, image_states = self._get_state()

        # ---- Restore in-memory state -------------------------------------
        for backup in self._session.frame_backups:
            state = image_states.get(backup.image_path)
            if state is None:
                continue
            state.mask = backup.old_mask
            state.mask_saved = backup.old_mask_saved
            state.mask_logits = backup.old_mask_logits
            state.mask_candidates = []
            state.mask_scores = None
            state.selected_mask_idx = 0
            if backup.old_mask_saved:
                state.saved_version = state.state_version
            else:
                state.saved_version = -1
            state.undo_stack.clear()
            state.redo_stack.clear()

        # ---- Revert disk -------------------------------------------------
        if self._session.saved_to_disk and save_dir is not None:
            for mask_path, old_bytes in self._session.old_disk_masks.items():
                if old_bytes is None:
                    if mask_path.exists():
                        mask_path.unlink()
                else:
                    mask_path.write_bytes(old_bytes)

        self._session.undone = True
        self._update_ui()

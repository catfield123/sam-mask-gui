"""Undo/redo logic and state-version tracking."""

from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt

from src.models import ImageState


class UndoController:
    """Manages undo/redo stacks, version tokens, and the unsaved-changes flag.

    The controller is stateless except for a global version counter; all
    per-image state lives inside :class:`ImageState`.

    Args:
        - image_viewer: ``ImageViewerWidget`` used to push visual updates.
        - mask_selector: ``MaskSelectorWidget`` used to restore variant selection.
    """

    MAX_HISTORY = 50

    def __init__(self, image_viewer, mask_selector):
        """Initialise the undo controller.

        Args:
            - image_viewer: ``ImageViewerWidget`` used to push visual updates.
            - mask_selector: ``MaskSelectorWidget`` used to restore variant selection.
        """
        self._viewer = image_viewer
        self._mask_selector = mask_selector
        self._version_counter: int = 0

    # ------------------------------------------------------------------
    # Version helpers
    # ------------------------------------------------------------------

    def next_version(self) -> int:
        """Return the next unique version number.

        Returns:
            - int: Monotonically increasing version id.
        """
        self._version_counter += 1
        return self._version_counter

    def sync_unsaved(self, state: ImageState) -> None:
        """Recompute ``has_unsaved_changes`` from version tokens and update the viewer.

        Args:
            - state (ImageState): The image state to check.
        """
        state.has_unsaved_changes = state.state_version != state.saved_version
        self._viewer.set_unsaved_changes(state.has_unsaved_changes)

    # ------------------------------------------------------------------
    # Undo entry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def make_entry(state: ImageState) -> tuple:
        """Snapshot the current state into an undo/redo entry.

        Args:
            - state (ImageState): Source state.

        Returns:
            - tuple: Serialised snapshot that can be pushed onto a stack.
        """
        return (
            state.keypoints.copy(),
            state.mask_logits.copy() if state.mask_logits is not None else None,
            state.mask is not None and state.mask_saved and not state.keypoints,
            state.mask.copy() if state.mask is not None else None,
            list(state.mask_candidates),
            state.mask_scores.copy() if state.mask_scores is not None else None,
            state.selected_mask_idx,
            state.state_version,
        )

    def push_undo(self, state: ImageState) -> None:
        """Save the current state to the undo stack and assign a new version.

        Args:
            - state (ImageState): The image state to snapshot.
        """
        state.undo_stack.append(self.make_entry(state))
        if len(state.undo_stack) > self.MAX_HISTORY:
            state.undo_stack.pop(0)
        state.redo_stack.clear()
        state.state_version = self.next_version()
        self.sync_unsaved(state)

    # ------------------------------------------------------------------
    # Undo / Redo
    # ------------------------------------------------------------------

    def undo(self, state: Optional[ImageState], predictor=None) -> None:
        """Restore the previous state from the undo stack.

        Args:
            - state (ImageState | None): Current image state, or ``None`` to no-op.
            - predictor: ``SAM2PredictorWrapper`` used to recompute logits
              if they are missing.
        """
        if state is None or not state.undo_stack:
            return

        state.redo_stack.append(self.make_entry(state))
        if len(state.redo_stack) > self.MAX_HISTORY:
            state.redo_stack.pop(0)

        self._restore(state, state.undo_stack.pop(), predictor)

    def redo(self, state: Optional[ImageState], predictor=None) -> None:
        """Re-apply the last undone action from the redo stack.

        Args:
            - state (ImageState | None): Current image state, or ``None`` to no-op.
            - predictor: ``SAM2PredictorWrapper`` used to recompute logits
              if they are missing.
        """
        if state is None or not state.redo_stack:
            return

        state.undo_stack.append(self.make_entry(state))
        if len(state.undo_stack) > self.MAX_HISTORY:
            state.undo_stack.pop(0)

        self._restore(state, state.redo_stack.pop(), predictor)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _restore(self, state: ImageState, entry: tuple, predictor) -> None:
        """Apply a saved entry to *state* and refresh the UI.

        Args:
            - state (ImageState): Target image state.
            - entry (tuple): Previously saved snapshot.
            - predictor: SAM2 predictor for optional logits recomputation.
        """
        (kp, logits, _loaded, mask, candidates, scores, idx, ver) = entry

        state.keypoints = kp
        state.mask_logits = logits.copy() if logits is not None else None
        state.mask = mask.copy() if mask is not None else None
        state.mask_candidates = list(candidates)
        state.mask_scores = scores.copy() if scores is not None else None
        state.selected_mask_idx = idx
        state.state_version = ver

        self._apply_to_ui(state, predictor)

    def _apply_to_ui(self, state: ImageState, predictor) -> None:
        """Push the restored state into the viewer and mask selector.

        Args:
            - state (ImageState): State that was just restored.
            - predictor: SAM2 predictor (may be ``None``).
        """
        self._viewer.set_keypoints(state.keypoints)

        if state.mask is not None:
            self._viewer.set_mask(state.mask)
            if state.mask_logits is None and predictor is not None:
                state.mask_logits = predictor.mask_to_logits(state.mask)
        else:
            self._viewer.set_mask(None)

        # Restore mask-selector without triggering on_mask_variant_selected
        if state.mask_candidates and self._viewer.image is not None:
            self._mask_selector.mask_list.blockSignals(True)
            try:
                self._mask_selector.set_masks(
                    state.mask_candidates,
                    state.mask_scores if state.mask_scores is not None else np.array([]),
                    self._viewer.image,
                )
                for row in range(self._mask_selector.mask_list.count()):
                    item = self._mask_selector.mask_list.item(row)
                    if item and item.data(Qt.ItemDataRole.UserRole) == state.selected_mask_idx:
                        self._mask_selector.mask_list.setCurrentRow(row)
                        break
            finally:
                self._mask_selector.mask_list.blockSignals(False)
        elif self._viewer.image is not None:
            self._mask_selector.set_masks([], np.array([]), self._viewer.image)

        self.sync_unsaved(state)

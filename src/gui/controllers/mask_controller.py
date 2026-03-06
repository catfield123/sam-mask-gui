"""Mask prediction, keypoint handling, brush integration, grow/shrink, and mask I/O."""

from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
from PyQt6.QtWidgets import QMessageBox

from src.gui.controllers.undo_controller import UndoController
from src.logging_config import get_logger
from src.models import ImageState, Keypoint, KeypointType
from src.models.session_models import BatchSession, FrameBackup
from src.services import ImageService, MaskService

logger = get_logger(__name__)

if TYPE_CHECKING:
    from src.sam2 import SAM2PredictorWrapper  # noqa: F401
    from src.sam3 import SAM3PredictorWrapper  # noqa: F401


class MaskController:
    """Orchestrates mask generation from keypoints, brush stroke integration,
    mask candidate selection, and save/load operations.

    Args:
        - image_viewer: ``ImageViewerWidget`` for visual updates.
        - mask_selector: ``MaskSelectorWidget`` for candidate list.
        - undo_ctrl (UndoController): Undo/redo controller.
        - get_state (callable): Returns ``(current_path, image_states)``
          giving access to the shared application state.
    """

    def __init__(self, image_viewer, mask_selector, undo_ctrl: UndoController, get_state):
        """Initialise the mask controller.

        Args:
            - image_viewer: ``ImageViewerWidget`` for visual updates.
            - mask_selector: ``MaskSelectorWidget`` for candidate list.
            - undo_ctrl (UndoController): Undo/redo controller.
            - get_state (callable): Returns ``(current_path, image_states)``.
        """
        self._viewer = image_viewer
        self._selector = mask_selector
        self._undo = undo_ctrl
        self._get_state = get_state

    @staticmethod
    def _supports_text_prompt_prediction(predictor) -> bool:
        """Return whether the predictor exposes the SAM3 text-prompt API.

        Args:
            predictor: The predictor instance to check.

        Returns:
            bool: True if the predictor supports text-prompt segmentation.
        """
        return (
            predictor is not None and hasattr(predictor, "predict_mask_from_text") and hasattr(predictor, "load_image")
        )

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def _current(self) -> tuple:
        """Return the current image path and image states dict.

        Returns:
            tuple: ``(current_image_path, image_states_dict)``.
        """
        return self._get_state()

    def _current_state(self) -> Optional[ImageState]:
        """Return the image state for the active image, or None.

        Returns:
            ImageState | None: State for the current image, or None.
        """
        path, states = self._current()
        if path is None:
            return None
        return states.get(path)

    # ------------------------------------------------------------------
    # Keypoint callbacks
    # ------------------------------------------------------------------

    def on_keypoint_added(self, x: int, y: int, kp_type: int, predictor):
        """Handle a new keypoint placed by the user.

        Args:
            - x (int): X in image space.
            - y (int): Y in image space.
            - kp_type (int): Keypoint type value (1 = positive, 0 = negative).
            - predictor (SAM2PredictorWrapper | None): Active predictor.
        """
        state = self._current_state()
        if state is None:
            return

        self._undo.push_undo(state)

        kp = Keypoint(x, y, KeypointType(kp_type))
        state.keypoints.append(kp)

        # When switching from a prompt-generated mask to SAM2 clicks, recompute the
        # iterative logits from the currently visible mask on the first click.
        if (
            state.mask is not None
            and predictor is not None
            and (state.mask_logits is None or len(state.keypoints) == 1)
        ):
            state.mask_logits = predictor.mask_to_logits(state.mask)

        self._viewer.set_keypoints(state.keypoints)
        self.update_mask(predictor)

    # ------------------------------------------------------------------
    # Mask prediction
    # ------------------------------------------------------------------

    def update_mask(self, predictor, parent_widget=None):
        """Regenerate the mask from the current keypoints.

        Args:
            - predictor (SAM2PredictorWrapper | None): Active predictor.
            - parent_widget (QWidget | None): Parent for error message boxes.
        """
        state = self._current_state()
        if state is None or predictor is None:
            return

        if not state.keypoints:
            if state.mask is None:
                state.mask_candidates = []
                state.mask_scores = None
                state.selected_mask_idx = 0
                state.mask_logits = None
                state.mask_logits_all = None
                self._viewer.set_mask(None)
                if self._viewer.image is not None:
                    self._selector.set_masks([], np.array([]), self._viewer.image)
            return

        try:
            point_coords = [(kp.x, kp.y) for kp in state.keypoints]
            point_labels = [kp.type.value for kp in state.keypoints]

            masks, scores, all_logits = predictor.predict_mask(
                point_coords,
                point_labels,
                multimask_output=True,
                mask_input=state.mask_logits,
            )
            state.mask_candidates = masks
            state.mask_scores = scores
            state.mask_logits_all = all_logits

            if len(masks) > 0:
                best_idx = 0
                if scores is not None and len(scores) == len(masks) and len(scores) > 0:
                    best_idx = int(np.argmax(scores))
                state.selected_mask_idx = best_idx
                state.mask = masks[best_idx]
                if all_logits is not None and len(all_logits) > best_idx:
                    state.mask_logits = all_logits[best_idx : best_idx + 1]
            else:
                state.mask = None
                state.selected_mask_idx = 0
                state.mask_logits = None
                state.mask_logits_all = None

            if self._viewer.image is not None:
                self._selector.set_masks(masks, scores, self._viewer.image)
                self._selector.select_mask(state.selected_mask_idx)

            self._viewer.set_mask(state.mask)
        except Exception as e:
            if parent_widget:
                QMessageBox.critical(parent_widget, "Error", f"Failed to generate mask: {e}")

    def segment_by_prompt(
        self,
        prompt: str,
        predictor,
        image_path: Optional[Path] = None,
        parent_widget=None,
    ):
        """Segment the current image using a text prompt (SAM3 only).

        Args:
            - prompt (str): Text prompt describing what to segment.
            - predictor: Active predictor (must be SAM3PredictorWrapper).
            - image_path (Path | None): Current image path to load into SAM3.
            - parent_widget (QWidget | None): Parent for error message boxes.

        Raises:
            ValueError: If no image is selected when loading into SAM3.
        """
        state = self._current_state()
        if state is None or predictor is None:
            return

        if not self._supports_text_prompt_prediction(predictor):
            if parent_widget:
                QMessageBox.warning(
                    parent_widget, "Error", "Text prompt segmentation is only available with SAM3 model."
                )
            return

        if not prompt or not prompt.strip():
            if parent_widget:
                QMessageBox.warning(parent_widget, "Error", "Please enter a text prompt.")
            return

        logger.info("segment_by_prompt: prompt=%r, image_path=%s", prompt.strip(), image_path)
        self._undo.push_undo(state)

        try:
            if image_path is None:
                raise ValueError("No image selected.")

            predictor.load_image(str(image_path))
            state.original_size = predictor.get_original_size()
            state.scaled_size = predictor.get_scaled_size()
            state.scale_factor = predictor.get_scale_factor()
            # Use SAM3 text prompt API
            masks, scores = predictor.predict_mask_from_text(prompt.strip())

            if len(masks) == 0:
                if parent_widget:
                    QMessageBox.information(parent_widget, "No Results", "No masks found for the given prompt.")
                state.mask_candidates = []
                state.mask_scores = None
                state.selected_mask_idx = 0
                state.mask_logits = None
                state.mask_logits_all = None
                self._viewer.set_mask(None)
                if self._viewer.image is not None:
                    self._selector.set_masks([], np.array([]), self._viewer.image)
                return

            state.mask_candidates = masks
            state.mask_scores = scores
            state.keypoints = []

            best_idx = 0
            if scores is not None and len(scores) == len(masks) and len(scores) > 0:
                best_idx = int(np.argmax(scores))

            state.selected_mask_idx = best_idx
            state.mask = masks[best_idx]
            state.mask_logits = None  # SAM3 doesn't use logits in the same way
            state.mask_logits_all = None

            if self._viewer.image is not None:
                self._selector.set_masks(masks, scores, self._viewer.image)
                self._selector.select_mask(best_idx)

            self._viewer.set_keypoints([])
            self._viewer.set_mask(state.mask)
        except Exception as e:
            logger.error("Error segmenting by prompt: %s", e, exc_info=True)
            if parent_widget:
                QMessageBox.critical(parent_widget, "Error", f"Failed to segment by prompt: {e}")

    def on_mask_variant_selected(self, mask_idx: int):
        """Handle the user picking a different mask candidate.

        Args:
            - mask_idx (int): Index into ``state.mask_candidates``.
        """
        state = self._current_state()
        if state is None or len(state.mask_candidates) == 0:
            return

        if 0 <= mask_idx < len(state.mask_candidates):
            state.selected_mask_idx = mask_idx
            state.mask = state.mask_candidates[mask_idx]

            if state.mask_logits_all is not None and 0 <= mask_idx < len(state.mask_logits_all):
                state.mask_logits = state.mask_logits_all[mask_idx : mask_idx + 1]
            else:
                state.mask_logits = None

            self._viewer.set_mask(state.mask)

    # ------------------------------------------------------------------
    # Brush stroke integration
    # ------------------------------------------------------------------

    def on_brush_stroke_started(self):
        """Save undo state at the beginning of a brush stroke."""
        state = self._current_state()
        if state is not None:
            self._undo.push_undo(state)

    def on_brush_stroke_finished(self, mask: np.ndarray, predictor, update_counter_cb):
        """Apply the completed brush mask and convert to logits.

        Args:
            - mask (np.ndarray): Final mask from the brush engine.
            - predictor (SAM2PredictorWrapper | None): Active predictor.
            - update_counter_cb (callable): Callback to refresh the mask counter.
        """
        state = self._current_state()
        if state is None or predictor is None:
            return

        state.mask = mask
        state.mask_candidates = []
        state.mask_scores = None
        state.selected_mask_idx = 0
        if self._viewer.image is not None:
            self._selector.set_masks([], np.array([]), self._viewer.image)

        state.mask_logits = predictor.mask_to_logits(mask)

        self._viewer.set_mask(state.mask)
        self._undo.sync_unsaved(state)
        update_counter_cb()

    # ------------------------------------------------------------------
    # Grow / shrink mask
    # ------------------------------------------------------------------

    def grow_current_mask(self, pixels: int, predictor, update_counter_cb):
        """Apply morphological grow/shrink to the currently displayed mask.

        Pushes undo state before modifying so that Ctrl+Z restores the
        previous mask.

        Args:
            - pixels (int): Number of pixels to grow (>0) or shrink (<0).
            - predictor: ``SAM2PredictorWrapper`` for logits recomputation.
            - update_counter_cb (callable): Refreshes the mask counter.
        """
        state = self._current_state()
        if state is None or state.mask is None or pixels == 0:
            return

        self._undo.push_undo(state)

        state.mask = MaskService.grow_mask(state.mask, pixels)
        state.mask_candidates = []
        state.mask_scores = None
        state.selected_mask_idx = 0

        if predictor is not None:
            state.mask_logits = predictor.mask_to_logits(state.mask)
        else:
            state.mask_logits = None

        self._viewer.set_mask(state.mask)
        if self._viewer.image is not None:
            self._selector.set_masks([], np.array([]), self._viewer.image)
        self._undo.sync_unsaved(state)
        update_counter_cb()

    @staticmethod
    def grow_masks_batch(
        image_paths: List[Path],
        image_states: Dict[Path, ImageState],
        pixels: int,
    ) -> BatchSession:
        """Apply morphological grow/shrink to multiple images at once.

        Creates a :class:`BatchSession` containing :class:`FrameBackup`
        entries for every modified frame, suitable for batch revert / save.

        This method modifies ``image_states`` in-place but does **not**
        touch the viewer or any UI widgets.

        Args:
            - image_paths (list[Path]): Ordered image paths to process.
            - image_states (dict): ``Path`` -> ``ImageState`` mapping.
            - pixels (int): Grow (>0) or shrink (<0) amount.

        Returns:
            - BatchSession: Session with backup data for all affected frames.
        """
        session = BatchSession(operation_type="grow_mask")

        for idx, img_path in enumerate(image_paths):
            state = image_states.get(img_path)
            if state is None or state.mask is None:
                continue

            backup = FrameBackup(
                image_path=img_path,
                frame_idx=idx,
                old_mask=state.mask.copy(),
                old_mask_saved=state.mask_saved,
                old_mask_logits=(state.mask_logits.copy() if state.mask_logits is not None else None),
            )

            new_mask = MaskService.grow_mask(state.mask, pixels)
            backup.new_mask = new_mask

            # Update in-memory state
            state.mask = new_mask
            state.mask_logits = None
            state.mask_candidates = []
            state.mask_scores = None
            state.selected_mask_idx = 0
            # Clear per-image undo/redo to avoid inconsistency
            state.undo_stack.clear()
            state.redo_stack.clear()

            session.frame_backups.append(backup)

        return session

    @staticmethod
    def segment_masks_by_prompt_batch(
        image_paths: List[Path],
        image_states: Dict[Path, ImageState],
        prompt: str,
        predictor,
    ) -> BatchSession:
        """Apply text prompt segmentation to multiple images at once.

        Creates a :class:`BatchSession` containing :class:`FrameBackup`
        entries for every modified frame, suitable for batch revert / save.

        This method modifies ``image_states`` in-place but does **not**
        touch the viewer or any UI widgets.

        Args:
            - image_paths (list[Path]): Ordered image paths to process.
            - image_states (dict): ``Path`` -> ``ImageState`` mapping.
            - prompt (str): Text prompt for segmentation.
            - predictor: ``SAM3PredictorWrapper`` for prompt segmentation.

        Returns:
            BatchSession: Session with backup data for all affected frames.

        Raises:
            ValueError: If predictor does not support text prompts or prompt is empty.
        """
        if not MaskController._supports_text_prompt_prediction(predictor):
            raise ValueError("Prompt batch segmentation requires SAM3PredictorWrapper")

        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        session = BatchSession(operation_type="prompt_batch")
        prompt = prompt.strip()

        for idx, img_path in enumerate(image_paths):
            state = image_states.get(img_path)
            if state is None:
                continue

            # Snapshot before modification
            backup = FrameBackup(
                image_path=img_path,
                frame_idx=idx,
                old_mask=state.mask.copy() if state.mask is not None else None,
                old_mask_saved=state.mask_saved,
                old_mask_logits=(state.mask_logits.copy() if state.mask_logits is not None else None),
            )

            try:
                # Load image and segment by prompt
                scaled_img, original_size, scale_factor = predictor.load_image(str(img_path))

                # Segment using prompt
                masks, scores = predictor.predict_mask_from_text(prompt)

                if len(masks) > 0:
                    # Use the first (best) mask
                    new_mask = masks[0]
                    backup.new_mask = new_mask

                    # Update in-memory state
                    state.mask = new_mask
                    state.mask_candidates = masks
                    state.mask_scores = scores
                    state.selected_mask_idx = 0
                else:
                    # No masks found - keep old mask or set to None
                    backup.new_mask = None
                    state.mask = None
                    state.mask_candidates = []
                    state.mask_scores = None
                    state.selected_mask_idx = 0

                state.mask_logits = None
                state.mask_logits_all = None
                # Clear per-image undo/redo to avoid inconsistency
                state.undo_stack.clear()
                state.redo_stack.clear()

                session.frame_backups.append(backup)
            except Exception as e:
                logger.error("Error processing %s: %s", img_path.name, e)
                # Still add backup even if processing failed
                backup.new_mask = state.mask
                session.frame_backups.append(backup)

        return session

    # ------------------------------------------------------------------
    # Save / load / clear
    # ------------------------------------------------------------------

    def save_current_mask(
        self,
        save_dir: Optional[Path],
        mask_service: Optional[MaskService],
        sort_combo,
        image_list,
        update_list_cb,
        update_counter_cb,
        sort_cb,
        parent_widget=None,
    ):
        """Save the active mask to disk.

        Args:
            - save_dir (Path | None): Target directory for masks.
            - mask_service (MaskService | None): Service performing the I/O.
            - sort_combo: ``QComboBox`` for the current sort mode.
            - image_list: ``QListWidget`` of images.
            - update_list_cb (callable): Refreshes list item labels.
            - update_counter_cb (callable): Refreshes the mask counter.
            - sort_cb (callable): Re-sorts the image list.
            - parent_widget (QWidget | None): Parent for error dialogs.
        """
        path, _ = self._current()
        state = self._current_state()
        if path is None or save_dir is None or mask_service is None:
            QMessageBox.warning(parent_widget, "Warning", "Please set a save folder first.")
            return
        if state is None or state.mask is None:
            QMessageBox.warning(parent_widget, "Warning", "No mask to save. Please create a mask first.")
            return

        try:
            was_saved = state.mask_saved

            next_image_path = None
            if not was_saved and sort_combo.currentIndex() == 1:
                row = image_list.currentRow()
                if row < image_list.count() - 1:
                    nxt = image_list.item(row + 1)
                    if nxt:
                        next_image_path = nxt.data(0x0100)  # Qt.ItemDataRole.UserRole

            mask_path = ImageService.get_mask_path(path, save_dir)
            if mask_service.save_mask(state.mask, mask_path):
                state.mask_saved = True
                state.saved_version = state.state_version
                self._undo.sync_unsaved(state)
                update_list_cb()
                update_counter_cb()

                if sort_combo.currentIndex() in (1, 2):
                    sort_cb(sort_combo.currentIndex())

                if not was_saved and next_image_path and sort_combo.currentIndex() == 1:
                    for i in range(image_list.count()):
                        item = image_list.item(i)
                        if item and item.data(0x0100) == next_image_path:
                            image_list.setCurrentRow(i)
                            break
            else:
                QMessageBox.critical(parent_widget, "Error", f"Failed to save mask to {mask_path}")
        except Exception as e:
            QMessageBox.critical(parent_widget, "Error", f"Failed to save mask: {e}")

    def clear_current_mask(self, update_counter_cb):
        """Clear the active mask, keypoints, and undo history.

        Args:
            - update_counter_cb (callable): Callback to refresh the mask counter.
        """
        state = self._current_state()
        if state is None:
            return

        state.mask = None
        state.mask_candidates = []
        state.mask_scores = None
        state.mask_logits = None
        state.mask_logits_all = None
        state.selected_mask_idx = 0
        state.keypoints = []
        state.undo_stack = []
        state.redo_stack = []

        self._viewer.set_mask(None)
        self._viewer.set_keypoints([])
        if self._viewer.image is not None:
            self._selector.set_masks([], np.array([]), self._viewer.image)

        update_counter_cb()

    def check_and_load_masks_for_all(
        self,
        save_dir: Optional[Path],
        mask_service: Optional[MaskService],
        image_list,
        image_states: Dict[Path, ImageState],
        update_counter_cb,
    ):
        """Scan all image states and load masks that exist on disk.

        Args:
            - save_dir (Path | None): Directory with saved masks.
            - mask_service (MaskService | None): Mask I/O service.
            - image_list: ``QListWidget`` of images.
            - image_states (dict): Path -> ImageState mapping.
            - update_counter_cb (callable): Refreshes the mask counter.
        """
        if save_dir is None or mask_service is None:
            return

        for i in range(image_list.count()):
            item = image_list.item(i)
            if item is None:
                continue
            img_path = item.data(0x0100)
            if img_path is None:
                continue
            state = image_states.get(img_path)
            if state is None:
                continue

            mask_path = ImageService.get_mask_path(img_path, save_dir)
            if mask_path.exists():
                if not state.mask_saved:
                    state.mask_saved = True
                    if not item.text().startswith("✓"):
                        item.setText(f"✓ {img_path.name}")
                if state.mask is None:
                    try:
                        mask_full = mask_service.load_mask(mask_path)
                        if mask_full is not None:
                            state.mask = mask_full
                    except Exception:
                        pass

        update_counter_cb()

    def check_and_load_mask_for_current(
        self,
        save_dir: Optional[Path],
        mask_service: Optional[MaskService],
        predictor,
        update_list_cb,
    ):
        """Load the mask for the currently selected image if it exists on disk.

        Args:
            - save_dir (Path | None): Mask directory.
            - mask_service (MaskService | None): Mask I/O service.
            - predictor (SAM2PredictorWrapper | None): Active predictor.
            - update_list_cb (callable): Refreshes list item labels.
        """
        path, _ = self._current()
        state = self._current_state()
        if path is None or state is None or save_dir is None or mask_service is None:
            return

        mask_path = ImageService.get_mask_path(path, save_dir)
        if not mask_path.exists():
            return

        state.mask_saved = True
        update_list_cb()

        if state.mask is not None:
            return

        try:
            mask_full = mask_service.load_mask(mask_path)
            if mask_full is None:
                return
            state.mask = mask_full
            if self._viewer.image is not None and predictor is not None:
                state.mask = predictor.downscale_mask(state.mask)
                state.mask_logits = predictor.mask_to_logits(state.mask)
                self._viewer.set_mask(state.mask)
        except Exception:
            pass

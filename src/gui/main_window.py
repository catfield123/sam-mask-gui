"""Main application window — thin orchestrator that wires panels and controllers."""

import gc
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import cv2

from src.logging_config import get_logger

logger = get_logger(__name__)
import numpy as np
import torch
from PyQt6.QtCore import QEventLoop, Qt
from PyQt6.QtGui import QColor, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QColorDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QVBoxLayout,
    QWidget,
)

from src.gui.dialogs import ShortcutsDialog
from src.gui.controllers import (
    BatchSessionController,
    ImageListController,
    MaskController,
    SettingsController,
    UndoController,
    PropagationController,
)
from src.gui.panels import CenterPanel, LeftPanel, RightPanel
from src.gui.workers import ModelLoadWorker, ThumbnailLoaderWorker
from src.models import ImageState
from src.models.keypoint import Keypoint
from src.models.session_models import BatchSession, FrameBackup
from src.services import ConfigService, MaskService
from src.utils.check_packages import check_sam2_installed, check_sam3_installed

if TYPE_CHECKING:
    from src.sam2 import SAM2PredictorWrapper
    from src.sam3 import SAM3PredictorWrapper


class MainWindow(QMainWindow):
    """Top-level window that composes UI panels and delegates logic to controllers."""

    def __init__(self):
        """Initialise the window, build UI, create controllers, and load config."""
        super().__init__()
        logger.debug("MainWindow __init__ started")
        self.setWindowTitle("SAM2/SAM3 Image Segmentation")
        self.setGeometry(100, 100, 1400, 900)

        # Shared application state
        config_path = Path(__file__).parent.parent.parent / "config.json"
        self.config_service = ConfigService(config_path)
        self.images_dir: Optional[Path] = None
        self.save_dir: Optional[Path] = None
        self.image_states: Dict[Path, ImageState] = {}
        self.current_image_path: Optional[Path] = None
        self.predictor: Optional[SAM2PredictorWrapper] = None
        self.mask_service: Optional[MaskService] = None
        self.checkpoint_path: str = ""
        self.sam3_predictor: Optional[SAM3PredictorWrapper] = None
        self.sam3_checkpoint_path: Optional[str] = None
        self.sam3_bpe_path: Optional[str] = None
        self.max_side: int = 1024
        self.keep_models_loaded: bool = False
        self._active_predictor_kind: Optional[str] = None
        self._is_prompt_batch_running: bool = False
        self._sam2_last_error: Optional[str] = None
        self._sam3_last_error: Optional[str] = None

        # Background thumbnail loader
        self._thumb_worker = ThumbnailLoaderWorker(self)
        self._thumb_worker.start()

        # Build UI panels
        self._build_ui()

        # Controllers
        self._undo_ctrl = UndoController(
            self._center.image_viewer,
            self._right.mask_selector,
        )
        self._mask_ctrl = MaskController(
            self._center.image_viewer,
            self._right.mask_selector,
            self._undo_ctrl,
            self._get_mask_state,
        )
        self._list_ctrl = ImageListController(
            self._left.image_list,
            self._left.mask_counter_label,
            self._left.sort_combo,
            self._thumb_worker,
            self._get_list_state,
        )
        self._settings_ctrl = SettingsController(
            self.config_service,
            self._get_window_state,
            self._set_window_state,
        )
        self._batch_ctrl = BatchSessionController(
            self._get_batch_state,
            self._on_batch_ui_update,
        )
        self._propagation_ctrl = PropagationController(
            self._get_propagation_state,
            self._on_batch_ui_update,
            self._batch_ctrl,
        )

        # Wire signals
        self._wire_signals()
        self._setup_shortcuts()
        self._settings_ctrl.load_config(self._load_images, self._check_all_masks)
        self._update_prompt_ui_availability()

        # Check packages and show warnings
        logger.info("Checking SAM2/SAM3 package availability at startup")
        sam2_installed, sam2_msg = check_sam2_installed()
        sam3_installed, sam3_msg = check_sam3_installed()
        logger.info("Startup package check: SAM2=%s, SAM3=%s", sam2_installed, sam3_installed)
        if not sam3_installed:
            logger.warning("SAM3 is not installed or failed to import: %s", sam3_msg)

        if not sam2_installed:
            logger.warning("SAM2 is not installed: %s", sam2_msg)
            QMessageBox.warning(
                self,
                "SAM2 Package Warning",
                f"{sam2_msg}\n\nThe SAM2 status badge will stay gray until the package is importable.",
            )

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        """Assemble the panel layout and the menu bar."""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)

        top_controls = QHBoxLayout()
        top_controls.setContentsMargins(0, 0, 0, 0)
        self._sam2_status_dot, self._sam2_status_badge = self._create_model_status_widget("SAM2")
        self._sam3_status_dot, self._sam3_status_badge = self._create_model_status_widget("SAM3")
        top_controls.addWidget(self._sam2_status_badge)
        top_controls.addWidget(self._sam3_status_badge)
        top_controls.addSpacing(12)
        self._keep_models_checkbox = QCheckBox("Keep SAM2 and SAM3 loaded in memory")
        self._keep_models_checkbox.setToolTip(
            "Uses more memory, but avoids reloading models when switching between SAM2 and SAM3."
        )
        self._keep_models_checkbox.toggled.connect(self._on_keep_models_toggled)
        top_controls.addWidget(self._keep_models_checkbox)
        top_controls.addStretch()
        main_layout.addLayout(top_controls)

        # Three-panel layout
        panels_layout = QHBoxLayout()
        panels_layout.setContentsMargins(0, 0, 0, 0)
        panels_layout.setSpacing(5)

        self._left = LeftPanel()
        self._center = CenterPanel()
        self._right = RightPanel(self._center.image_viewer)

        panels_layout.addWidget(self._left)
        panels_layout.addWidget(self._center, stretch=1)
        panels_layout.addWidget(self._right)

        main_layout.addLayout(panels_layout, stretch=1)

        self._build_menu()
        self._center.prev_button.setToolTip("Previous image (←)")
        self._center.next_button.setToolTip("Next image (→)")
        self._center.clear_button.setToolTip("Clear current mask")
        self._refresh_model_status_indicators()

    def _create_model_status_widget(self, model_name: str):
        """Create a compact status badge with a coloured square and label."""
        container = QWidget(self)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        dot = QFrame(container)
        dot.setFixedSize(12, 12)
        dot.setFrameShape(QFrame.Shape.StyledPanel)
        layout.addWidget(dot)

        label = QLabel(model_name, container)
        layout.addWidget(label)

        return dot, container

    def _build_menu(self):
        """Create the application menu bar."""
        mb = self.menuBar()
        fm = mb.addMenu("File")

        act_open = fm.addAction("Open Images Folder...")
        act_open.setShortcut(QKeySequence("Ctrl+O"))
        act_open.setToolTip("Open a folder with images (Ctrl+O)")
        act_open.triggered.connect(self._open_images_folder)

        act_save_folder = fm.addAction("Set Save Folder...")
        act_save_folder.setToolTip("Choose where to save mask files")
        act_save_folder.triggered.connect(self._set_save_folder)
        fm.addSeparator()
        act_settings = fm.addAction("Settings...")
        act_settings.setToolTip("SAM2/SAM3 paths, max_side, and display options")
        act_settings.triggered.connect(self.show_settings)
        fm.addSeparator()
        fm.addAction("Exit").triggered.connect(self.close)

        hm = mb.addMenu("Help")
        act_shortcuts = hm.addAction("Keyboard shortcuts...")
        act_shortcuts.setToolTip("Show all keyboard shortcuts")
        act_shortcuts.triggered.connect(self._show_shortcuts_dialog)

    def _wire_signals(self):
        """Connect panel/widget signals to controller methods."""
        viewer = self._center.image_viewer

        # Image list
        self._left.image_list.currentItemChanged.connect(self._on_image_selected)
        self._left.sort_combo.currentIndexChanged.connect(self._list_ctrl.sort_image_list)
        self._thumb_worker.thumbnail_loaded.connect(self._list_ctrl.on_thumbnail_loaded)

        # Viewer → mask / undo
        viewer.keypoint_added.connect(self._on_keypoint_added)
        viewer.brush_stroke_started.connect(self._mask_ctrl.on_brush_stroke_started)
        viewer.brush_stroke_finished.connect(self._on_brush_stroke_finished)

        # Mask selector
        self._right.mask_selector.mask_selected.connect(self._mask_ctrl.on_mask_variant_selected)

        # Navigation
        self._center.prev_button.clicked.connect(self._go_previous)
        self._center.next_button.clicked.connect(self._go_next)
        self._center.clear_button.clicked.connect(
            lambda: self._mask_ctrl.clear_current_mask(self._list_ctrl.update_mask_counter)
        )

        # Prompt segmentation (right panel - SAM3 only)
        self._right.prompt_segment_btn.clicked.connect(self._segment_by_prompt)

        # Prompt batch (left panel - SAM3 only)
        self._left.prompt_batch_btn.clicked.connect(self._segment_selected_by_prompt)

        # Grow / shrink — single image (right panel)
        self._right.grow_apply_btn.clicked.connect(self._grow_current_mask)

        # Select All / Deselect All toggle (left panel)
        self._left.select_all_btn.clicked.connect(self._toggle_select_all)

        # Batch action buttons (left panel)
        self._left.revert_btn.clicked.connect(lambda: self._batch_ctrl.revert_all(self))
        self._left.save_all_btn.clicked.connect(lambda: self._batch_ctrl.save_all(self))
        self._left.grow_selected_btn.clicked.connect(self._grow_selected_masks)
        self._left.segment_video_btn.clicked.connect(lambda: self._propagation_ctrl.start_propagation(self))

        # Re-evaluate action buttons when sort mode or selection changes
        self._left.sort_combo.currentIndexChanged.connect(self._update_action_buttons)
        self._left.image_list.itemSelectionChanged.connect(self._update_action_buttons)

        # Zoom / brush / colour sliders
        viewer.zoom_changed.connect(self._sync_zoom_slider)
        viewer.brush_size_changed.connect(self._sync_brush_slider)
        viewer.brush_max_changed.connect(self._on_brush_max_changed)
        self._right.zoom_slider.valueChanged.connect(lambda v: viewer.set_zoom(v / 100.0))
        self._right.brush_slider.valueChanged.connect(viewer.set_brush_size)
        self._right.brush_slider.sliderPressed.connect(lambda: viewer.show_center_brush_preview(True))
        self._right.brush_slider.sliderReleased.connect(lambda: viewer.show_center_brush_preview(False))
        self._right.mask_color_btn.clicked.connect(self._pick_mask_colour)
        self._right.opacity_slider.valueChanged.connect(self._on_opacity_changed)

    def _setup_shortcuts(self):
        """Register global keyboard shortcuts."""
        QShortcut(QKeySequence("Ctrl+S"), self).activated.connect(self._save_mask)
        QShortcut(QKeySequence("Ctrl+Z"), self).activated.connect(self._undo)
        QShortcut(QKeySequence("Ctrl+Y"), self).activated.connect(self._redo)
        QShortcut(QKeySequence("G"), self).activated.connect(self._grow_current_mask)

    # ------------------------------------------------------------------
    # State accessor callbacks (passed to controllers)
    # ------------------------------------------------------------------

    def _get_mask_state(self):
        """Return ``(current_image_path, image_states)`` for MaskController."""
        return self.current_image_path, self.image_states

    def _get_list_state(self):
        """Return ``(images_dir, save_dir, image_states, mask_service)`` for ImageListController."""
        return self.images_dir, self.save_dir, self.image_states, self.mask_service

    def _get_window_state(self):
        """Return a snapshot dict of window-level state for SettingsController."""
        return {
            "checkpoint_path": self.checkpoint_path,
            "sam3_checkpoint_path": self.sam3_checkpoint_path,
            "sam3_bpe_path": self.sam3_bpe_path,
            "keep_models_loaded": self.keep_models_loaded,
            "max_side": self.max_side,
            "predictor": self.predictor,
            "sam3_predictor": self.sam3_predictor,
            "mask_service": self.mask_service,
            "images_dir": self.images_dir,
            "save_dir": self.save_dir,
            "current_image_path": self.current_image_path,
            "release_predictors_cb": self._release_all_predictors,
        }

    def _set_window_state(self, updates: dict):
        """Apply a dict of state updates to window attributes.

        Args:
            - updates (dict): Mapping of attribute names to new values.
        """
        sam2_keys = {"checkpoint_path"}
        sam3_keys = {"sam3_checkpoint_path", "sam3_bpe_path"}
        if sam2_keys.intersection(updates):
            self._sam2_last_error = None
        if sam3_keys.intersection(updates):
            self._sam3_last_error = None
        for k, v in updates.items():
            setattr(self, k, v)
        if hasattr(self, "_keep_models_checkbox") and "keep_models_loaded" in updates:
            self._keep_models_checkbox.blockSignals(True)
            self._keep_models_checkbox.setChecked(bool(self.keep_models_loaded))
            self._keep_models_checkbox.blockSignals(False)
        self._refresh_model_status_indicators()

    def _get_propagation_state(self):
        """Return the state tuple expected by :class:`PropagationController`.

        Returns:
            tuple: ``(checkpoint_path, device, images_dir, save_dir,
                image_states, sort_index, image_list, current_image_path,
                predictor)``.
        """
        return (
            self.checkpoint_path,
            "cuda",
            self.images_dir,
            self.save_dir,
            self.image_states,
            self._left.sort_combo.currentIndex(),
            self._left.image_list,
            self.current_image_path,
            self.predictor,
        )

    def _get_batch_state(self):
        """Return ``(save_dir, image_states)`` for BatchSessionController."""
        return self.save_dir, self.image_states

    # ------------------------------------------------------------------
    # Batch UI refresh
    # ------------------------------------------------------------------

    def _on_batch_ui_update(self):
        """Refresh UI elements after any batch operation (propagate masks or grow)."""
        self._list_ctrl.update_image_list()
        self._list_ctrl.update_mask_counter()
        self._update_action_buttons()

        # Refresh the currently displayed image if it was affected
        if self.current_image_path is not None:
            self._load_current_image()

    def _update_action_buttons(self, *_args):
        """Enable / disable left-panel action buttons based on current state."""
        running = self._propagation_ctrl.is_running or self._is_prompt_batch_running
        has_session = self._batch_ctrl.has_session
        has_blocking_session = has_session and not self._batch_ctrl.is_saved
        sort_by_name = self._left.sort_combo.currentIndex() == 0
        has_selected = len(self._left.image_list.selectedItems()) > 0
        has_sam2 = self._has_valid_sam2_checkpoint()
        has_sam3 = self._can_use_sam3()

        self._left.segment_video_btn.setEnabled(sort_by_name and not has_blocking_session and not running and has_sam2)
        self._left.revert_btn.setEnabled(has_session and not running)
        self._left.save_all_btn.setEnabled(has_session and not running)
        self._left.grow_selected_btn.setEnabled(has_selected and not has_blocking_session and not running)

        # Enable prompt batch button for SAM3
        self._left.prompt_batch_btn.setEnabled(has_sam3 and has_selected and not has_blocking_session and not running)

    # ------------------------------------------------------------------
    # Select All / Deselect All
    # ------------------------------------------------------------------

    def _toggle_select_all(self):
        """Toggle between selecting all and deselecting all images."""
        lst = self._left.image_list
        total = lst.count()
        if total == 0:
            return

        all_selected = len(lst.selectedItems()) == total
        if all_selected:
            lst.clearSelection()
            self._left.select_all_btn.setText("Select All")
        else:
            lst.selectAll()
            self._left.select_all_btn.setText("Deselect All")

    # ------------------------------------------------------------------
    # Grow / shrink mask
    # ------------------------------------------------------------------

    def _grow_current_mask(self):
        """Apply grow/shrink from the right-panel controls to the current mask."""
        pixels = self._right.grow_pixels
        if pixels == 0:
            return
        if self.predictor is None and self.sam3_predictor is None:
            if not self._ensure_default_predictor_loaded():
                return
        display_predictor = self._get_display_predictor()
        self._mask_ctrl.grow_current_mask(
            pixels,
            display_predictor,
            self._list_ctrl.update_mask_counter,
        )

    def _grow_selected_masks(self):
        """Apply grow/shrink to all selected images with a progress bar.

        Runs on the main thread but calls ``processEvents`` after each
        frame so the progress dialog stays responsive and the user can
        cancel at any time.
        """
        from src.models.session_models import BatchSession, FrameBackup

        pixels = self._right.grow_pixels
        if pixels == 0:
            QMessageBox.information(
                self,
                "Nothing to Do",
                "The grow/shrink value is 0. Adjust the slider or enter a value first.",
            )
            return

        selected_paths = self._get_selected_image_paths()
        if not selected_paths:
            QMessageBox.information(
                self,
                "No Selection",
                "Please select one or more images in the list first.",
            )
            return

        n = len(selected_paths)
        direction = "grow" if pixels > 0 else "shrink"
        answer = QMessageBox.question(
            self,
            "Grow/Shrink Masks",
            f"{direction.capitalize()} masks by {abs(pixels)} px for {n} image(s)?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        # Mark all selected list items as "pending"
        self._set_list_items_prefix(selected_paths, "⏳")

        # Modal progress dialog with Cancel support
        progress = QProgressDialog(
            "Growing/shrinking masks…",
            "Cancel",
            0,
            n,
            self,
        )
        progress.setWindowTitle("Batch Mask Grow")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        session = BatchSession(operation_type="grow_mask")

        for i, img_path in enumerate(selected_paths):
            if progress.wasCanceled():
                break

            progress.setLabelText(f"Processing {img_path.name}… ({i + 1}/{n})")
            QApplication.processEvents()

            state = self.image_states.get(img_path)
            if state is None or state.mask is None:
                # Mark as done even if skipped (no mask)
                self._set_list_item_text_for_path(img_path, f"— {img_path.name}")
                progress.setValue(i + 1)
                continue

            # Snapshot before modification
            backup = FrameBackup(
                image_path=img_path,
                frame_idx=i,
                old_mask=state.mask.copy(),
                old_mask_saved=state.mask_saved,
                old_mask_logits=(state.mask_logits.copy() if state.mask_logits is not None else None),
            )

            new_mask = MaskService.grow_mask(state.mask, pixels)
            backup.new_mask = new_mask

            # Apply to in-memory state
            state.mask = new_mask
            state.mask_logits = None
            state.mask_candidates = []
            state.mask_scores = None
            state.selected_mask_idx = 0
            state.undo_stack.clear()
            state.redo_stack.clear()

            session.frame_backups.append(backup)

            # Update list item indicator to "done"
            self._set_list_item_text_for_path(img_path, f"✔ {img_path.name}")
            progress.setValue(i + 1)
            QApplication.processEvents()

        progress.close()

        if session.frame_backups:
            self._batch_ctrl.set_session(session)

        self._on_batch_ui_update()

    # ------------------------------------------------------------------
    # List item helpers
    # ------------------------------------------------------------------

    def _set_list_items_prefix(self, paths: List[Path], prefix: str):
        """Set a prefix indicator on every list item whose path is in *paths*.

        Args:
            - paths (list[Path]): Image paths to update.
            - prefix (str): Unicode prefix (e.g. ``"⏳"``).
        """
        path_set = set(paths)
        for i in range(self._left.image_list.count()):
            item = self._left.image_list.item(i)
            if item is None:
                continue
            p = item.data(Qt.ItemDataRole.UserRole)
            if p in path_set:
                item.setText(f"{prefix} {p.name}")

    def _set_list_item_text_for_path(self, img_path: Path, text: str):
        """Find the list item for *img_path* and set its display text.

        Args:
            - img_path (Path): Image path stored in item data.
            - text (str): New display text.
        """
        for i in range(self._left.image_list.count()):
            item = self._left.image_list.item(i)
            if item is not None and item.data(Qt.ItemDataRole.UserRole) == img_path:
                item.setText(text)
                break

    # ------------------------------------------------------------------
    # Thin slot wrappers
    # ------------------------------------------------------------------

    def _on_image_selected(self, current, _previous):
        """Forward image-list selection to the list controller."""
        path = self._list_ctrl.on_image_selected(current, self._load_current_image)
        if path is not None:
            self.current_image_path = path

    def _on_keypoint_added(self, x, y, kp_type):
        """Forward keypoint-added signal to the mask controller."""
        needs_reload = self.predictor is None or self._active_predictor_kind != "sam2"
        if not self._ensure_sam2_loaded(reload_current=needs_reload):
            return
        self._mask_ctrl.on_keypoint_added(x, y, kp_type, self.predictor)

    def _on_brush_stroke_finished(self, mask):
        """Forward brush-stroke-finished signal to the mask controller."""
        if self.predictor is None and self.sam3_predictor is None:
            if not self._ensure_default_predictor_loaded():
                return
        display_predictor = self._get_display_predictor()
        self._mask_ctrl.on_brush_stroke_finished(
            mask,
            display_predictor,
            self._list_ctrl.update_mask_counter,
        )

    def _save_mask(self):
        """Save the current mask to disk via the mask controller."""
        self._mask_ctrl.save_current_mask(
            self.save_dir,
            self.mask_service,
            self._left.sort_combo,
            self._left.image_list,
            self._list_ctrl.update_image_list,
            self._list_ctrl.update_mask_counter,
            self._list_ctrl.sort_image_list,
            parent_widget=self,
        )

    def _undo(self):
        """Undo the last mask/keypoint change."""
        state = self.image_states.get(self.current_image_path) if self.current_image_path else None
        if self.predictor is None and self.sam3_predictor is None:
            self._ensure_default_predictor_loaded()
        self._undo_ctrl.undo(state, self._get_display_predictor())
        self._list_ctrl.update_mask_counter()

    def _redo(self):
        """Redo the last undone change."""
        state = self.image_states.get(self.current_image_path) if self.current_image_path else None
        if self.predictor is None and self.sam3_predictor is None:
            self._ensure_default_predictor_loaded()
        self._undo_ctrl.redo(state, self._get_display_predictor())
        self._list_ctrl.update_mask_counter()

    def _go_previous(self):
        """Navigate to the previous image in the list."""
        row = self._left.image_list.currentRow()
        if row > 0:
            self._left.image_list.setCurrentRow(row - 1)

    def _go_next(self):
        """Navigate to the next image in the list."""
        row = self._left.image_list.currentRow()
        if row < self._left.image_list.count() - 1:
            self._left.image_list.setCurrentRow(row + 1)

    # ------------------------------------------------------------------
    # Helper: selected image paths
    # ------------------------------------------------------------------

    def _get_selected_image_paths(self) -> List[Path]:
        """Return the file paths of all currently selected list items.

        Returns:
            - list[Path]: Ordered list of selected image paths.
        """
        paths: List[Path] = []
        for item in self._left.image_list.selectedItems():
            p = item.data(Qt.ItemDataRole.UserRole)
            if p is not None:
                paths.append(p)
        return paths

    # ------------------------------------------------------------------
    # Slider / colour sync
    # ------------------------------------------------------------------

    def _sync_zoom_slider(self, zoom: float):
        """Update the zoom slider and label to match the viewer's zoom level.

        Args:
            - zoom (float): Current zoom factor (1.0 = 100%).
        """
        pct = int(zoom * 100)
        s = self._right.zoom_slider
        s.blockSignals(True)
        s.setValue(max(s.minimum(), min(s.maximum(), pct)))
        self._right.zoom_label.setText(f"{pct}%")
        s.blockSignals(False)

    def _sync_brush_slider(self, size: int):
        """Update the brush slider and label to match the viewer's brush size.

        Args:
            - size (int): Current brush diameter in pixels.
        """
        s = self._right.brush_slider
        s.blockSignals(True)
        s.setValue(max(s.minimum(), min(s.maximum(), size)))
        self._right.brush_label.setText(f"{size}px")
        s.blockSignals(False)

    def _on_brush_max_changed(self, max_size: int):
        """Adjust the brush slider's maximum when the image resolution changes.

        Args:
            - max_size (int): New maximum brush size.
        """
        s = self._right.brush_slider
        s.blockSignals(True)
        s.setMaximum(max_size)
        if s.value() > max_size:
            s.setValue(max_size)
            self._right.brush_label.setText(f"{max_size}px")
        s.blockSignals(False)

    def _pick_mask_colour(self):
        """Open a colour picker dialog and apply the chosen mask overlay colour."""
        r, g, b = self._center.image_viewer.mask_color
        colour = QColorDialog.getColor(QColor(r, g, b), self, "Choose mask colour")
        if colour.isValid():
            nr, ng, nb = colour.red(), colour.green(), colour.blue()
            self._center.image_viewer.set_mask_color(nr, ng, nb)
            self._right.mask_color_btn.setStyleSheet(
                f"background-color: rgb({nr}, {ng}, {nb}); border: 1px solid #666;"
            )

    def _on_opacity_changed(self, value: int):
        """Apply a new mask opacity from the slider.

        Args:
            - value (int): Alpha value in range [20, 255].
        """
        self._center.image_viewer.set_mask_alpha(value)
        self._right.opacity_label.setText(f"{round(value / 255 * 100)}%")

    # ------------------------------------------------------------------
    # Prompt segmentation
    # ------------------------------------------------------------------

    @staticmethod
    def _release_predictor_resources(predictor) -> None:
        """Drop a predictor and aggressively release its memory."""
        if predictor is not None and hasattr(predictor, "release"):
            predictor.release()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _release_all_predictors(self) -> None:
        """Release both predictor runtimes and clear the active predictor state."""
        logger.info("Releasing all predictors (SAM2, SAM3)")
        self._release_predictor_resources(self.predictor)
        self._release_predictor_resources(self.sam3_predictor)
        self.predictor = None
        self.sam3_predictor = None
        self.mask_service = None
        self._active_predictor_kind = None
        self._refresh_model_status_indicators()
        logger.debug("All predictors released")

    def _get_display_predictor(self):
        """Return the predictor that should currently drive the viewer."""
        if self._active_predictor_kind == "sam3" and self.sam3_predictor is not None:
            return self.sam3_predictor
        if self._active_predictor_kind == "sam2" and self.predictor is not None:
            return self.predictor
        return self.predictor or self.sam3_predictor

    def _with_loading_progress(self, title: str, label: str, loader: Callable[[], Any]):
        """Show a busy progress dialog while a model loads in a worker thread."""
        logger.info("Starting model load with progress dialog: %s", title)
        progress = QProgressDialog(label, None, 0, 0, self)
        progress.setWindowTitle(title)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setCancelButton(None)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.setRange(0, 0)
        progress.show()
        progress.repaint()
        if hasattr(progress, "forceShow"):
            progress.forceShow()
        QApplication.processEvents()

        result: dict[str, Any] = {"model": None, "error": None}
        loop = QEventLoop(self)
        worker = ModelLoadWorker(loader, self)
        worker.loaded.connect(lambda model: result.__setitem__("model", model))
        worker.error.connect(lambda message: result.__setitem__("error", message))
        worker.finished.connect(loop.quit)
        worker.start()
        try:
            loop.exec()
        finally:
            worker.wait()
            worker.deleteLater()
            progress.close()
            QApplication.processEvents()

        if result["error"] is not None:
            logger.error("Model load failed (%s): %s", title, result["error"])
            raise RuntimeError(result["error"])
        logger.info("Model load completed: %s", title)
        return result["model"]

    @staticmethod
    def _status_colour(state: str) -> str:
        """Map a logical model state to a badge colour."""
        if state == "loaded":
            return "#2e9d49"
        if state == "not_loaded":
            return "#c93c3c"
        return "#7a7a7a"

    @staticmethod
    def _apply_status_badge(dot: QFrame, container: QWidget, state: str, tooltip: str) -> None:
        """Update the visual state and tooltip for one model badge."""
        colour = MainWindow._status_colour(state)
        dot.setStyleSheet(f"background-color: {colour}; border: 1px solid #555; border-radius: 2px;")
        container.setToolTip(tooltip)
        dot.setToolTip(tooltip)
        for label in container.findChildren(QLabel):
            label.setToolTip(tooltip)

    def _validate_sam2_setup(self) -> tuple[bool, str]:
        """Return whether SAM2 is configured well enough to be loaded."""
        logger.debug("Validating SAM2 setup (checkpoint_path=%s)", self.checkpoint_path)
        sam2_installed, sam2_msg = check_sam2_installed()
        if not sam2_installed:
            logger.info("SAM2 validation failed: package not installed — %s", sam2_msg)
            return False, sam2_msg or "SAM2 package is not installed."
        if not self.checkpoint_path:
            logger.info("SAM2 validation failed: checkpoint path not configured")
            return False, "SAM2 checkpoint path is not configured."
        ckpt_path = Path(self.checkpoint_path)
        if not ckpt_path.exists():
            logger.info("SAM2 validation failed: checkpoint file does not exist: %s", ckpt_path)
            return False, f"SAM2 checkpoint file does not exist: {ckpt_path}"
        if not ckpt_path.is_file():
            logger.info("SAM2 validation failed: checkpoint path is not a file: %s", ckpt_path)
            return False, f"SAM2 checkpoint path is not a file: {ckpt_path}"
        logger.debug("SAM2 setup valid: %s", ckpt_path)
        return True, "SAM2 is configured and ready to load."

    def _has_valid_sam2_checkpoint(self) -> bool:
        """Return True when the configured SAM2 checkpoint exists on disk."""
        valid, _ = self._validate_sam2_setup()
        return valid

    def _validate_sam3_setup(self) -> tuple[bool, str]:
        """Return whether SAM3 is configured well enough to be loaded."""
        logger.debug(
            "Validating SAM3 setup (checkpoint_path=%s, bpe_path=%s)",
            self.sam3_checkpoint_path,
            self.sam3_bpe_path,
        )
        sam3_installed, sam3_msg = check_sam3_installed()
        if not sam3_installed:
            logger.info("SAM3 validation failed: package not installed — %s", sam3_msg)
            return False, sam3_msg or "SAM3 package is not installed."
        if not self.sam3_checkpoint_path:
            logger.info("SAM3 validation failed: checkpoint path not configured")
            return False, "SAM3 checkpoint path is not configured."
        ckpt_path = Path(self.sam3_checkpoint_path)
        if not ckpt_path.exists():
            logger.info("SAM3 validation failed: checkpoint file does not exist: %s", ckpt_path)
            return False, f"SAM3 checkpoint file does not exist: {ckpt_path}"
        if not ckpt_path.is_file():
            logger.info("SAM3 validation failed: checkpoint path is not a file: %s", ckpt_path)
            return False, f"SAM3 checkpoint path is not a file: {ckpt_path}"
        if self.sam3_bpe_path:
            bpe_path = Path(self.sam3_bpe_path)
            if not bpe_path.exists():
                logger.info("SAM3 validation failed: BPE file does not exist: %s", bpe_path)
                return False, f"SAM3 BPE file does not exist: {bpe_path}"
            if not bpe_path.is_file():
                logger.info("SAM3 validation failed: BPE path is not a file: %s", bpe_path)
                return False, f"SAM3 BPE path is not a file: {bpe_path}"
            if not bpe_path.name.endswith(".txt.gz"):
                logger.info("SAM3 validation failed: invalid BPE file name: %s", bpe_path.name)
                return (
                    False,
                    "SAM3 BPE file has an invalid name. Expected something like 'bpe_simple_vocab_16e6.txt.gz'.",
                )
        logger.debug("SAM3 setup valid: checkpoint=%s", ckpt_path)
        return True, "SAM3 is configured and ready to load."

    def _can_use_sam3(self) -> bool:
        """Return True when SAM3 is installed and a checkpoint path is configured."""
        valid, _ = self._validate_sam3_setup()
        return valid

    def _get_sam2_status(self) -> tuple[str, str]:
        """Return ``(state, tooltip)`` for the SAM2 status badge."""
        if self.predictor is not None:
            return "loaded", "SAM2 loaded in memory."
        if self._sam2_last_error:
            return "unavailable", f"SAM2 initialisation error:\n{self._sam2_last_error}"
        valid, message = self._validate_sam2_setup()
        if not valid:
            return "unavailable", message
        return "not_loaded", "SAM2 is available but not currently loaded in memory."

    def _get_sam3_status(self) -> tuple[str, str]:
        """Return ``(state, tooltip)`` for the SAM3 status badge."""
        if self.sam3_predictor is not None:
            return "loaded", "SAM3 loaded in memory."
        if self._sam3_last_error:
            return "unavailable", f"SAM3 initialisation error:\n{self._sam3_last_error}"
        valid, message = self._validate_sam3_setup()
        if not valid:
            return "unavailable", message
        return "not_loaded", "SAM3 is available but not currently loaded in memory."

    def _refresh_model_status_indicators(self) -> None:
        """Refresh the top badges that summarise SAM2 and SAM3 availability."""
        if not hasattr(self, "_sam2_status_dot"):
            return
        sam2_state, sam2_tooltip = self._get_sam2_status()
        sam3_state, sam3_tooltip = self._get_sam3_status()
        logger.debug("Model status: SAM2=%s, SAM3=%s", sam2_state, sam3_state)
        self._apply_status_badge(
            self._sam2_status_dot,
            self._sam2_status_badge,
            sam2_state,
            sam2_tooltip,
        )
        self._apply_status_badge(
            self._sam3_status_dot,
            self._sam3_status_badge,
            sam3_state,
            sam3_tooltip,
        )

    def _build_sam2_predictor(self):
        """Create SAM2 with the stored configuration."""
        logger.info("Building SAM2 predictor (checkpoint=%s, max_side=%s)", self.checkpoint_path, self.max_side)
        from src.sam2 import SAM2PredictorWrapper

        predictor = SAM2PredictorWrapper(self.checkpoint_path)
        predictor.set_max_side(self.max_side)
        logger.info("SAM2 predictor built successfully")
        return predictor

    def _build_sam3_predictor(self):
        """Create SAM3 with the stored configuration and BPE fallback."""
        logger.info(
            "Building SAM3 predictor (checkpoint=%s, bpe_path=%s)",
            self.sam3_checkpoint_path,
            self.sam3_bpe_path,
        )
        from src.sam3 import SAM3PredictorWrapper

        try:
            predictor = SAM3PredictorWrapper(
                checkpoint_path=self.sam3_checkpoint_path,
                bpe_path=self.sam3_bpe_path,
            )
        except ValueError as exc:
            if self.sam3_bpe_path and "Invalid SAM3 BPE file" in str(exc):
                logger.warning("SAM3 BPE invalid, retrying without BPE: %s", exc)
                predictor = SAM3PredictorWrapper(
                    checkpoint_path=self.sam3_checkpoint_path,
                    bpe_path=None,
                )
            else:
                raise
        except Exception as exc:
            logger.error("SAM3 predictor construction failed: %s", exc, exc_info=True)
            raise
        predictor.set_max_side(self.max_side)
        logger.info("SAM3 predictor built successfully")
        return predictor

    def _ensure_sam2_loaded(self, reload_current: bool = False) -> bool:
        """Load SAM2 on demand and release SAM3 if it is currently active."""
        logger.debug("_ensure_sam2_loaded(reload_current=%s)", reload_current)
        if not self._has_valid_sam2_checkpoint():
            logger.info("SAM2 not loaded: checkpoint not valid, showing warning")
            QMessageBox.warning(
                self,
                "SAM2 Not Configured",
                "Please set a valid SAM2 checkpoint path in Settings.",
            )
            return False
        if self.predictor is None:
            logger.info("SAM2 predictor is None, loading SAM2 model")
            if self.sam3_predictor is not None and not self.keep_models_loaded:
                logger.debug("Releasing SAM3 to free memory before loading SAM2")
                self._release_predictor_resources(self.sam3_predictor)
                self.sam3_predictor = None
            try:
                self._sam2_last_error = None
                predictor = self._with_loading_progress(
                    "Loading SAM2",
                    "Loading SAM2 model…",
                    self._build_sam2_predictor,
                )
            except Exception as exc:
                self._sam2_last_error = str(exc)
                logger.error("Failed to load SAM2 model: %s", exc, exc_info=True)
                self._refresh_model_status_indicators()
                QMessageBox.critical(self, "Error", f"Failed to load SAM2 model:\n\n{exc}")
                return False
            self.predictor = predictor
            self._sam2_last_error = None
            logger.info("SAM2 model loaded successfully")
        if self.predictor is not None:
            self.mask_service = MaskService(self.predictor)
        self._active_predictor_kind = "sam2"
        self._refresh_model_status_indicators()
        self._update_prompt_ui_availability()
        if reload_current and self.current_image_path is not None:
            self._load_current_image(self.current_image_path)
        return True

    def _ensure_sam3_loaded(self, reload_current: bool = False) -> bool:
        """Load SAM3 on demand and release SAM2 if it is currently active."""
        logger.debug("_ensure_sam3_loaded(reload_current=%s)", reload_current)
        if not self._can_use_sam3():
            valid, msg = self._validate_sam3_setup()
            logger.info("SAM3 not loaded: validation failed — %s", msg)
            QMessageBox.warning(
                self,
                "SAM3 Not Configured",
                "Prompt segmentation is unavailable until SAM3 and its checkpoint are configured.",
            )
            return False
        if self.sam3_predictor is None:
            logger.info("SAM3 predictor is None, loading SAM3 model")
            if self.predictor is not None and not self.keep_models_loaded:
                logger.debug("Releasing SAM2 to free memory before loading SAM3")
                self._release_predictor_resources(self.predictor)
                self.predictor = None
            try:
                self._sam3_last_error = None
                sam3_predictor = self._with_loading_progress(
                    "Loading SAM3",
                    "Loading SAM3 model…",
                    self._build_sam3_predictor,
                )
            except Exception as exc:
                self._sam3_last_error = str(exc)
                logger.error("Failed to load SAM3 model: %s", exc, exc_info=True)
                self._refresh_model_status_indicators()
                QMessageBox.critical(self, "Error", f"Failed to load SAM3 model:\n\n{exc}")
                return False
            self.sam3_predictor = sam3_predictor
            self._sam3_last_error = None
            logger.info("SAM3 model loaded successfully")
        if self.sam3_predictor is not None:
            self.mask_service = MaskService(self.sam3_predictor)
        self._active_predictor_kind = "sam3"
        self._refresh_model_status_indicators()
        self._update_prompt_ui_availability()
        if reload_current and self.current_image_path is not None:
            self._load_current_image(self.current_image_path)
        return True

    def _ensure_default_predictor_loaded(self) -> bool:
        """Load the default viewer predictor when nothing is active yet."""
        logger.debug("_ensure_default_predictor_loaded (active=%s)", self._active_predictor_kind)
        if self._active_predictor_kind == "sam2" and self.predictor is not None:
            logger.debug("Default predictor: SAM2 already loaded")
            return True
        if self._active_predictor_kind == "sam3" and self.sam3_predictor is not None:
            logger.debug("Default predictor: SAM3 already loaded")
            return True
        if self._has_valid_sam2_checkpoint():
            logger.debug("Loading default predictor: trying SAM2 first")
            return self._ensure_sam2_loaded(reload_current=False)
        if self._can_use_sam3():
            logger.debug("Loading default predictor: trying SAM3")
            return self._ensure_sam3_loaded(reload_current=False)
        logger.warning("No valid SAM2 or SAM3 setup; default predictor could not be loaded")
        return False

    def _try_keep_both_models_loaded(self) -> bool:
        """Load both models immediately so future switching is instant."""
        if not self._has_valid_sam2_checkpoint() or not self._can_use_sam3():
            QMessageBox.warning(
                self,
                "Cannot Keep Both Models",
                "Configure valid SAM2 and SAM3 checkpoints first.",
            )
            return False

        previous_kind = self._active_predictor_kind
        previous_predictor = self.predictor
        previous_sam3_predictor = self.sam3_predictor
        previous_mask_service = self.mask_service
        previous_sam2_error = self._sam2_last_error
        previous_sam3_error = self._sam3_last_error
        loaded_sam2_here = False
        loaded_sam3_here = False

        try:
            if self.predictor is None:
                self._sam2_last_error = None
                predictor = self._with_loading_progress(
                    "Loading SAM2",
                    "Loading SAM2 model…",
                    self._build_sam2_predictor,
                )
                self.predictor = predictor
                loaded_sam2_here = True
            if self.sam3_predictor is None:
                self._sam3_last_error = None
                self.sam3_predictor = self._with_loading_progress(
                    "Loading SAM3",
                    "Loading SAM3 model…",
                    self._build_sam3_predictor,
                )
                loaded_sam3_here = True

            if previous_kind == "sam3" and self.sam3_predictor is not None:
                self.mask_service = MaskService(self.sam3_predictor)
            elif self.predictor is not None:
                self.mask_service = MaskService(self.predictor)
            self._refresh_model_status_indicators()
            return True
        except Exception as exc:
            if loaded_sam2_here and self.predictor is not None:
                self._release_predictor_resources(self.predictor)
            if loaded_sam3_here and self.sam3_predictor is not None:
                self._release_predictor_resources(self.sam3_predictor)
            self.predictor = previous_predictor
            self.sam3_predictor = previous_sam3_predictor
            self.mask_service = previous_mask_service
            self._active_predictor_kind = previous_kind
            self._sam2_last_error = previous_sam2_error
            self._sam3_last_error = previous_sam3_error
            if previous_predictor is None and previous_sam3_predictor is not None:
                self._sam2_last_error = str(exc)
            elif previous_sam3_predictor is None and previous_predictor is not None:
                self._sam3_last_error = str(exc)
            elif previous_predictor is None and previous_sam3_predictor is None:
                error_text = str(exc)
                if "sam3" in error_text.lower():
                    self._sam3_last_error = error_text
                elif "sam2" in error_text.lower():
                    self._sam2_last_error = error_text
            self._refresh_model_status_indicators()
            QMessageBox.critical(
                self,
                "Failed to Load Both Models",
                f"Could not keep both models loaded:\n\n{exc}",
            )
            return False

    def _disable_keep_both_mode(self) -> None:
        """Disable dual-model residency while keeping SAM2 loaded when available."""
        had_sam2 = self.predictor is not None
        had_sam3 = self.sam3_predictor is not None

        if had_sam2 and had_sam3:
            self._release_predictor_resources(self.sam3_predictor)
            self.sam3_predictor = None

        if had_sam2:
            predictor = self.predictor
            if predictor is not None:
                self.mask_service = MaskService(predictor)
            self._active_predictor_kind = "sam2"
            if had_sam3 and self.current_image_path is not None:
                self._load_current_image(self.current_image_path)
        elif had_sam3:
            sam3_predictor = self.sam3_predictor
            if sam3_predictor is not None:
                self.mask_service = MaskService(sam3_predictor)
            self._active_predictor_kind = "sam3"

        self._refresh_model_status_indicators()
        self._update_prompt_ui_availability()

    def _on_keep_models_toggled(self, checked: bool):
        """Handle the top checkbox controlling dual-model residency."""
        if checked:
            answer = QMessageBox.question(
                self,
                "Keep Both Models Loaded",
                "Keeping both models loaded uses much more memory.\n\nContinue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                self._keep_models_checkbox.blockSignals(True)
                self._keep_models_checkbox.setChecked(False)
                self._keep_models_checkbox.blockSignals(False)
                return

            self.keep_models_loaded = True
            if not self._try_keep_both_models_loaded():
                self.keep_models_loaded = False
                self._keep_models_checkbox.blockSignals(True)
                self._keep_models_checkbox.setChecked(False)
                self._keep_models_checkbox.blockSignals(False)
            self._settings_ctrl.save_config()
            return

        self.keep_models_loaded = False
        self._settings_ctrl.save_config()
        self._disable_keep_both_mode()

    @staticmethod
    def _rescale_mask_between_sizes(
        mask: np.ndarray,
        source_size: tuple[int, int],
        target_size: tuple[int, int],
    ) -> np.ndarray:
        """Resize a binary mask between two scaled resolutions."""
        if mask is None:
            return mask
        if source_size == target_size:
            return mask
        h, w = target_size
        return np.asarray(
            cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST),
            dtype=np.uint8,
        )

    def _rescale_state_for_new_scale(
        self,
        state: ImageState,
        new_scaled_size: tuple[int, int],
        new_scale_factor: float,
    ) -> None:
        """Project stored keypoints and masks from the previous scale to the new one."""
        old_scaled_size = state.scaled_size

        if old_scaled_size is None or old_scaled_size == new_scaled_size:
            return

        if old_scaled_size[0] <= 0 or old_scaled_size[1] <= 0:
            return

        ratio_x = new_scaled_size[1] / float(old_scaled_size[1])
        ratio_y = new_scaled_size[0] / float(old_scaled_size[0])

        if state.keypoints:
            rescaled_keypoints: List[Keypoint] = []
            max_x = max(new_scaled_size[1] - 1, 0)
            max_y = max(new_scaled_size[0] - 1, 0)
            for kp in state.keypoints:
                rescaled_keypoints.append(
                    Keypoint(
                        x=max(0, min(max_x, int(round(kp.x * ratio_x)))),
                        y=max(0, min(max_y, int(round(kp.y * ratio_y)))),
                        type=kp.type,
                    )
                )
            state.keypoints = rescaled_keypoints

        if state.mask is not None:
            state.mask = self._rescale_mask_between_sizes(
                state.mask,
                old_scaled_size,
                new_scaled_size,
            )

        if state.mask_candidates:
            state.mask_candidates = [
                self._rescale_mask_between_sizes(mask, old_scaled_size, new_scaled_size)
                for mask in state.mask_candidates
            ]

        # Logits are tied to the previous scaled resolution and must be recomputed.
        state.mask_logits = None
        state.mask_logits_all = None
        state.scale_factor = new_scale_factor

    def _update_prompt_ui_availability(self):
        """Show prompt controls only when SAM3 is configured and loaded."""
        has_sam3 = self._can_use_sam3()
        self._right.set_prompt_visible(has_sam3)
        self._left.set_prompt_batch_visible(has_sam3)
        self._update_action_buttons()

    def _segment_by_prompt(self):
        """Segment current image using text prompt."""
        prompt = self._right.get_prompt_text()
        if not prompt:
            QMessageBox.warning(self, "Warning", "Please enter a text prompt.")
            return
        if not self._ensure_sam3_loaded(reload_current=True):
            return
        self._mask_ctrl.segment_by_prompt(
            prompt,
            self.sam3_predictor,
            self.current_image_path,
            parent_widget=self,
        )
        self._list_ctrl.update_mask_counter()

    def _segment_selected_by_prompt(self):
        """Segment selected images using text prompt."""
        if self._is_prompt_batch_running or self._propagation_ctrl.is_running:
            return
        self._is_prompt_batch_running = True
        self._update_action_buttons()

        prompt = self._left.get_prompt_batch_text()
        if not prompt:
            self._is_prompt_batch_running = False
            self._update_action_buttons()
            QMessageBox.warning(self, "Warning", "Please enter a text prompt.")
            return

        selected_paths = self._get_selected_image_paths()
        if not selected_paths:
            self._is_prompt_batch_running = False
            self._update_action_buttons()
            QMessageBox.information(
                self,
                "No Selection",
                "Please select one or more images in the list first.",
            )
            return

        if not self._ensure_sam3_loaded(reload_current=False):
            self._is_prompt_batch_running = False
            self._update_action_buttons()
            return

        n = len(selected_paths)
        answer = QMessageBox.question(
            self,
            "Segment by Prompt",
            f"Segment {n} image(s) using prompt '{prompt}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if answer != QMessageBox.StandardButton.Yes:
            self._is_prompt_batch_running = False
            self._update_action_buttons()
            return

        prompt = prompt.strip()
        self._set_list_items_prefix(selected_paths, "⏳")
        progress = QProgressDialog("Segmenting by prompt…", "Cancel", 0, n, self)
        progress.setWindowTitle("Batch Prompt Segmentation")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        session = BatchSession(operation_type="prompt_batch")
        errors: List[str] = []

        try:
            for idx, img_path in enumerate(selected_paths):
                if progress.wasCanceled():
                    break

                progress.setLabelText(f"Processing {img_path.name}… ({idx + 1}/{n})")
                QApplication.processEvents()

                state = self.image_states.get(img_path)
                if state is None:
                    self._set_list_item_text_for_path(img_path, img_path.name)
                    progress.setValue(idx + 1)
                    continue

                backup = FrameBackup(
                    image_path=img_path,
                    frame_idx=idx,
                    old_mask=state.mask.copy() if state.mask is not None else None,
                    old_mask_saved=state.mask_saved,
                    old_mask_logits=(state.mask_logits.copy() if state.mask_logits is not None else None),
                )

                try:
                    _, original_size, scale_factor = self.sam3_predictor.load_image(str(img_path))
                    scaled_size = self.sam3_predictor.get_scaled_size()
                    masks, scores = self.sam3_predictor.predict_mask_from_text(prompt)

                    mask_candidates: List[np.ndarray] = []
                    for mask in masks:
                        mask_np = np.asarray(mask, dtype=np.uint8)
                        if mask_np.ndim > 2:
                            mask_np = np.squeeze(mask_np)
                        if mask_np.ndim == 2:
                            mask_candidates.append(mask_np.copy())

                    if not mask_candidates:
                        errors.append(f"{img_path.name}: no mask candidates returned.")
                        self._set_list_item_text_for_path(img_path, img_path.name)
                        progress.setValue(idx + 1)
                        continue

                    scores_np = np.asarray(scores, dtype=np.float32).reshape(-1)
                    if scores_np.size == 0:
                        scores_np = np.ones(len(mask_candidates), dtype=np.float32)
                    elif scores_np.size < len(mask_candidates):
                        pad = np.ones(len(mask_candidates) - scores_np.size, dtype=np.float32)
                        scores_np = np.concatenate([scores_np, pad], axis=0)
                    elif scores_np.size > len(mask_candidates):
                        scores_np = scores_np[: len(mask_candidates)]

                    best_idx = int(np.argmax(scores_np)) if scores_np.size > 0 else 0
                    best_mask = mask_candidates[best_idx]
                    if not np.any(best_mask):
                        errors.append(f"{img_path.name}: prompt returned an empty mask.")
                        self._set_list_item_text_for_path(img_path, img_path.name)
                        progress.setValue(idx + 1)
                        continue

                    state.mask = best_mask.copy()
                    state.keypoints = []
                    state.mask_candidates = [mask.copy() for mask in mask_candidates]
                    state.mask_scores = scores_np
                    state.selected_mask_idx = best_idx
                    state.mask_logits = None
                    state.mask_logits_all = None
                    state.original_size = original_size
                    state.scaled_size = scaled_size
                    state.scale_factor = scale_factor
                    state.undo_stack.clear()
                    state.redo_stack.clear()
                    state.state_version = self._undo_ctrl.next_version()
                    self._undo_ctrl.sync_unsaved(state)

                    backup.new_mask = state.mask.copy()
                    session.frame_backups.append(backup)
                except Exception as exc:
                    errors.append(f"{img_path.name}: {exc}")

                self._set_list_item_text_for_path(img_path, img_path.name)
                progress.setValue(idx + 1)
                QApplication.processEvents()
        finally:
            progress.close()
            self._is_prompt_batch_running = False
            self._update_action_buttons()

        if session.frame_backups:
            self._batch_ctrl.set_session(session)

        self._on_batch_ui_update()

        if errors:
            shown_errors = "\n".join(errors[:12])
            extra = "" if len(errors) <= 12 else f"\n... and {len(errors) - 12} more."
            QMessageBox.warning(
                self,
                "Batch Prompt Segmentation",
                f"Some images were not segmented:\n\n{shown_errors}{extra}",
            )

    # ------------------------------------------------------------------
    # Folder actions
    # ------------------------------------------------------------------

    def _open_images_folder(self):
        """Prompt the user to select an images folder and load its contents."""
        start_dir = str(self.images_dir) if self.images_dir and self.images_dir.exists() else ""
        folder = QFileDialog.getExistingDirectory(self, "Select Images Folder", start_dir)
        if folder:
            self.images_dir = Path(folder)
            self._settings_ctrl.save_config()
            self._load_images()

    def _set_save_folder(self):
        """Prompt the user to select a save folder for masks."""
        start_dir = str(self.save_dir) if self.save_dir and self.save_dir.exists() else ""
        folder = QFileDialog.getExistingDirectory(self, "Select Save Folder", start_dir)
        if folder:
            self.save_dir = Path(folder)
            self._settings_ctrl.save_config()
            self._list_ctrl.update_image_list()
            self._check_all_masks()
            if self.current_image_path:
                if self.predictor is None and self.sam3_predictor is None:
                    self._ensure_default_predictor_loaded()
                display_predictor = self._get_display_predictor()
                self._mask_ctrl.check_and_load_mask_for_current(
                    self.save_dir,
                    self.mask_service,
                    display_predictor,
                    self._list_ctrl.update_image_list,
                )

    def show_settings(self):
        """Open the settings dialog (also called from ``main.py`` on first launch)."""
        updated = self._settings_ctrl.show_settings(self, self._load_current_image)
        self._update_prompt_ui_availability()
        return updated

    def _show_shortcuts_dialog(self) -> None:
        """Open the keyboard shortcuts dialog."""
        dlg = ShortcutsDialog(self)
        dlg.exec()

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    def _load_images(self):
        """Scan the images directory and populate the image list."""
        logger.debug("_load_images: calling list controller")
        self._list_ctrl.load_images(self._on_image_selected, parent_widget=self)
        self._update_action_buttons()
        logger.debug("_load_images: done")

    def _check_all_masks(self):
        """Scan all image states and load saved masks from disk."""
        logger.debug("_check_all_masks: scanning for saved masks")
        self._mask_ctrl.check_and_load_masks_for_all(
            self.save_dir,
            self.mask_service,
            self._left.image_list,
            self.image_states,
            self._list_ctrl.update_mask_counter,
        )

    def _load_current_image(self, img_path=None):
        """Load the specified (or current) image into the viewer.

        Args:
            - img_path (Path | None): Image to load; uses
              ``self.current_image_path`` when ``None``.
        """
        if img_path is not None:
            self.current_image_path = img_path
        logger.debug("_load_current_image: path=%s", self.current_image_path)

        if self.predictor is None and self.sam3_predictor is None:
            if not self._ensure_default_predictor_loaded():
                logger.debug("_load_current_image: no predictor, aborting")
                return

        display_predictor = self._get_display_predictor()
        if self.current_image_path is None or display_predictor is None:
            logger.debug("_load_current_image: no path or predictor")
            return

        state = self.image_states.get(self.current_image_path)
        if state is None:
            logger.debug("_load_current_image: no state for path")
            return

        viewer = self._center.image_viewer

        try:
            old_scaled_size = state.scaled_size
            logger.debug("_load_current_image: loading image with predictor")
            scaled_img, original_size, scale_factor = display_predictor.load_image(str(self.current_image_path))
            new_scaled_size = display_predictor.get_scaled_size()
            self._rescale_state_for_new_scale(state, new_scaled_size, scale_factor)
            state.original_size = original_size
            state.scaled_size = new_scaled_size
            state.scale_factor = scale_factor

            if state.mask is not None:
                mh, mw = state.mask.shape[:2]
                ih, iw = scaled_img.shape[:2]
                if mh != ih or mw != iw:
                    if state.original_size is not None and (mh, mw) == state.original_size:
                        state.mask = display_predictor.downscale_mask(state.mask)
                    elif old_scaled_size is not None:
                        state.mask = self._rescale_mask_between_sizes(
                            state.mask,
                            old_scaled_size,
                            new_scaled_size,
                        )
                    if state.mask_logits is None and display_predictor is self.predictor:
                        state.mask_logits = display_predictor.mask_to_logits(state.mask)

            viewer.set_image(scaled_img)
            viewer.set_keypoints(state.keypoints)

            if self.save_dir and state.mask is None:
                self._mask_ctrl.check_and_load_mask_for_current(
                    self.save_dir,
                    self.mask_service,
                    display_predictor,
                    self._list_ctrl.update_image_list,
                )
                state = self.image_states.get(self.current_image_path)

            if state.mask is not None:
                viewer.set_mask(state.mask)
                if state.mask_logits is None and display_predictor is self.predictor:
                    state.mask_logits = display_predictor.mask_to_logits(state.mask)
                if len(state.mask_candidates) > 0:
                    scores = state.mask_scores if state.mask_scores is not None else np.array([])
                    self._right.mask_selector.set_masks(
                        state.mask_candidates,
                        scores,
                        scaled_img,
                    )
                    self._right.mask_selector.select_mask(state.selected_mask_idx)
            elif state.keypoints:
                if self.predictor is None and self._has_valid_sam2_checkpoint():
                    if self._ensure_sam2_loaded(reload_current=True):
                        return
                self._mask_ctrl.update_mask(self.predictor, parent_widget=self)
            else:
                viewer.set_mask(None)
                self._right.mask_selector.set_masks([], np.array([]), scaled_img)

            if state.state_version == 0:
                state.state_version = self._undo_ctrl.next_version()
                state.saved_version = state.state_version if state.mask_saved else -1

            self._undo_ctrl.sync_unsaved(state)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load image: {e}")

    # ------------------------------------------------------------------
    # Qt overrides
    # ------------------------------------------------------------------

    def keyPressEvent(self, event):
        """Arrow keys navigate the image list."""
        if event.key() == Qt.Key.Key_Left:
            self._go_previous()
        elif event.key() == Qt.Key.Key_Right:
            self._go_next()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        """Stop background threads on window close."""
        if self._thumb_worker.isRunning():
            self._thumb_worker.stop()
        super().closeEvent(event)

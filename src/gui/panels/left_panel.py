"""Left sidebar panel: image list with sorting, mask counter, and action buttons."""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class LeftPanel(QWidget):
    """Panel containing the image list, sort selector, mask counter,
    Select All toggle, and action buttons (Revert/Save, Grow Mask,
    Propagate Masks).

    All child widgets are exposed as public attributes so that controllers
    can wire signals and read/write state.
    """

    def __init__(self, parent=None):
        """Initialise the left panel with image list, sort selector, mask counter,
        multi-selection toggle, and action buttons.

        Args:
            - parent (QWidget | None): Parent widget.
        """
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        # Mask counter
        self.mask_counter_label = QLabel("0/0")
        self.mask_counter_label.setStyleSheet("font-weight: bold; font-size: 11pt; padding: 5px;")
        self.mask_counter_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.mask_counter_label)

        # Sort selector
        sort_layout = QHBoxLayout()
        sort_layout.addWidget(QLabel("Sort:"))
        self.sort_combo = QComboBox()
        self.sort_combo.addItems(["By name", "Unmasked first", "Masked first"])
        sort_layout.addWidget(self.sort_combo)
        layout.addLayout(sort_layout)

        # Select All / Deselect All toggle button
        self.select_all_btn = QPushButton("Select All")
        self.select_all_btn.setToolTip("Select or deselect all images in the list")
        layout.addWidget(self.select_all_btn)

        # Image list — ExtendedSelection allows Ctrl+click for multi-select
        self.image_list = QListWidget()
        self.image_list.setMaximumWidth(300)
        self.image_list.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection,
        )
        layout.addWidget(self.image_list)

        # ------------------------------------------------------------------
        # Action buttons — three rows
        # ------------------------------------------------------------------

        # Row 1: Revert | Save All
        revert_save_row = QHBoxLayout()
        revert_save_row.setSpacing(4)

        self.revert_btn = QPushButton("Revert")
        self.revert_btn.setToolTip("Revert the last batch operation (propagate masks or mask grow)")
        self.revert_btn.setEnabled(False)
        revert_save_row.addWidget(self.revert_btn, stretch=1)

        self.save_all_btn = QPushButton("Save All")
        self.save_all_btn.setToolTip("Save all modified masks from the last batch operation to disk")
        self.save_all_btn.setEnabled(False)
        revert_save_row.addWidget(self.save_all_btn, stretch=1)

        layout.addLayout(revert_save_row)

        # Row 2: Grow Mask For Selected Images
        self.grow_selected_btn = QPushButton("Grow Mask For Selected")
        self.grow_selected_btn.setToolTip("Apply the grow/shrink value from the right panel to all selected images")
        self.grow_selected_btn.setEnabled(False)
        layout.addWidget(self.grow_selected_btn)

        # Row 3: Propagate Masks
        self.segment_video_btn = QPushButton("Propagate Masks")
        self.segment_video_btn.setToolTip(
            "Propagate masks from annotated key-frames to all images (requires 'By name' sort order)"
        )
        self.segment_video_btn.setEnabled(False)
        layout.addWidget(self.segment_video_btn)

        # ------------------------------------------------------------------
        # Prompt Batch group (visible only for SAM3)
        # ------------------------------------------------------------------
        self.prompt_batch_group = QGroupBox("Prompt Batch")
        pg = QVBoxLayout(self.prompt_batch_group)
        pg.setContentsMargins(6, 6, 6, 6)
        pg.setSpacing(5)

        self.prompt_batch_edit = QLineEdit()
        self.prompt_batch_edit.setPlaceholderText("Enter text prompt...")
        self.prompt_batch_edit.setToolTip("Text description of what to segment (SAM3 only)")
        pg.addWidget(self.prompt_batch_edit)

        self.prompt_batch_btn = QPushButton("Segment Selected by Prompt")
        self.prompt_batch_btn.setToolTip("Segment all selected images using the text prompt (SAM3 only)")
        self.prompt_batch_btn.setEnabled(False)
        pg.addWidget(self.prompt_batch_btn)

        # Initially hidden (shown only when SAM3 is selected)
        self.prompt_batch_group.setVisible(False)
        layout.addWidget(self.prompt_batch_group)

    def get_prompt_batch_text(self) -> str:
        """Return the text prompt from the batch prompt input field."""
        return self.prompt_batch_edit.text().strip()

    def set_prompt_batch_visible(self, visible: bool):
        """Show or hide the prompt batch group.

        Args:
            - visible (bool): True to show, False to hide.
        """
        self.prompt_batch_group.setVisible(visible)

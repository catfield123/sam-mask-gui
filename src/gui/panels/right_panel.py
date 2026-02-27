"""Right sidebar panel: mask selector, grow/shrink controls, and display settings."""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from src.gui.widgets import MaskSelectorWidget


class RightPanel(QWidget):
    """Panel containing the mask variant selector, grow/shrink controls,
    and display-settings group.

    All child widgets are exposed as public attributes so that controllers
    can wire signals and read/write state.
    """

    def __init__(self, image_viewer, parent=None):
        """Initialise the right panel.

        Args:
            - image_viewer: The ``ImageViewerWidget`` whose current brush
              size is used to initialise the brush slider.
            - parent (QWidget | None): Parent widget.
        """
        super().__init__(parent)
        self.setMinimumWidth(200)
        self.setMaximumWidth(260)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        # Mask selector
        self.mask_selector = MaskSelectorWidget()
        layout.addWidget(self.mask_selector, stretch=1)

        # Grow / Shrink Mask group
        grow_group = QGroupBox("Grow / Shrink Mask")
        gg = QVBoxLayout(grow_group)
        gg.setContentsMargins(6, 6, 6, 6)
        gg.setSpacing(5)

        # Slider row: slider + spin box
        grow_row = QHBoxLayout()
        self.grow_slider = QSlider(Qt.Orientation.Horizontal)
        self.grow_slider.setMinimum(-20)
        self.grow_slider.setMaximum(20)
        self.grow_slider.setValue(0)
        self.grow_slider.setToolTip("Pixels to grow (+) or shrink (−) the mask boundary")
        grow_row.addWidget(self.grow_slider, stretch=1)

        self.grow_spinbox = QSpinBox()
        self.grow_spinbox.setMinimum(-999)
        self.grow_spinbox.setMaximum(999)
        self.grow_spinbox.setValue(0)
        self.grow_spinbox.setSuffix(" px")
        self.grow_spinbox.setFixedWidth(72)
        self.grow_spinbox.setToolTip("Enter a custom pixel value (beyond slider range if needed)")
        grow_row.addWidget(self.grow_spinbox)
        gg.addLayout(grow_row)

        # Sync slider ↔ spin box
        self.grow_slider.valueChanged.connect(self._on_grow_slider_changed)
        self.grow_spinbox.valueChanged.connect(self._on_grow_spinbox_changed)

        # Apply button
        self.grow_apply_btn = QPushButton("Grow Mask")
        self.grow_apply_btn.setToolTip(
            "Apply the grow/shrink operation to the current image's mask (G)"
        )
        gg.addWidget(self.grow_apply_btn)

        layout.addWidget(grow_group)

        # Prompt Segmentation group (visible only for SAM3)
        self.prompt_group = QGroupBox("Prompt Segmentation")
        pg = QVBoxLayout(self.prompt_group)
        pg.setContentsMargins(6, 6, 6, 6)
        pg.setSpacing(5)

        self.prompt_edit = QLineEdit()
        self.prompt_edit.setPlaceholderText("Enter text prompt...")
        self.prompt_edit.setToolTip("Text description of what to segment (SAM3 only)")
        pg.addWidget(self.prompt_edit)

        self.prompt_segment_btn = QPushButton("Segment by prompt")
        self.prompt_segment_btn.setToolTip("Segment the current image using the text prompt")
        pg.addWidget(self.prompt_segment_btn)

        # Initially hidden (shown only when SAM3 is selected)
        self.prompt_group.setVisible(False)
        layout.addWidget(self.prompt_group)

        # Settings group
        settings_group = QGroupBox("Settings")
        sg = QVBoxLayout(settings_group)
        sg.setContentsMargins(6, 6, 6, 6)
        sg.setSpacing(5)

        # Zoom slider
        zoom_row = QHBoxLayout()
        zoom_row.addWidget(QLabel("Zoom:"))
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setMinimum(100)
        self.zoom_slider.setMaximum(500)
        self.zoom_slider.setValue(100)
        zoom_row.addWidget(self.zoom_slider, stretch=1)
        self.zoom_label = QLabel("100%")
        self.zoom_label.setFixedWidth(46)
        zoom_row.addWidget(self.zoom_label)
        sg.addLayout(zoom_row)

        # Brush size slider
        brush_row = QHBoxLayout()
        brush_row.addWidget(QLabel("Brush:"))
        self.brush_slider = QSlider(Qt.Orientation.Horizontal)
        self.brush_slider.setMinimum(2)
        self.brush_slider.setMaximum(2000)
        self.brush_slider.setValue(image_viewer.brush_size)
        brush_row.addWidget(self.brush_slider, stretch=1)
        self.brush_label = QLabel(f"{image_viewer.brush_size}px")
        self.brush_label.setFixedWidth(46)
        brush_row.addWidget(self.brush_label)
        sg.addLayout(brush_row)

        # Mask colour button
        color_row = QHBoxLayout()
        color_row.addWidget(QLabel("Mask color:"))
        self.mask_color_btn = QPushButton()
        self.mask_color_btn.setFixedSize(36, 22)
        self.mask_color_btn.setToolTip("Choose mask overlay colour")
        self.mask_color_btn.setStyleSheet("background-color: rgb(255, 0, 0); border: 1px solid #666;")
        color_row.addWidget(self.mask_color_btn)
        color_row.addStretch()
        sg.addLayout(color_row)

        # Mask opacity slider
        opacity_row = QHBoxLayout()
        opacity_row.addWidget(QLabel("Opacity:"))
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setMinimum(20)
        self.opacity_slider.setMaximum(255)
        self.opacity_slider.setValue(128)
        self.opacity_slider.setToolTip("Mask overlay opacity (minimum ~8%)")
        opacity_row.addWidget(self.opacity_slider, stretch=1)
        self.opacity_label = QLabel("50%")
        self.opacity_label.setFixedWidth(38)
        opacity_row.addWidget(self.opacity_label)
        sg.addLayout(opacity_row)

        layout.addWidget(settings_group)

    # ------------------------------------------------------------------
    # Internal sync helpers
    # ------------------------------------------------------------------

    def _on_grow_slider_changed(self, value: int):
        """Keep the spin box in sync when the slider moves."""
        self.grow_spinbox.blockSignals(True)
        self.grow_spinbox.setValue(value)
        self.grow_spinbox.blockSignals(False)

    def _on_grow_spinbox_changed(self, value: int):
        """Keep the slider in sync when the spin box changes.

        If the value exceeds the slider range, the slider is clamped
        to its nearest bound while the spin box retains the exact value.
        """
        self.grow_slider.blockSignals(True)
        clamped = max(self.grow_slider.minimum(), min(self.grow_slider.maximum(), value))
        self.grow_slider.setValue(clamped)
        self.grow_slider.blockSignals(False)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def grow_pixels(self) -> int:
        """Return the current grow/shrink value from the spin box."""
        return self.grow_spinbox.value()

    def get_prompt_text(self) -> str:
        """Return the text prompt from the prompt input field."""
        return self.prompt_edit.text().strip()

    def set_prompt_visible(self, visible: bool):
        """Show or hide the prompt segmentation group.

        Args:
            - visible (bool): True to show, False to hide.
        """
        self.prompt_group.setVisible(visible)

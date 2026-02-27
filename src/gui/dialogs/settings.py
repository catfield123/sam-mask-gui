"""Settings dialog for configuring SAM2/SAM3 checkpoints and scaling."""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QWidget,
)

from src.utils.check_packages import check_sam3_installed

# Help text for each setting (tooltip and click-to-show dialog)
_SETTING_HELP = {
    "sam2_checkpoint": (
        "Path to the SAM2 model weights file (.pt). Required for point-based "
        "segmentation and Propagate Masks."
    ),
    "sam3_checkpoint": (
        "Path to the SAM3 model weights file (.pt). Required for text-prompt "
        "segmentation. Weights are available on Hugging Face (facebook/sam3) "
        "after access approval."
    ),
    "sam3_bpe": (
        "Optional. BPE tokenizer vocabulary for SAM3 text prompts. "
        "Leave empty to use the built-in vocabulary."
    ),
    "max_side": (
        "Maximum length in pixels for the longer side of an image while loading. "
        "Images are scaled down for display and inference to save memory. "
        "Masks are always saved at the original image resolution. Use 0 for no limit."
    ),
}


class SettingsDialog(QDialog):
    """Modal dialog that lets the user configure checkpoints and max-side.

    Args:
        - parent (QWidget | None): Parent widget.
        - sam2_checkpoint_path (str): Initial SAM2 checkpoint file path.
        - sam3_checkpoint_path (str | None): Initial SAM3 checkpoint path.
        - sam3_bpe_path (str | None): Initial SAM3 BPE tokenizer path.
        - max_side (int): Initial max-side value.
    """

    def __init__(
        self,
        parent=None,
        sam2_checkpoint_path: str = "",
        sam3_checkpoint_path: str | None = None,
        sam3_bpe_path: str | None = None,
        max_side: int = 1024,
    ):
        """Initialise the settings dialog.

        Args:
            - parent (QWidget | None): Parent widget.
            - sam2_checkpoint_path (str): Pre-filled SAM2 checkpoint file path.
            - sam3_checkpoint_path (str | None): Pre-filled SAM3 checkpoint path.
            - sam3_bpe_path (str | None): Pre-filled SAM3 BPE tokenizer path.
            - max_side (int): Pre-filled maximum image side length.
        """
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)

        layout = QFormLayout(self)
        sam3_installed, _ = check_sam3_installed()

        # SAM2 checkpoint path
        self.sam2_checkpoint_label = QLabel("SAM2 Checkpoint path:")
        self.sam2_checkpoint_edit = QLineEdit(sam2_checkpoint_path)
        self.sam2_checkpoint_button = QPushButton("Browse...")
        self.sam2_checkpoint_button.clicked.connect(self._browse_sam2_checkpoint)
        sam2_checkpoint_layout = QHBoxLayout()
        sam2_checkpoint_layout.addWidget(self.sam2_checkpoint_edit)
        sam2_checkpoint_layout.addWidget(self.sam2_checkpoint_button)
        layout.addRow(
            self._make_label_with_help(self.sam2_checkpoint_label, _SETTING_HELP["sam2_checkpoint"]),
            sam2_checkpoint_layout,
        )

        # SAM3 checkpoint path
        self.sam3_checkpoint_label = QLabel("SAM3 Checkpoint path (optional):")
        self.sam3_checkpoint_edit = QLineEdit(sam3_checkpoint_path or "")
        self.sam3_checkpoint_button = QPushButton("Browse...")
        self.sam3_checkpoint_button.clicked.connect(self._browse_sam3_checkpoint)
        sam3_checkpoint_layout = QHBoxLayout()
        sam3_checkpoint_layout.addWidget(self.sam3_checkpoint_edit)
        sam3_checkpoint_layout.addWidget(self.sam3_checkpoint_button)
        layout.addRow(
            self._make_label_with_help(self.sam3_checkpoint_label, _SETTING_HELP["sam3_checkpoint"]),
            sam3_checkpoint_layout,
        )

        # SAM3 BPE path
        self.sam3_bpe_label = QLabel("SAM3 BPE path (optional, .txt.gz):")
        self.sam3_bpe_edit = QLineEdit(sam3_bpe_path or "")
        self.sam3_bpe_button = QPushButton("Browse...")
        self.sam3_bpe_button.clicked.connect(self._browse_sam3_bpe)
        sam3_bpe_layout = QHBoxLayout()
        sam3_bpe_layout.addWidget(self.sam3_bpe_edit)
        sam3_bpe_layout.addWidget(self.sam3_bpe_button)
        layout.addRow(
            self._make_label_with_help(self.sam3_bpe_label, _SETTING_HELP["sam3_bpe"]),
            sam3_bpe_layout,
        )

        self.sam3_status_label = QLabel(
            "SAM3 package is not installed. Prompt features stay disabled until it is available."
        )
        self.sam3_status_label.setWordWrap(True)
        self.sam3_status_label.setVisible(not sam3_installed)
        layout.addRow("", self.sam3_status_label)

        # Max side
        self.max_side_spin = QSpinBox()
        self.max_side_spin.setMinimum(0)
        self.max_side_spin.setMaximum(10000)
        self.max_side_spin.setValue(max_side)
        self.max_side_spin.setSpecialValueText("No limit")
        max_side_label = QLabel("Max side size (0 - no limit):")
        layout.addRow(
            self._make_label_with_help(max_side_label, _SETTING_HELP["max_side"]),
            self.max_side_spin,
        )
        self._set_sam3_controls_enabled(sam3_installed)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    # ------------------------------------------------------------------
    # Public getters
    # ------------------------------------------------------------------

    def get_sam2_checkpoint_path(self) -> str:
        """Return the SAM2 checkpoint path entered by the user (stripped)."""
        return self.sam2_checkpoint_edit.text().strip()

    def get_sam3_checkpoint_path(self) -> str | None:
        """Return the SAM3 checkpoint path entered by the user."""
        path = self.sam3_checkpoint_edit.text().strip()
        return path if path else None

    def get_sam3_bpe_path(self) -> str | None:
        """Return the SAM3 BPE path entered by the user."""
        path = self.sam3_bpe_edit.text().strip()
        return path if path else None

    def get_max_side(self) -> int:
        """Return the max-side value selected by the user."""
        return self.max_side_spin.value()

    # ------------------------------------------------------------------
    # Private helpers / slots
    # ------------------------------------------------------------------

    @staticmethod
    def _make_label_with_help(label: QLabel, help_text: str) -> QWidget:
        """Build a row label widget with a '?' button that shows help on hover or click.

        Args:
            label: The label widget (e.g. "SAM2 Checkpoint path:").
            help_text: Description shown as tooltip and in the help dialog.

        Returns:
            QWidget: A widget containing the label and a '?' button.
        """
        wrap = QWidget()
        row = QHBoxLayout(wrap)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(label)
        btn = QPushButton("?")
        btn.setToolTip(help_text)
        btn.setFixedSize(22, 22)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        row.addWidget(btn)

        def show_help():
            QMessageBox.information(wrap.window(), "Settings help", help_text)

        btn.clicked.connect(show_help)
        return wrap

    def _set_sam3_controls_enabled(self, enabled: bool):
        """Enable SAM3 settings only when the package is installed."""
        self.sam3_checkpoint_label.setEnabled(enabled)
        self.sam3_checkpoint_edit.setEnabled(enabled)
        self.sam3_checkpoint_button.setEnabled(enabled)
        self.sam3_bpe_label.setEnabled(enabled)
        self.sam3_bpe_edit.setEnabled(enabled)
        self.sam3_bpe_button.setEnabled(enabled)

    def _browse_sam2_checkpoint(self):
        """Open a file dialog to select a SAM2 checkpoint file."""
        start_dir = self._start_dir_for_edit(self.sam2_checkpoint_edit)
        path, _ = QFileDialog.getOpenFileName(
            self, "Select SAM2 Checkpoint", start_dir, "PyTorch Files (*.pt)"
        )
        if path:
            self.sam2_checkpoint_edit.setText(path)

    def _browse_sam3_checkpoint(self):
        """Open a file dialog to select a SAM3 checkpoint file."""
        start_dir = self._start_dir_for_edit(self.sam3_checkpoint_edit)
        path, _ = QFileDialog.getOpenFileName(
            self, "Select SAM3 Checkpoint", start_dir, "PyTorch Files (*.pt);;All Files (*)"
        )
        if path:
            self.sam3_checkpoint_edit.setText(path)

    def _browse_sam3_bpe(self):
        """Open a file dialog to select a SAM3 BPE tokenizer file."""
        start_dir = self._start_dir_for_edit(self.sam3_bpe_edit)
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select SAM3 BPE Tokenizer",
            start_dir,
            "Gzipped Text Files (*.txt.gz);;All Files (*)",
        )
        if path:
            self.sam3_bpe_edit.setText(path)

    @staticmethod
    def _start_dir_for_edit(edit: QLineEdit) -> str:
        """Return the directory of the path in the edit, or empty string if none."""
        from pathlib import Path

        text = edit.text().strip()
        if not text:
            return ""
        p = Path(text)
        if p.exists():
            return str(p.parent) if p.is_file() else str(p)
        if p.parent.exists():
            return str(p.parent)
        return ""

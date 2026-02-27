"""Dialog that displays all keyboard shortcuts."""

from PyQt6.QtWidgets import QDialog, QVBoxLayout, QTextBrowser, QDialogButtonBox


SHORTCUTS_HTML = """
<h3>Keyboard</h3>
<table style="border-collapse: collapse;">
<tr><td style="padding: 4px 12px 4px 0;"><b>Ctrl+O</b></td><td>Open images folder</td></tr>
<tr><td style="padding: 4px 12px 4px 0;"><b>Ctrl+S</b></td><td>Save current mask</td></tr>
<tr><td style="padding: 4px 12px 4px 0;"><b>Ctrl+Z</b></td><td>Undo</td></tr>
<tr><td style="padding: 4px 12px 4px 0;"><b>Ctrl+Y</b></td><td>Redo</td></tr>
<tr><td style="padding: 4px 12px 4px 0;"><b>G</b></td><td>Grow / shrink current mask</td></tr>
<tr><td style="padding: 4px 12px 4px 0;"><b>←</b> / <b>→</b></td><td>Previous / next image</td></tr>
<tr><td style="padding: 4px 12px 4px 0;"><b>Shift</b></td><td>Brush mode (hold while painting)</td></tr>
<tr><td style="padding: 4px 12px 4px 0;"><b>Alt</b></td><td>Mask preview (hold to see mask without overlay)</td></tr>
</table>
<h3>Mouse</h3>
<table style="border-collapse: collapse;">
<tr><td style="padding: 4px 12px 4px 0;"><b>Ctrl + scroll</b></td><td>Zoom in/out</td></tr>
<tr><td style="padding: 4px 12px 4px 0;"><b>Shift + scroll</b></td><td>Change brush size</td></tr>
<tr><td style="padding: 4px 12px 4px 0;"><b>Middle button drag</b></td><td>Pan the image</td></tr>
</table>
<p>Other actions use the File menu or panel buttons.</p>
"""


class ShortcutsDialog(QDialog):
    """A dialog that shows all application keyboard shortcuts."""

    def __init__(self, parent=None):
        """Initialise the shortcuts dialog.

        Args:
            parent: Optional parent widget.
        """
        super().__init__(parent)
        self.setWindowTitle("Keyboard Shortcuts")
        self.setMinimumSize(400, 280)
        layout = QVBoxLayout(self)
        browser = QTextBrowser()
        browser.setHtml(SHORTCUTS_HTML)
        browser.setOpenExternalLinks(False)
        layout.addWidget(browser)
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        button_box.accepted.connect(self.accept)
        layout.addWidget(button_box)

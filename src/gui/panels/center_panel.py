"""Centre panel: image viewer with navigation and clear buttons."""

from PyQt6.QtWidgets import QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from src.gui.widgets import ImageViewerWidget


class CenterPanel(QWidget):
    """Panel containing the image viewer and navigation controls.

    All child widgets are exposed as public attributes so that controllers
    can wire signals and read/write state.
    """

    def __init__(self, parent=None):
        """Initialise the centre panel with image viewer and navigation buttons.

        Args:
            - parent (QWidget | None): Parent widget.
        """
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        self.image_viewer = ImageViewerWidget()
        layout.addWidget(self.image_viewer, stretch=1)

        # Navigation row
        nav_layout = QHBoxLayout()
        nav_layout.addStretch()

        self.prev_button = QPushButton("◀ Previous")
        nav_layout.addWidget(self.prev_button)

        self.next_button = QPushButton("Next ▶")
        nav_layout.addWidget(self.next_button)

        nav_layout.addStretch()

        self.clear_button = QPushButton("CLEAR")
        nav_layout.addWidget(self.clear_button)

        nav_layout.addStretch()
        layout.addLayout(nav_layout)

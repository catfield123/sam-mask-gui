"""Application entry point."""

import argparse
import logging
import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication

from src.gui.main_window import MainWindow
from src.logging_config import configure_logging

logger = logging.getLogger(__name__)


def main() -> None:
    """Launch the SAM2/SAM3 segmentation GUI application."""
    parser = argparse.ArgumentParser(description="SAM2/SAM3 Image Segmentation GUI")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging (verbose trace of operations).",
    )
    args = parser.parse_args()
    configure_logging(debug=args.debug)
    logger.info("Starting SAM2/SAM3 segmentation GUI")

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    if not window.checkpoint_path or not Path(window.checkpoint_path).exists():
        window.show_settings()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

"""Application entry point."""

import argparse
import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication

from src.gui.main_window import MainWindow
from src.logging_config import configure_logging, get_logger

logger = get_logger(__name__)


def main() -> None:
    """Launch the SAM2/SAM3 segmentation GUI application."""
    parser = argparse.ArgumentParser(description="SAM2/SAM3 Image Segmentation GUI")
    parser.add_argument(
        "--debug",
        action="store_true",
        default=True,
        help="Enable debug logging (default: on). Use --no-debug to reduce log volume.",
    )
    parser.add_argument(
        "--no-debug",
        action="store_false",
        dest="debug",
        help="Disable debug logging (INFO and above only).",
    )
    args = parser.parse_args()
    configure_logging(debug=args.debug)
    logger.info("app_started", log_level="DEBUG" if args.debug else "INFO")

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    if not window.checkpoint_path or not Path(window.checkpoint_path).exists():
        window.show_settings()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

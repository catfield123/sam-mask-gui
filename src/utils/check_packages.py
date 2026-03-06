"""Utility for checking if SAM2 and SAM3 packages are installed."""

import importlib
from typing import Dict, Optional, Tuple

from src.logging_config import get_logger
from src.utils.decord_stub import install_decord_stub_if_needed

logger = get_logger(__name__)


def check_sam2_installed() -> Tuple[bool, Optional[str]]:
    """Check if SAM2 package is installed.

    Returns:
        - tuple[bool, str | None]: (is_installed, error_message)
          If installed, returns (True, None).
          If not installed, returns (False, error_message).
    """
    logger.debug("Checking if SAM2 package is installed")
    try:
        import sam2  # noqa: F401

        logger.info("SAM2 package is installed")
        return True, None
    except ImportError as e:
        msg = f"SAM2 package not found. Install with: pip install sam2\nError: {e}"
        logger.error("SAM2 package not installed: %s", e)
        return False, msg


def check_sam3_installed() -> Tuple[bool, Optional[str]]:
    """Check if SAM3 package is installed and usable for image inference.

    We import only sam3.model_builder. Vendor's import chain pulls decord
    (model_builder → sam3_image → collator → sam3_image_dataset). We install
    a decord stub when decord is not available (e.g. macOS) so the import
    succeeds; our app only uses image path, not video.

    Returns:
        - tuple[bool, str | None]: (is_installed, error_message)
          If installed, returns (True, None).
          If not installed, returns (False, error_message).
    """
    logger.debug("Checking if SAM3 package is installed (model_builder path)")
    install_decord_stub_if_needed()
    try:
        importlib.import_module("sam3.model_builder")
        logger.info("SAM3 package is installed")
        return True, None
    except ImportError as e:
        msg = f"SAM3 package not found. Install with: pip install sam3\nError: {e}"
        logger.error("SAM3 package not installed: %s", e)
        return False, msg


def check_all_packages() -> Dict[str, Tuple[bool, Optional[str]]]:
    """Check installation status of both SAM2 and SAM3 packages.

    Returns:
        Dictionary with keys ``'sam2'`` and ``'sam3'``, each mapping to
        a tuple ``(is_installed, error_message)``. If installed, the
        message is ``None``.
    """
    logger.debug("Checking all packages (SAM2, SAM3)")
    result = {
        "sam2": check_sam2_installed(),
        "sam3": check_sam3_installed(),
    }
    logger.debug("Package check result: sam2=%s, sam3=%s", result["sam2"][0], result["sam3"][0])
    return result

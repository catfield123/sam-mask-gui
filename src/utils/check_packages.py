"""Utility for checking if SAM2 and SAM3 packages are installed."""

from typing import Dict, Optional, Tuple


def check_sam2_installed() -> Tuple[bool, Optional[str]]:
    """Check if SAM2 package is installed.

    Returns:
        - tuple[bool, str | None]: (is_installed, error_message)
          If installed, returns (True, None).
          If not installed, returns (False, error_message).
    """
    try:
        import sam2  # noqa: F401

        return True, None
    except ImportError as e:
        return False, f"SAM2 package not found. Install with: pip install sam2\nError: {e}"


def check_sam3_installed() -> Tuple[bool, Optional[str]]:
    """Check if SAM3 package is installed.

    Returns:
        - tuple[bool, str | None]: (is_installed, error_message)
          If installed, returns (True, None).
          If not installed, returns (False, error_message).
    """
    try:
        import sam3  # noqa: F401  # type: ignore[import-not-found]

        return True, None
    except ImportError as e:
        return False, f"SAM3 package not found. Install with: pip install sam3\nError: {e}"


def check_all_packages() -> Dict[str, Tuple[bool, Optional[str]]]:
    """Check installation status of both SAM2 and SAM3 packages.

    Returns:
        Dictionary with keys ``'sam2'`` and ``'sam3'``, each mapping to
        a tuple ``(is_installed, error_message)``. If installed, the
        message is ``None``.
    """
    return {
        "sam2": check_sam2_installed(),
        "sam3": check_sam3_installed(),
    }

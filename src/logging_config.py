"""Centralized logging configuration for the application."""

import logging
import sys
from typing import Optional


def configure_logging(level: Optional[str] = None, debug: bool = False) -> None:
    """Configure root and app loggers with a consistent format.

    Call this once at application startup, before creating the GUI.

    Args:
        level: Override log level (e.g. "INFO", "DEBUG"). If None, uses DEBUG
            when debug is True, else INFO.
        debug: When True, set level to DEBUG; ignored if level is provided.
    """
    if level is None:
        level = "DEBUG" if debug else "INFO"
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    root = logging.getLogger()
    root.setLevel(log_level)
    # Reduce noise from third-party loggers
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

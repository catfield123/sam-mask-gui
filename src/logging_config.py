"""Centralized structured logging configuration (python-observability patterns).

Configure once at application startup. Use get_logger(__name__) in modules.
Supports structured key-value logs and optional JSON output for production.

Public API:
    - configure_logging(): call once at startup
    - get_logger(name): use in every module
    - timed_operation(): context manager for timed operations (Pattern 7)
"""

__all__ = ["configure_logging", "get_logger", "timed_operation"]

import logging
import logging.config
import os
import sys
import time
from contextlib import contextmanager
from typing import Any, Iterator, Optional

import structlog


def configure_logging(
    level: Optional[str] = None,
    debug: bool = False,
    json_format: Optional[bool] = None,
) -> None:
    """Configure structured logging for the application (Pattern 1).

    Call this once at application startup, before creating the GUI.

    Args:
        level: Override log level (e.g. "INFO", "DEBUG"). If None, uses DEBUG
            when debug is True, else INFO.
        debug: When True, set level to DEBUG; ignored if level is provided.
        json_format: When True, emit JSON logs (machine-readable). When False,
            use human-readable console output. When None, follow LOG_FORMAT env
            (e.g. LOG_FORMAT=json) or default to console.
    """
    if level is None:
        level = "DEBUG" if debug else "INFO"
    log_level = getattr(logging, level.upper(), logging.INFO)

    use_json = json_format if json_format is not None else (os.environ.get("LOG_FORMAT", "").lower() == "json")

    # Shared processors for structlog event dict (Pattern 1)
    timestamper = structlog.processors.TimeStamper(fmt="iso")
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        timestamper,
    ]

    if use_json:
        renderer: Any = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())

    # Pre-chain for log records that did not come from structlog (e.g. third-party)
    foreign_pre_chain = [
        structlog.stdlib.add_log_level,
        timestamper,
    ]

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "structlog": {
                    "()": structlog.stdlib.ProcessorFormatter,
                    "processor": renderer,
                    "foreign_pre_chain": foreign_pre_chain,
                },
            },
            "handlers": {
                "default": {
                    "level": log_level,
                    "class": "logging.StreamHandler",
                    "stream": sys.stdout,
                    "formatter": "structlog",
                },
            },
            "loggers": {
                "": {
                    "handlers": ["default"],
                    "level": log_level,
                    "propagate": False,
                },
            },
        }
    )

    structlog.configure(
        processors=shared_processors + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Reduce noise from third-party loggers (semantic levels: keep app logs focused)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a structlog bound logger for the given module name (Pattern 2).

    Use in every module: logger = get_logger(__name__).
    Log with structured fields: logger.info("event_name", key=value, ...).
    """
    return structlog.get_logger(name)


@contextmanager
def timed_operation(operation: str, logger: Optional[structlog.stdlib.BoundLogger] = None, **extra: Any) -> Iterator[None]:
    """Context manager for timing and logging operations (Pattern 7).

    Usage:
        with timed_operation("load_model", checkpoint_path=path):
            load_model(path)
    """
    log = logger or structlog.get_logger(__name__)
    start = time.perf_counter()
    log.debug("operation_started", operation=operation, **extra)
    try:
        yield
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        log.error(
            "operation_failed",
            operation=operation,
            duration_ms=round(elapsed_ms, 2),
            error_type=type(e).__name__,
            error_message=str(e),
            **extra,
            exc_info=True,
        )
        raise
    else:
        elapsed_ms = (time.perf_counter() - start) * 1000
        log.info("operation_completed", operation=operation, duration_ms=round(elapsed_ms, 2), **extra)

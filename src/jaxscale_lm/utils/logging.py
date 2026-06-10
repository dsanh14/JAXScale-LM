"""Structured logging helpers.

Two output modes:

- human: ``2026-06-10 12:00:00 INFO trainer | step complete | step=5 loss=2.31``
- json:  one JSON object per line, suitable for ingestion.

Loggers are ordinary stdlib loggers; structure is carried via the ``extra``
mapping passed to log calls (wrapped by :func:`log_event`). Nothing in this
module may be called from inside a jitted function.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()) | {
    "message",
    "asctime",
}


class _HumanFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, tz=UTC).strftime("%Y-%m-%d %H:%M:%S")
        fields = {k: v for k, v in record.__dict__.items() if k not in _RESERVED}
        suffix = " | " + " ".join(f"{k}={v}" for k, v in fields.items()) if fields else ""
        return f"{ts} {record.levelname} {record.name} | {record.getMessage()}{suffix}"


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update({k: v for k, v in record.__dict__.items() if k not in _RESERVED})
        return json.dumps(payload, default=str)


def setup_logging(level: str = "INFO", json_format: bool = False) -> None:
    """Configure root logging once per process. Idempotent."""
    root = logging.getLogger()
    root.setLevel(level)
    # Replace existing handlers so repeated calls (tests, notebooks) don't
    # duplicate output.
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_JsonFormatter() if json_format else _HumanFormatter())
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Get a namespaced logger (``jaxscale.<name>``)."""
    return logging.getLogger(f"jaxscale.{name}")


def log_event(logger: logging.Logger, message: str, /, **fields: Any) -> None:
    """Log an INFO event with structured fields."""
    logger.info(message, extra=fields)

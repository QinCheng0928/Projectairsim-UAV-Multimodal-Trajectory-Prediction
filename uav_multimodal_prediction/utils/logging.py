"""Logging setup helpers."""

import logging
from typing import Optional


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a consistently formatted application logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def log_optional(logger: logging.Logger, message: str, value: Optional[object]) -> None:
    """Log a value only when it exists."""
    if value is not None:
        logger.info("%s: %s", message, value)

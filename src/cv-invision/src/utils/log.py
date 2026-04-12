import logging
from pathlib import Path
from datetime import datetime

from .const import LOGS_DIR, LOG_FILE


def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger for the given module name.
    Logs to both console and a timestamped log file in logs/.

    Args:
        name: Logger name — use __name__ from the calling module.

    Returns:
        logging.Logger: Configured logger instance.

    Usage:
        from src.utils.log import get_logger
        logger = get_logger(__name__)
        logger.info("Model loaded!")
        logger.warning("Low confidence detection")
        logger.error("Failed to load weights")
    """
    logger    = logging.getLogger(name)

    # Avoid adding duplicate handlers if logger already exists
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # ── formatter ────────────────────────────────────────────
    formatter = logging.Formatter(
        fmt      = "[%(asctime)s] [%(levelname)-8s] %(name)s — %(message)s",
        datefmt  = "%Y-%m-%d %H:%M:%S"
    )

    # ── console handler ───────────────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    # ── file handler ──────────────────────────────────────────
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger



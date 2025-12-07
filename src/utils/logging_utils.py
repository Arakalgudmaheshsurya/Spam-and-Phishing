# src/utils/logging_utils.py

from pathlib import Path
from loguru import logger

_LOGGER_CONFIGURED = False


def configure_logging(log_dir: str = "logs") -> None:
    """
    Configure loguru logger to log to both stdout and a file.
    Idempotent: safe to call multiple times.
    """
    global _LOGGER_CONFIGURED
    if _LOGGER_CONFIGURED:
        return

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Remove default handlers (so we don't double-log)
    logger.remove()

    # Console
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level="INFO",
        backtrace=False,
        diagnose=False,
    )

    # File
    logger.add(
        log_path / "app.log",
        rotation="10 MB",
        retention="14 days",
        level="INFO",
        backtrace=False,
        diagnose=False,
        enqueue=True,
        encoding="utf-8",
    )

    _LOGGER_CONFIGURED = True


def get_logger():
    """
    Return the shared loguru logger. Make sure configure_logging()
    was called once at app startup.
    """
    return logger

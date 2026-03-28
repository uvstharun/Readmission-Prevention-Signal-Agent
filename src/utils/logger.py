"""
Structured logging for the readmission prevention agent.
Logs to both console and file with structured format.
"""
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional

_loggers = {}


def get_logger(name: str, log_level: Optional[str] = None) -> logging.Logger:
    """Get or create a named logger with structured formatting."""
    if name in _loggers:
        return _loggers[name]

    from src.utils.config import config

    level_str = log_level or config.LOG_LEVEL
    level = getattr(logging, level_str.upper(), logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        _loggers[name] = logger
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    try:
        os.makedirs(os.path.dirname(config.LOG_FILE), exist_ok=True)
        file_handler = RotatingFileHandler(
            config.LOG_FILE,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception:
        pass  # If log file can't be created, just use console

    logger.propagate = False
    _loggers[name] = logger
    return logger


def log_agent_action(
    agent_name: str,
    patient_id: str,
    action: str,
    result: str,
    level: str = "INFO",
):
    """Log a structured agent action."""
    logger = get_logger(f"agent.{agent_name}")
    msg = f"patient={patient_id} | action={action} | result={result}"
    getattr(logger, level.lower(), logger.info)(msg)

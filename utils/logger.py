"""
VisionCred Logging Utility
===========================
Centralized logging configuration for the entire pipeline.
"""

import logging
import sys
from src.config import LOG_FORMAT, LOG_DATE_FORMAT, LOG_LEVEL


def get_logger(name: str) -> logging.Logger:
    """
    Create a configured logger instance.
    
    Args:
        name: Logger name (typically module __name__)
    
    Returns:
        Configured logging.Logger instance
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

        # Console handler with colored output
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

        formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)

    return logger

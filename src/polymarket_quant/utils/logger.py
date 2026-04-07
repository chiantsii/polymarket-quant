import logging
import os

def get_logger(name: str) -> logging.Logger:
    """Provides a standardized logger instance."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        level = os.getenv("LOG_LEVEL", "INFO")
        logger.setLevel(level)
    return logger
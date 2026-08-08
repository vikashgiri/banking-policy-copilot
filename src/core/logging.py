import logging
import sys


def get_logger(name: str) -> logging.Logger:
    # Create logger with module name
    logger = logging.getLogger(name)

    # Avoid adding handlers multiple times
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        # Show logs on console
        handler = logging.StreamHandler(sys.stdout)

        # Log format: time | level | module | message
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
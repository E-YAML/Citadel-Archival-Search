import logging
import sys
from typing import Union
from loguru import logger


class InterceptHandler(logging.Handler):
    """
    Custom logging handler that redirects standard library logging
    events to loguru for unified structured output.
    """
    def emit(self, record: logging.LogRecord) -> None:
        # Find corresponding Loguru level if it exists
        try:
            level: Union[str, int] = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where the logged message originated
        frame = logging.currentframe()
        depth = 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logging() -> None:
    """
    Configures loguru to output structured JSON to stdout and intercepts
    all standard framework loggers (e.g. FastAPI, Uvicorn).
    """
    # Clear handlers for the root logger
    logging.root.handlers = []
    
    # Configure basic configuration with our InterceptHandler
    logging.basicConfig(handlers=[InterceptHandler()], level=logging.INFO)

    # Intercept specific framework loggers
    interceptable_loggers = [
        "uvicorn",
        "uvicorn.access",
        "uvicorn.error",
        "fastapi",
        "gunicorn",
    ]
    for logger_name in interceptable_loggers:
        logging_logger = logging.getLogger(logger_name)
        logging_logger.handlers = [InterceptHandler()]
        logging_logger.propagate = False

    # Configure Loguru to serialize all output to JSON and send to stdout
    logger.remove()
    logger.add(
        sys.stdout,
        level="INFO",
        serialize=True,  # Enables structured JSON logging output
        backtrace=True,
        diagnose=True,
    )


# Automatically execute setup upon import to ensure logging is configured early
setup_logging()

"""Modern logging setup with structured logging and rich formatting."""

import logging
import sys
from pathlib import Path
from typing import Any, Dict

import structlog
from loguru import logger
from rich.console import Console
from rich.logging import RichHandler

from .config import settings


def setup_logging() -> None:
    """Configure structured logging with rich formatting and file output."""
    
    # Remove default loguru handler
    logger.remove()
    
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Console handler with rich formatting
    console = Console()
    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=True,
        markup=True,
        rich_tracebacks=True
    )
    
    # Configure loguru with multiple handlers
    logger.add(
        rich_handler,
        level=settings.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=False,
    )
    
    # File handler for all logs
    logger.add(
        logs_dir / "app.log",
        rotation="100 MB",
        retention="30 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {extra} | {message}",
        serialize=settings.log_format == "json"
    )
    
    # Error file handler
    logger.add(
        logs_dir / "errors.log",
        rotation="50 MB",
        retention="60 days",
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {extra} | {message}",
        serialize=True,  # Always use JSON for errors
        backtrace=True,
        diagnose=True
    )
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="ISO"),
            structlog.dev.ConsoleRenderer() if settings.debug else structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level.upper())
        ),
        logger_factory=structlog.WriteLoggerFactory(),
        context_class=dict,
        cache_logger_on_first_use=False,
    )


def get_logger(name: str) -> Any:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


def log_function_call(func_name: str, **kwargs: Any) -> None:
    """Log function call with parameters."""
    logger.info(f"Calling {func_name}", **kwargs)


def log_error(error: Exception, context: Dict[str, Any] = None) -> None:
    """Log error with context."""
    context = context or {}
    logger.error(
        f"Error occurred: {error}",
        error_type=type(error).__name__,
        error_message=str(error),
        **context
    )


def log_performance(operation: str, duration: float, **kwargs: Any) -> None:
    """Log performance metrics."""
    logger.info(
        f"Performance: {operation}",
        duration_seconds=duration,
        **kwargs
    ) 
"""Colorized logging helpers."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from colorama import Back, Fore, Style
from colorama import init as colorama_init


class ColorLogFormatter(logging.Formatter):
    """Terminal-friendly formatter with level-based coloring."""

    def format(self, record: logging.LogRecord) -> str:
        """Render one log record with timestamp, context, and ANSI colors."""
        timestamp = datetime.now(UTC).isoformat()
        message = (
            f"{timestamp} | {record.levelname:<8} | {record.name} | "
            f"{record.getMessage()}"
        )
        if record.exc_info:
            message = f"{message}\n{self.formatException(record.exc_info)}"

        if record.levelno >= logging.CRITICAL:
            color = Fore.WHITE + Back.RED + Style.BRIGHT
        elif record.levelno >= logging.ERROR:
            color = Fore.RED + Style.BRIGHT
        elif record.levelno >= logging.WARNING:
            color = Fore.YELLOW + Style.BRIGHT
        elif record.levelno >= logging.INFO:
            color = Fore.WHITE
        else:
            color = Fore.CYAN
        return f"{color}{message}{Style.RESET_ALL}"


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging with colorized terminal output."""
    colorama_init(autoreset=True)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level.upper())
    handler = logging.StreamHandler()
    handler.setFormatter(ColorLogFormatter())
    root.addHandler(handler)

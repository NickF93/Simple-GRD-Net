import logging
import os
import re

import torch
from colorama import Back, Fore, Style

from grdnet.core.logging import ColorLogFormatter, configure_logging
from grdnet.core.reproducibility import set_global_seed


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def test_color_log_formatter_payload() -> None:
    formatter = ColorLogFormatter()
    logger = logging.getLogger("grdnet.test")
    record = logger.makeRecord(
        name="grdnet.test",
        level=logging.INFO,
        fn=__file__,
        lno=1,
        msg="hello",
        args=(),
        exc_info=None,
    )
    plain = _strip_ansi(formatter.format(record))
    assert "INFO" in plain
    assert "grdnet.test" in plain
    assert "hello" in plain
    assert " | " in plain


def test_color_log_formatter_uses_expected_level_colors_and_exception_block() -> None:
    formatter = ColorLogFormatter()
    logger = logging.getLogger("grdnet.levels")

    records = [
        logger.makeRecord(
            name="grdnet.levels",
            level=logging.DEBUG,
            fn=__file__,
            lno=1,
            msg="debug",
            args=(),
            exc_info=None,
        ),
        logger.makeRecord(
            name="grdnet.levels",
            level=logging.WARNING,
            fn=__file__,
            lno=1,
            msg="warn",
            args=(),
            exc_info=None,
        ),
        logger.makeRecord(
            name="grdnet.levels",
            level=logging.ERROR,
            fn=__file__,
            lno=1,
            msg="err",
            args=(),
            exc_info=None,
        ),
        logger.makeRecord(
            name="grdnet.levels",
            level=logging.CRITICAL,
            fn=__file__,
            lno=1,
            msg="crit",
            args=(),
            exc_info=(ValueError, ValueError("boom"), None),
        ),
    ]

    rendered = [formatter.format(record) for record in records]
    assert rendered[0].startswith(Fore.CYAN)
    assert rendered[1].startswith(Fore.YELLOW + Style.BRIGHT)
    assert rendered[2].startswith(Fore.RED + Style.BRIGHT)
    assert rendered[3].startswith(Fore.WHITE + Back.RED + Style.BRIGHT)
    assert "ValueError: boom" in rendered[3]


def test_configure_logging_sets_color_formatter() -> None:
    configure_logging("debug")
    root = logging.getLogger()
    assert root.level == logging.DEBUG
    assert root.handlers
    assert isinstance(root.handlers[0].formatter, ColorLogFormatter)


def test_set_global_seed_sets_expected_state() -> None:
    set_global_seed(seed=123, deterministic=True)
    assert os.environ["PYTHONHASHSEED"] == "123"
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False

    set_global_seed(seed=123, deterministic=False)
    assert torch.backends.cudnn.deterministic is False
    assert torch.backends.cudnn.benchmark is True

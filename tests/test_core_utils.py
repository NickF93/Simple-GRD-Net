import logging
import os
import re

import torch

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

import json
import logging
import os

import torch

from grdnet.core.logging import JsonLogFormatter, configure_logging
from grdnet.core.reproducibility import set_global_seed


def test_json_log_formatter_payload() -> None:
    formatter = JsonLogFormatter()
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
    payload = json.loads(formatter.format(record))
    assert payload["level"] == "INFO"
    assert payload["logger"] == "grdnet.test"
    assert payload["message"] == "hello"
    assert "timestamp" in payload


def test_configure_logging_sets_json_formatter() -> None:
    configure_logging("debug")
    root = logging.getLogger()
    assert root.level == logging.DEBUG
    assert root.handlers
    assert isinstance(root.handlers[0].formatter, JsonLogFormatter)


def test_set_global_seed_sets_expected_state() -> None:
    set_global_seed(seed=123, deterministic=True)
    assert os.environ["PYTHONHASHSEED"] == "123"
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False

    set_global_seed(seed=123, deterministic=False)
    assert torch.backends.cudnn.deterministic is False
    assert torch.backends.cudnn.benchmark is True

from __future__ import annotations
from argparse import Namespace
from typing import TYPE_CHECKING

from hermes_onnx_asr.cli import handle_command

if TYPE_CHECKING:
    import pytest


def test_list_models_prints_default_and_quantizations(capsys: pytest.CaptureFixture[str]) -> None:
    assert handle_command(Namespace(onnx_asr_command="list-models")) == 0
    output = capsys.readouterr().out
    assert "gigaam-v3-e2e-rnnt (default): int8, fp32 [certified]" in output
    assert "nemo-fastconformer-ru-rnnt: int8, fp32 [pending smoke]" in output

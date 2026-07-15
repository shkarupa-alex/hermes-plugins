from __future__ import annotations
import os
import socket
from pathlib import Path
from typing import NoReturn

import pytest

from hermes_onnx_asr.config import OnnxAsrSettings
from hermes_onnx_asr.pipeline import load_pipeline


@pytest.mark.asr_live
def test_installed_pipeline_warmup_does_not_attempt_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir = Path(value) if (value := os.environ.get("HERMES_ASR_RELEASE_MODEL_DIR")) else None
    if model_dir is None:
        pytest.skip("HERMES_ASR_RELEASE_MODEL_DIR is not configured")

    def blocked(*_args: object, **_kwargs: object) -> NoReturn:
        raise AssertionError("offline ONNX ASR runtime attempted network access")

    monkeypatch.setattr(socket, "socket", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr("hermes_onnx_asr.catalog.snapshot_download", blocked)
    pipeline = load_pipeline(OnnxAsrSettings(model_dir=model_dir))
    assert pipeline.audit.roles

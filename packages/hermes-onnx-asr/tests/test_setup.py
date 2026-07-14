from __future__ import annotations
from typing import Any

import pytest

from hermes_onnx_asr import setup


@pytest.mark.parametrize("vad_seconds", [None, 0.0, 20.0])
def test_setup_writes_real_yaml_null_and_preserves_unrelated_settings(
    monkeypatch: pytest.MonkeyPatch,
    vad_seconds: float | None,
) -> None:
    saved: list[dict[str, Any]] = []
    monkeypatch.setattr(
        setup,
        "read_raw_config",
        lambda: {"stt": {"onnx_asr": {"runtime": {"queue_depth": 2}}}, "unrelated": True},
    )

    def save_config(config: dict[str, Any], **_kwargs: object) -> None:
        saved.append(config)

    monkeypatch.setattr(setup, "save_config", save_config)
    setup.write_setup_config("gigaam-v3-e2e-rnnt", vad_seconds)
    assert saved[0]["unrelated"] is True
    onnx_config = saved[0]["stt"]["onnx_asr"]
    assert onnx_config["vad"]["min_audio_seconds"] == vad_seconds
    assert onnx_config["runtime"]["queue_depth"] == 2

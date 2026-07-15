from __future__ import annotations

import pytest
from pydantic import ValidationError

from hermes_onnx_asr.config import DEFAULT_MODEL, OnnxAsrSettings, load_settings


def test_defaults_match_specification() -> None:
    settings = OnnxAsrSettings()
    assert settings.model == DEFAULT_MODEL == "gigaam-v3-e2e-rnnt"
    assert settings.quantization == "int8"
    assert settings.vad.min_audio_seconds == 20
    assert settings.runtime.concurrency == 1
    assert settings.runtime.queue_depth == 4
    assert settings.audio.max_duration_seconds is None


@pytest.mark.parametrize("threshold", [None, 0, 20])
def test_vad_threshold_semantics_accept_null_zero_and_positive(threshold: float | None) -> None:
    settings = OnnxAsrSettings(vad={"min_audio_seconds": threshold})  # type: ignore[arg-type]
    assert settings.vad.min_audio_seconds == threshold


@pytest.mark.parametrize("threshold", [-1, -0.1])
def test_vad_threshold_rejects_negative_values(threshold: float) -> None:
    with pytest.raises(ValidationError, match="non-negative"):
        OnnxAsrSettings(vad={"min_audio_seconds": threshold})  # type: ignore[arg-type]


def test_vad_hysteresis_must_be_ordered() -> None:
    with pytest.raises(ValidationError, match="negative_threshold"):
        OnnxAsrSettings(vad={"threshold": 0.5, "negative_threshold": 0.5})  # type: ignore[arg-type]


def test_cpu_provider_is_not_user_configurable() -> None:
    with pytest.raises(ValidationError, match="providers"):
        OnnxAsrSettings(providers=["CoreMLExecutionProvider"])  # type: ignore[call-arg]


def test_t_one_uses_its_only_supported_quantization_when_not_explicit() -> None:
    assert OnnxAsrSettings(model="t-tech/t-one").quantization is None
    assert OnnxAsrSettings(model="t-tech/t-one", quantization="int8").quantization == "int8"


def test_pydantic_settings_nested_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HERMES_ONNX_ASR__VAD__MIN_AUDIO_SECONDS", "0")
    assert OnnxAsrSettings().vad.min_audio_seconds == 0


def test_environment_overrides_profile_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "hermes_onnx_asr.config.load_config_readonly",
        lambda: {"stt": {"onnx_asr": {"vad": {"min_audio_seconds": 20}}}},
    )
    monkeypatch.setenv("HERMES_ONNX_ASR__VAD__MIN_AUDIO_SECONDS", "0")
    assert load_settings().vad.min_audio_seconds == 0

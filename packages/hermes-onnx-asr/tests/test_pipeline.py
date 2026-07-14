from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from hermes_onnx_asr.config import OnnxAsrSettings
from hermes_onnx_asr.errors import OnnxAsrError
from hermes_onnx_asr.pipeline import CPU_PROVIDERS, audit_cpu_sessions, load_pipeline, recognize


class FakeSession:
    def __init__(self, providers: list[str] | None = None) -> None:
        self.providers = providers or CPU_PROVIDERS

    def get_providers(self) -> list[str]:
        return self.providers


class BaseModel:
    def __init__(self, transcript: str = "текст") -> None:
        self.asr = SimpleNamespace(
            _encoder=FakeSession(),
            _decoder=FakeSession(),
            _joiner=FakeSession(),
        )
        self.resampler = SimpleNamespace(_preprocessors={8000: FakeSession()})
        self.transcript = transcript
        self.calls: list[tuple[Path, str]] = []

    def recognize(self, path: Path, *, channel: str, **_kwargs: object) -> str:
        self.calls.append((path, channel))
        return self.transcript


class VadModel:
    def __init__(self, base: BaseModel, texts: list[str]) -> None:
        self.asr = base.asr
        self.resampler = base.resampler
        self.vad = SimpleNamespace(_model=FakeSession())
        self.texts = texts

    def recognize(self, _path: Path, *, channel: str, **_kwargs: object) -> list[SimpleNamespace]:
        assert channel == "mean"
        return [SimpleNamespace(text=text) for text in self.texts]


def test_cpu_audit_rejects_non_cpu_session() -> None:
    model = BaseModel()
    model.asr._encoder = FakeSession(["CoreMLExecutionProvider"])
    with pytest.raises(OnnxAsrError) as caught:
        audit_cpu_sessions(model, None)
    assert caught.value.code == "cpu_provider_violation"


@pytest.mark.parametrize(
    ("threshold", "duration", "expected_vad"),
    [(None, 100, False), (0, 0, True), (20, 19.999, False), (20, 20, True)],
)
def test_duration_selects_base_or_vad(
    threshold: float | None,
    duration: float,
    expected_vad: object,
) -> None:
    base = BaseModel()
    vad = VadModel(base, [" один ", "", "два"])
    settings = OnnxAsrSettings(vad={"min_audio_seconds": threshold})  # type: ignore[arg-type]
    pipeline = SimpleNamespace(settings=settings, base_model=base, vad_model=vad)
    result = recognize(pipeline, Path("audio.wav"), duration, None)  # type: ignore[arg-type]
    assert result["vad_applied"] is expected_vad
    assert result["transcript"] == ("один два" if expected_vad else "текст")
    if not expected_vad:
        assert base.calls == [(Path("audio.wav"), "mean")]


def test_empty_transcript_is_deterministic_failure() -> None:
    base = BaseModel("")
    settings = OnnxAsrSettings(vad={"min_audio_seconds": None})  # type: ignore[arg-type]
    pipeline = SimpleNamespace(settings=settings, base_model=base, vad_model=None)
    with pytest.raises(OnnxAsrError) as caught:
        recognize(pipeline, Path("audio.wav"), 1, None)  # type: ignore[arg-type]
    assert caught.value.code == "no_speech_detected"


def test_model_construction_maps_negative_threshold_and_cpu_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    base = BaseModel()
    observed: dict[str, Any] = {}

    def with_vad(vad: object, **kwargs: object) -> VadModel:
        observed["vad"] = vad
        observed["vad_kwargs"] = kwargs
        return VadModel(base, ["текст"])

    base.with_vad = with_vad  # type: ignore[attr-defined]

    def load_model(*args: object, **kwargs: object) -> BaseModel:
        observed["model_args"] = args
        observed["model_kwargs"] = kwargs
        return base

    vad = object()
    monkeypatch.setattr("onnx_asr.load_model", load_model)

    def load_vad(*_args: object, **_kwargs: object) -> object:
        return vad

    def resolve_bundle(*_args: object, **_kwargs: object) -> Path:
        return tmp_path

    monkeypatch.setattr("onnx_asr.load_vad", load_vad)
    monkeypatch.setattr("hermes_onnx_asr.pipeline._resolve_bundle", resolve_bundle)
    settings = OnnxAsrSettings(model_dir=tmp_path)
    pipeline = load_pipeline(settings)
    assert pipeline.vad_model is not None
    assert observed["vad_kwargs"]["neg_threshold"] == 0.35
    assert "negative_threshold" not in observed["vad_kwargs"]
    model_kwargs = observed["model_kwargs"]
    assert model_kwargs["providers"] == CPU_PROVIDERS
    assert model_kwargs["asr_config"]["providers"] == CPU_PROVIDERS
    assert model_kwargs["resampler_config"]["providers"] == CPU_PROVIDERS
    assert model_kwargs["preprocessor_config"] == {"use_numpy_preprocessors": True, "max_concurrent_workers": 1}

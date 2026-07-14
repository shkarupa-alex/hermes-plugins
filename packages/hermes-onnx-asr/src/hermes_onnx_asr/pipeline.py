# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownVariableType=false
from __future__ import annotations
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import onnx_asr
import onnxruntime as ort

from hermes_onnx_asr.catalog import bundle_path, fetch_bundle, verify_bundle
from hermes_onnx_asr.errors import OnnxAsrError, safe_error

if TYPE_CHECKING:
    from pathlib import Path

    from onnx_asr.onnx import OnnxSessionOptions

    from hermes_onnx_asr.config import OnnxAsrSettings

CPU_PROVIDERS = ["CPUExecutionProvider"]


@dataclass(frozen=True)
class SessionAudit:
    roles: tuple[str, ...]
    providers: dict[str, tuple[str, ...]]


@dataclass
class Pipeline:
    settings: OnnxAsrSettings
    base_model: Any
    vad_model: Any | None
    audit: SessionAudit


_LOAD_LOCK = threading.Lock()


def _session_options(settings: OnnxAsrSettings) -> ort.SessionOptions:
    options = ort.SessionOptions()
    options.intra_op_num_threads = settings.runtime.intra_op_num_threads
    options.inter_op_num_threads = settings.runtime.inter_op_num_threads
    return options


def _resolve_bundle(settings: OnnxAsrSettings, alias: str, *, kind: str = "model") -> Path:
    typed_kind = "vad" if kind == "vad" else "model"
    quantization = None if typed_kind == "vad" else settings.quantization
    path = bundle_path(settings.model_dir, alias, quantization, kind=typed_kind)
    try:
        verify_bundle(path, alias, quantization, kind=typed_kind)
    except OnnxAsrError:
        if not settings.allow_runtime_download:
            raise
        path = fetch_bundle(settings.model_dir, alias, quantization, kind=typed_kind)
    return path


def load_pipeline(settings: OnnxAsrSettings) -> Pipeline:
    with _LOAD_LOCK:
        model_path = _resolve_bundle(settings, settings.model)
        options = _session_options(settings)
        asr_config: OnnxSessionOptions = {"sess_options": options, "providers": CPU_PROVIDERS}
        resampler_config: OnnxSessionOptions = {"sess_options": options, "providers": CPU_PROVIDERS}
        try:
            base_model = onnx_asr.load_model(
                settings.model,
                path=model_path,
                quantization=settings.quantization,
                providers=CPU_PROVIDERS,
                sess_options=options,
                asr_config=asr_config,
                preprocessor_config={"use_numpy_preprocessors": True, "max_concurrent_workers": 1},
                resampler_config=resampler_config,
            )
            vad_model = None
            if settings.vad.min_audio_seconds is not None:
                vad_path = _resolve_bundle(settings, settings.vad.engine, kind="vad")
                vad = onnx_asr.load_vad(
                    settings.vad.engine,
                    path=vad_path,
                    providers=CPU_PROVIDERS,
                    sess_options=options,
                )
                vad_model = base_model.with_vad(
                    vad,
                    threshold=settings.vad.threshold,
                    neg_threshold=settings.vad.negative_threshold,
                    min_speech_duration_ms=settings.vad.min_speech_duration_ms,
                    max_speech_duration_s=settings.vad.max_speech_duration_s,
                    min_silence_duration_ms=settings.vad.min_silence_duration_ms,
                    speech_pad_ms=settings.vad.speech_pad_ms,
                )
                if vad_model.asr is not base_model.asr or vad_model.resampler is not base_model.resampler:
                    _raise_wrapper_identity_error()
            audit = audit_cpu_sessions(base_model, vad_model)
        except OnnxAsrError:
            raise
        except Exception as exc:
            raise safe_error("model_load_failed") from exc
    return Pipeline(settings=settings, base_model=base_model, vad_model=vad_model, audit=audit)


def _raise_wrapper_identity_error() -> None:
    raise safe_error("model_load_failed")


@runtime_checkable
class _Session(Protocol):
    def get_providers(self) -> list[str]: ...


def _add_session(
    sessions: dict[str, _Session],
    role: str,
    candidate: object,
) -> None:
    if candidate is None:
        return
    if not isinstance(candidate, _Session):
        raise safe_error("cpu_provider_violation")
    if role in sessions:
        raise safe_error("cpu_provider_violation")
    sessions[role] = candidate


def audit_cpu_sessions(  # noqa: C901 - explicit pinned onnx-asr session-role introspector
    base_model: Any,  # noqa: ANN401 - pinned upstream adapters expose private session fields
    vad_model: Any | None,  # noqa: ANN401 - pinned upstream adapters expose private session fields
) -> SessionAudit:
    sessions: dict[str, _Session] = {}
    asr = base_model.asr
    asr_roles = {
        "_model": "asr.model",
        "_encoder": "asr.encoder",
        "_decoder": "asr.decoder",
        "_joiner": "asr.joiner",
        "_decoder_joint": "asr.decoder_joint",
    }
    for attribute, role in asr_roles.items():
        if hasattr(asr, attribute):
            _add_session(sessions, role, getattr(asr, attribute))
    resampler = base_model.resampler
    preprocessors = getattr(resampler, "_preprocessors", None)
    if not isinstance(preprocessors, dict):
        raise safe_error("cpu_provider_violation")
    for rate, session in sorted(preprocessors.items()):
        _add_session(sessions, f"resampler.{rate}", session)
    if vad_model is not None:
        vad = vad_model.vad
        _add_session(sessions, "vad.silero", getattr(vad, "_model", None))
    if not sessions:
        raise safe_error("cpu_provider_violation")
    if "rnnt" in type(asr).__name__.lower():
        required = {"asr.encoder", "asr.decoder", "asr.joiner"}
        if not required.issubset(sessions):
            raise safe_error("cpu_provider_violation")
    providers: dict[str, tuple[str, ...]] = {}
    for role, session in sessions.items():
        observed = tuple(session.get_providers())
        providers[role] = observed
        if observed != tuple(CPU_PROVIDERS):
            raise safe_error("cpu_provider_violation")
    return SessionAudit(roles=tuple(sessions), providers=providers)


def recognize(pipeline: Pipeline, wav_path: Path, duration: float, language: str | None) -> dict[str, object]:
    threshold = pipeline.settings.vad.min_audio_seconds
    use_vad = threshold is not None and duration >= threshold
    kwargs = {"language": language} if language else {}
    if use_vad:
        if pipeline.vad_model is None:
            raise safe_error("vad_not_installed")
        raw_segments = pipeline.vad_model.recognize(wav_path, channel="mean", **kwargs)
        texts: list[str] = []
        segment_count = 0
        for segment in raw_segments:
            text = str(segment.text).strip()
            if text:
                texts.append(text)
                segment_count += 1
        transcript = " ".join(texts)
    else:
        transcript = str(pipeline.base_model.recognize(wav_path, channel="mean", **kwargs)).strip()
        segment_count = 1 if transcript else 0
    if not transcript:
        raise safe_error("no_speech_detected")
    return {
        "success": True,
        "transcript": transcript,
        "provider": "onnx_asr",
        "model": pipeline.settings.model,
        "vad_applied": use_vad,
        "audio_seconds": round(duration, 3),
        "segments": segment_count,
    }

from __future__ import annotations
from pathlib import Path
from typing import Any, Self, cast

from hermes_cli.config import load_config_readonly
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

DEFAULT_MODEL = "gigaam-v3-e2e-rnnt"


class VadSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_audio_seconds: float | None = 20
    engine: str = "silero"
    threshold: float = 0.5
    negative_threshold: float = 0.35
    min_speech_duration_ms: float = 250
    max_speech_duration_s: float = 20
    min_silence_duration_ms: float = 100
    speech_pad_ms: float = 30

    @model_validator(mode="after")
    def validate_values(self) -> Self:
        if self.min_audio_seconds is not None and self.min_audio_seconds < 0:
            raise ValueError("vad.min_audio_seconds must be non-negative or null")
        if self.engine != "silero":
            raise ValueError("only the silero VAD engine is supported")
        if not 0 <= self.negative_threshold < self.threshold <= 1:
            raise ValueError("VAD thresholds must satisfy 0 <= negative_threshold < threshold <= 1")
        durations = (
            self.min_speech_duration_ms,
            self.max_speech_duration_s,
            self.min_silence_duration_ms,
            self.speech_pad_ms,
        )
        if any(value <= 0 for value in durations):
            raise ValueError("VAD durations must be positive")
        return self


class AudioSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_duration_seconds: float | None = None
    temp_safety_margin_bytes: int = 1_073_741_824

    @model_validator(mode="after")
    def validate_values(self) -> Self:
        if self.max_duration_seconds is not None and self.max_duration_seconds <= 0:
            raise ValueError("audio.max_duration_seconds must be positive or null")
        if self.temp_safety_margin_bytes < 0:
            raise ValueError("audio.temp_safety_margin_bytes must be non-negative")
        return self


class RuntimeSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    concurrency: int = 1
    queue_depth: int = 4
    ffmpeg_timeout_seconds: float = 3600
    transcription_timeout_seconds: float = 21_600
    intra_op_num_threads: int = 0
    inter_op_num_threads: int = 0
    stale_temp_ttl_seconds: int = 86_400

    @model_validator(mode="after")
    def validate_values(self) -> Self:
        if self.concurrency != 1:
            raise ValueError("v1 requires runtime.concurrency to be 1")
        if self.queue_depth < 0:
            raise ValueError("runtime.queue_depth must be non-negative")
        if self.ffmpeg_timeout_seconds <= 0 or self.transcription_timeout_seconds <= 0:
            raise ValueError("runtime timeouts must be positive")
        if self.intra_op_num_threads < 0 or self.inter_op_num_threads < 0:
            raise ValueError("ORT thread counts must be non-negative")
        if self.stale_temp_ttl_seconds <= 0:
            raise ValueError("runtime.stale_temp_ttl_seconds must be positive")
        return self


class OnnxAsrSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="HERMES_ONNX_ASR__",
        env_nested_delimiter="__",
        extra="forbid",
    )

    model: str = DEFAULT_MODEL
    quantization: str | None = "int8"
    model_dir: Path = Field(default=Path("~/.hermes/models/onnx-asr"))
    allow_runtime_download: bool = False
    language: str | None = None
    vad: VadSettings = Field(default_factory=VadSettings)
    audio: AudioSettings = Field(default_factory=AudioSettings)
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        del cls, settings_cls
        return env_settings, init_settings, dotenv_settings, file_secret_settings

    @model_validator(mode="after")
    def normalize(self) -> Self:
        if not self.model.strip():
            raise ValueError("model must not be empty")
        if self.model == "t-tech/t-one" and "quantization" not in self.model_fields_set:
            self.quantization = None
        if self.quantization is not None and not self.quantization.strip():
            self.quantization = None
        self.model_dir = self.model_dir.expanduser()
        return self


def load_settings() -> OnnxAsrSettings:
    config = TypeAdapter(dict[str, object]).validate_python(load_config_readonly() or {})
    stt = TypeAdapter(dict[str, object]).validate_python(config.get("stt") or {})
    validated = TypeAdapter(dict[str, object]).validate_python(stt.get("onnx_asr") or {})
    return OnnxAsrSettings(**cast("dict[str, Any]", validated))

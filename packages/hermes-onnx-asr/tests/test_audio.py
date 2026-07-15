from __future__ import annotations
import wave
from typing import TYPE_CHECKING

import pytest

from hermes_onnx_asr.audio import enforce_duration_limit, is_compatible_pcm_wav, wav_duration
from hermes_onnx_asr.config import OnnxAsrSettings
from hermes_onnx_asr.errors import OnnxAsrError

if TYPE_CHECKING:
    from pathlib import Path


def write_wav(path: Path, *, seconds: float = 1, channels: int = 2, sample_rate: int = 16_000) -> None:
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(channels)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(b"\0\0" * channels * int(sample_rate * seconds))


def test_pcm_wav_uses_direct_path_and_header_duration(tmp_path: Path) -> None:
    source = tmp_path / "stereo.wav"
    write_wav(source, seconds=1.25)
    assert is_compatible_pcm_wav(source) is True
    assert wav_duration(source) == pytest.approx(1.25)


def test_pcm_wav_with_unsupported_sample_rate_requires_normalization(tmp_path: Path) -> None:
    source = tmp_path / "unsupported-rate.wav"
    write_wav(source, sample_rate=12_000)
    assert is_compatible_pcm_wav(source) is False


def test_optional_duration_limit(tmp_path: Path) -> None:
    source = tmp_path / "audio.wav"
    write_wav(source)
    enforce_duration_limit(wav_duration(source), OnnxAsrSettings())
    with pytest.raises(OnnxAsrError) as caught:
        enforce_duration_limit(1, OnnxAsrSettings(audio={"max_duration_seconds": 0.5}))  # type: ignore[arg-type]
    assert caught.value.code == "audio_too_long"

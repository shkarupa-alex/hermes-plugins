from __future__ import annotations
import contextlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import wave
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, cast

from pydantic import BaseModel, ConfigDict, TypeAdapter

from hermes_onnx_asr.errors import OnnxAsrError, safe_error

if TYPE_CHECKING:
    from collections.abc import Callable

    from hermes_onnx_asr.config import OnnxAsrSettings

MAX_STDERR_BYTES = 65_536
NORMALIZED_BYTES_PER_SECOND = 16_000 * 2
MIN_WAV_BYTES = 44
SUPPORTED_SAMPLE_RATES = frozenset({8_000, 11_025, 16_000, 22_050, 24_000, 32_000, 44_100, 48_000})


class _ProbeStream(BaseModel):
    model_config = ConfigDict(extra="ignore")

    duration: str | float | None = None


class _ProbePayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    streams: list[_ProbeStream] = []
    format: _ProbeStream | None = None


def wav_duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as audio:
            rate = audio.getframerate()
            return audio.getnframes() / rate if rate > 0 else 0
    except (OSError, wave.Error) as exc:
        raise safe_error("decode_failed") from exc


def is_compatible_pcm_wav(path: Path) -> bool:
    try:
        with wave.open(str(path), "rb") as audio:
            return (
                audio.getcomptype() == "NONE"
                and audio.getsampwidth() in {1, 2, 3, 4}
                and audio.getframerate() in SUPPORTED_SAMPLE_RATES
            )
    except (OSError, wave.Error):
        return False


def probe_duration(path: Path) -> float | None:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None
    try:
        completed = subprocess.run(  # noqa: S603
            [
                ffprobe,
                "-v",
                "error",
                "-protocol_whitelist",
                "file,pipe,crypto,data",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=duration:format=duration",
                "-of",
                "json",
                str(path),
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    try:
        raw_payload = TypeAdapter(dict[str, object]).validate_python(json.loads(completed.stdout))
        payload = _ProbePayload.model_validate(raw_payload)
        value = (
            payload.streams[0].duration if payload.streams else (payload.format.duration if payload.format else None)
        )
        return float(value) if value is not None else None
    except (TypeError, ValueError, json.JSONDecodeError, IndexError):
        return None


def ensure_temp_space(directory: Path, duration: float | None, safety_margin: int) -> None:
    if duration is None:
        return
    required = int(duration * NORMALIZED_BYTES_PER_SECOND) + safety_margin
    if shutil.disk_usage(directory).free < required:
        raise safe_error("insufficient_temp_space")


def _terminate_process_tree(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    if sys.platform == "win32":
        taskkill = shutil.which("taskkill") or str(
            Path(os.environ.get("SYSTEMROOT", "C:/Windows")) / "System32/taskkill.exe"
        )
        subprocess.run(  # noqa: S603
            [taskkill, "/PID", str(process.pid), "/T", "/F"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)


def _start_capped_reader(stream: BinaryIO) -> tuple[threading.Thread, Callable[[], bytes]]:
    captured = bytearray()

    def read() -> None:
        while chunk := stream.read(8192):
            captured.extend(chunk)
            overflow = len(captured) - MAX_STDERR_BYTES
            if overflow > 0:
                del captured[:overflow]

    thread = threading.Thread(target=read, name="hermes-onnx-asr-ffmpeg-stderr", daemon=True)
    thread.start()
    return thread, lambda: bytes(captured)


def normalize_audio(source: Path, destination: Path, settings: OnnxAsrSettings, timeout: float) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise safe_error("ffmpeg_missing")
    duration = probe_duration(source)
    ensure_temp_space(destination.parent, duration, settings.audio.temp_safety_margin_bytes)
    command = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-protocol_whitelist",
        "file,pipe,crypto,data",
        "-i",
        str(source),
        "-map",
        "0:a:0",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(destination),
    ]
    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if sys.platform == "win32" else 0
    try:
        process = subprocess.Popen(  # noqa: S603
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            shell=False,
            start_new_session=sys.platform != "win32",
            creationflags=creationflags,
        )
    except OSError as exc:
        raise safe_error("ffmpeg_missing") from exc
    if process.stderr is None:
        _terminate_process_tree(process)
        raise safe_error("decode_failed")
    stderr_thread, get_stderr = _start_capped_reader(cast("BinaryIO", process.stderr))
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        _terminate_process_tree(process)
        stderr_thread.join(timeout=1)
        raise safe_error("ffmpeg_timeout") from exc
    stderr_thread.join(timeout=1)
    stderr_text = get_stderr().decode(errors="replace").lower()
    if process.returncode != 0:
        if "matches no streams" in stderr_text or "does not contain any stream" in stderr_text:
            raise safe_error("no_audio_stream")
        if "no space left on device" in stderr_text:
            raise safe_error("insufficient_temp_space")
        raise safe_error("decode_failed")
    if not destination.is_file() or destination.stat().st_size <= MIN_WAV_BYTES:
        raise safe_error("decode_failed")


def enforce_duration_limit(duration: float, settings: OnnxAsrSettings) -> None:
    limit = settings.audio.max_duration_seconds
    if limit is not None and duration > limit:
        raise safe_error("audio_too_long")


def validate_source(path: Path) -> None:
    if not path.is_file():
        raise OnnxAsrError("transcription_failed", "Audio input is unavailable.")

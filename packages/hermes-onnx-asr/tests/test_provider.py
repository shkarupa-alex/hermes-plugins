# pyright: reportPrivateUsage=false
from __future__ import annotations
import threading
import time
import wave
from concurrent.futures import Future
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from hermes_onnx_asr.catalog import certified_model_names
from hermes_onnx_asr.config import OnnxAsrSettings, RuntimeSettings
from hermes_onnx_asr.errors import OnnxAsrError, safe_error
from hermes_onnx_asr.provider import OnnxAsrProvider, _Job, _own_source, _Scheduler, failure_result

if TYPE_CHECKING:
    from hermes_onnx_asr.pipeline import Pipeline


def test_provider_advertises_only_release_smoked_models() -> None:
    advertised = tuple(item["id"] for item in OnnxAsrProvider().list_models())
    assert advertised == certified_model_names()


def test_failure_envelope_is_stable_and_path_free() -> None:
    result = failure_result(safe_error("model_not_installed"))
    assert result == {
        "success": False,
        "transcript": "",
        "provider": "onnx_asr",
        "error": "The selected speech recognition model is not installed.",
        "error_code": "model_not_installed",
    }


def test_queue_depth_zero_rejects_second_job_until_first_really_finishes(monkeypatch: pytest.MonkeyPatch) -> None:
    started = threading.Event()
    release = threading.Event()

    def execute(_scheduler: _Scheduler, _job: _Job) -> dict[str, object]:
        started.set()
        release.wait(timeout=2)
        return {"success": True, "transcript": "ok", "provider": "onnx_asr"}

    monkeypatch.setattr(_Scheduler, "_execute", execute)
    scheduler = _Scheduler(queue_depth=0)
    settings = OnnxAsrSettings(runtime=RuntimeSettings(queue_depth=0))
    first_future: Future[dict[str, object]] = Future()
    first = _Job(Path("one.wav"), settings, None, time.monotonic() + 10, first_future)
    second = _Job(Path("two.wav"), settings, None, time.monotonic() + 10, Future())
    try:
        assert scheduler.submit(first) is True
        assert started.wait(timeout=1)
        assert scheduler.submit(second) is False
        release.set()
        assert first_future.result(timeout=1)["success"] is True
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline and not scheduler.submit(second):
            time.sleep(0.01)
        assert second.future.result(timeout=1)["success"] is True
    finally:
        release.set()
        scheduler.shutdown()


def test_one_provider_rejects_profile_switch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    provider = OnnxAsrProvider()
    active = tmp_path / "one"
    monkeypatch.setattr("hermes_onnx_asr.provider.get_hermes_home", lambda: active)
    provider._bind_profile()
    monkeypatch.setattr("hermes_onnx_asr.provider.get_hermes_home", lambda: tmp_path / "two")
    with pytest.raises(OnnxAsrError) as caught:
        provider._bind_profile()
    assert caught.value.code == "configuration_invalid"


def test_expired_waiter_is_removed_from_fifo_and_releases_admission(monkeypatch: pytest.MonkeyPatch) -> None:
    started = threading.Event()
    release = threading.Event()

    def execute(_scheduler: _Scheduler, job: _Job) -> dict[str, object]:
        if job.source.name == "running.wav":
            started.set()
            release.wait(timeout=2)
        return {"success": True, "transcript": "ok", "provider": "onnx_asr"}

    monkeypatch.setattr(_Scheduler, "_execute", execute)
    scheduler = _Scheduler(queue_depth=1)
    settings = OnnxAsrSettings(runtime=RuntimeSettings(queue_depth=1))
    running = _Job(Path("running.wav"), settings, None, time.monotonic() + 10, Future())
    expired = _Job(Path("expired.wav"), settings, None, time.monotonic() + 0.01, Future())
    replacement = _Job(Path("replacement.wav"), settings, None, time.monotonic() + 10, Future())
    try:
        assert scheduler.submit(running)
        assert started.wait(timeout=1)
        assert scheduler.submit(expired)
        time.sleep(0.02)
        assert scheduler.cancel_queued(expired)
        assert scheduler.queued == 0
        assert scheduler.submit(replacement)
    finally:
        release.set()
        scheduler.shutdown()


def test_running_pcm_job_owns_wav_after_caller_removes_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.wav"
    with wave.open(str(source), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(b"\0\0" * 16_000)
    started = threading.Event()
    release = threading.Event()
    observed: dict[str, Path] = {}

    def get_pipeline(_scheduler: _Scheduler, _settings: OnnxAsrSettings) -> Pipeline:
        return cast("Pipeline", object())

    monkeypatch.setattr(_Scheduler, "_get_pipeline", get_pipeline)

    def recognize(_pipeline: object, wav_path: Path, _duration: float, _language: str | None) -> dict[str, object]:
        observed["path"] = wav_path
        started.set()
        release.wait(timeout=2)
        assert wav_path.is_file()
        return {"success": True, "transcript": "ok", "provider": "onnx_asr"}

    monkeypatch.setattr("hermes_onnx_asr.provider.recognize", recognize)
    scheduler = _Scheduler(queue_depth=0)
    future: Future[dict[str, object]] = Future()
    job = _Job(source, OnnxAsrSettings(), None, time.monotonic() + 10, future)
    try:
        assert scheduler.submit(job)
        assert started.wait(timeout=1)
        assert observed["path"] != source
        source.unlink()
        assert observed["path"].is_file()
        release.set()
        assert future.result(timeout=1)["success"] is True
    finally:
        release.set()
        scheduler.shutdown()


def test_transcribe_owns_source_before_worker_claim_and_waiter_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.wav"
    with wave.open(str(source), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(b"\0\0" * 16_000)
    settings = OnnxAsrSettings(runtime=RuntimeSettings(queue_depth=0, transcription_timeout_seconds=0.05))
    started = threading.Event()
    release = threading.Event()
    observed: dict[str, Path] = {}

    monkeypatch.setattr("hermes_onnx_asr.provider.load_settings", lambda: settings)
    monkeypatch.setattr("hermes_onnx_asr.provider.get_hermes_home", lambda: tmp_path)

    def execute(_scheduler: _Scheduler, job: _Job) -> dict[str, object]:
        observed["source"] = job.source
        started.set()
        release.wait(timeout=2)
        assert job.source.is_file()
        return {"success": True, "transcript": "ok", "provider": "onnx_asr"}

    monkeypatch.setattr(_Scheduler, "_execute", execute)
    provider = OnnxAsrProvider()
    try:
        result = provider.transcribe(str(source))
        assert started.is_set()
        assert result["error_code"] == "asr_timeout"
        assert observed["source"] != source
        source.unlink()
        assert observed["source"].is_file()
    finally:
        release.set()
        provider.shutdown()


def test_full_queue_is_rejected_before_source_staging(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    scheduler = _Scheduler(queue_depth=0)
    assert scheduler.reserve()
    provider = OnnxAsrProvider()
    provider._scheduler = scheduler
    monkeypatch.setattr("hermes_onnx_asr.provider.get_hermes_home", lambda: tmp_path)

    def must_not_stage(*_args: object) -> tuple[Path, object]:
        raise AssertionError("source staging ran for a full queue")

    monkeypatch.setattr("hermes_onnx_asr.provider._own_source", must_not_stage)
    try:
        result = provider.transcribe(str(tmp_path / "multi-hour.wav"))
        assert result["error_code"] == "asr_queue_full"
    finally:
        scheduler.release_reserved()
        provider.shutdown()


def test_source_copy_enospc_has_specific_safe_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"RIFF")

    def hardlink_unavailable(_source: Path, _destination: Path) -> None:
        raise OSError

    monkeypatch.setattr("hermes_onnx_asr.provider.os.link", hardlink_unavailable)

    def no_space(*_args: object) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("hermes_onnx_asr.provider.shutil.copyfile", no_space)
    with pytest.raises(OnnxAsrError) as caught:
        _own_source(source, 0)
    assert caught.value.code == "insufficient_temp_space"


def test_staging_failure_releases_reserved_capacity(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings = OnnxAsrSettings(runtime=RuntimeSettings(queue_depth=0))
    monkeypatch.setattr("hermes_onnx_asr.provider.load_settings", lambda: settings)
    monkeypatch.setattr("hermes_onnx_asr.provider.get_hermes_home", lambda: tmp_path)
    provider = OnnxAsrProvider()
    try:
        result = provider.transcribe(str(tmp_path / "missing.wav"))
        assert result["error_code"] == "transcription_failed"
        assert provider._scheduler is not None
        assert provider._scheduler.reserve()
        provider._scheduler.release_reserved()
    finally:
        provider.shutdown()

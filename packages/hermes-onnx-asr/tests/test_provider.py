# pyright: reportPrivateUsage=false
from __future__ import annotations
import threading
import time
from concurrent.futures import Future
from pathlib import Path

import pytest

from hermes_onnx_asr.config import OnnxAsrSettings, RuntimeSettings
from hermes_onnx_asr.errors import OnnxAsrError, safe_error
from hermes_onnx_asr.provider import OnnxAsrProvider, _Job, _Scheduler, failure_result


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

from __future__ import annotations
import atexit
import errno
import importlib.util
import os
import shutil
import tempfile
import threading
import time
from collections import deque
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.transcription_provider import TranscriptionProvider
from hermes_constants import get_hermes_home
from pydantic import ValidationError

from hermes_onnx_asr.audio import (
    enforce_duration_limit,
    is_compatible_pcm_wav,
    normalize_audio,
    validate_source,
    wav_duration,
)
from hermes_onnx_asr.catalog import certified_model_names, model_languages
from hermes_onnx_asr.config import DEFAULT_MODEL, OnnxAsrSettings, load_settings
from hermes_onnx_asr.errors import OnnxAsrError, safe_error
from hermes_onnx_asr.pipeline import Pipeline, load_pipeline, recognize


@dataclass
class _Job:
    source: Path
    settings: OnnxAsrSettings
    language: str | None
    deadline: float
    future: Future[dict[str, object]]
    owned_source: tempfile.TemporaryDirectory[str] | None = None

    def cleanup(self) -> None:
        owned_source = self.owned_source
        self.owned_source = None
        if owned_source is not None:
            owned_source.cleanup()


def _own_source(source: Path, safety_margin: int) -> tuple[Path, tempfile.TemporaryDirectory[str]]:
    """Acquire a caller-independent path after reservation and before worker publication."""
    validate_source(source)
    try:
        owned_source = tempfile.TemporaryDirectory(prefix="hermes-onnx-asr-source-")
    except OSError as exc:
        if exc.errno == errno.ENOSPC:
            raise safe_error("insufficient_temp_space") from exc
        raise
    staged = Path(owned_source.name) / f"input{source.suffix}"
    try:
        try:
            os.link(source, staged)
        except OSError:
            required = source.stat().st_size + safety_margin
            if shutil.disk_usage(owned_source.name).free < required:
                raise safe_error("insufficient_temp_space") from None
            shutil.copyfile(source, staged)
    except OSError as exc:
        owned_source.cleanup()
        if exc.errno == errno.ENOSPC:
            raise safe_error("insufficient_temp_space") from exc
        raise
    except Exception:
        owned_source.cleanup()
        raise
    return staged, owned_source


class _Scheduler:
    def __init__(self, queue_depth: int) -> None:
        self._jobs: deque[_Job] = deque()
        self._state_lock = threading.Lock()
        self._condition = threading.Condition(self._state_lock)
        self._admission = threading.BoundedSemaphore(1 + queue_depth)
        self._pipeline: Pipeline | None = None
        self._pipeline_key: tuple[object, ...] | None = None
        self._shutting_down = False
        self._worker = threading.Thread(target=self._run, name="hermes-onnx-asr", daemon=True)
        self._worker.start()

    @property
    def queued(self) -> int:
        with self._state_lock:
            return len(self._jobs)

    @property
    def pipeline(self) -> Pipeline | None:
        return self._pipeline

    @property
    def is_shutting_down(self) -> bool:
        return self._shutting_down

    def submit(self, job: _Job) -> bool:
        if not self.reserve():
            return False
        return self.submit_reserved(job)

    def reserve(self) -> bool:
        """Reserve bounded admission before potentially expensive source staging."""
        with self._state_lock:
            return not (self._shutting_down or not self._admission.acquire(blocking=False))

    def release_reserved(self) -> None:
        self._admission.release()

    def submit_reserved(self, job: _Job) -> bool:
        """Publish a job after its caller has acquired a reservation."""
        with self._state_lock:
            if self._shutting_down:
                self._admission.release()
                return False
            self._jobs.append(job)
            self._condition.notify()
            return True

    def cancel_queued(self, job: _Job) -> bool:
        """Remove an expired job only if the worker has not claimed it."""
        with self._state_lock:
            try:
                self._jobs.remove(job)
            except ValueError:
                return False
            self._admission.release()
            job.cleanup()
            return True

    def _key(self, settings: OnnxAsrSettings) -> tuple[object, ...]:
        vad = settings.vad
        runtime = settings.runtime
        return (
            settings.model,
            settings.quantization,
            settings.model_dir.resolve(),
            vad.min_audio_seconds,
            vad.threshold,
            vad.negative_threshold,
            vad.min_speech_duration_ms,
            vad.max_speech_duration_s,
            vad.min_silence_duration_ms,
            vad.speech_pad_ms,
            runtime.intra_op_num_threads,
            runtime.inter_op_num_threads,
        )

    def _get_pipeline(self, settings: OnnxAsrSettings) -> Pipeline:
        key = self._key(settings)
        if self._pipeline is not None and self._pipeline_key == key:
            return self._pipeline
        if self._pipeline is not None and self.queued > 0:
            raise safe_error("model_switch_busy")
        replacement = load_pipeline(settings)
        self._pipeline = replacement
        self._pipeline_key = key
        return replacement

    def _execute(self, job: _Job) -> dict[str, object]:
        validate_source(job.source)
        remaining = job.deadline - time.monotonic()
        if remaining <= 0:
            raise safe_error("asr_timeout")
        with tempfile.TemporaryDirectory(prefix="hermes-onnx-asr-") as work_dir:
            wav_path = Path(work_dir) / "input.wav"
            if is_compatible_pcm_wav(job.source):
                # ORT inference is not cancellable. Keep a worker-owned path alive after
                # the caller's waiter times out and is free to remove its source file.
                try:
                    os.link(job.source, wav_path)
                except OSError:
                    shutil.copyfile(job.source, wav_path)
            else:
                normalize_audio(
                    job.source,
                    wav_path,
                    job.settings,
                    min(job.settings.runtime.ffmpeg_timeout_seconds, remaining),
                )
            duration = wav_duration(wav_path)
            enforce_duration_limit(duration, job.settings)
            pipeline = self._get_pipeline(job.settings)
            return recognize(pipeline, wav_path, duration, job.language)

    def _run(self) -> None:
        while True:
            with self._state_lock:
                while not self._jobs and not self._shutting_down:
                    self._condition.wait()
                if not self._jobs and self._shutting_down:
                    return
                job = self._jobs.popleft()
            try:
                if time.monotonic() >= job.deadline:
                    result = failure_result(safe_error("asr_timeout"))
                else:
                    try:
                        result = self._execute(job)
                    except OnnxAsrError as exc:
                        result = failure_result(exc)
                    except Exception:  # noqa: BLE001 - provider boundary must return the Hermes envelope
                        result = failure_result(safe_error("transcription_failed"))
                job.future.set_result(result)
            finally:
                self._admission.release()
                job.cleanup()

    def shutdown(self, grace_seconds: float = 30) -> None:
        with self._state_lock:
            if self._shutting_down:
                return
            self._shutting_down = True
            while self._jobs:
                job = self._jobs.popleft()
                job.future.set_result(failure_result(safe_error("provider_shutting_down")))
                self._admission.release()
                job.cleanup()
            self._condition.notify_all()
        self._worker.join(timeout=grace_seconds)


def failure_result(error: OnnxAsrError) -> dict[str, object]:
    return {
        "success": False,
        "transcript": "",
        "provider": "onnx_asr",
        "error": error.message,
        "error_code": error.code,
    }


class OnnxAsrProvider(TranscriptionProvider):
    def __init__(self) -> None:
        self._profile_root: Path | None = None
        self._scheduler: _Scheduler | None = None
        self._lock = threading.Lock()
        atexit.register(self.shutdown)

    @property
    def name(self) -> str:
        return "onnx_asr"

    @property
    def display_name(self) -> str:
        return "ONNX ASR (CPU, offline)"

    def default_model(self) -> str:
        return DEFAULT_MODEL

    def list_models(self) -> list[dict[str, Any]]:
        return [
            {"id": alias, "display": alias, "languages": model_languages(alias)} for alias in certified_model_names()
        ]

    def get_setup_schema(self) -> dict[str, Any]:
        return {
            "name": self.display_name,
            "badge": "local",
            "tag": "Russian-capable, CPUExecutionProvider only",
            "env_vars": [],
        }

    def is_available(self) -> bool:
        try:
            if importlib.util.find_spec("onnx_asr") is None or importlib.util.find_spec("onnxruntime") is None:
                return False
            load_settings()
        except Exception:  # noqa: BLE001 - availability probes must never raise
            return False
        return True

    def _bind_profile(self) -> None:
        current = get_hermes_home().resolve()
        with self._lock:
            if self._profile_root is None:
                self._profile_root = current
            elif self._profile_root != current:
                raise safe_error("configuration_invalid")

    def _get_scheduler(self, settings: OnnxAsrSettings) -> _Scheduler:
        with self._lock:
            if self._scheduler is None:
                self._scheduler = _Scheduler(settings.runtime.queue_depth)
            return self._scheduler

    def transcribe(  # noqa: PLR0911 - provider boundary maps each admission failure explicitly
        self,
        file_path: str,
        *,
        model: str | None = None,
        language: str | None = None,
        **extra: Any,  # noqa: ANN401 - exact Hermes TranscriptionProvider contract
    ) -> dict[str, Any]:
        del extra
        try:
            self._bind_profile()
            settings = load_settings()
            if model is not None:
                settings = settings.model_copy(update={"model": model})
            effective_language = language if language is not None else settings.language
            scheduler = self._get_scheduler(settings)
            timeout = settings.runtime.transcription_timeout_seconds
            deadline = time.monotonic() + timeout
            if not scheduler.reserve():
                code = "provider_shutting_down" if scheduler.is_shutting_down else "asr_queue_full"
                return failure_result(safe_error(code))
            try:
                source, owned_source = _own_source(Path(file_path), settings.audio.temp_safety_margin_bytes)
            except Exception:
                scheduler.release_reserved()
                raise
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                owned_source.cleanup()
                scheduler.release_reserved()
                return failure_result(safe_error("asr_timeout"))
            future: Future[dict[str, object]] = Future()
            job = _Job(
                source=source,
                settings=settings,
                language=effective_language,
                deadline=deadline,
                future=future,
                owned_source=owned_source,
            )
            if not scheduler.submit_reserved(job):
                job.cleanup()
                return failure_result(safe_error("provider_shutting_down"))
            try:
                return future.result(timeout=remaining)
            except FutureTimeoutError:
                scheduler.cancel_queued(job)
                return failure_result(safe_error("asr_timeout"))
        except (ValidationError, ValueError):
            return failure_result(safe_error("configuration_invalid"))
        except OnnxAsrError as exc:
            return failure_result(exc)
        except Exception:  # noqa: BLE001 - provider boundary must not leak dependency exceptions
            return failure_result(safe_error("transcription_failed"))

    def shutdown(self) -> None:
        scheduler = self._scheduler
        if scheduler is not None:
            scheduler.shutdown()

    def scheduler_diagnostics(self) -> dict[str, object]:
        scheduler = self._scheduler
        return {
            "queued": scheduler.queued if scheduler is not None else 0,
            "shutting_down": scheduler.is_shutting_down if scheduler is not None else False,
            "pipeline_loaded": scheduler.pipeline is not None if scheduler is not None else False,
        }


provider = OnnxAsrProvider()

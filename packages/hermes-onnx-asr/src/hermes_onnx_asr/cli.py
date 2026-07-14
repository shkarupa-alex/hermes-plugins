from __future__ import annotations
import os
import shutil
import tempfile
import time
import wave
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import ValidationError

from hermes_onnx_asr.catalog import bundle_path, fetch_bundle, load_catalog, verify_bundle
from hermes_onnx_asr.compat import check_compatibility, check_requirements
from hermes_onnx_asr.config import load_settings
from hermes_onnx_asr.errors import OnnxAsrError
from hermes_onnx_asr.pipeline import load_pipeline
from hermes_onnx_asr.provider import provider
from hermes_onnx_asr.setup import interactive_setup

if TYPE_CHECKING:
    import argparse


def setup_parser(parser: argparse.ArgumentParser) -> None:
    commands = parser.add_subparsers(dest="onnx_asr_command", required=True)
    commands.add_parser("setup", help="Run the interactive setup wizard")
    commands.add_parser("list-models", help="List the pinned model catalog")
    fetch = commands.add_parser("fetch", help="Download and verify a model bundle")
    fetch.add_argument("model")
    fetch.add_argument("--quantization", default="int8")
    fetch_vad = commands.add_parser("fetch-vad", help="Download and verify a VAD bundle")
    fetch_vad.add_argument("engine", nargs="?", default="silero")
    commands.add_parser("warmup", help="Load models and run a local inference smoke test")
    commands.add_parser("doctor", help="Check dependencies, bundles, CPU providers, and temporary storage")
    transcribe = commands.add_parser("transcribe", help="Transcribe a local audio file without Hermes' 25 MiB gate")
    transcribe.add_argument("file", type=Path)
    transcribe.add_argument("--model")
    transcribe.add_argument("--language")


def handle_command(args: argparse.Namespace) -> int:  # noqa: C901, PLR0911 - argparse subcommand dispatcher
    command = args.onnx_asr_command
    if command == "setup":
        interactive_setup()
        return 0
    if command == "list-models":
        return _list_models()
    if command == "fetch":
        settings = load_settings()
        quantization = None if args.quantization.lower() in {"none", "fp32"} else args.quantization
        try:
            path = fetch_bundle(settings.model_dir, args.model, quantization)
        except OnnxAsrError as exc:
            print(f"Fetch failed [{exc.code}]: {exc.message}")
            return 1
        print(f"Installed: {path}")
        return 0
    if command == "fetch-vad":
        settings = load_settings()
        try:
            path = fetch_bundle(settings.model_dir, args.engine, None, kind="vad")
        except OnnxAsrError as exc:
            print(f"Fetch failed [{exc.code}]: {exc.message}")
            return 1
        print(f"Installed: {path}")
        return 0
    if command == "warmup":
        return _warmup()
    if command == "doctor":
        return _doctor()
    if command == "transcribe":
        result = provider.transcribe(str(args.file), model=args.model, language=args.language)
        if result["success"]:
            print(result["transcript"])
            return 0
        print(f"Error [{result.get('error_code')}]: {result['error']}")
        return 1
    raise ValueError(f"unsupported onnx-asr command: {command}")


def _list_models() -> int:
    catalog = load_catalog()
    for entry in catalog.models:
        marker = " (default)" if entry.alias == "gigaam-v3-e2e-rnnt" else ""
        quantizations = ", ".join(value or "fp32" for value in entry.quantizations)
        print(f"{entry.alias}{marker}: {quantizations}")
    return 0


def _warmup() -> int:
    try:
        settings = load_settings()
        pipeline = load_pipeline(settings)
        with tempfile.TemporaryDirectory(prefix="hermes-onnx-asr-warmup-") as directory:
            sample = Path(directory) / "silence.wav"
            with wave.open(str(sample), "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(16_000)
                audio.writeframes(b"\0\0" * 4_000)
            pipeline.base_model.recognize(sample, channel="mean")
    except (OnnxAsrError, OSError, ValidationError, ValueError) as exc:
        print(f"Warm-up failed: {exc}")
        return 1
    print(f"Warm-up passed; audited {len(pipeline.audit.roles)} CPU-only ONNX sessions")
    return 0


def _cleanup_stale_temp(ttl_seconds: int) -> int:
    root = Path(tempfile.gettempdir())
    now = time.time()
    removed = 0
    for path in root.glob("hermes-onnx-asr-*"):
        try:
            stat_result = path.lstat()
            if path.is_symlink() or not path.is_dir() or now - stat_result.st_mtime < ttl_seconds:
                continue
            if hasattr(os, "getuid") and stat_result.st_uid != os.getuid():
                continue
            shutil.rmtree(path)
            removed += 1
        except OSError:
            continue
    return removed


def _doctor() -> int:
    compatible, message = check_compatibility()
    print(f"Hermes:         {message}")
    if not compatible:
        return 1
    print(f"Dependencies:   {'ready' if check_requirements() else 'version mismatch'}")
    try:
        settings = load_settings()
    except (ValidationError, ValueError) as exc:
        print(f"Configuration:  invalid ({exc})")
        return 1
    print("Provider:       onnx_asr")
    print(f"Model:          {settings.model} / {settings.quantization or 'fp32'}")
    threshold = settings.vad.min_audio_seconds
    vad_status = "disabled" if threshold is None else f"{settings.vad.engine}, threshold {threshold:g}s"
    print(f"VAD:            {vad_status}")
    print("Execution:      CPUExecutionProvider only")
    print(f"Runtime fetch:  {'enabled' if settings.allow_runtime_download else 'disabled'}")
    print(f"ffmpeg:         {'ready' if shutil.which('ffmpeg') else 'missing (needed for non-PCM audio)'}")
    model_path = bundle_path(settings.model_dir, settings.model, settings.quantization)
    try:
        verify_bundle(model_path, settings.model, settings.quantization)
        if threshold is not None:
            verify_bundle(
                bundle_path(settings.model_dir, settings.vad.engine, None, kind="vad"),
                settings.vad.engine,
                None,
                kind="vad",
            )
    except OnnxAsrError as exc:
        print(f"Model files:    not ready [{exc.code}]")
        return 1
    print("Model files:    ready, immutable revision")
    removed = _cleanup_stale_temp(settings.runtime.stale_temp_ttl_seconds)
    free_gib = shutil.disk_usage(tempfile.gettempdir()).free // (1024**3)
    print(f"Temp storage:   {free_gib} GiB free; {removed} stale removed")
    return _warmup()

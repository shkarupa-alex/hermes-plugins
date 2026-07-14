from __future__ import annotations
from typing import Any, cast

from hermes_cli.config import read_raw_config, save_config
from hermes_cli.setup import print_header, print_info, print_success, prompt, prompt_yes_no

from hermes_onnx_asr.catalog import fetch_bundle, load_catalog
from hermes_onnx_asr.config import DEFAULT_MODEL, OnnxAsrSettings, VadSettings


def _mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if isinstance(value, dict):
        return cast("dict[str, Any]", value)
    result: dict[str, Any] = {}
    parent[key] = result
    return result


def write_setup_config(model: str, vad_seconds: float | None) -> None:
    config = read_raw_config()
    stt = _mapping(config, "stt")
    stt["provider"] = "onnx_asr"
    onnx_config = _mapping(stt, "onnx_asr")
    onnx_config["model"] = model
    onnx_config["quantization"] = "int8"
    vad_config = _mapping(onnx_config, "vad")
    vad_config["min_audio_seconds"] = vad_seconds
    save_config(config, strip_defaults=False)


def interactive_setup() -> None:
    print_header("ONNX ASR — локальное распознавание речи")
    print_info("Модели работают только через CPUExecutionProvider. Секреты и API-ключи не нужны.")
    aliases = {entry.alias for entry in load_catalog().models}
    model = prompt("Модель", default=DEFAULT_MODEL).strip()
    if model not in aliases:
        raise ValueError(f"Неподдерживаемая модель: {model}")
    threshold = prompt("Silero VAD от N секунд (off — отключить)", default="20").strip().lower()
    if threshold in {"off", "none", "null"}:
        vad_seconds = None
    else:
        try:
            vad_seconds = float(threshold)
        except ValueError as exc:
            raise ValueError("Порог VAD должен быть неотрицательным числом или off") from exc
        if vad_seconds < 0:
            raise ValueError("Порог VAD должен быть неотрицательным числом")
    write_setup_config(model, vad_seconds)
    settings = OnnxAsrSettings(model=model, vad=VadSettings(min_audio_seconds=vad_seconds))
    if prompt_yes_no("Скачать модель и Silero VAD сейчас?", default=True):
        print_info("Скачивание может занять несколько минут и несколько гигабайт.")
        fetch_bundle(settings.model_dir, model, settings.quantization)
        if settings.vad.min_audio_seconds is not None:
            fetch_bundle(settings.model_dir, settings.vad.engine, None, kind="vad")
    print_success("ONNX ASR настроен. Выполните `hermes onnx-asr doctor` для проверки.")

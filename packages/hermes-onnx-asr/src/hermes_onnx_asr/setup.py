from __future__ import annotations
from typing import Any, cast

from hermes_cli.config import read_raw_config, save_config
from hermes_cli.setup import print_header, print_info, print_success, prompt, prompt_yes_no

from hermes_onnx_asr.catalog import (
    catalog_model_entries,
    certified_model_names,
    fetch_bundle,
    model_entry,
    upstream_model_names,
)
from hermes_onnx_asr.config import DEFAULT_MODEL, OnnxAsrSettings, VadSettings


def _mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if isinstance(value, dict):
        return cast("dict[str, Any]", value)
    result: dict[str, Any] = {}
    parent[key] = result
    return result


def write_setup_config(model: str, quantization: str | None, vad_seconds: float | None) -> None:
    config = read_raw_config()
    stt = _mapping(config, "stt")
    stt["provider"] = "onnx_asr"
    onnx_config = _mapping(stt, "onnx_asr")
    onnx_config["model"] = model
    onnx_config["quantization"] = quantization
    vad_config = _mapping(onnx_config, "vad")
    vad_config["min_audio_seconds"] = vad_seconds
    save_config(config, strip_defaults=False)


def _select_model() -> str:
    upstream = upstream_model_names()
    entries = catalog_model_entries()
    aliases = tuple(entry.alias for entry in entries)
    certified = set(certified_model_names())
    print_info("Модели, объявленные установленной версией onnx-asr:")
    for index, entry in enumerate(entries, start=1):
        alias = entry.alias
        quantizations = ", ".join(value or "fp32" for value in entry.quantizations)
        status = "certified" if alias in certified else "pending smoke"
        marker = ", по умолчанию" if alias == DEFAULT_MODEL else ""
        print_info(f"  {index:>2}. {alias} [{quantizations}; {status}{marker}]")
    for alias in upstream:
        if alias not in aliases:
            print_info(f"   -. {alias} [pending catalog]")
    default_choice = str(aliases.index(DEFAULT_MODEL) + 1)
    raw_model = prompt("Модель — номер или точное имя", default=default_choice).strip()
    selected = aliases[int(raw_model) - 1] if raw_model.isdigit() and 1 <= int(raw_model) <= len(aliases) else raw_model
    if selected in certified:
        return selected
    if selected in aliases:
        raise ValueError(f"Модель ещё не прошла обязательный release smoke: {selected}")
    raise ValueError(f"Неподдерживаемая модель: {raw_model}")


def _select_quantization(model: str) -> str | None:
    supported_quantizations = model_entry(model).quantizations
    if len(supported_quantizations) > 1:
        raw_quantization = prompt("Квантование (int8 или fp32)", default="int8").strip().lower()
        quantization = None if raw_quantization in {"fp32", "none"} else raw_quantization
        if quantization not in supported_quantizations:
            raise ValueError(f"Неподдерживаемое квантование для {model}: {raw_quantization}")
        return quantization
    only = supported_quantizations[0]
    print_info(f"Для {model} доступно только {only or 'fp32'}.")
    return only


def _select_vad_threshold() -> float | None:
    threshold = prompt("Silero VAD от N секунд (off — отключить)", default="20").strip().lower()
    if threshold in {"off", "none", "null"}:
        return None
    try:
        vad_seconds = float(threshold)
    except ValueError as exc:
        raise ValueError("Порог VAD должен быть неотрицательным числом или off") from exc
    if vad_seconds < 0:
        raise ValueError("Порог VAD должен быть неотрицательным числом")
    return vad_seconds


def interactive_setup() -> None:
    print_header("ONNX ASR — локальное распознавание речи")
    print_info("Модели работают только через CPUExecutionProvider. Секреты и API-ключи не нужны.")
    model = _select_model()
    quantization = _select_quantization(model)
    vad_seconds = _select_vad_threshold()
    write_setup_config(model, quantization, vad_seconds)
    settings = OnnxAsrSettings(
        model=model,
        quantization=quantization,
        vad=VadSettings(min_audio_seconds=vad_seconds),
    )
    if prompt_yes_no("Скачать модель и Silero VAD сейчас?", default=True):
        print_info("Скачивание может занять несколько минут и несколько гигабайт.")
        fetch_bundle(settings.model_dir, model, settings.quantization)
        if settings.vad.min_audio_seconds is not None:
            fetch_bundle(settings.model_dir, settings.vad.engine, None, kind="vad")
    print_success("ONNX ASR настроен. Выполните `hermes onnx-asr doctor` для проверки.")

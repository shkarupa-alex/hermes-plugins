# pyright: reportPrivateUsage=false
from __future__ import annotations
from typing import Any

import pytest

from hermes_onnx_asr import setup
from hermes_onnx_asr.config import DEFAULT_MODEL


@pytest.mark.parametrize("vad_seconds", [None, 0.0, 20.0])
def test_setup_writes_real_yaml_null_and_preserves_unrelated_settings(
    monkeypatch: pytest.MonkeyPatch,
    vad_seconds: float | None,
) -> None:
    saved: list[dict[str, Any]] = []
    monkeypatch.setattr(
        setup,
        "read_raw_config",
        lambda: {"stt": {"onnx_asr": {"runtime": {"queue_depth": 2}}}, "unrelated": True},
    )

    def save_config(config: dict[str, Any], **_kwargs: object) -> None:
        saved.append(config)

    monkeypatch.setattr(setup, "save_config", save_config)
    setup.write_setup_config("gigaam-v3-e2e-rnnt", "int8", vad_seconds)
    assert saved[0]["unrelated"] is True
    onnx_config = saved[0]["stt"]["onnx_asr"]
    assert onnx_config["vad"]["min_audio_seconds"] == vad_seconds
    assert onnx_config["quantization"] == "int8"
    assert onnx_config["runtime"]["queue_depth"] == 2


def test_model_picker_uses_upstream_registry_and_accepts_number(monkeypatch: pytest.MonkeyPatch) -> None:
    output: list[str] = []
    defaults: list[str] = []

    def choose(_label: str, *, default: str) -> str:
        defaults.append(default)
        return default

    monkeypatch.setattr(setup, "print_info", output.append)
    monkeypatch.setattr(setup, "prompt", choose)
    assert setup._select_model() == DEFAULT_MODEL
    assert defaults == ["6"]
    assert any("nemo-parakeet-tdt-0.6b-v3" in line for line in output)
    assert any("nemo-canary-1b-v2" in line for line in output)
    assert any("t-tech/t-one" in line for line in output)


def test_t_one_picker_uses_its_only_supported_fp32(monkeypatch: pytest.MonkeyPatch) -> None:
    def ignore(_message: str) -> None:
        pass

    monkeypatch.setattr(setup, "print_info", ignore)
    assert setup._select_quantization("t-tech/t-one") is None


@pytest.mark.parametrize(
    ("system", "machine", "expected"),
    [
        ("Linux", "x86_64", "fp32"),
        ("Darwin", "arm64", "fp32"),
        ("Linux", "aarch64", "int8"),
        ("Linux", "riscv64", "int8"),
    ],
)
def test_quantization_guidance_uses_safe_platform_default(
    monkeypatch: pytest.MonkeyPatch,
    system: str,
    machine: str,
    expected: str,
) -> None:
    monkeypatch.setattr(setup.platform, "system", lambda: system)
    monkeypatch.setattr(setup.platform, "machine", lambda: machine)
    default, guidance = setup._quantization_guidance()
    assert default == expected
    combined = " ".join(guidance)
    assert "int8" in combined
    assert "fp32" in combined


def test_model_picker_rejects_upstream_model_pending_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    def ignore(_message: str) -> None:
        pass

    def choose(_label: str, *, default: str) -> str:
        del default
        return "nemo-canary-1b-v2"

    monkeypatch.setattr(setup, "print_info", ignore)
    monkeypatch.setattr(setup, "prompt", choose)
    with pytest.raises(ValueError, match="release smoke"):
        setup._select_model()

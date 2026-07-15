# pyright: reportPrivateUsage=false
from __future__ import annotations
from typing import Any

import pytest
from agent.transcription_provider import TranscriptionProvider
from agent.transcription_registry import _reset_for_tests, register_provider
from tools.transcription_tools import _dispatch_to_plugin_provider


class RecordingProvider(TranscriptionProvider):
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None, str | None]] = []

    @property
    def name(self) -> str:
        return "onnx_asr"

    def transcribe(
        self,
        file_path: str,
        *,
        model: str | None = None,
        language: str | None = None,
        **extra: Any,  # noqa: ANN401 - exact Hermes provider contract
    ) -> dict[str, Any]:
        del extra
        self.calls.append((file_path, model, language))
        return {"success": True, "transcript": "текст", "provider": self.name}


@pytest.mark.parametrize("source_platform", ["telegram", "vk"])
def test_all_gateway_platforms_reach_the_same_registered_provider(
    monkeypatch: pytest.MonkeyPatch,
    source_platform: str,
) -> None:
    provider = RecordingProvider()
    _reset_for_tests()
    register_provider(provider)

    def discovery_stub(**_kwargs: object) -> None:
        return None

    monkeypatch.setattr("hermes_cli.plugins._ensure_plugins_discovered", discovery_stub)
    result = _dispatch_to_plugin_provider(
        f"/{source_platform}/voice.ogg",
        "onnx_asr",
        {"onnx_asr": {"model": "gigaam-v3-e2e-rnnt"}},
        model="gigaam-v3-e2e-rnnt",
        language=None,
    )
    assert result is not None
    assert result["success"] is True
    assert provider.calls == [(f"/{source_platform}/voice.ogg", "gigaam-v3-e2e-rnnt", None)]
    _reset_for_tests()

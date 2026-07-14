from __future__ import annotations

from agent.transcription_provider import TranscriptionProvider

from hermes_onnx_asr.plugin import register


class ContextRecorder:
    def __init__(self) -> None:
        self.provider: object | None = None
        self.command: dict[str, object] = {}

    def register_transcription_provider(self, provider: object) -> None:
        self.provider = provider

    def register_cli_command(self, **kwargs: object) -> None:
        self.command = kwargs


def test_register_exposes_transcription_and_cli_contract() -> None:
    context = ContextRecorder()
    register(context)
    assert isinstance(context.provider, TranscriptionProvider)
    assert context.provider.name == "onnx_asr"  # type: ignore[union-attr]
    assert context.command["name"] == "onnx-asr"
    assert callable(context.command["setup_fn"])
    assert callable(context.command["handler_fn"])

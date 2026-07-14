from __future__ import annotations
from typing import Protocol

from hermes_onnx_asr.cli import handle_command, setup_parser
from hermes_onnx_asr.compat import check_compatibility
from hermes_onnx_asr.provider import provider


class PluginContext(Protocol):
    def register_transcription_provider(self, provider: object) -> None: ...

    def register_cli_command(self, **kwargs: object) -> None: ...


def register(ctx: PluginContext) -> None:
    compatible, message = check_compatibility()
    if not compatible:
        raise RuntimeError(f"hermes-onnx-asr is incompatible with this Hermes installation: {message}")
    ctx.register_transcription_provider(provider)
    ctx.register_cli_command(
        name="onnx-asr",
        help="CPU-only ONNX speech recognition",
        description="Configure, fetch, diagnose, and use the ONNX ASR transcription provider",
        setup_fn=setup_parser,
        handler_fn=handle_command,
    )

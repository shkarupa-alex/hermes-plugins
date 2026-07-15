from __future__ import annotations
import inspect
from importlib.metadata import PackageNotFoundError, version

from agent.transcription_provider import TranscriptionProvider
from hermes_cli.plugins import PluginContext
from packaging.version import Version

MIN_HERMES = Version("0.18.2")
MAX_HERMES = Version("0.19")


EXPECTED_TRANSCRIBE_SHAPE = (
    ("self", inspect.Parameter.POSITIONAL_OR_KEYWORD, False),
    ("file_path", inspect.Parameter.POSITIONAL_OR_KEYWORD, False),
    ("model", inspect.Parameter.KEYWORD_ONLY, True),
    ("language", inspect.Parameter.KEYWORD_ONLY, True),
    ("extra", inspect.Parameter.VAR_KEYWORD, False),
)


def check_compatibility() -> tuple[bool, str]:
    try:
        installed = Version(version("hermes-agent"))
    except (ImportError, PackageNotFoundError) as exc:
        return False, f"Hermes Agent is unavailable: {exc}"
    if not MIN_HERMES <= installed < MAX_HERMES:
        return False, f"Hermes Agent {installed} is outside the tested range >=0.18.2,<0.19"
    parameters = inspect.signature(TranscriptionProvider.transcribe).parameters
    shape = tuple(
        (parameter.name, parameter.kind, parameter.default is not inspect.Parameter.empty)
        for parameter in parameters.values()
    )
    if shape != EXPECTED_TRANSCRIBE_SHAPE:
        return False, f"Hermes TranscriptionProvider.transcribe contract has changed: {shape}"
    from hermes_onnx_asr.provider import OnnxAsrProvider  # noqa: PLC0415

    provider_shape = tuple(
        (parameter.name, parameter.kind, parameter.default is not inspect.Parameter.empty)
        for parameter in inspect.signature(OnnxAsrProvider.transcribe).parameters.values()
    )
    if provider_shape != EXPECTED_TRANSCRIBE_SHAPE:
        return False, f"OnnxAsrProvider.transcribe signature drifted: {provider_shape}"
    if not callable(getattr(PluginContext, "register_transcription_provider", None)):
        return False, "Hermes does not expose register_transcription_provider"
    return True, f"Hermes Agent {installed} transcription contract is compatible"


def check_requirements() -> bool:
    compatible, _ = check_compatibility()
    if not compatible:
        return False
    try:
        return version("onnx-asr") == "0.12.0" and version("onnxruntime") == "1.23.2"
    except PackageNotFoundError:
        return False

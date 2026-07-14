from __future__ import annotations
import inspect
from importlib.metadata import PackageNotFoundError, version

from agent.transcription_provider import TranscriptionProvider
from hermes_cli.plugins import PluginContext
from packaging.version import Version

MIN_HERMES = Version("0.18.2")
MAX_HERMES = Version("0.19")


def check_compatibility() -> tuple[bool, str]:
    try:
        installed = Version(version("hermes-agent"))
    except (ImportError, PackageNotFoundError) as exc:
        return False, f"Hermes Agent is unavailable: {exc}"
    if not MIN_HERMES <= installed < MAX_HERMES:
        return False, f"Hermes Agent {installed} is outside the tested range >=0.18.2,<0.19"
    required = {"file_path", "model", "language", "extra"}
    parameters = inspect.signature(TranscriptionProvider.transcribe).parameters
    if not required.issubset(parameters):
        return False, "Hermes TranscriptionProvider.transcribe contract has changed"
    if not callable(getattr(PluginContext, "register_transcription_provider", None)):
        return False, "Hermes does not expose register_transcription_provider"
    return True, f"Hermes Agent {installed} transcription contract is compatible"


def check_requirements() -> bool:
    compatible, _ = check_compatibility()
    if not compatible:
        return False
    try:
        return version("onnx-asr") == "0.11.0" and version("onnxruntime") == "1.23.2"
    except PackageNotFoundError:
        return False

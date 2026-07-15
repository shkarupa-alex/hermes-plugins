from __future__ import annotations
import inspect
from importlib.metadata import PackageNotFoundError, version

from gateway.platforms.base import BasePlatformAdapter
from packaging.version import Version

MIN_HERMES = Version("0.18.2")
MAX_HERMES = Version("0.19")


def _shape(function: object) -> tuple[tuple[str, int, bool], ...]:
    return tuple(
        (parameter.name, int(parameter.kind), parameter.default is not inspect.Parameter.empty)
        for parameter in inspect.signature(function).parameters.values()  # type: ignore[arg-type]
    )


def check_compatibility() -> tuple[bool, str]:
    try:
        installed = Version(version("hermes-agent"))
    except PackageNotFoundError:
        return False, "hermes-agent is not installed"
    if not MIN_HERMES <= installed < MAX_HERMES:
        return False, f"hermes-agent {installed} is outside the tested range >=0.18.2,<0.19"
    expected = {
        "connect": (("self", 1, False), ("is_reconnect", 3, True)),
        "send": (
            ("self", 1, False),
            ("chat_id", 1, False),
            ("content", 1, False),
            ("reply_to", 1, True),
            ("metadata", 1, True),
        ),
        "send_typing": (("self", 1, False), ("chat_id", 1, False), ("metadata", 1, True)),
        "stop_typing": (("self", 1, False), ("chat_id", 1, False)),
        "edit_message": (
            ("self", 1, False),
            ("chat_id", 1, False),
            ("message_id", 1, False),
            ("content", 1, False),
            ("finalize", 3, True),
        ),
        "send_clarify": (
            ("self", 1, False),
            ("chat_id", 1, False),
            ("question", 1, False),
            ("choices", 1, False),
            ("clarify_id", 1, False),
            ("session_key", 1, False),
            ("metadata", 1, True),
        ),
        "send_slash_confirm": (
            ("self", 1, False),
            ("chat_id", 1, False),
            ("title", 1, False),
            ("message", 1, False),
            ("session_key", 1, False),
            ("confirm_id", 1, False),
            ("metadata", 1, True),
        ),
        "send_image_file": (
            ("self", 1, False),
            ("chat_id", 1, False),
            ("image_path", 1, False),
            ("caption", 1, True),
            ("reply_to", 1, True),
            ("metadata", 1, True),
            ("kwargs", 4, False),
        ),
        "send_image": (
            ("self", 1, False),
            ("chat_id", 1, False),
            ("image_url", 1, False),
            ("caption", 1, True),
            ("reply_to", 1, True),
            ("metadata", 1, True),
        ),
        "send_document": (
            ("self", 1, False),
            ("chat_id", 1, False),
            ("file_path", 1, False),
            ("caption", 1, True),
            ("file_name", 1, True),
            ("reply_to", 1, True),
            ("metadata", 1, True),
            ("kwargs", 4, False),
        ),
        "send_voice": (
            ("self", 1, False),
            ("chat_id", 1, False),
            ("audio_path", 1, False),
            ("caption", 1, True),
            ("reply_to", 1, True),
            ("metadata", 1, True),
            ("kwargs", 4, False),
        ),
        "supports_draft_streaming": (
            ("self", 1, False),
            ("chat_type", 1, True),
            ("metadata", 1, True),
        ),
        "prefers_fresh_final_streaming": (
            ("self", 1, False),
            ("content", 1, False),
            ("metadata", 1, True),
        ),
    }
    from hermes_vk_community.adapter import VkCommunityAdapter  # noqa: PLC0415

    for name, expected_shape in expected.items():
        method = getattr(BasePlatformAdapter, name, None)
        if method is None or _shape(method) != expected_shape:
            return False, f"BasePlatformAdapter.{name} signature changed: {_shape(method) if method else 'missing'}"
        adapter_method = getattr(VkCommunityAdapter, name, None)
        if adapter_method is None or _shape(adapter_method) != expected_shape:
            return False, f"VkCommunityAdapter.{name} signature drifted from the pinned Hermes call site"
    exec_shape = (
        ("self", 1, False),
        ("chat_id", 1, False),
        ("command", 1, False),
        ("session_key", 1, False),
        ("description", 1, True),
        ("metadata", 1, True),
    )
    if _shape(VkCommunityAdapter.send_exec_approval) != exec_shape:
        return False, "VkCommunityAdapter.send_exec_approval signature drifted from gateway/run.py"
    return True, f"Hermes {installed} contract is compatible"


def check_requirements() -> bool:
    compatible, _message = check_compatibility()
    return compatible

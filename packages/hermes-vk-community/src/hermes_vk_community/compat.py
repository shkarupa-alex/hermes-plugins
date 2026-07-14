from __future__ import annotations
import inspect
from importlib.metadata import PackageNotFoundError, version

from gateway.platforms.base import BasePlatformAdapter
from packaging.version import Version

MIN_HERMES = Version("0.18.2")
MAX_HERMES = Version("0.19")


def check_compatibility() -> tuple[bool, str]:
    try:
        installed = Version(version("hermes-agent"))
    except PackageNotFoundError:
        return False, "hermes-agent is not installed"
    if not MIN_HERMES <= installed < MAX_HERMES:
        return False, f"hermes-agent {installed} is outside the tested range >=0.18.2,<0.19"
    expected = {"self", "chat_id", "content", "reply_to", "metadata"}
    actual = set(inspect.signature(BasePlatformAdapter.send).parameters)
    if actual != expected:
        return False, f"BasePlatformAdapter.send signature changed: {sorted(actual)}"
    connect = set(inspect.signature(BasePlatformAdapter.connect).parameters)
    if connect != {"self", "is_reconnect"}:
        return False, f"BasePlatformAdapter.connect signature changed: {sorted(connect)}"
    return True, f"Hermes {installed} contract is compatible"


def check_requirements() -> bool:
    compatible, _message = check_compatibility()
    return compatible

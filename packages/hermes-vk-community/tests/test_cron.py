# pyright: reportPrivateUsage=false
from __future__ import annotations
import asyncio
import importlib
from typing import TYPE_CHECKING, Any, cast

import pytest
from gateway.config import HomeChannel, Platform, PlatformConfig
from gateway.platform_registry import PlatformEntry, platform_registry
from gateway.platforms.base import SendResult

from hermes_vk_community import adapter as adapter_module
from hermes_vk_community.adapter import VkCommunityAdapter
from hermes_vk_community.plugin import build_adapter, register, send_standalone

if TYPE_CHECKING:
    from pathlib import Path


class RegistryContext:
    def register_platform(self, **kwargs: Any) -> None:  # noqa: ANN401 - Hermes registration contract
        platform_registry.register(PlatformEntry(**kwargs))

    def register_cli_command(self, **_kwargs: object) -> None:
        pass


def _config(tmp_path: Path) -> PlatformConfig:
    register(RegistryContext())
    return PlatformConfig(
        enabled=True,
        extra={"group_id": 123, "allowed_user_ids": [456], "storage": {"path": str(tmp_path / "state.sqlite3")}},
    )


@pytest.fixture
def delivery_client(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    events: list[str] = []

    class Client:
        def __init__(self, _token: str, **_kwargs: object) -> None:
            events.append("created")

        async def open(self) -> None:
            events.append("opened")

        async def close(self) -> None:
            events.append("closed")

    async def verify_group(_self: VkCommunityAdapter) -> None:
        events.append("verified")

    def forbidden_lock(*_args: object, **_kwargs: object) -> None:
        pytest.fail("a cron sender must not acquire or release the Long Poll receiver lock")

    monkeypatch.setattr(adapter_module, "VkApiClient", Client)

    def profile_token(_name: str) -> str:
        return "profile-token"

    monkeypatch.setattr(adapter_module, "get_secret", profile_token)
    monkeypatch.setattr(VkCommunityAdapter, "_verify_group", verify_group)
    monkeypatch.setattr(VkCommunityAdapter, "_acquire_platform_lock", forbidden_lock)
    monkeypatch.setattr(VkCommunityAdapter, "_release_platform_lock", forbidden_lock)
    return events


@pytest.mark.asyncio
async def test_standalone_cron_reuses_adapter_send_and_closes_resources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    delivery_client: list[str],
) -> None:
    config = _config(tmp_path)
    sent: list[tuple[str, str]] = []

    async def send(self: VkCommunityAdapter, chat_id: str, content: str) -> SendResult:
        assert self._storage is not None
        assert self._poll_task is None
        sent.append((chat_id, content))
        return SendResult(success=True, message_id="42")

    monkeypatch.setattr(VkCommunityAdapter, "send", send)
    result = await send_standalone(config, "456", "**Отчёт**")
    assert result == {"success": True, "message_id": "42", "media_delivered": False}
    assert sent == [("456", "**Отчёт**")]
    assert delivery_client == ["created", "opened", "verified", "closed"]


@pytest.mark.asyncio
@pytest.mark.parametrize("chat_id", ["999", "2000000001", "-456", "vk.com/id456"])
async def test_standalone_rejects_non_allowlisted_target_before_io(
    tmp_path: Path,
    delivery_client: list[str],
    chat_id: str,
) -> None:
    assert "error" in await send_standalone(_config(tmp_path), chat_id, "Отчёт")
    assert delivery_client == []


@pytest.mark.asyncio
async def test_standalone_cancellation_closes_resources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    delivery_client: list[str],
) -> None:
    async def cancelled(_self: VkCommunityAdapter, _chat_id: str, _content: str) -> SendResult:
        raise asyncio.CancelledError

    monkeypatch.setattr(VkCommunityAdapter, "send", cancelled)
    with pytest.raises(asyncio.CancelledError):
        await send_standalone(_config(tmp_path), "456", "Отчёт")
    assert delivery_client[-1] == "closed"


@pytest.mark.asyncio
async def test_standalone_does_not_report_partial_delivery_as_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    delivery_client: list[str],
) -> None:
    async def partial(_self: VkCommunityAdapter, _chat_id: str, _content: str) -> SendResult:
        return SendResult(success=True, message_id="42", raw_response={"partial_delivery": {"failed_segment": "text"}})

    monkeypatch.setattr(VkCommunityAdapter, "send", partial)
    assert "error" in await send_standalone(_config(tmp_path), "456", "Отчёт")
    assert delivery_client[-1] == "closed"


def test_cron_lists_vk_and_resolves_home_and_explicit_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gateway import config as gateway_config

    # Hermes moved delivery helpers into a module but retains scheduler aliases.
    try:
        delivery = cast("Any", importlib.import_module("cron.scheduler_delivery"))
    except ImportError:
        delivery = cast("Any", importlib.import_module("cron.scheduler"))
    config = _config(tmp_path)
    home = HomeChannel(platform=Platform("vk"), chat_id="456", name="VK reports")

    class GatewayConfig:
        def get_connected_platforms(self) -> list[Platform]:
            return [Platform("vk")]

        def get_home_channel(self, platform: Platform) -> HomeChannel | None:
            return home if str(platform.value) == "vk" else None

    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda: None)
    monkeypatch.setattr(gateway_config, "load_gateway_config", GatewayConfig)

    def no_env(*_args: object, **_kwargs: object) -> str:
        return ""

    if hasattr(delivery, "_get_config_home_channel"):
        monkeypatch.setattr(delivery, "_home_env_lookup", no_env)
    else:
        # Old Hermes resolves bare platforms only through the env mirror.
        monkeypatch.setenv("VK_HOME_CHANNEL", "456")
    targets = delivery.cron_delivery_targets()
    assert {"id": "vk", "name": "Vk", "home_target_set": True, "home_env_var": "VK_HOME_CHANNEL"} in targets
    for target in ("vk", "vk:456"):
        resolved = delivery._resolve_delivery_targets({"deliver": target})
        assert len(resolved) == 1
        assert resolved[0]["platform"] == "vk"
        assert resolved[0]["chat_id"] == "456"
        assert resolved[0]["thread_id"] is None
    assert isinstance(build_adapter(config), VkCommunityAdapter)


@pytest.mark.asyncio
async def test_exec_approval_accepts_current_hermes_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = build_adapter(_config(tmp_path))
    captured: list[tuple[str, str, str, str]] = []

    async def keyboard(
        _self: VkCommunityAdapter,
        _chat_id: str,
        _body: str,
        buttons: list[tuple[str, str, str, str]],
        _session: str,
        _metadata: object,
    ) -> SendResult:
        captured.extend(buttons)
        return SendResult(success=True, message_id="42")

    monkeypatch.setattr(VkCommunityAdapter, "_send_keyboard", keyboard)
    assert instance.supports_exec_approval_buttons()
    result = await instance.send_exec_approval(
        "456",
        "command",
        "session",
        None,
        allow_permanent=False,
        allow_session=False,
        smart_denied=True,
    )
    assert result.success
    assert [row[2] for row in captured] == ["once", "deny"]

from __future__ import annotations
from typing import TYPE_CHECKING

from gateway.config import PlatformConfig
from gateway.platform_registry import PlatformEntry, platform_registry

from hermes_vk_community.adapter import VkCommunityAdapter
from hermes_vk_community.plugin import build_adapter, is_connected, register

if TYPE_CHECKING:
    import pytest


class ContextRecorder:
    def __init__(self) -> None:
        self.platform: dict[str, object] = {}
        self.command: dict[str, object] = {}

    def register_platform(self, **kwargs: object) -> None:
        self.platform = kwargs

    def register_cli_command(self, **kwargs: object) -> None:
        self.command = kwargs


def test_register_exposes_pinned_hermes_contract() -> None:
    context = ContextRecorder()
    register(context)
    assert context.platform["name"] == "vk"
    assert context.platform["required_env"] == ["VK_COMMUNITY_TOKEN"]
    assert context.platform["max_message_length"] == 4096
    assert context.platform["allow_update_command"] is False
    assert callable(context.platform["apply_yaml_config_fn"])
    assert callable(context.platform["setup_fn"])
    assert callable(context.platform["is_connected"])
    assert context.command["name"] == "vk"


def test_adapter_factory_builds_real_hermes_subclass() -> None:
    context = ContextRecorder()
    register(context)
    platform_registry.register(
        PlatformEntry(
            name="vk",
            label="VK Community",
            adapter_factory=build_adapter,
            check_fn=lambda: True,
        )
    )
    config = PlatformConfig(
        enabled=True,
        extra={
            "group_id": 123,
            "allowed_user_ids": [456],
            "allow_from": ["456"],
            "_vk_validation_errors": [],
        },
    )
    adapter = build_adapter(config)
    assert isinstance(adapter, VkCommunityAdapter)
    assert adapter.enforces_own_access_policy is True
    assert adapter.splits_long_messages is True
    assert adapter.allow_from == ["456"]


def test_connected_status_requires_profile_token(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_token(_name: str) -> None:
        return None

    def present_token(_name: str) -> str:
        return "token"

    monkeypatch.setattr("hermes_vk_community.plugin.get_env_value", missing_token)
    assert is_connected(PlatformConfig(enabled=True)) is False
    monkeypatch.setattr("hermes_vk_community.plugin.get_env_value", present_token)
    assert is_connected(PlatformConfig(enabled=True)) is True

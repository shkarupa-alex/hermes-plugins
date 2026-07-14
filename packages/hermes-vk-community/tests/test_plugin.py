from __future__ import annotations

from gateway.config import PlatformConfig
from gateway.platform_registry import PlatformEntry, platform_registry

from hermes_vk_community.adapter import VkCommunityAdapter
from hermes_vk_community.plugin import build_adapter, register


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

from __future__ import annotations
from typing import TYPE_CHECKING, Protocol

from hermes_vk_community.adapter import VkCommunityAdapter
from hermes_vk_community.cli import handle_command, setup_parser
from hermes_vk_community.compat import check_requirements
from hermes_vk_community.config import apply_yaml_config, validate_config

if TYPE_CHECKING:
    from gateway.config import PlatformConfig


class PluginContext(Protocol):
    def register_platform(self, **kwargs: object) -> None: ...

    def register_cli_command(self, **kwargs: object) -> None: ...


def build_adapter(config: PlatformConfig) -> VkCommunityAdapter:
    return VkCommunityAdapter(config)


def register(ctx: PluginContext) -> None:
    ctx.register_platform(
        name="vk",
        label="VK Community",
        adapter_factory=build_adapter,
        check_fn=check_requirements,
        validate_config=validate_config,
        apply_yaml_config_fn=apply_yaml_config,
        required_env=["VK_COMMUNITY_TOKEN"],
        max_message_length=4096,
        allow_update_command=False,
        pii_safe=True,
        emoji="💬",
        platform_hint=(
            "You are chatting via a VK Community bot. Write normal Markdown; the adapter safely renders it "
            "for VK. Long responses are split automatically."
        ),
    )
    ctx.register_cli_command(
        name="vk",
        help="VK Community plugin diagnostics",
        description="Configure, diagnose, and probe the VK Community adapter",
        setup_fn=setup_parser,
        handler_fn=handle_command,
    )

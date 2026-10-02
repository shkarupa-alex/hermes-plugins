from __future__ import annotations
from typing import TYPE_CHECKING, Protocol

from hermes_cli.config import get_env_value

from hermes_vk_community.adapter import VkCommunityAdapter
from hermes_vk_community.cli import handle_command, setup_parser
from hermes_vk_community.compat import check_requirements, supports_cron_delivery
from hermes_vk_community.config import apply_yaml_config, validate_config
from hermes_vk_community.setup import interactive_setup

if TYPE_CHECKING:
    from gateway.config import PlatformConfig


class PluginContext(Protocol):
    def register_platform(self, **kwargs: object) -> None: ...

    def register_cli_command(self, **kwargs: object) -> None: ...


def build_adapter(config: PlatformConfig) -> VkCommunityAdapter:
    return VkCommunityAdapter(config)


def is_connected(_config: PlatformConfig) -> bool:
    """Report profile-scoped credential presence to Hermes setup and gateway discovery."""
    return bool((get_env_value("VK_COMMUNITY_TOKEN") or "").strip())


def register(ctx: PluginContext) -> None:
    cron: dict[str, object] = {}
    if supports_cron_delivery():
        cron = {"cron_deliver_env_var": "VK_HOME_CHANNEL", "standalone_sender_fn": send_standalone}
    ctx.register_platform(
        name="vk",
        label="VK Community",
        adapter_factory=build_adapter,
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        apply_yaml_config_fn=apply_yaml_config,
        **cron,
        required_env=["VK_COMMUNITY_TOKEN"],
        setup_fn=interactive_setup,
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


async def send_standalone(  # noqa: PLR0913 - exact Hermes standalone sender contract
    config: PlatformConfig,
    chat_id: str,
    message: str,
    *,
    thread_id: str | None = None,
    media_files: list[tuple[str, bool]] | None = None,
    force_document: bool = False,
) -> dict[str, object]:
    """Deliver a cron report without starting a second Long Poll receiver."""
    if not supports_cron_delivery():
        return {"error": "VK cron delivery requires a current Hermes Git host with safe partial-delivery routing"}
    if thread_id:
        return {"error": "VK private messages do not support threads"}
    if not config.enabled or not validate_config(config):
        return {"error": "VK configuration is disabled or invalid"}
    adapter = build_adapter(config)
    return await adapter.send_once(chat_id, message, media_files=media_files, force_document=force_document)

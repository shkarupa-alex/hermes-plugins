from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hermes_vk_community.plugin import PluginContext

__all__ = ["register"]


def register(ctx: PluginContext) -> None:
    # Hermes scans general entry points while its config module is still
    # initializing. Import CLI/setup and runtime dependencies only when the
    # PluginManager actually registers this plugin.
    from hermes_vk_community.plugin import register as register_plugin  # noqa: PLC0415

    register_plugin(ctx)

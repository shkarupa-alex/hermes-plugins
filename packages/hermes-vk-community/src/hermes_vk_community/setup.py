from __future__ import annotations
import asyncio
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from hermes_cli.config import get_env_value, save_env_value, write_platform_config_field
from hermes_cli.setup import print_error, print_header, print_info, print_success, prompt, prompt_yes_no
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from hermes_vk_community.client import VkApiClient
from hermes_vk_community.config import API_VERSION
from hermes_vk_community.models import CommunityLongPollSettings, Group, GroupsResponse, User

if TYPE_CHECKING:
    from collections.abc import Callable


VK_BOTS_GUIDE = "https://dev.vk.com/ru/api/bots/getting-started"
VK_COMMUNITY_URL = "https://vk.com/groups"


class ResolvedScreenName(BaseModel):
    model_config = ConfigDict(extra="ignore")

    object_id: int = Field(gt=0)
    type: str


class VkSetupResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    group_id: int = Field(gt=0)
    group_name: str
    allowed_user_ids: list[int] = Field(min_length=1)


def screen_name_from_reference(value: str) -> str:
    candidate = value.strip()
    if not candidate:
        raise ValueError("VK link or ID is empty")
    if "://" in candidate:
        parsed = urlparse(candidate)
        hostname = (parsed.hostname or "").lower().rstrip(".")
        if hostname not in {"vk.com", "www.vk.com", "m.vk.com"}:
            raise ValueError("expected a vk.com link")
        candidate = parsed.path.strip("/").split("/", 1)[0]
    candidate = candidate.strip().lstrip("@").split("?", 1)[0]
    if not candidate:
        raise ValueError("VK link has no screen name or numeric ID")
    return candidate


def numeric_id_from_reference(value: str, *, prefixes: tuple[str, ...]) -> int | None:
    candidate = screen_name_from_reference(value)
    if candidate.isdecimal():
        return int(candidate)
    lowered = candidate.lower()
    for prefix in prefixes:
        suffix = lowered.removeprefix(prefix)
        if suffix != lowered and suffix.isdecimal():
            return int(suffix)
    return None


async def _resolve_object(client: VkApiClient, value: str, *, expected_type: str) -> int:
    prefixes = ("id",) if expected_type == "user" else ("club", "public", "event")
    numeric = numeric_id_from_reference(value, prefixes=prefixes)
    if numeric is not None:
        return numeric
    payload = await client.call("utils.resolveScreenName", {"screen_name": screen_name_from_reference(value)})
    resolved = ResolvedScreenName.model_validate(payload)
    if resolved.type != expected_type:
        raise ValueError(f"VK reference resolves to {resolved.type!r}, expected {expected_type!r}")
    return resolved.object_id


async def inspect_vk_setup(token: str, group_ref: str, user_refs: list[str]) -> VkSetupResult:
    """Validate credentials, resolve human-friendly VK links, and test Long Poll."""
    clean_token = token.strip()
    if not clean_token:
        raise ValueError("VK community token is empty")
    if not user_refs:
        raise ValueError("at least one allowed VK user is required")

    client = VkApiClient(clean_token)
    try:
        group_id = await _resolve_object(client, group_ref, expected_type="group")
        groups_payload = await client.call("groups.getById", {"group_id": group_id})
        if isinstance(groups_payload, dict):
            groups = GroupsResponse.model_validate(groups_payload).groups
        else:
            groups = TypeAdapter(list[Group]).validate_python(groups_payload)
        if not groups or groups[0].id != group_id:
            raise ValueError("VK token cannot access the selected community")

        allowed_user_ids: list[int] = []
        for reference in user_refs:
            user_id = await _resolve_object(client, reference, expected_type="user")
            if user_id not in allowed_user_ids:
                allowed_user_ids.append(user_id)
        users_payload = await client.call("users.get", {"user_ids": allowed_user_ids})
        users = TypeAdapter(list[User]).validate_python(users_payload)
        visible_user_ids = {user.id for user in users}
        if any(user_id not in visible_user_ids for user_id in allowed_user_ids):
            raise ValueError("one or more allowed VK users could not be resolved")

        long_poll_payload = await client.call("groups.getLongPollSettings", {"group_id": group_id})
        validate_long_poll_capabilities(CommunityLongPollSettings.model_validate(long_poll_payload))
        await client.get_long_poll_lease(group_id)
        return VkSetupResult(
            group_id=group_id,
            group_name=groups[0].name,
            allowed_user_ids=allowed_user_ids,
        )
    finally:
        await client.close()


def validate_long_poll_capabilities(settings: CommunityLongPollSettings) -> None:
    """Reject a superficially enabled Long Poll setup that cannot deliver messages."""
    if not settings.is_enabled:
        raise ValueError("Community Long Poll is disabled")
    if settings.api_version != API_VERSION:
        raise ValueError(
            f"Community Long Poll API version must be {API_VERSION}, got {settings.api_version or 'unset'}",
        )
    if settings.events.message_new != 1:
        raise ValueError("Community Long Poll event 'message_new' (incoming messages) is disabled")


def interactive_setup() -> None:
    """Guide the user through a profile-scoped VK Community setup."""
    print_header("VK Community")
    print_info("Hermes connects through a VK community bot using Community Long Poll.")
    print_info(f"VK bot guide: {VK_BOTS_GUIDE}")
    print_info("")
    print_info("1. Create a community (a private community is fine) and enable Messages.")
    print_info("2. Settings → API usage → Long Poll API: enable it, select API 5.199,")
    print_info("   then enable the incoming-message event (message_new).")
    print_info("3. Settings → API usage → Access tokens: create a community token.")
    print_info("   Select: community management, community messages, photos, documents.")
    print_info("   Stories, wall, and market permissions are not required.")

    existing_token = get_env_value("VK_COMMUNITY_TOKEN") or ""
    if existing_token:
        print_success("A VK community token is already stored in the active Hermes profile.")
        if not prompt_yes_no("Reconfigure VK Community?", default=False):
            return

    token = prompt("VK community access token", password=True) or existing_token
    if not token:
        print_error("VK setup cancelled: token is required.")
        return

    print_info("")
    print_info("Community ID: copy a link such as https://vk.com/club123456789.")
    print_info("A custom vk.com/community-name link also works; the wizard resolves it.")
    group_ref = prompt("VK community link or numeric ID")
    print_info("")
    print_info("Allowed users: paste profile links, screen names, or numeric IDs.")
    print_info("Example: https://vk.com/id123456 or https://vk.com/username")
    raw_users = prompt("Allowed VK users (comma-separated)")
    user_refs = [item.strip() for item in raw_users.split(",") if item.strip()]

    print_info("Validating the token, IDs, permissions, and Community Long Poll...")
    try:
        result = asyncio.run(inspect_vk_setup(token, group_ref, user_refs))
    except Exception as exc:  # noqa: BLE001 - setup boundary must return a friendly diagnostic
        print_error(f"VK setup validation failed: {exc}")
        print_info("Nothing was saved. Check the token permissions and Long Poll settings, then retry.")
        return

    save_env_value("VK_COMMUNITY_TOKEN", token)
    write_setup_config(write_platform_config_field, result)
    print_success(f"VK Community configured: {result.group_name} (ID {result.group_id})")
    print_success("Only these VK user IDs are allowed: " + ", ".join(map(str, result.allowed_user_ids)))
    print_info("The token was stored as VK_COMMUNITY_TOKEN in the active Hermes profile's .env.")
    print_info("Non-secret IDs were stored under platforms.vk in config.yaml.")


def write_setup_config(
    writer: Callable[..., None],
    result: VkSetupResult,
) -> None:
    writer("vk", "enabled", True, raw=True)  # noqa: FBT003 - Hermes config writer accepts arbitrary values
    writer("vk", "group_id", result.group_id, raw=True)
    writer("vk", "allowed_user_ids", result.allowed_user_ids, raw=True)
    writer("vk", "typing_indicator", True, raw=True)  # noqa: FBT003 - Hermes config writer accepts arbitrary values

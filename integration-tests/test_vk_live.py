from __future__ import annotations
import base64
import os
import secrets
from typing import TYPE_CHECKING

import pytest
from gateway.config import PlatformConfig
from gateway.platform_registry import PlatformEntry, platform_registry
from pydantic import TypeAdapter

from hermes_vk_community.client import VkApiClient
from hermes_vk_community.plugin import build_adapter
from hermes_vk_community.storage import VkStorage

if TYPE_CHECKING:
    from pathlib import Path

    from hermes_vk_community.adapter import VkCommunityAdapter

PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def _credentials() -> tuple[str, int, int]:
    token = os.environ.get("VK_COMMUNITY_TOKEN", "").strip()
    group_id = os.environ.get("VK_GROUP_ID", "").strip()
    peer_id = os.environ.get("VK_TEST_PEER_ID", "").strip()
    if not token or not group_id or not peer_id:
        pytest.skip("private VK release-test credentials are not configured")
    return token, int(group_id), int(peer_id)


def _adapter(group_id: int, peer_id: int) -> VkCommunityAdapter:
    if not platform_registry.is_registered("vk"):
        platform_registry.register(
            PlatformEntry(name="vk", label="VK Community", adapter_factory=build_adapter, check_fn=lambda: True)
        )
    return build_adapter(
        PlatformConfig(
            enabled=True,
            extra={
                "group_id": group_id,
                "allowed_user_ids": [peer_id],
                "allow_from": [str(peer_id)],
                "_vk_validation_errors": [],
            },
        )
    )


@pytest.mark.vk_live
@pytest.mark.asyncio
async def test_vk_live_text_typing_edit_and_long_poll() -> None:
    token, group_id, peer_id = _credentials()
    client = VkApiClient(token)
    marker = f"Hermes VK release test {secrets.token_hex(6)}"
    message_id: int | None = None
    try:
        lease = await client.get_long_poll_lease(group_id)
        assert lease.key
        assert lease.ts
        assert lease.server.startswith("https://")
        assert (
            await client.call(
                "messages.setActivity",
                {"peer_id": peer_id, "group_id": group_id, "type": "typing"},
            )
            == 1
        )
        response = await client.call(
            "messages.send",
            {"peer_id": peer_id, "random_id": secrets.randbelow(2_147_483_647) + 1, "message": marker},
        )
        if isinstance(response, int):
            message_id = response
        elif isinstance(response, dict):
            candidate = TypeAdapter(dict[str, object]).validate_python(response).get("message_id")
            if not isinstance(candidate, int):
                raise TypeError("messages.send returned no numeric message id")
            message_id = candidate
        else:
            raise TypeError("messages.send returned no numeric message id")
        edited = marker + " edited"
        await client.call("messages.edit", {"peer_id": peer_id, "message_id": message_id, "message": edited})
        readback = await client.call("messages.getById", {"message_ids": message_id})
        assert isinstance(readback, dict)
        assert readback["items"][0]["text"] == edited
    finally:
        if message_id is not None:
            await client.call(
                "messages.delete",
                {"peer_id": peer_id, "message_ids": message_id, "delete_for_all": True},
            )
        await client.close()


@pytest.mark.vk_live
@pytest.mark.asyncio
async def test_vk_live_photo_upload_and_visible_delivery(tmp_path: Path) -> None:
    token, group_id, peer_id = _credentials()
    adapter = _adapter(group_id, peer_id)
    storage = VkStorage(tmp_path / "state.sqlite3")
    await storage.open()
    client = VkApiClient(token)
    adapter._client = client  # pyright: ignore[reportPrivateUsage]
    adapter._storage = storage  # pyright: ignore[reportPrivateUsage]
    image = tmp_path / "release-test.png"
    image.write_bytes(PNG_1X1)
    result = None
    try:
        result = await adapter.send_image_file(peer_id.__str__(), str(image), "Hermes VK photo release test")
        assert result.success
        assert result.message_id
    finally:
        if result is not None and result.message_id:
            await adapter.delete_message(str(peer_id), result.message_id)
        await client.close()
        await storage.close()

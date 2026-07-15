from __future__ import annotations
import json
import os
import secrets
from typing import TYPE_CHECKING, cast

import pytest
from gateway.config import PlatformConfig
from gateway.platform_registry import PlatformEntry, platform_registry
from PIL import Image
from pydantic import TypeAdapter

from hermes_vk_community.client import VkApiClient
from hermes_vk_community.plugin import build_adapter
from hermes_vk_community.storage import VkStorage

if TYPE_CHECKING:
    from pathlib import Path

    from hermes_vk_community.adapter import VkCommunityAdapter


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


def _message_id(response: object) -> int:
    if isinstance(response, int):
        return response
    if isinstance(response, dict):
        candidate = cast("dict[str, object]", response).get("message_id")
        if isinstance(candidate, int):
            return candidate
    raise TypeError("messages.send returned no numeric message id")


def _mapping(value: object) -> dict[str, object] | None:
    return cast("dict[str, object]", value) if isinstance(value, dict) else None


def _dimension(value: object) -> int:
    return value if isinstance(value, int) else 0


def _latest_incoming_message_id(history: object, peer_id: int) -> int:
    if not isinstance(history, dict):
        raise TypeError("messages.getHistory returned no object")
    items = cast("dict[str, object]", history).get("items")
    if not isinstance(items, list):
        raise TypeError("messages.getHistory returned no items")
    for raw_message in cast("list[object]", items):
        if not isinstance(raw_message, dict):
            continue
        message = cast("dict[str, object]", raw_message)
        message_id = message.get("id")
        if message.get("from_id") == peer_id and isinstance(message_id, int):
            return message_id
    raise AssertionError("no recent incoming VK message was found for the test peer")


def _photo_url(message: dict[str, object]) -> str:
    attachments = message.get("attachments")
    if not isinstance(attachments, list):
        raise TypeError("VK message contains no attachment list")
    for raw_attachment in cast("list[object]", attachments):
        attachment = _mapping(raw_attachment)
        if attachment is None or attachment.get("type") != "photo":
            continue
        photo = _mapping(attachment.get("photo"))
        if photo is None:
            continue
        sizes = photo.get("sizes")
        if not isinstance(sizes, list):
            continue
        candidates = [
            size
            for raw_size in cast("list[object]", sizes)
            if (size := _mapping(raw_size)) is not None and isinstance(size.get("url"), str)
        ]
        if candidates:
            largest = max(
                candidates,
                key=lambda size: _dimension(size.get("width")) * _dimension(size.get("height")),
            )
            return cast("str", largest["url"])
    raise AssertionError("VK message contains no downloadable photo")


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
        message_id = _message_id(response)
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
async def test_vk_live_dm_reply_formatting_and_buttons() -> None:
    token, _group_id, peer_id = _credentials()
    client = VkApiClient(token)
    sent_id: int | None = None
    try:
        incoming_id = _latest_incoming_message_id(
            await client.call("messages.getHistory", {"peer_id": peer_id, "count": 100}),
            peer_id,
        )
        message = "😀 Жирный live test"
        format_data = {
            "version": 1,
            "items": [{"type": "bold", "offset": 2, "length": len("Жирный")}],
        }
        keyboard = {
            "inline": True,
            "buttons": [
                [
                    {
                        "action": {"type": "text", "label": "Live OK", "payload": '{"live":true}'},
                        "color": "secondary",
                    }
                ]
            ],
        }
        sent_id = _message_id(
            await client.call(
                "messages.send",
                {
                    "peer_id": peer_id,
                    "random_id": secrets.randbelow(2_147_483_647) + 1,
                    "message": message,
                    "reply_to": incoming_id,
                    "format_data": json.dumps(format_data, ensure_ascii=False, separators=(",", ":")),
                    "keyboard": json.dumps(keyboard, ensure_ascii=False, separators=(",", ":")),
                    "disable_mentions": True,
                    "dont_parse_links": True,
                },
            )
        )
        readback = TypeAdapter(dict[str, object]).validate_python(
            await client.call("messages.getById", {"message_ids": sent_id})
        )
        items = TypeAdapter(list[dict[str, object]]).validate_python(readback.get("items"))
        assert items[0]["text"] == message
        assert items[0].get("format_data")
        assert items[0].get("keyboard")
        reply = TypeAdapter(dict[str, object]).validate_python(items[0].get("reply_message"))
        assert reply["id"] == incoming_id
    finally:
        if sent_id is not None:
            await client.call(
                "messages.delete",
                {"peer_id": peer_id, "message_ids": sent_id, "delete_for_all": True},
            )
        await client.close()


@pytest.mark.vk_live
@pytest.mark.asyncio
async def test_vk_live_prepared_outbox_recovery_after_storage_restart(tmp_path: Path) -> None:
    token, group_id, peer_id = _credentials()
    path = tmp_path / "restart.sqlite3"
    initial = VkStorage(path)
    await initial.open()
    marker = f"Hermes VK restart test {secrets.token_hex(6)}"
    await initial.prepare_outbox(peer_id, [marker], None)
    await initial.close()

    storage = VkStorage(path)
    await storage.open()
    client = VkApiClient(token)
    adapter = _adapter(group_id, peer_id)
    adapter._client = client  # pyright: ignore[reportPrivateUsage]
    adapter._storage = storage  # pyright: ignore[reportPrivateUsage]
    sent_id: int | None = None
    try:
        await adapter._recover_prepared_outbox()  # pyright: ignore[reportPrivateUsage]
        assert await storage.prepared_outbox() == []
        db = storage._connection()  # pyright: ignore[reportPrivateUsage]
        async with db.execute("SELECT state,returned_message_id FROM outbox") as cursor:
            row = await cursor.fetchone()
        assert row is not None
        assert row[0] == "sent"
        sent_id = int(row[1])
    finally:
        if sent_id is None:
            history = TypeAdapter(dict[str, object]).validate_python(
                await client.call("messages.getHistory", {"peer_id": peer_id, "count": 20})
            )
            items = TypeAdapter(list[dict[str, object]]).validate_python(history.get("items"))
            matching = [item["id"] for item in items if item.get("text") == marker and isinstance(item.get("id"), int)]
            sent_id = cast("int", matching[0]) if matching else None
        if sent_id is not None:
            await client.call(
                "messages.delete",
                {"peer_id": peer_id, "message_ids": sent_id, "delete_for_all": True},
            )
        await client.close()
        await storage.close()


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
    Image.new("RGB", (64, 64), color=(51, 102, 153)).save(image)
    result = None
    downloaded = None
    try:
        result = await adapter.send_image_file(peer_id.__str__(), str(image), "Hermes VK photo release test")
        assert result.success
        assert result.message_id
        readback = TypeAdapter(dict[str, object]).validate_python(
            await client.call("messages.getById", {"message_ids": int(result.message_id)})
        )
        items = TypeAdapter(list[dict[str, object]]).validate_python(readback.get("items"))
        downloaded = await client.download_media(_photo_url(items[0]))
        assert downloaded.path.is_file()
        assert downloaded.path.stat().st_size > 0
    finally:
        if downloaded is not None:
            downloaded.cleanup()
        if result is not None and result.message_id:
            await adapter.delete_message(str(peer_id), result.message_id)
        await client.close()
        await storage.close()

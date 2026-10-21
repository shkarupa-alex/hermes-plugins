# pyright: reportPrivateUsage=false
from __future__ import annotations
import json
from typing import TYPE_CHECKING, cast
from unittest.mock import Mock

import pytest
from gateway.config import PlatformConfig
from gateway.platform_registry import PlatformEntry, platform_registry

from hermes_vk_community.plugin import build_adapter
from hermes_vk_community.storage import InboxRecord

if TYPE_CHECKING:
    from gateway.platforms.base import MessageEvent

    from hermes_vk_community.client import VkApiClient
    from hermes_vk_community.storage import InboxState, VkStorage


class StorageSpy:
    def __init__(self) -> None:
        self.marks: list[tuple[int, str, str | None]] = []

    async def mark_inbox(self, row_id: int, state: InboxState, error: str | None = None) -> None:
        self.marks.append((row_id, state, error))


@pytest.mark.asyncio
@pytest.mark.parametrize("with_context", [False, True], ids=["plain-text", "reply-and-forward"])
async def test_vk_message_survives_native_admission(monkeypatch: pytest.MonkeyPatch, *, with_context: bool) -> None:
    if not platform_registry.is_registered("vk"):
        platform_registry.register(
            PlatformEntry(name="vk", label="VK Community", adapter_factory=build_adapter, check_fn=lambda: True)
        )
    adapter = build_adapter(
        PlatformConfig(
            enabled=True,
            extra={"group_id": 123, "allowed_user_ids": [456], "allow_from": ["456"], "_vk_validation_errors": []},
        )
    )
    storage = StorageSpy()
    adapter._storage = cast("VkStorage", storage)
    adapter._client = cast("VkApiClient", object())
    received: list[MessageEvent] = []

    async def receive(event: MessageEvent) -> None:
        received.append(event)

    monkeypatch.setattr(adapter, "handle_message", receive)
    message: dict[str, object] = {"id": 10, "date": 1, "peer_id": 456, "from_id": 456, "text": "hello"}
    if with_context:
        message.update(
            reply_message={"id": 9, "from_id": 456, "text": "original question"},
            fwd_messages=[{"id": 8, "from_id": 789, "text": "forwarded context"}],
        )
    record = InboxRecord(
        id=1,
        normalized_json=json.dumps(
            {"type": "message_new", "group_id": 123, "event_id": "evt-1", "object": {"message": message}}
        ),
    )
    await adapter._dispatch_record(record)
    assert storage.marks == [(1, "dispatched", None)]
    assert len(received) == 1
    event = received[0]
    # VK context is already projected into text; duplicating it in free-form
    # metadata makes current Hermes reject the entire message before admission.
    assert event.metadata == {}
    if with_context:
        assert "original question" in event.text
        assert "forwarded context" in event.text
        assert event.reply_to_message_id == "9"
    else:
        assert event.text == "hello"

    # Legacy Hermes releases have no durable native codec. They still exercise
    # the adapter assertions above; current hosts also prove the actual roundtrip.
    codec = pytest.importorskip("gateway.session_envelope")
    runner = Mock()
    runner.config.multiplex_profiles = False
    runner._adapter_flag.return_value = False
    runner._adapter_profile_for_source.return_value = None
    runner._is_user_authorized_for_source.return_value = True
    runner._adapter_for_source.return_value = adapter
    runner.session_store._generate_session_key.return_value = "test-vk-route"
    replay = cast("MessageEvent", codec.restore_native(codec._snapshot_native(runner, event, provenance=None)))
    assert replay.text == event.text
    assert replay.message_id == event.message_id
    assert replay.reply_to_message_id == event.reply_to_message_id
    assert replay.source.to_dict() == event.source.to_dict()

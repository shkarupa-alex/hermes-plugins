# pyright: reportPrivateUsage=false
from __future__ import annotations
from typing import TYPE_CHECKING, cast

import pytest
from gateway.config import PlatformConfig
from gateway.platform_registry import PlatformEntry, platform_registry

from hermes_vk_community.adapter import (
    VkCommunityAdapter,
    _attachment_candidate,
    _is_retryable_poll_error,
    _is_retryable_send_error,
)
from hermes_vk_community.errors import VkApiError, VkDeliveryUnknownError
from hermes_vk_community.models import VkAttachment
from hermes_vk_community.plugin import build_adapter
from hermes_vk_community.storage import InboxRecord, VkStorage

if TYPE_CHECKING:
    from hermes_vk_community.client import VkApiClient
    from hermes_vk_community.storage import InboxState


class StorageSpy:
    def __init__(self) -> None:
        self.marks: list[tuple[int, str, str | None]] = []

    async def mark_inbox(self, row_id: int, state: InboxState, error: str | None = None) -> None:
        self.marks.append((row_id, state, error))


class ClientSpy:
    def __init__(self) -> None:
        self.download_calls = 0

    async def download_media(self, url: str) -> None:
        del url
        self.download_calls += 1
        raise AssertionError("unauthorized attachment reached media I/O")


def _adapter() -> VkCommunityAdapter:
    if not platform_registry.is_registered("vk"):
        platform_registry.register(
            PlatformEntry(
                name="vk",
                label="VK Community",
                adapter_factory=build_adapter,
                check_fn=lambda: True,
            )
        )
    return build_adapter(
        PlatformConfig(
            enabled=True,
            extra={
                "group_id": 123,
                "allowed_user_ids": [456],
                "allow_from": ["456"],
                "_vk_validation_errors": [],
            },
        )
    )


@pytest.mark.asyncio
async def test_unauthorized_sender_is_rejected_before_media_io() -> None:
    adapter = _adapter()
    storage = StorageSpy()
    client = ClientSpy()
    adapter._storage = cast("VkStorage", storage)
    adapter._client = cast("VkApiClient", client)
    record = InboxRecord(
        id=1,
        normalized_json="""{
          "type":"message_new","group_id":123,"event_id":"evt-1",
          "object":{"message":{"id":10,"date":1,"peer_id":999,"from_id":999,"text":"",
          "attachments":[{"type":"audio_message","audio_message":{"link_ogg":"https://cdn.userapi.com/a.ogg"}}]}}
        }""",
    )
    await adapter._dispatch_record(record)
    assert client.download_calls == 0
    assert storage.marks == [(1, "quarantined", "sender is not authorized")]


def test_audio_message_prefers_ogg_and_marks_voice() -> None:
    attachment = VkAttachment.model_validate(
        {
            "type": "audio_message",
            "audio_message": {
                "link_ogg": "https://cdn.userapi.com/a.ogg",
                "link_mp3": "https://cdn.userapi.com/a.mp3",
            },
        }
    )
    assert _attachment_candidate(attachment) == (
        "https://cdn.userapi.com/a.ogg",
        "voice.ogg",
        "audio",
        True,
    )


def test_retries_only_definitely_rejected_send_attempts() -> None:
    assert _is_retryable_send_error(VkApiError(6, "too many requests"))
    assert _is_retryable_send_error(VkApiError(10, "internal error"))
    assert not _is_retryable_send_error(VkApiError(914, "message too long"))
    assert not _is_retryable_send_error(VkDeliveryUnknownError("timed out"))


def test_poll_retries_transport_errors_but_not_protocol_errors() -> None:
    assert _is_retryable_poll_error(TimeoutError())
    assert _is_retryable_poll_error(OSError())
    assert not _is_retryable_poll_error(ValueError("invalid lease host"))

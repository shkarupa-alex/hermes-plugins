# pyright: reportPrivateUsage=false
from __future__ import annotations
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
from gateway.config import PlatformConfig
from gateway.platform_registry import PlatformEntry, platform_registry

from hermes_vk_community.adapter import (
    VkCommunityAdapter,
    _attachment_candidate,
    _Interaction,
    _is_retryable_poll_error,
    _is_retryable_send_error,
)
from hermes_vk_community.errors import VkApiError, VkDeliveryUnknownError
from hermes_vk_community.models import InteractionPayload, VkAttachment, VkMessage
from hermes_vk_community.plugin import build_adapter
from hermes_vk_community.storage import InboxRecord, VkStorage

if TYPE_CHECKING:
    from pathlib import Path

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


@pytest.mark.asyncio
async def test_failed_platform_lock_does_not_poison_reconnect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _adapter()

    def secret(_name: str) -> str:
        return "token"

    monkeypatch.setattr("hermes_vk_community.adapter.get_secret", secret)

    def reject_lock(*_args: object) -> bool:
        cast("Any", adapter)._platform_lock_identity = "123"
        return False

    monkeypatch.setattr(adapter, "_acquire_platform_lock", reject_lock)
    assert not await adapter.connect()
    assert adapter._platform_lock_identity is None


@pytest.mark.asyncio
async def test_keyboard_payload_is_bound_and_consumed_once(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _adapter()
    resolved: list[tuple[str, str]] = []

    def resolve(clarify_id: str, value: str) -> None:
        resolved.append((clarify_id, value))

    monkeypatch.setattr(
        "hermes_vk_community.adapter.resolve_gateway_clarify",
        resolve,
    )
    nonce = "n" * 24
    adapter._interactions[nonce] = _Interaction(
        group="group",
        peer_id=456,
        user_id=456,
        session_key="session",
        kind="clarify",
        value="вариант",
        target_id="clarify-id",
        expires_at=10**12,
    )
    payload = InteractionPayload.model_validate({"v": 1, "n": nonce}).model_dump_json(by_alias=True)
    message = VkMessage(id=1, date=1, peer_id=456, from_id=456, payload=payload)
    assert await adapter._consume_interaction(message)
    assert not await adapter._consume_interaction(message)
    assert resolved == [("clarify-id", "вариант")]


@pytest.mark.asyncio
async def test_mixed_attachments_expose_only_voice_to_stt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    adapter = _adapter()

    class DownloadClient:
        async def download_media(self, url: str) -> SimpleNamespace:
            return SimpleNamespace(data=url.encode(), content_type="application/octet-stream")

    cached_index = 0

    def fake_cache(*_args: object, **_kwargs: object) -> SimpleNamespace:
        nonlocal cached_index
        cached_index += 1
        return SimpleNamespace(
            path=str(tmp_path / f"media-{cached_index}"),
            media_type="audio/ogg" if cached_index == 1 else "image/jpeg",
            context_note=lambda: "[cached]",
        )

    adapter._client = cast("VkApiClient", DownloadClient())
    monkeypatch.setattr("hermes_vk_community.adapter.cache_media_bytes", fake_cache)
    attachments = [
        VkAttachment.model_validate(
            {"type": "audio_message", "audio_message": {"link_ogg": "https://cdn.userapi.com/a.ogg"}}
        ),
        VkAttachment.model_validate({"type": "photo", "photo": {"sizes": [{"url": "https://cdn.userapi.com/a.jpg"}]}}),
    ]
    paths, media_types, is_voice, _text = await adapter._cache_attachments(attachments)
    assert is_voice
    assert paths == [str(tmp_path / "media-1")]
    assert media_types == ["audio/ogg"]


@pytest.mark.asyncio
async def test_error_914_progressively_reduces_and_caches_limit(tmp_path: Path) -> None:
    adapter = _adapter()
    storage = VkStorage(tmp_path / "state.sqlite3")
    await storage.open()

    class LengthClient:
        def __init__(self) -> None:
            self.lengths: list[int] = []

        async def call(self, method: str, params: dict[str, object]) -> int:
            assert method == "messages.send"
            message = str(params["message"])
            self.lengths.append(len(message))
            if len(self.lengths) == 1:
                raise VkApiError(914, "message is too long")
            return len(self.lengths)

    client = LengthClient()
    adapter._storage = storage
    adapter._client = cast("VkApiClient", client)
    try:
        result = await adapter.send("456", "x" * 1000)
        assert result.success
        assert adapter._effective_limit == 500
        assert client.lengths == [1000, 500, 500]
    finally:
        await storage.close()

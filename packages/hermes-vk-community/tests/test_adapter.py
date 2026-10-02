# pyright: reportPrivateUsage=false
from __future__ import annotations
import asyncio
import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
from gateway.config import PlatformConfig
from gateway.platform_registry import PlatformEntry, platform_registry
from gateway.platforms.base import SendResult

from hermes_vk_community.adapter import (
    VkCommunityAdapter,
    _attachment_candidate,
    _attachment_description,
    _geo_context,
    _Interaction,
    _is_retryable_poll_error,
    _is_retryable_send_error,
    _source_prefix_for_rendered,
)
from hermes_vk_community.errors import VkApiError, VkDeliveryUnknownError, VkHttpError, VkLongPollProtocolError
from hermes_vk_community.models import InteractionPayload, LongPollLease, LongPollResponse, VkAttachment, VkMessage
from hermes_vk_community.plugin import build_adapter
from hermes_vk_community.renderer import RenderedTableSegment, RenderedTextSegment, RichVkRenderer
from hermes_vk_community.storage import InboxRecord, VkStorage

if TYPE_CHECKING:
    from pathlib import Path

    from hermes_vk_community.client import VkApiClient
    from hermes_vk_community.storage import InboxState, OutboxRecord


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


def test_non_downloadable_attachments_preserve_useful_context() -> None:
    poll = VkAttachment.model_validate({"type": "poll", "poll": {"question": "Куда идём?"}})
    article = VkAttachment.model_validate(
        {
            "type": "article",
            "article": {"title": "Новости", "url": "https://vk.com/@example-news"},
        }
    )
    assert _attachment_description(poll) == "Опрос: Куда идём?"
    assert _attachment_description(article) == "Статья: Новости — https://vk.com/@example-news"


def test_geo_context_handles_structured_and_unrecognized_coordinates() -> None:
    assert _geo_context({"coordinates": {"latitude": 55.7558, "longitude": 37.6173}}) == (
        "[Геолокация: 55.755800, 37.617300]"
    )
    assert _geo_context({"place": {"title": "Москва"}}) == ("[Геолокация без распознанных координат]")


def test_retries_only_definitely_rejected_send_attempts() -> None:
    assert _is_retryable_send_error(VkApiError(6, "too many requests"))
    assert _is_retryable_send_error(VkApiError(10, "internal error"))
    assert not _is_retryable_send_error(VkApiError(914, "message too long"))
    assert not _is_retryable_send_error(VkDeliveryUnknownError("timed out"))


def test_poll_retries_transport_and_remote_protocol_errors() -> None:
    assert _is_retryable_poll_error(TimeoutError())
    assert _is_retryable_poll_error(OSError())
    assert _is_retryable_poll_error(VkLongPollProtocolError("malformed response"))
    assert not _is_retryable_poll_error(ValueError("invalid lease host"))


@pytest.mark.asyncio
async def test_poll_loop_recovers_from_malformed_remote_response(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _adapter()
    adapter.settings.long_poll.retry_min_seconds = 0.001
    adapter.settings.long_poll.retry_max_seconds = 0.001
    adapter._running = True
    attempts = 0
    fatal_errors: list[str] = []

    async def poll_once() -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise VkLongPollProtocolError("malformed response")
        adapter._running = False

    def record_fatal(error_code: str, *_args: object, **_kwargs: object) -> None:
        fatal_errors.append(error_code)

    monkeypatch.setattr(adapter, "_poll_once", poll_once)
    monkeypatch.setattr(adapter, "_set_fatal_error", record_fatal)

    await adapter._poll_loop()

    assert attempts == 2
    assert fatal_errors == []


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
            path = tmp_path / f"download-{abs(hash(url))}"
            path.write_bytes(url.encode())
            return SimpleNamespace(
                path=path,
                content_type="application/octet-stream",
                cleanup=lambda: path.unlink(missing_ok=True),
            )

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
    paths, media_types, is_voice, text = await adapter._cache_attachments(attachments)
    assert is_voice
    assert paths == [str(tmp_path / "media-1")]
    assert media_types == ["audio/ogg"]
    assert text == "[cached]"


@pytest.mark.asyncio
async def test_voice_attachment_reaches_stt_without_cached_file_note(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    adapter = _adapter()

    class DownloadClient:
        async def download_media(self, _url: str) -> SimpleNamespace:
            path = tmp_path / "voice.ogg"
            path.write_bytes(b"voice")
            return SimpleNamespace(
                path=path,
                content_type="audio/ogg",
                cleanup=lambda: path.unlink(missing_ok=True),
            )

    cached_path = str(tmp_path / "cached-voice.ogg")

    def fake_cache(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            path=cached_path,
            media_type="audio/ogg",
            context_note=lambda: "[audio 'voice.ogg' saved at: /cache/voice.ogg]",
        )

    monkeypatch.setattr(
        "hermes_vk_community.adapter.cache_media_bytes",
        fake_cache,
    )
    adapter._client = cast("VkApiClient", DownloadClient())
    attachments = [
        VkAttachment.model_validate(
            {"type": "audio_message", "audio_message": {"link_ogg": "https://cdn.userapi.com/a.ogg"}}
        )
    ]

    paths, media_types, is_voice, text = await adapter._cache_attachments(attachments)

    assert paths == [cached_path]
    assert media_types == ["audio/ogg"]
    assert is_voice
    assert text == ""


@pytest.mark.asyncio
async def test_chat_send_retries_after_adapter_reconnects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _adapter()
    storage = VkStorage(tmp_path / "state.sqlite3")
    await storage.open()
    calls: list[object] = []

    class Client:
        async def call(self, _method: str, params: dict[str, object]) -> int:
            calls.append(params)
            return 42

    async def reconnect(_delay: float) -> None:
        adapter._client = cast("VkApiClient", Client())

    adapter._storage = storage
    monkeypatch.setattr("hermes_vk_community.adapter.asyncio.sleep", reconnect)
    try:
        result = await adapter._send_with_retry("456", "reply", base_delay=0)
        assert result.success
        assert len(calls) == 1
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_permanent_target_failure_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _adapter()
    original = adapter.send
    calls: list[str] = []

    async def send(chat_id: str, content: str, **_kwargs: object) -> SendResult:
        calls.append(content)
        return await original(chat_id, content)

    monkeypatch.setattr(adapter, "send", send)
    result = await adapter._send_with_retry("999", "reply")
    assert not result.success
    assert result.error_kind == "forbidden"
    assert calls == ["reply"]


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [429, 503])
async def test_transient_http_send_reuses_the_persisted_random_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    from tenacity import wait_none

    adapter = _adapter()
    storage = VkStorage(tmp_path / "state.sqlite3")
    await storage.open()
    random_ids: list[object] = []

    class Client:
        async def call(self, _method: str, params: dict[str, object]) -> int:
            random_ids.append(params["random_id"])
            if len(random_ids) < 3:
                raise VkHttpError(status, "API")
            return 42

    def no_wait(**_kwargs: object) -> wait_none:
        return wait_none()

    monkeypatch.setattr("hermes_vk_community.adapter.wait_random_exponential", no_wait)
    adapter._storage = storage
    adapter._client = cast("VkApiClient", Client())
    try:
        result = await adapter._send_with_retry("456", "reply")
        assert result.success
        assert len(random_ids) == 3
        assert len(set(random_ids)) == 1
        assert len(await storage.diagnostic_rows(outbox_state="sent")) == 1
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("lane", ["send", "recovery"])
@pytest.mark.parametrize("failure", ["before", "after_commit", "cancel_before", "cancel_after_commit"])
async def test_rechunk_failure_terminalizes_all_durable_replacements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lane: str, failure: str
) -> None:
    adapter = _adapter()
    adapter._effective_limit = 512
    storage = VkStorage(tmp_path / "state.sqlite3")
    await storage.open()
    adapter._storage = storage
    adapter._client = cast("Any", object())
    original_rechunk = adapter._rechunk_rejected

    async def broken_rechunk(record: OutboxRecord) -> list[tuple[OutboxRecord, int, int]]:
        if "after_commit" in failure:
            await original_rechunk(record)
        if failure.startswith("cancel"):
            raise asyncio.CancelledError
        raise OSError("rechunk failed")

    async def too_long(_self: VkCommunityAdapter, _params: dict[str, object]) -> object:
        raise VkApiError(914, "too long")

    monkeypatch.setattr(adapter, "_rechunk_rejected", broken_rechunk)
    monkeypatch.setattr(VkCommunityAdapter, "_send_chunk", too_long)
    try:
        if lane == "recovery":
            await storage.prepare_outbox(456, ["x" * 512, "tail 1", "tail 2"], None)
            operation = adapter._recover_prepared_outbox()
        else:
            operation = adapter._send_with_retry("456", "x" * 1100, base_delay=0)
        if failure.startswith("cancel"):
            with pytest.raises(asyncio.CancelledError):
                await operation
        else:
            result = await operation
            if lane == "send":
                assert isinstance(result, SendResult)
                assert not result.success
                assert result.error_kind == "internal"
        await storage.close()
        await storage.open()
        assert await storage.prepared_outbox() == []
        assert (await storage.counts())["outbox_delivery_unknown"] == 0
        assert len(await storage.diagnostic_rows(outbox_state="failed")) == (5 if "after_commit" in failure else 3)
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("lane", ["send", "recovery"])
async def test_lost_rechunk_ownership_preserves_the_other_senders_recoverable_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lane: str
) -> None:
    adapter = _adapter()
    adapter._effective_limit = 512
    storage = VkStorage(tmp_path / "state.sqlite3")
    await storage.open()
    other = VkStorage(storage.path)
    await other.open(recover_inflight=False)
    adapter._storage = storage
    adapter._client = cast("Any", object())
    original_rechunk = adapter._rechunk_rejected
    replacements: list[OutboxRecord] = []
    calls: list[dict[str, object]] = []

    async def another_sender_rechunks_first(record: OutboxRecord) -> list[tuple[OutboxRecord, int, int]]:
        replacements.extend(await other.prepare_outbox(456, ["x" * 256, "x" * 256], None, replace_rejected=record))
        return await original_rechunk(record)

    async def too_long(params: dict[str, object]) -> object:
        calls.append(params)
        raise VkApiError(914, "too long")

    monkeypatch.setattr(adapter, "_rechunk_rejected", another_sender_rechunks_first)
    adapter._send_chunk = too_long
    try:
        if lane == "recovery":
            await storage.prepare_outbox(456, ["x" * 512, "tail"], None)
            await adapter._recover_prepared_outbox()
        else:
            result = await adapter._send_with_retry("456", "x" * 600, base_delay=0)
            assert not result.success
            assert result.raw_response
            assert result.raw_response["delivery_unknown"]
            assert result.raw_response["partial_overflow"]
        assert len(calls) == 1
        prepared = await other.prepared_outbox()
        assert [row.id for row in prepared[:2]] == [row.id for row in replacements]
        assert len(prepared) == 3  # replacements and the unchanged original tail
        await storage.close()
        await storage.open()
        assert [row.id for row in await storage.prepared_outbox()] == [row.id for row in prepared]
    finally:
        await other.close()
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("access", ["allowlist_removed", "pairing_disabled", "pairing_revoked", "pairing_active"])
async def test_recovery_rechecks_access_and_continues_authorized_invocations(tmp_path: Path, access: str) -> None:
    config = PlatformConfig(
        enabled=True,
        extra={
            "group_id": 123,
            "allowed_user_ids": [456, 999] if access == "allowlist_removed" else [456],
            "pairing": {"enabled": True},
        },
    )
    storage = VkStorage(tmp_path / "state.sqlite3")
    await storage.open()
    calls: list[dict[str, object]] = []

    async def transport(params: dict[str, object]) -> object:
        calls.append(params)
        return 42

    try:
        if access.startswith("pairing"):
            await storage.create_pairing_code("OLD", 60)
            assert await storage.consume_pairing_code("OLD", 999)
        admitted = build_adapter(config)
        admitted._storage = storage
        assert await admitted._delivery_target_error("999") is None
        await storage.prepare_outbox(999, ["queued report", "queued tail"], None)
        await storage.prepare_outbox(456, ["still authorized"], None)
        if access == "pairing_revoked":
            async with storage._transaction() as db:
                await db.execute("DELETE FROM paired_users WHERE user_id=999")
        await storage.close()
        await storage.open()
        config.extra["allowed_user_ids"] = [456]
        config.extra["pairing"] = {"enabled": access != "pairing_disabled"}
        restarted = build_adapter(config)
        restarted._storage = storage
        restarted._client = cast("Any", object())
        restarted._send_chunk = transport
        if access != "pairing_active":
            assert not (await restarted.send("999", "ordinary send is forbidden")).success
        await restarted._recover_prepared_outbox()
        assert [call["peer_id"] for call in calls] == ([999, 999, 456] if access == "pairing_active" else [456])
        assert len(await storage.diagnostic_rows(outbox_state="failed")) == (0 if access == "pairing_active" else 2)
        assert await storage.prepared_outbox() == []
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("lane", ["send", "direct", "recovery"])
async def test_cancellation_before_dispatch_is_not_delivery_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lane: str
) -> None:
    adapter = _adapter()
    adapter._effective_limit = 256
    storage = VkStorage(tmp_path / "state.sqlite3")
    await storage.open()
    adapter._storage = storage
    adapter._client = cast("Any", object())
    started = asyncio.Event()
    original_mark = storage.mark_outbox

    async def blocked_mark(row_id: int, state: str, **kwargs: Any) -> None:  # noqa: ANN401 - adapter write contract
        if state == "sending":
            started.set()
            await asyncio.Event().wait()
        await original_mark(row_id, state, **kwargs)

    async def forbidden(*_args: object, **_kwargs: object) -> object:
        pytest.fail("cancelled request reached the transport")

    monkeypatch.setattr(storage, "mark_outbox", blocked_mark)
    monkeypatch.setattr(VkCommunityAdapter, "_send_chunk", forbidden)
    if lane == "send":
        operation = adapter.send("456", "x" * 600)
    elif lane == "direct":
        operation = adapter._send_direct(456, "caption")
    else:
        await storage.prepare_outbox(456, ["head", "tail 1", "tail 2"], None)
        operation = adapter._recover_prepared_outbox()
    task = asyncio.create_task(operation)
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert (await storage.counts())["outbox_delivery_unknown"] == 0
        assert await storage.prepared_outbox() == []
        rows = await storage.diagnostic_rows(outbox_state="failed")
        assert len(rows) == (1 if lane == "direct" else 3)
        assert "before dispatch" in str(rows[0]["error"])
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("lane", ["send", "direct"])
@pytest.mark.parametrize("repeat_cancel", [False, True])
async def test_cancelled_initial_outbox_commit_cannot_be_recovered(
    tmp_path: Path, lane: str, *, repeat_cancel: bool
) -> None:
    adapter = _adapter()
    adapter._effective_limit = 256
    storage = VkStorage(tmp_path / "state.sqlite3")
    await storage.open()
    adapter._storage = storage
    adapter._client = cast("Any", object())
    calls: list[dict[str, object]] = []
    loop = asyncio.get_running_loop()
    cancel_count = 0

    async def transport(params: dict[str, object]) -> object:
        calls.append(params)
        return 42

    def cancel_on_commit(statement: str) -> None:
        nonlocal cancel_count
        if statement == "COMMIT" and (cancel_count == 0 or (repeat_cancel and cancel_count == 1)):
            cancel_count += 1
            loop.call_soon_threadsafe(task.cancel)

    adapter._send_chunk = transport
    await cast("Any", storage._connection()).set_trace_callback(cancel_on_commit)
    operation = (
        adapter.send("456", "x" * 600) if lane == "send" else adapter._send_direct(456, "caption", attachment="doc1_1")
    )
    task = asyncio.create_task(operation)
    try:
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cancel_count == (2 if repeat_cancel else 1)
        assert calls == []
        assert len(await storage.diagnostic_rows(outbox_state="failed")) == (3 if lane == "send" else 1)
        await storage.close()
        await storage.open()
        assert await storage.prepared_outbox() == []
        assert (await storage.counts())["outbox_delivery_unknown"] == 0
        await adapter._recover_prepared_outbox()
        assert calls == []
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await storage.close()


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


@pytest.mark.asyncio
@pytest.mark.parametrize("confirmed_replacements", [0, 1, 2])
async def test_rechunked_invocation_recovers_in_order_after_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, confirmed_replacements: int
) -> None:
    adapter = _adapter()
    adapter._effective_limit = 512
    path = tmp_path / "state.sqlite3"
    storage = VkStorage(path)
    await storage.open()
    delivered: list[str] = []
    requests = 0

    class ProcessCrash(BaseException):
        pass

    async def transport(_self: VkCommunityAdapter, params: dict[str, object]) -> object:
        nonlocal requests
        requests += 1
        if len(str(params["message"])) > 256:
            raise VkApiError(914, "message too long")
        delivered.append(str(params["message"]))
        return requests

    original_mark = storage.mark_outbox

    async def crash_before_dispatch(
        row_id: int, state: str, *, message_id: str | None = None, error: str | None = None
    ) -> None:
        if state == "sending" and requests > 0 and len(delivered) == confirmed_replacements:
            raise ProcessCrash
        await original_mark(row_id, state, message_id=message_id, error=error)

    monkeypatch.setattr(VkCommunityAdapter, "_send_chunk", transport)
    monkeypatch.setattr(storage, "mark_outbox", crash_before_dispatch)
    adapter._storage = storage
    adapter._client = cast("Any", object())
    content = "a" * 512 + "b" * 512 + "c" * 76
    try:
        with pytest.raises(ProcessCrash):
            await adapter.send("456", content)
    finally:
        await storage.close()

    reopened = VkStorage(path)
    await reopened.open()
    adapter._storage = reopened
    try:
        pending = await reopened.prepared_outbox()
        assert [len(row.wire_content) for row in pending] == [256, 256, 512, 76][confirmed_replacements:]
        assert len({row.invocation_id for row in pending}) == 1
        await adapter._recover_prepared_outbox()
        assert "".join(delivered) == content
        assert await reopened.prepared_outbox() == []
        assert len(await reopened.diagnostic_rows(outbox_state="sent")) == 5
        assert len(await reopened.diagnostic_rows(outbox_state="failed")) == 2
    finally:
        await reopened.close()


@pytest.mark.asyncio
async def test_repeated_rechunking_preserves_one_logical_invocation(tmp_path: Path) -> None:
    adapter = _adapter()
    adapter._effective_limit = 512
    storage = VkStorage(tmp_path / "state.sqlite3")
    await storage.open()
    delivered: list[str] = []

    class LengthClient:
        async def call(self, _method: str, params: dict[str, object]) -> int:
            text = str(params["message"])
            if len(text) > 256:
                raise VkApiError(914, "message too long")
            delivered.append(text)
            return len(delivered)

    adapter._storage = storage
    adapter._client = cast("VkApiClient", LengthClient())
    content = "a" * 512 + "b" * 512 + "c" * 76
    try:
        result = await adapter.send("456", content)
        assert result.success
        assert "".join(delivered) == content
        rows = await storage.diagnostic_rows(outbox_state="sent")
        assert len({row["invocation_id"] for row in rows}) == 1
        assert sorted(int(cast("int", row["chunk_index"])) for row in rows) == list(range(5))
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_failed_chunk_makes_unsent_tail_terminal(tmp_path: Path) -> None:
    adapter = _adapter()
    storage = VkStorage(tmp_path / "state.sqlite3")
    await storage.open()

    class FailingClient:
        async def call(self, _method: str, _params: dict[str, object]) -> object:
            raise VkApiError(7, "permission denied")

    adapter._storage = storage
    adapter._client = cast("VkApiClient", FailingClient())
    try:
        result = await adapter.send("456", "x" * 5000)
        assert not result.success
        assert await storage.prepared_outbox() == []
        db = storage._connection()
        async with db.execute("SELECT state FROM outbox ORDER BY id") as cursor:
            states = [row[0] for row in await cursor.fetchall()]
        assert states[0] == "failed"
        assert set(states[1:]) == {"failed"}
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_rich_text_table_sequence_sends_text_photo_text_in_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    adapter = _adapter()
    adapter._renderer = RichVkRenderer()
    storage = VkStorage(tmp_path / "state.sqlite3")
    await storage.open()

    class RecordingClient:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        async def call(self, method: str, params: dict[str, object]) -> int:
            assert method == "messages.send"
            self.calls.append(params)
            return len(self.calls)

    client = RecordingClient()

    async def upload_photo(peer_id: int, path: Path) -> str:
        assert peer_id == 456
        assert path.suffix == ".jpg"
        return "photo1_2"

    adapter._storage = storage
    adapter._client = cast("VkApiClient", client)
    monkeypatch.setattr(adapter, "_upload_photo", upload_photo)
    try:
        result = await adapter.send(
            "456",
            "До **жирного**.\n\n| Поле | Значение |\n|---|---|\n| План | Pro |\n\nПосле.",  # noqa: RUF001
        )
        assert result.success
        assert len(client.calls) == 3
        assert client.calls[0]["message"] == "До жирного."
        rich = json.loads(str(client.calls[0]["format_data"]))
        assert rich["items"] == [{"type": "bold", "offset": 3, "length": 7}]
        assert client.calls[1]["attachment"] == "photo1_2"
        assert client.calls[1]["message"] == ""
        assert client.calls[2]["message"] == "После."
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_recovery_stops_after_earlier_chunk_failure(tmp_path: Path) -> None:
    adapter = _adapter()
    storage = VkStorage(tmp_path / "state.sqlite3")
    await storage.open()
    await storage.prepare_outbox(456, ["one", "two"], None)

    class FailingClient:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def call(self, _method: str, params: dict[str, object]) -> object:
            self.calls.append(str(params["message"]))
            raise VkApiError(7, "permission denied")

    client = FailingClient()
    adapter._storage = storage
    adapter._client = cast("VkApiClient", client)
    try:
        await adapter._recover_prepared_outbox()
        assert client.calls == ["one"]
        assert await storage.prepared_outbox() == []
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_failed_three_records_history_gap_and_commits_fresh_cursor() -> None:
    adapter = _adapter()
    admitted: list[tuple[int, list[object], str]] = []

    class PollClient:
        async def poll(self, *_args: object, **_kwargs: object) -> LongPollResponse:
            return LongPollResponse(failed=3)

        async def get_long_poll_lease(self, _group_id: int) -> LongPollLease:
            return LongPollLease(key="new", server="https://lp.vk.com", ts="200")

    class PollStorage:
        async def cursor(self, _group_id: int) -> str:
            return "100"

        async def admit_batch(self, group_id: int, updates: list[object], ts: str) -> list[object]:
            admitted.append((group_id, updates, ts))
            return []

    adapter._client = cast("VkApiClient", PollClient())
    adapter._storage = cast("VkStorage", PollStorage())
    adapter._lease = LongPollLease(key="old", server="https://lp.vk.com", ts="100")
    await adapter._poll_once()
    assert adapter._history_gap_count == 1
    assert adapter._lease.ts == "200"
    assert admitted == [(123, [], "200")]


def test_stream_recovery_maps_rendered_bold_span_without_duplicate_text() -> None:
    source = "**" + "x" * 5000 + "**"
    rendered = RichVkRenderer().render_markdown(source)
    segment = rendered.segments[0]
    assert isinstance(segment, RenderedTextSegment)
    prefix = _source_prefix_for_rendered(source, segment.source_offsets, 4096)
    assert prefix == "**" + "x" * 4096
    assert source[len(prefix) :].count("x") == 904


@pytest.mark.asyncio
async def test_partial_table_delivery_is_reported_as_visible_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    adapter = _adapter()
    first = tmp_path / "page-1.jpg"
    second = tmp_path / "page-2.jpg"
    first.touch()
    second.touch()

    def render_pages(_table: RenderedTableSegment, _directory: Path) -> list[Path]:
        return [first, second]

    monkeypatch.setattr("hermes_vk_community.adapter.render_table_jpegs", render_pages)

    async def upload_photo(*_args: object) -> str:
        return "photo1_1"

    monkeypatch.setattr(adapter, "_upload_photo", upload_photo)
    results = iter(
        [
            SendResult(success=True, message_id="101", retryable=False),
            SendResult(success=False, error="failed", retryable=False),
        ]
    )

    async def send_direct(*_args: object, **_kwargs: object) -> SendResult:
        return next(results)

    monkeypatch.setattr(adapter, "_send_direct", send_direct)
    result = await adapter._send_table_segment(
        456,
        RenderedTableSegment(("a",), (("b",),)),
        None,
    )
    assert result.success
    assert result.message_id == "101"
    assert result.raw_response["partial_delivery"]["delivered_chunks"] == 1


@pytest.mark.asyncio
async def test_edit_overflow_returns_exact_ast_source_prefix(tmp_path: Path) -> None:
    adapter = _adapter()
    adapter._effective_limit = 256
    storage = VkStorage(tmp_path / "state.sqlite3")
    await storage.open()

    class EditThenFailClient:
        async def call(self, method: str, _params: dict[str, object]) -> int:
            if method == "messages.edit":
                return 1
            raise VkApiError(7, "permission denied")

    adapter._storage = storage
    adapter._client = cast("VkApiClient", EditThenFailClient())
    source = "**" + "x" * 500 + "**"
    try:
        result = await adapter.edit_message("456", "99", source, finalize=True)
        assert not result.success
        raw = cast("dict[str, object]", result.raw_response)
        assert raw["partial_overflow"] is True
        prefix = str(raw["delivered_prefix"])
        assert prefix == "**" + "x" * 256
        assert source.startswith(prefix)
        assert source[len(prefix) :].lstrip() == "x" * 244 + "**"
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_edit_uses_safe_formatting_flags_and_does_not_duplicate_last_id(tmp_path: Path) -> None:
    adapter = _adapter()
    adapter._effective_limit = 256
    storage = VkStorage(tmp_path / "state.sqlite3")
    await storage.open()

    class RecordingClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, object]]] = []

        async def call(self, method: str, params: dict[str, object]) -> int:
            self.calls.append((method, params))
            return 100 + len(self.calls)

    client = RecordingClient()
    adapter._storage = storage
    adapter._client = cast("VkApiClient", client)
    try:
        result = await adapter.edit_message("456", "99", "x" * 800, finalize=True)
        assert result.success
        assert result.message_id == "104"
        continuations = cast("tuple[str, ...]", cast("Any", result).continuation_message_ids)
        assert continuations == ("102", "103")
        edit_method, edit_params = client.calls[0]
        assert edit_method == "messages.edit"
        assert edit_params["disable_mentions"] is True
        assert edit_params["dont_parse_links"] is (not adapter.settings.formatting.parse_link_previews)
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_final_edit_with_table_sends_ordered_fresh_final_and_removes_preview(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    adapter = _adapter()
    adapter._renderer = RichVkRenderer()
    storage = VkStorage(tmp_path / "state.sqlite3")
    await storage.open()
    page = tmp_path / "table.jpg"
    page.touch()

    def render_pages(_table: RenderedTableSegment, _directory: Path) -> list[Path]:
        return [page]

    monkeypatch.setattr("hermes_vk_community.adapter.render_table_jpegs", render_pages)

    async def upload_photo(*_args: object) -> str:
        return "photo1_2"

    monkeypatch.setattr(adapter, "_upload_photo", upload_photo)

    class RecordingClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, object]]] = []

        async def call(self, method: str, params: dict[str, object]) -> int:
            self.calls.append((method, params))
            return 200 + len(self.calls)

    client = RecordingClient()
    adapter._storage = storage
    adapter._client = cast("VkApiClient", client)
    content = "До.\n\n| A | B |\n|---|---|\n| 1 | 2 |\n\nПосле."  # noqa: RUF001
    try:
        assert not adapter.supports_draft_streaming(chat_type="private", metadata={}, chat_id="456")
        assert adapter.prefers_fresh_final_streaming(content)
        result = await adapter.edit_message("456", "99", content, finalize=True)
        assert result.success
        assert [method for method, _params in client.calls] == [
            "messages.send",
            "messages.send",
            "messages.send",
            "messages.delete",
        ]
        assert client.calls[0][1]["message"] == "До."
        assert client.calls[1][1]["attachment"] == "photo1_2"
        assert client.calls[2][1]["message"] == "После."
        assert client.calls[3][1]["message_ids"] == 99
    finally:
        await storage.close()

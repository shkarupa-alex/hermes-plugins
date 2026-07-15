# pyright: reportPrivateUsage=false
from __future__ import annotations
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
from hermes_vk_community.errors import VkApiError, VkDeliveryUnknownError
from hermes_vk_community.models import InteractionPayload, LongPollLease, LongPollResponse, VkAttachment, VkMessage
from hermes_vk_community.plugin import build_adapter
from hermes_vk_community.renderer import RenderedTableSegment, RenderedTextSegment, RichVkRenderer
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

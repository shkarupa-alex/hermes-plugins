# pyright: reportPrivateUsage=false
from __future__ import annotations
import asyncio
import json
from typing import TYPE_CHECKING

import aiosqlite
import pytest

from hermes_vk_community.storage import MAX_NORMALIZED_JSON_LENGTH, VkStorage, canonical_json

if TYPE_CHECKING:
    from pathlib import Path

    from hermes_vk_community.models import JsonObject


@pytest.mark.asyncio
async def test_inbox_deduplicates_and_commits_cursor(tmp_path: Path) -> None:
    storage = VkStorage(tmp_path / "state.sqlite3")
    await storage.open()
    update: JsonObject = {
        "type": "message_new",
        "event_id": "evt-1",
        "group_id": 1,
        "object": {"message": {"peer_id": 2, "conversation_message_id": 3}},
    }
    first = await storage.admit_batch(1, [update], "10")
    second = await storage.admit_batch(1, [update], "11")
    assert len(first) == 1
    assert second == []
    assert await storage.cursor(1) == "11"
    assert len(await storage.received()) == 1
    await storage.close()


@pytest.mark.asyncio
async def test_sending_rows_become_delivery_unknown_after_restart(tmp_path: Path) -> None:
    path = tmp_path / "state.sqlite3"
    storage = VkStorage(path)
    await storage.open()
    record = (await storage.prepare_outbox(2, ["hello"], None))[0]
    await storage.mark_outbox(record.id, "sending")
    await storage.close()
    reopened = VkStorage(path)
    await reopened.open()
    assert (await reopened.counts())["outbox_delivery_unknown"] == 1
    await reopened.close()


@pytest.mark.asyncio
async def test_standalone_connection_preserves_another_writers_inflight_request(tmp_path: Path) -> None:
    path = tmp_path / "state.sqlite3"
    writer = VkStorage(path)
    standalone = VkStorage(path)
    await writer.open()
    try:
        record = (await writer.prepare_outbox(2, ["hello"], None))[0]
        await writer.mark_outbox(record.id, "sending")
        await standalone.open(recover_inflight=False)
        assert len(await writer.diagnostic_rows(outbox_state="sending")) == 1
        assert (await standalone.counts())["outbox_delivery_unknown"] == 0
    finally:
        await standalone.close()
        await writer.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("prefix_state", ["sending", "delivery_unknown", "failed", "sent"])
async def test_recovery_requires_a_recoverable_or_confirmed_prefix(tmp_path: Path, prefix_state: str) -> None:
    storage = VkStorage(tmp_path / "state.sqlite3")
    await storage.open()
    try:
        chunks = await storage.prepare_outbox(2, ["head", "tail 1", "tail 2"], None)
        unrelated = (await storage.prepare_outbox(3, ["unrelated"], None))[0]
        await storage.mark_outbox(chunks[0].id, prefix_state)
        recovered = await storage.prepared_outbox()
        expected = [*chunks[1:], unrelated] if prefix_state == "sent" else [unrelated]
        assert [row.id for row in recovered] == [row.id for row in expected]
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_concurrent_inbox_outbox_and_pairing_writes_are_independent(tmp_path: Path) -> None:
    storage = VkStorage(tmp_path / "state.sqlite3")
    await storage.open()
    update: JsonObject = {"type": "message_new", "event_id": "concurrent", "group_id": 1}
    try:
        results = await asyncio.gather(
            *(storage.prepare_outbox(2, [f"report {i}"], None) for i in range(10)),
            storage.admit_batch(1, [update], "42"),
            storage.create_pairing_code("PAIR", 60),
        )
        assert len(await storage.prepared_outbox()) == 10
        assert len(await storage.received()) == 1
        assert await storage.cursor(1) == "42"
        assert await storage.consume_pairing_code("PAIR", 123)
        assert len(results) == 12
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_cancelled_transaction_cannot_commit_or_rollback_another_writer(tmp_path: Path) -> None:
    storage = VkStorage(tmp_path / "state.sqlite3")
    await storage.open()
    started = asyncio.Event()

    async def interrupted_writer() -> None:
        async with storage._transaction() as db:
            await db.execute("INSERT INTO paired_users(user_id,paired_at_ms) VALUES(123,0)")
            started.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(interrupted_writer())
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        reader = asyncio.create_task(storage.is_paired(123))
        other_writer = asyncio.create_task(storage.create_pairing_code("SAFE", 60))
        await asyncio.sleep(0)
        assert not reader.done()
        assert not other_writer.done()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not await reader
        await other_writer
        assert await storage.consume_pairing_code("SAFE", 456)
        assert await storage.is_paired(456)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await storage.close()


@pytest.mark.asyncio
async def test_oversized_update_is_quarantined_before_cursor_advances(tmp_path: Path) -> None:
    storage = VkStorage(tmp_path / "state.sqlite3")
    await storage.open()
    update: JsonObject = {
        "type": "message_new",
        "event_id": "huge",
        "group_id": 1,
        "object": {"message": {"peer_id": 2, "text": "x" * MAX_NORMALIZED_JSON_LENGTH}},
    }
    inserted = await storage.admit_batch(1, [update], "99")
    assert inserted
    assert await storage.cursor(1) == "99"
    assert await storage.received() == []
    db = storage._connection()
    async with db.execute("SELECT state,error,length(normalized_json) FROM inbox") as cursor:
        row = await cursor.fetchone()
    assert row is not None
    assert row[:2] == ("quarantined", "normalized update exceeds 262144 characters")
    assert 0 < row[2] < 1024
    await storage.close()


@pytest.mark.asyncio
async def test_pairing_code_is_hashed_expiring_and_one_time(tmp_path: Path) -> None:
    storage = VkStorage(tmp_path / "state.sqlite3")
    await storage.open()
    await storage.create_pairing_code("VK-SECRET", 600)
    db = storage._connection()
    async with db.execute("SELECT code_sha256 FROM pairing_codes") as cursor:
        row = await cursor.fetchone()
    assert row
    assert row[0] != "VK-SECRET"
    assert await storage.consume_pairing_code("VK-SECRET", 42)
    assert not await storage.consume_pairing_code("VK-SECRET", 43)
    assert await storage.is_paired(42)
    await storage.close()


@pytest.mark.asyncio
async def test_prepared_outbox_retains_recoverable_wire_payload(tmp_path: Path) -> None:
    storage = VkStorage(tmp_path / "state.sqlite3")
    await storage.open()
    original = (await storage.prepare_outbox(2, ["hello"], "7"))[0]
    recovered = (await storage.prepared_outbox())[0]
    assert recovered.id == original.id
    assert recovered.wire_content == "hello"
    assert recovered.reply_target == "7"
    await storage.close()


@pytest.mark.asyncio
async def test_random_id_collision_is_rejected_and_resampled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    storage = VkStorage(tmp_path / "state.sqlite3")
    await storage.open()
    samples = iter([40, 40, 41])

    def next_random_id(_limit: int) -> int:
        return next(samples)

    monkeypatch.setattr("hermes_vk_community.storage.secrets.randbelow", next_random_id)
    first = (await storage.prepare_outbox(2, ["first"], None))[0]
    second = (await storage.prepare_outbox(2, ["second"], None))[0]
    assert first.random_id == 41
    assert second.random_id == 42
    await storage.close()


@pytest.mark.asyncio
async def test_prepared_outbox_retains_format_data(tmp_path: Path) -> None:
    storage = VkStorage(tmp_path / "state.sqlite3")
    await storage.open()
    rich: dict[str, object] = {
        "version": 1,
        "items": [{"type": "bold", "offset": 0, "length": 5}],
    }
    await storage.prepare_outbox(2, ["hello"], None, format_data=[rich])
    recovered = (await storage.prepared_outbox())[0]
    assert recovered.format_data == rich
    await storage.close()


@pytest.mark.asyncio
async def test_schema_v2_migrates_format_data_column(tmp_path: Path) -> None:
    path = tmp_path / "state.sqlite3"
    storage = VkStorage(path)
    await storage.open()
    await storage.close()

    legacy = await aiosqlite.connect(path)
    await legacy.execute("ALTER TABLE outbox DROP COLUMN format_data_json")
    await legacy.execute("UPDATE schema_meta SET version=2 WHERE singleton=1")
    await legacy.commit()
    await legacy.close()

    migrated = VkStorage(path)
    await migrated.open()
    db = migrated._connection()
    async with db.execute("PRAGMA table_info(outbox)") as cursor:
        columns = {str(row[1]) for row in await cursor.fetchall()}
    assert "format_data_json" in columns
    async with db.execute("SELECT version FROM schema_meta WHERE singleton=1") as cursor:
        row = await cursor.fetchone()
    assert row == (3,)
    await migrated.close()


@pytest.mark.asyncio
async def test_nonrecoverable_outbox_is_not_replayed_as_plain_text(tmp_path: Path) -> None:
    storage = VkStorage(tmp_path / "vk.sqlite3")
    await storage.open()
    await storage.prepare_outbox(2, ["media caption"], None, recoverable=False)
    assert await storage.prepared_outbox() == []
    await storage.close()

    reopened = VkStorage(tmp_path / "vk.sqlite3")
    await reopened.open()
    assert await reopened.prepared_outbox() == []
    assert (await reopened.counts())["outbox_delivery_unknown"] == 1
    await reopened.close()


@pytest.mark.asyncio
async def test_terminalized_outbox_tail_is_not_recovered(tmp_path: Path) -> None:
    storage = VkStorage(tmp_path / "vk.sqlite3")
    await storage.open()
    records = await storage.prepare_outbox(2, ["first", "second", "third"], None)
    await storage.terminalize_outbox_failure(
        records[0],
        "delivery_unknown",
        "request timed out",
        records[1:],
        "blocked by ambiguous prefix",
    )
    assert await storage.prepared_outbox() == []
    db = storage._connection()
    async with db.execute("SELECT state FROM outbox ORDER BY id") as cursor:
        assert [row[0] for row in await cursor.fetchall()] == ["delivery_unknown", "failed", "failed"]
    await storage.close()


def test_canonical_json_keeps_unicode_and_order_is_stable() -> None:
    encoded = canonical_json({"б": "😀", "a": 1})  # noqa: RUF001
    assert encoded == '{"a":1,"б":"😀"}'  # noqa: RUF001
    assert json.loads(encoded) == {"a": 1, "б": "😀"}  # noqa: RUF001

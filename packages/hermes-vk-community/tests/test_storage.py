# pyright: reportPrivateUsage=false
from __future__ import annotations
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

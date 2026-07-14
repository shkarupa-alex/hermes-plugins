from __future__ import annotations
import json
from typing import TYPE_CHECKING

import pytest

from hermes_vk_community.storage import VkStorage, canonical_json

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


def test_canonical_json_keeps_unicode_and_order_is_stable() -> None:
    encoded = canonical_json({"б": "😀", "a": 1})  # noqa: RUF001
    assert encoded == '{"a":1,"б":"😀"}'  # noqa: RUF001
    assert json.loads(encoded) == {"a": 1, "б": "😀"}  # noqa: RUF001

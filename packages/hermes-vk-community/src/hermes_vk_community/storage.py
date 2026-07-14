from __future__ import annotations
import hashlib
import json
import secrets
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import aiosqlite

if TYPE_CHECKING:
    from pathlib import Path

    from hermes_vk_community.models import JsonObject

InboxState = Literal["received", "dispatched", "completed", "quarantined"]
MAX_NORMALIZED_JSON_LENGTH = 262_144

SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_meta (
  singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
  version INTEGER NOT NULL
);
INSERT OR IGNORE INTO schema_meta(singleton, version) VALUES(1, 1);
CREATE TABLE IF NOT EXISTS long_poll_cursor (
  group_id INTEGER PRIMARY KEY,
  ts TEXT NOT NULL,
  updated_at_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS inbox (
  id INTEGER PRIMARY KEY,
  group_id INTEGER NOT NULL,
  event_id TEXT,
  peer_id INTEGER,
  conversation_message_id INTEGER,
  canonical_sha256 TEXT NOT NULL,
  state TEXT NOT NULL CHECK(state IN ('received','dispatched','completed','quarantined')),
  normalized_json TEXT NOT NULL CHECK(length(normalized_json) <= 262144),
  attempts INTEGER NOT NULL DEFAULT 0,
  error TEXT CHECK(error IS NULL OR length(error) <= 2048),
  created_at_ms INTEGER NOT NULL,
  updated_at_ms INTEGER NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS inbox_event_id_uq
  ON inbox(group_id, event_id) WHERE event_id IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS inbox_cmid_uq
  ON inbox(group_id, peer_id, conversation_message_id)
  WHERE peer_id IS NOT NULL AND conversation_message_id IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS inbox_hash_uq ON inbox(group_id, canonical_sha256);
CREATE TABLE IF NOT EXISTS outbox (
  id INTEGER PRIMARY KEY,
  invocation_id TEXT NOT NULL,
  peer_id INTEGER NOT NULL,
  content_sha256 TEXT NOT NULL,
  chunk_index INTEGER NOT NULL,
  random_id INTEGER NOT NULL CHECK(random_id BETWEEN 1 AND 2147483647),
  reply_target TEXT,
  state TEXT NOT NULL CHECK(state IN
    ('prepared','sending','sent','partial_delivery','delivery_unknown','failed')),
  attempt_count INTEGER NOT NULL DEFAULT 0,
  returned_message_id TEXT,
  error TEXT CHECK(error IS NULL OR length(error) <= 2048),
  created_at_ms INTEGER NOT NULL,
  updated_at_ms INTEGER NOT NULL,
  UNIQUE(invocation_id, chunk_index),
  UNIQUE(peer_id, random_id)
);
"""


@dataclass(frozen=True, slots=True)
class InboxRecord:
    id: int
    normalized_json: str


@dataclass(frozen=True, slots=True)
class OutboxRecord:
    id: int
    invocation_id: str
    peer_id: int
    chunk_index: int
    random_id: int


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class VkStorage:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._db: aiosqlite.Connection | None = None

    async def open(self) -> None:
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.path)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA foreign_keys=ON")
        await self._db.execute("PRAGMA synchronous=FULL")
        await self._db.execute("PRAGMA busy_timeout=5000")
        await self._db.executescript(SCHEMA)
        await self._db.execute("UPDATE outbox SET state='delivery_unknown' WHERE state='sending'")
        await self._db.commit()

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
        self._db = None

    def _connection(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("storage is not open")
        return self._db

    async def cursor(self, group_id: int) -> str | None:
        db = self._connection()
        async with db.execute("SELECT ts FROM long_poll_cursor WHERE group_id=?", (group_id,)) as cursor:
            row = await cursor.fetchone()
        return str(row[0]) if row else None

    async def admit_batch(self, group_id: int, updates: list[JsonObject], ts: str) -> list[int]:
        db = self._connection()
        now = _now_ms()
        inserted: list[int] = []
        await db.execute("BEGIN IMMEDIATE")
        try:
            for update in updates:
                normalized = canonical_json(update)
                if len(normalized) > MAX_NORMALIZED_JSON_LENGTH:
                    continue
                peer_id, conversation_message_id = _message_identifiers(update)
                digest = hashlib.sha256(normalized.encode()).hexdigest()
                cursor = await db.execute(
                    """INSERT OR IGNORE INTO inbox(
                    group_id,event_id,peer_id,conversation_message_id,canonical_sha256,state,
                    normalized_json,created_at_ms,updated_at_ms) VALUES(?,?,?,?,?,'received',?,?,?)""",
                    (
                        group_id,
                        update.get("event_id"),
                        peer_id,
                        conversation_message_id,
                        digest,
                        normalized,
                        now,
                        now,
                    ),
                )
                if cursor.rowcount:
                    inserted.append(int(cursor.lastrowid or 0))
            await db.execute(
                """INSERT INTO long_poll_cursor(group_id,ts,updated_at_ms) VALUES(?,?,?)
                ON CONFLICT(group_id) DO UPDATE SET ts=excluded.ts,updated_at_ms=excluded.updated_at_ms""",
                (group_id, ts, now),
            )
            await db.commit()
        except BaseException:
            await db.rollback()
            raise
        return inserted

    async def received(self) -> list[InboxRecord]:
        db = self._connection()
        async with db.execute("SELECT id,normalized_json FROM inbox WHERE state='received' ORDER BY id") as cursor:
            rows = await cursor.fetchall()
        return [InboxRecord(id=int(row[0]), normalized_json=str(row[1])) for row in rows]

    async def mark_inbox(self, row_id: int, state: InboxState, error: str | None = None) -> None:
        db = self._connection()
        await db.execute(
            "UPDATE inbox SET state=?,attempts=attempts+1,error=?,updated_at_ms=? WHERE id=?",
            (state, (error or "")[:2048] or None, _now_ms(), row_id),
        )
        await db.commit()

    async def prepare_outbox(self, peer_id: int, chunks: list[str], reply_target: str | None) -> list[OutboxRecord]:
        db = self._connection()
        invocation_id = uuid.uuid4().hex
        now = _now_ms()
        records: list[OutboxRecord] = []
        await db.execute("BEGIN IMMEDIATE")
        try:
            for index, chunk in enumerate(chunks):
                while True:
                    random_id = secrets.randbelow(2_147_483_647) + 1
                    cursor = await db.execute(
                        """INSERT OR IGNORE INTO outbox(invocation_id,peer_id,content_sha256,chunk_index,
                        random_id,reply_target,state,created_at_ms,updated_at_ms)
                        VALUES(?,?,?,?,?,?,'prepared',?,?)""",
                        (
                            invocation_id,
                            peer_id,
                            hashlib.sha256(chunk.encode()).hexdigest(),
                            index,
                            random_id,
                            reply_target,
                            now,
                            now,
                        ),
                    )
                    if cursor.rowcount:
                        records.append(
                            OutboxRecord(int(cursor.lastrowid or 0), invocation_id, peer_id, index, random_id)
                        )
                        break
            await db.commit()
        except BaseException:
            await db.rollback()
            raise
        return records

    async def mark_outbox(
        self, row_id: int, state: str, *, message_id: str | None = None, error: str | None = None
    ) -> None:
        db = self._connection()
        await db.execute(
            """UPDATE outbox SET state=?,attempt_count=attempt_count+1,returned_message_id=?,error=?,updated_at_ms=?
            WHERE id=?""",
            (state, message_id, (error or "")[:2048] or None, _now_ms(), row_id),
        )
        await db.commit()

    async def counts(self) -> dict[str, int]:
        db = self._connection()
        result: dict[str, int] = {}
        for table, state in (("inbox", "dispatched"), ("outbox", "delivery_unknown")):
            async with db.execute(f"SELECT count(*) FROM {table} WHERE state=?", (state,)) as cursor:  # noqa: S608
                row = await cursor.fetchone()
            result[f"{table}_{state}"] = int(row[0]) if row else 0
        return result


def _now_ms() -> int:
    return time.time_ns() // 1_000_000


def _message_identifiers(update: JsonObject) -> tuple[int | None, int | None]:
    event_object = update.get("object")
    if not isinstance(event_object, dict):
        return None, None
    message = event_object.get("message")
    if not isinstance(message, dict):
        return None, None
    peer_id = message.get("peer_id")
    conversation_message_id = message.get("conversation_message_id")
    return (
        peer_id if isinstance(peer_id, int) else None,
        conversation_message_id if isinstance(conversation_message_id, int) else None,
    )

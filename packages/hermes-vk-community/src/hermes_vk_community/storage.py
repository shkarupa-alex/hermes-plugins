from __future__ import annotations
import asyncio
import hashlib
import json
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

import aiosqlite

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable
    from pathlib import Path

    from hermes_vk_community.models import JsonObject

InboxState = Literal["received", "dispatched", "completed", "quarantined"]
MAX_NORMALIZED_JSON_LENGTH = 262_144
SCHEMA_VERSION = 3
PREVIOUS_SCHEMA_VERSION = 2
UPDATE_FIELDS = ("type", "object", "group_id", "event_id")

SCHEMA = """
CREATE TABLE schema_meta (
  singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
  version INTEGER NOT NULL
);
INSERT INTO schema_meta(singleton, version) VALUES(1, 3);
CREATE TABLE long_poll_cursor (
  group_id INTEGER PRIMARY KEY,
  ts TEXT NOT NULL,
  updated_at_ms INTEGER NOT NULL
);
CREATE TABLE inbox (
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
CREATE UNIQUE INDEX inbox_event_id_uq
  ON inbox(group_id, event_id) WHERE event_id IS NOT NULL;
CREATE UNIQUE INDEX inbox_cmid_uq
  ON inbox(group_id, peer_id, conversation_message_id)
  WHERE peer_id IS NOT NULL AND conversation_message_id IS NOT NULL;
CREATE UNIQUE INDEX inbox_hash_uq ON inbox(group_id, canonical_sha256);
CREATE TABLE outbox (
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
  wire_content TEXT CHECK(wire_content IS NULL OR length(wire_content) <= 262144),
  format_data_json TEXT CHECK(format_data_json IS NULL OR length(format_data_json) <= 262144),
  error TEXT CHECK(error IS NULL OR length(error) <= 2048),
  created_at_ms INTEGER NOT NULL,
  updated_at_ms INTEGER NOT NULL,
  UNIQUE(invocation_id, chunk_index),
  UNIQUE(peer_id, random_id)
);
CREATE TABLE pairing_codes (
  code_sha256 TEXT PRIMARY KEY,
  expires_at_ms INTEGER NOT NULL,
  consumed_at_ms INTEGER,
  consumed_by_user_id INTEGER,
  created_at_ms INTEGER NOT NULL
);
CREATE TABLE paired_users (
  user_id INTEGER PRIMARY KEY,
  paired_at_ms INTEGER NOT NULL
);
CREATE TABLE media_orphans (
  id INTEGER PRIMARY KEY,
  kind TEXT NOT NULL,
  peer_id INTEGER NOT NULL,
  upload_sha256 TEXT NOT NULL,
  error TEXT NOT NULL CHECK(length(error) <= 2048),
  created_at_ms INTEGER NOT NULL
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
    wire_content: str = ""
    reply_target: str | None = None
    format_data: dict[str, object] | None = None


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def normalize_update(update: JsonObject) -> JsonObject:
    """Keep only the stable VK update fields used by the transport contract."""
    return {field: update[field] for field in UPDATE_FIELDS if field in update}


class VkStorage:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._db: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    async def open(self, *, recover_inflight: bool = True) -> None:
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.path)
        self.path.chmod(0o600)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA foreign_keys=ON")
        await self._db.execute("PRAGMA synchronous=FULL")
        await self._db.execute("PRAGMA busy_timeout=5000")
        await self._migrate()
        if recover_inflight:
            await self._db.execute("UPDATE outbox SET state='delivery_unknown' WHERE state='sending'")
        await self._db.commit()

    async def _migrate(self) -> None:
        db = self._connection()
        async with db.execute("SELECT name FROM sqlite_master WHERE type='table'") as cursor:
            existing = {str(row[0]) for row in await cursor.fetchall() if not str(row[0]).startswith("sqlite_")}
        if not existing:
            await db.executescript(f"BEGIN IMMEDIATE;\n{SCHEMA}\nCOMMIT;")
            return
        if "schema_meta" not in existing:
            raise RuntimeError("VK storage has a partial or unsupported schema (schema_meta is missing)")
        async with db.execute("SELECT version FROM schema_meta WHERE singleton=1") as cursor:
            row = await cursor.fetchone()
        if row is not None and int(row[0]) == 1:
            await db.execute("BEGIN IMMEDIATE")
            try:
                await db.execute(
                    "ALTER TABLE outbox ADD COLUMN wire_content TEXT "
                    "CHECK(wire_content IS NULL OR length(wire_content) <= 262144)"
                )
                await db.execute(
                    "UPDATE outbox SET state='failed',error='legacy prepared row has no recoverable payload' "
                    "WHERE state='prepared'"
                )
                await db.execute(
                    "CREATE TABLE pairing_codes (code_sha256 TEXT PRIMARY KEY,expires_at_ms INTEGER NOT NULL,"
                    "consumed_at_ms INTEGER,consumed_by_user_id INTEGER,created_at_ms INTEGER NOT NULL)"
                )
                await db.execute(
                    "CREATE TABLE paired_users (user_id INTEGER PRIMARY KEY,paired_at_ms INTEGER NOT NULL)"
                )
                await db.execute(
                    "CREATE TABLE media_orphans (id INTEGER PRIMARY KEY,kind TEXT NOT NULL,peer_id INTEGER NOT NULL,"
                    "upload_sha256 TEXT NOT NULL,error TEXT NOT NULL CHECK(length(error) <= 2048),"
                    "created_at_ms INTEGER NOT NULL)"
                )
                await db.execute("UPDATE schema_meta SET version=2 WHERE singleton=1")
                await db.commit()
            except BaseException:
                await db.rollback()
                raise
            row = (PREVIOUS_SCHEMA_VERSION,)
        if row is not None and int(row[0]) == PREVIOUS_SCHEMA_VERSION:
            await db.execute("BEGIN IMMEDIATE")
            try:
                await db.execute(
                    "ALTER TABLE outbox ADD COLUMN format_data_json TEXT "
                    "CHECK(format_data_json IS NULL OR length(format_data_json) <= 262144)"
                )
                await db.execute("UPDATE schema_meta SET version=3 WHERE singleton=1")
                await db.commit()
            except BaseException:
                await db.rollback()
                raise
            row = (SCHEMA_VERSION,)
        if row is None or int(row[0]) != SCHEMA_VERSION:
            observed = "missing" if row is None else str(row[0])
            raise RuntimeError(f"VK storage schema version {observed} is unsupported; expected {SCHEMA_VERSION}")
        await db.execute(
            "CREATE TABLE IF NOT EXISTS media_orphans (id INTEGER PRIMARY KEY,kind TEXT NOT NULL,"
            "peer_id INTEGER NOT NULL,upload_sha256 TEXT NOT NULL,error TEXT NOT NULL "
            "CHECK(length(error) <= 2048),created_at_ms INTEGER NOT NULL)"
        )
        await db.commit()
        async with db.execute("SELECT name FROM sqlite_master WHERE type='table'") as cursor:
            existing = {str(item[0]) for item in await cursor.fetchall() if not str(item[0]).startswith("sqlite_")}
        required = {
            "long_poll_cursor",
            "inbox",
            "outbox",
            "pairing_codes",
            "paired_users",
            "media_orphans",
        }
        missing = required - existing
        if missing:
            raise RuntimeError(f"VK storage schema is incomplete; missing: {', '.join(sorted(missing))}")

    async def close(self) -> None:
        async with self._lock:
            if self._db is not None:
                await self._db.close()
            self._db = None

    def _connection(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("storage is not open")
        return self._db

    @asynccontextmanager
    async def _transaction(
        self, *, on_cancel: Callable[[aiosqlite.Connection], Awaitable[None]] | None = None
    ) -> AsyncGenerator[aiosqlite.Connection, None]:
        # aiosqlite queues statements, not whole coroutine transactions. Every
        # writer owns the connection until commit/rollback; readers use the same
        # lock so they cannot act on another coroutine's uncommitted records.
        async with self._lock:
            db = self._connection()
            try:
                await db.execute("BEGIN IMMEDIATE")
                yield db
                await db.commit()
            except asyncio.CancelledError:

                async def settle() -> None:
                    # aiosqlite's worker may have committed before cancellation
                    # reached the waiter. FIFO rollback settles that outcome;
                    # preparation still owns its invocation and can retire it.
                    await db.rollback()
                    if on_cancel is not None:
                        await on_cancel(db)
                        await db.commit()

                cleanup = asyncio.create_task(settle())
                while not cleanup.done():
                    try:
                        await asyncio.shield(cleanup)
                    except asyncio.CancelledError:
                        # Retain the connection lock even if shutdown cancels
                        # this task again while the worker finishes cleanup.
                        continue
                cleanup.result()
                raise
            except BaseException:
                await db.rollback()
                raise

    async def cursor(self, group_id: int) -> str | None:
        async with self._lock:
            db = self._connection()
            async with db.execute("SELECT ts FROM long_poll_cursor WHERE group_id=?", (group_id,)) as cursor:
                row = await cursor.fetchone()
            return str(row[0]) if row else None

    async def admit_batch(self, group_id: int, updates: list[JsonObject], ts: str) -> list[int]:
        now = _now_ms()
        inserted: list[int] = []
        async with self._transaction() as db:
            for update in updates:
                stable_update = normalize_update(update)
                normalized = canonical_json(stable_update)
                digest = hashlib.sha256(normalized.encode()).hexdigest()
                if len(normalized) > MAX_NORMALIZED_JSON_LENGTH:
                    normalized = canonical_json(
                        {
                            "event_id": stable_update.get("event_id"),
                            "group_id": stable_update.get("group_id", group_id),
                            "quarantined": "normalized update exceeds storage limit",
                            "sha256": digest,
                            "type": stable_update.get("type"),
                        }
                    )
                    state = "quarantined"
                    error = "normalized update exceeds 262144 characters"
                else:
                    state = "received"
                    error = None
                peer_id, conversation_message_id = _message_identifiers(stable_update)
                cursor = await db.execute(
                    """INSERT OR IGNORE INTO inbox(
                    group_id,event_id,peer_id,conversation_message_id,canonical_sha256,state,
                    normalized_json,error,created_at_ms,updated_at_ms) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                    (
                        group_id,
                        stable_update.get("event_id"),
                        peer_id,
                        conversation_message_id,
                        digest,
                        state,
                        normalized,
                        error,
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
        return inserted

    async def received(self) -> list[InboxRecord]:
        async with self._lock:
            db = self._connection()
            async with db.execute("SELECT id,normalized_json FROM inbox WHERE state='received' ORDER BY id") as cursor:
                rows = await cursor.fetchall()
            return [InboxRecord(id=int(row[0]), normalized_json=str(row[1])) for row in rows]

    async def mark_inbox(self, row_id: int, state: InboxState, error: str | None = None) -> None:
        async with self._transaction() as db:
            await db.execute(
                "UPDATE inbox SET state=?,attempts=attempts+1,error=?,updated_at_ms=? WHERE id=?",
                (state, (error or "")[:2048] or None, _now_ms(), row_id),
            )

    async def prepare_outbox(  # noqa: PLR0913 - atomic replacement retains the original delivery contract
        self,
        peer_id: int,
        chunks: list[str],
        reply_target: str | None,
        *,
        recoverable: bool = True,
        format_data: list[dict[str, object] | None] | None = None,
        replace_rejected: OutboxRecord | None = None,
    ) -> list[OutboxRecord]:
        invocation_id = uuid.uuid4().hex
        now = _now_ms()
        records: list[OutboxRecord] = []
        formats = format_data or [None] * len(chunks)
        if len(formats) != len(chunks):
            raise ValueError("format_data must align with chunks")

        async def cancel_preparation(db: aiosqlite.Connection) -> None:
            await db.execute(
                "UPDATE outbox SET state='failed',error=?,updated_at_ms=? WHERE invocation_id=? AND state=?",
                (
                    "outbox preparation cancelled before dispatch",
                    _now_ms(),
                    invocation_id,
                    "prepared" if recoverable else "sending",
                ),
            )

        async with self._transaction(on_cancel=cancel_preparation) as db:
            first_index = 0
            if replace_rejected is not None:
                if not recoverable or not chunks:
                    raise ValueError("a rejected text chunk requires recoverable replacements")
                async with db.execute(
                    "SELECT chunk_index FROM outbox WHERE id=? AND invocation_id=? AND peer_id=? "
                    "AND state IN ('sending','delivery_unknown')",
                    (replace_rejected.id, replace_rejected.invocation_id, peer_id),
                ) as cursor:
                    row = await cursor.fetchone()
                if row is None:
                    raise RuntimeError("rejected outbox chunk is no longer owned by this invocation")
                first_index = int(row[0])
                # Retain the rejected wire request for diagnostics, but replace
                # its logical position atomically. Recovery then sees one ordered
                # invocation, including the replacements and original tail.
                await db.execute(
                    "UPDATE outbox SET invocation_id=?,chunk_index=0,state='failed',error=?,updated_at_ms=? WHERE id=?",
                    (invocation_id, "VK rejected chunk at cached limit", now, replace_rejected.id),
                )
                invocation_id = replace_rejected.invocation_id
                async with db.execute(
                    "SELECT id,chunk_index FROM outbox WHERE invocation_id=? AND chunk_index>? "
                    "ORDER BY chunk_index DESC",
                    (invocation_id, first_index),
                ) as cursor:
                    tail = await cursor.fetchall()
                for row_id, old_index in tail:
                    await db.execute(
                        "UPDATE outbox SET chunk_index=? WHERE id=?",
                        (int(old_index) + len(chunks) - 1, row_id),
                    )
            for index, (chunk, chunk_format) in enumerate(zip(chunks, formats, strict=True)):
                chunk_index = first_index + index
                format_json = canonical_json(chunk_format) if chunk_format else None
                while True:
                    random_id = secrets.randbelow(2_147_483_647) + 1
                    cursor = await db.execute(
                        """INSERT OR IGNORE INTO outbox(invocation_id,peer_id,content_sha256,chunk_index,
                        random_id,reply_target,state,wire_content,format_data_json,created_at_ms,updated_at_ms)
                        VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            invocation_id,
                            peer_id,
                            hashlib.sha256(chunk.encode()).hexdigest(),
                            chunk_index,
                            random_id,
                            reply_target,
                            "prepared" if recoverable else "sending",
                            chunk,
                            format_json,
                            now,
                            now,
                        ),
                    )
                    if cursor.rowcount:
                        records.append(
                            OutboxRecord(
                                int(cursor.lastrowid or 0),
                                invocation_id,
                                peer_id,
                                chunk_index,
                                random_id,
                                chunk,
                                reply_target,
                                chunk_format,
                            )
                        )
                        break
        return records

    async def prepared_outbox(self) -> list[OutboxRecord]:
        async with self._lock:
            db = self._connection()
            # Another process can still own an in-flight prefix. Recover its tail
            # only after that prefix is confirmed, or replay the prepared prefix
            # first using its persisted random_id.
            async with db.execute(
                "SELECT pending.id,pending.invocation_id,pending.peer_id,pending.chunk_index,pending.random_id,"
                "pending.wire_content,pending.reply_target,pending.format_data_json "
                "FROM outbox AS pending WHERE pending.state='prepared' AND pending.wire_content IS NOT NULL "
                "AND NOT EXISTS (SELECT 1 FROM outbox AS earlier "
                "WHERE earlier.invocation_id=pending.invocation_id AND earlier.chunk_index<pending.chunk_index "
                "AND (earlier.state NOT IN ('prepared','sent') "
                "OR (earlier.state='prepared' AND earlier.wire_content IS NULL))) "
                "ORDER BY (SELECT MIN(first.id) FROM outbox AS first "
                "WHERE first.invocation_id=pending.invocation_id),pending.chunk_index"
            ) as cursor:
                rows = await cursor.fetchall()
            return [
                OutboxRecord(
                    id=int(row[0]),
                    invocation_id=str(row[1]),
                    peer_id=int(row[2]),
                    chunk_index=int(row[3]),
                    random_id=int(row[4]),
                    wire_content=str(row[5]),
                    reply_target=str(row[6]) if row[6] is not None else None,
                    format_data=cast("dict[str, object]", json.loads(str(row[7]))) if row[7] is not None else None,
                )
                for row in rows
            ]

    async def create_pairing_code(self, code: str, ttl_seconds: int) -> None:
        async with self._transaction() as db:
            now = _now_ms()
            digest = hashlib.sha256(code.strip().encode()).hexdigest()
            await db.execute(
                "INSERT INTO pairing_codes(code_sha256,expires_at_ms,created_at_ms) VALUES(?,?,?)",
                (digest, now + ttl_seconds * 1000, now),
            )

    async def consume_pairing_code(self, code: str, user_id: int) -> bool:
        now = _now_ms()
        digest = hashlib.sha256(code.strip().encode()).hexdigest()
        async with self._transaction() as db:
            cursor = await db.execute(
                "UPDATE pairing_codes SET consumed_at_ms=?,consumed_by_user_id=? "
                "WHERE code_sha256=? AND consumed_at_ms IS NULL AND expires_at_ms>=?",
                (now, user_id, digest, now),
            )
            consumed = bool(cursor.rowcount)
            if consumed:
                await db.execute(
                    "INSERT OR REPLACE INTO paired_users(user_id,paired_at_ms) VALUES(?,?)", (user_id, now)
                )
        return consumed

    async def is_paired(self, user_id: int) -> bool:
        async with self._lock:
            db = self._connection()
            async with db.execute("SELECT 1 FROM paired_users WHERE user_id=?", (user_id,)) as cursor:
                return await cursor.fetchone() is not None

    async def record_media_orphan(self, kind: str, peer_id: int, upload_fingerprint: str, error: str) -> None:
        async with self._transaction() as db:
            await db.execute(
                "INSERT INTO media_orphans(kind,peer_id,upload_sha256,error,created_at_ms) VALUES(?,?,?,?,?)",
                (kind[:32], peer_id, upload_fingerprint[:64], error[:2048], _now_ms()),
            )

    async def mark_outbox(
        self, row_id: int, state: str, *, message_id: str | None = None, error: str | None = None
    ) -> None:
        async with self._transaction() as db:
            await db.execute(
                """UPDATE outbox SET state=?,attempt_count=attempt_count+CASE WHEN ?='sending' THEN 1 ELSE 0 END,
                returned_message_id=?,error=?,updated_at_ms=?
                WHERE id=?""",
                (state, state, message_id, (error or "")[:2048] or None, _now_ms(), row_id),
            )

    async def terminalize_outbox_failure(
        self,
        record: OutboxRecord,
        state: str,
        error: str,
        tail_records: list[OutboxRecord],
        tail_error: str,
    ) -> None:
        """Atomically terminate a failed chunk and every later prepared chunk."""
        now = _now_ms()
        async with self._transaction() as db:
            await db.execute(
                "UPDATE outbox SET state=?,error=?,updated_at_ms=? WHERE id=?",
                (state, (error or "delivery failed")[:2048], now, record.id),
            )
            # A rechunk transaction may have committed new records before its
            # caller was cancelled. Include the durable invocation, rather than
            # relying only on the caller's earlier in-memory snapshot.
            tail_ids = [record.id, *(item.id for item in tail_records)]
            placeholders = ",".join("?" for _ in tail_ids)
            await db.execute(
                f"UPDATE outbox SET state='failed',error=?,updated_at_ms=? "  # noqa: S608 - IDs use placeholders
                f"WHERE state='prepared' AND (invocation_id=? OR id IN ({placeholders}))",
                ((tail_error or "unsent tail is terminal")[:2048], now, record.invocation_id, *tail_ids),
            )

    async def counts(self) -> dict[str, int]:
        async with self._lock:
            db = self._connection()
            result: dict[str, int] = {}
            for table, state in (("inbox", "dispatched"), ("outbox", "delivery_unknown")):
                async with db.execute(f"SELECT count(*) FROM {table} WHERE state=?", (state,)) as cursor:  # noqa: S608
                    row = await cursor.fetchone()
                result[f"{table}_{state}"] = int(row[0]) if row else 0
            async with db.execute("SELECT count(*) FROM media_orphans") as cursor:
                row = await cursor.fetchone()
            result["media_orphans"] = int(row[0]) if row else 0
            return result

    async def diagnostic_rows(
        self, *, inbox_state: str | None = None, outbox_state: str | None = None
    ) -> list[dict[str, object]]:
        async with self._lock:
            db = self._connection()
            result: list[dict[str, object]] = []
            if inbox_state is not None:
                async with db.execute(
                    "SELECT id,group_id,event_id,peer_id,attempts,error,updated_at_ms "
                    "FROM inbox WHERE state=? ORDER BY id",
                    (inbox_state,),
                ) as cursor:
                    result.extend(
                        {
                            "kind": "inbox",
                            "id": int(row[0]),
                            "group_id": int(row[1]),
                            "event_id": row[2],
                            "peer_id": row[3],
                            "attempts": int(row[4]),
                            "error": row[5],
                            "updated_at_ms": int(row[6]),
                        }
                        for row in await cursor.fetchall()
                    )
            if outbox_state is not None:
                async with db.execute(
                    "SELECT id,invocation_id,peer_id,chunk_index,attempt_count,error,updated_at_ms "
                    "FROM outbox WHERE state=? ORDER BY id",
                    (outbox_state,),
                ) as cursor:
                    result.extend(
                        {
                            "kind": "outbox",
                            "id": int(row[0]),
                            "invocation_id": str(row[1]),
                            "peer_id": int(row[2]),
                            "chunk_index": int(row[3]),
                            "attempt_count": int(row[4]),
                            "error": row[5],
                            "updated_at_ms": int(row[6]),
                        }
                        for row in await cursor.fetchall()
                    )
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

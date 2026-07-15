from __future__ import annotations
import asyncio
import difflib
import hashlib
import logging
import secrets
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Never, cast

import aiohttp
import filetype
from agent.secret_scope import get_secret
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult, cache_media_bytes
from hermes_constants import get_hermes_home
from pydantic import TypeAdapter
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception,
    stop_after_attempt,
    stop_never,
    wait_random_exponential,
)
from tools.approval import resolve_gateway_approval
from tools.clarify_gateway import mark_awaiting_text, resolve_gateway_clarify

from hermes_vk_community.client import VkApiClient
from hermes_vk_community.config import PolicyEnvironment, VkSettings, settings_from_platform_config
from hermes_vk_community.errors import VkApiError, VkDeliveryUnknownError, VkHttpError
from hermes_vk_community.models import (
    DocumentUploadResponse,
    Group,
    GroupsResponse,
    InteractionPayload,
    KeyboardAction,
    KeyboardButton,
    LongPollLease,
    MessageNewObject,
    PhotoUploadResponse,
    SaveDocumentResponse,
    SavedPhoto,
    UploadServer,
    User,
    VkAttachment,
    VkKeyboard,
    VkUpdate,
)
from hermes_vk_community.renderer import PlainVkRenderer, split_message, split_message_with_spans
from hermes_vk_community.storage import InboxRecord, VkStorage
from tools import slash_confirm

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

logger = logging.getLogger(__name__)
HTTP_TOO_MANY_REQUESTS = 429
HTTP_SERVER_ERROR_MIN = 500
INTERACTION_TTL_SECONDS = 600
VK_TOO_LONG_ERROR = 914
MIN_MESSAGE_LIMIT = 256
MAX_APPROVAL_PREVIEW = 800
MAX_PAIRING_TEXT = 128
MAX_GEO_COORDINATES_LENGTH = 128


@dataclass(frozen=True, slots=True)
class _Interaction:
    group: str
    peer_id: int
    user_id: int
    session_key: str
    kind: str
    value: str
    target_id: str
    expires_at: float


class VkCommunityAdapter(BasePlatformAdapter):
    splits_long_messages = True
    supports_code_blocks = False

    def __init__(self, config: PlatformConfig) -> None:
        super().__init__(config=config, platform=Platform("vk"))
        self.settings: VkSettings = settings_from_platform_config(config)
        self._dm_policy = "allowlist"
        self._group_policy = "disabled"
        self._allow_from = set(self.settings.allow_from or [])
        self.allow_from = sorted(self._allow_from)
        self._client: VkApiClient | None = None
        self._storage: VkStorage | None = None
        self._poll_task: asyncio.Task[None] | None = None
        self._lease: LongPollLease | None = None
        self._typing_last: dict[str, float] = {}
        self._typing_cooldown: dict[str, float] = {}
        self._renderer = PlainVkRenderer()
        self._effective_limit = self.settings.max_message_length
        self._interactions: dict[str, _Interaction] = {}

    @property
    def enforces_own_access_policy(self) -> bool:
        return True

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        if is_reconnect:
            await self._stop_polling()
            if self._client is not None:
                await self._client.close()
        conflicts = PolicyEnvironment().conflicts()
        if conflicts:
            self._set_fatal_error(
                "vk_auth_policy_conflict",
                f"VK YAML-only authorization conflicts with: {', '.join(conflicts)}",
                retryable=False,
            )
            return False
        token = get_secret("VK_COMMUNITY_TOKEN")
        if not token:
            self._set_fatal_error("vk_missing_token", "VK_COMMUNITY_TOKEN is missing", retryable=False)
            return False
        lock_held = getattr(self, "_platform_lock_identity", None) is not None
        if not lock_held and not self._acquire_platform_lock("vk", str(self.settings.group_id), "VK community"):
            # Hermes 0.18.2 assigns the identity before attempting the lock. Clearing it is
            # essential; calling _release_platform_lock here would release the other process.
            self._platform_lock_identity = None
            return False
        try:
            self._client = VkApiClient(token, api_version=self.settings.api_version, media=self.settings.media)
            await self._client.open()
            await self._verify_group()
            storage_path = self.settings.resolve_storage_path(Path(get_hermes_home()))
            if self._storage is None:
                self._storage = VkStorage(storage_path)
                await self._storage.open()
            self._lease = await self._client.get_long_poll_lease(self.settings.group_id)
            self._running = True
            await self._recover_prepared_outbox()
            await self._dispatch_received()
            self._poll_task = asyncio.create_task(self._poll_loop(), name=f"vk-long-poll-{self.settings.group_id}")
            return True  # noqa: TRY300
        except (ValueError, VkApiError) as exc:
            message = _safe_api_error(exc) if isinstance(exc, VkApiError) else str(exc)[:512]
            self._set_fatal_error("vk_identity_mismatch", message, retryable=False)
            logger.exception("[vk] permanent connection failure: %s", type(exc).__name__)
            await self._close_resources(release_lock=True)
            return False
        except Exception:
            logger.exception("[vk] connection failed")
            await self._close_resources(release_lock=True)
            return False

    async def disconnect(self) -> None:
        self._running = False
        await self._stop_polling()
        await self._close_resources(release_lock=True)

    async def send(  # noqa: PLR0911
        self,
        chat_id: str,
        content: str,
        reply_to: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SendResult:
        del metadata
        if self._client is None or self._storage is None:
            return SendResult(
                success=False, error="VK adapter is not connected", retryable=True, error_kind="transient"
            )
        rendered = self._renderer.render_markdown(content)
        wire_chunks = split_message_with_spans(rendered.text, self._effective_limit)
        chunks = [chunk.text for chunk in wire_chunks]
        outbox = await self._storage.prepare_outbox(int(chat_id), chunks, reply_to)
        delivered: list[str] = []
        pending = list(zip(chunks, outbox, strict=True))
        index = 0
        while index < len(pending):
            chunk, record = pending[index]
            await self._storage.mark_outbox(record.id, "sending")
            try:
                response = await self._send_chunk(
                    {
                        "peer_id": int(chat_id),
                        "message": chunk,
                        "random_id": record.random_id,
                        "reply_to": int(reply_to) if reply_to else None,
                        "disable_mentions": self.settings.formatting.disable_mentions,
                        "dont_parse_links": not self.settings.formatting.parse_link_previews,
                    },
                )
                message_id = _message_id(response)
                delivered.append(message_id)
                await self._storage.mark_outbox(record.id, "sent", message_id=message_id)
            except VkDeliveryUnknownError:
                await self._storage.terminalize_outbox_failure(
                    record,
                    "delivery_unknown",
                    "request timed out",
                    [item[1] for item in pending[index + 1 :]],
                    "blocked after ambiguous earlier chunk",
                )
                if delivered:
                    return _partial_result(delivered, [item[0] for item in pending])
                return SendResult(
                    success=False,
                    error="VK delivery timed out after the request may have succeeded",
                    retryable=False,
                    error_kind="unknown",
                    raw_response={"delivery_unknown": True, "outbox_id": record.id},
                )
            except VkApiError as exc:
                if exc.code == VK_TOO_LONG_ERROR and self._effective_limit > MIN_MESSAGE_LIMIT:
                    await self._storage.mark_outbox(record.id, "failed", error="VK rejected chunk at cached limit")
                    self._effective_limit = max(MIN_MESSAGE_LIMIT, min(self._effective_limit - 1, len(chunk) // 2))
                    replacement_chunks = split_message(chunk, self._effective_limit)
                    replacement = await self._storage.prepare_outbox(int(chat_id), replacement_chunks, reply_to)
                    pending[index : index + 1] = list(zip(replacement_chunks, replacement, strict=True))
                    continue
                state = "partial_delivery" if delivered else "failed"
                await self._storage.terminalize_outbox_failure(
                    record,
                    state,
                    _safe_api_error(exc),
                    [item[1] for item in pending[index + 1 :]],
                    "blocked after failed earlier chunk",
                )
                if delivered:
                    return _partial_result(delivered, [item[0] for item in pending])
                return _api_error_result(exc)
            except Exception as exc:  # noqa: BLE001
                await self._storage.terminalize_outbox_failure(
                    record,
                    "delivery_unknown",
                    type(exc).__name__,
                    [item[1] for item in pending[index + 1 :]],
                    "blocked after ambiguous earlier chunk",
                )
                if delivered:
                    return _partial_result(delivered, [item[0] for item in pending])
                return SendResult(
                    success=False,
                    error="VK delivery timed out after the request may have started",
                    retryable=False,
                    error_kind="unknown",
                    raw_response={"delivery_unknown": True, "outbox_id": record.id},
                )
            index += 1
        return SendResult(
            success=True,
            message_id=delivered[-1] if delivered else None,
            continuation_message_ids=tuple(delivered[:-1]),
            retryable=False,
        )

    async def _send_direct(
        self,
        peer_id: int,
        content: str,
        *,
        reply_to: str | None = None,
        keyboard: str | None = None,
        attachment: str | None = None,
    ) -> SendResult:
        if self._client is None or self._storage is None:
            return SendResult(success=False, error="VK adapter is not connected", retryable=True)
        recoverable = keyboard is None and attachment is None
        record = (
            await self._storage.prepare_outbox(
                peer_id,
                [content],
                reply_to,
                recoverable=recoverable,
            )
        )[0]
        if recoverable:
            await self._storage.mark_outbox(record.id, "sending")
        try:
            payload = await self._send_chunk(
                {
                    "peer_id": peer_id,
                    "message": content,
                    "random_id": record.random_id,
                    "reply_to": int(reply_to) if reply_to else None,
                    "keyboard": keyboard,
                    "attachment": attachment,
                    "disable_mentions": self.settings.formatting.disable_mentions,
                    "dont_parse_links": not self.settings.formatting.parse_link_previews,
                }
            )
            message_id = _message_id(payload)
            await self._storage.mark_outbox(record.id, "sent", message_id=message_id)
            return SendResult(success=True, message_id=message_id, retryable=False)
        except VkDeliveryUnknownError:
            await self._storage.mark_outbox(record.id, "delivery_unknown", error="request timed out")
            return SendResult(
                success=False,
                error="VK delivery timed out after the request may have succeeded",
                retryable=False,
                error_kind="unknown",
                raw_response={"delivery_unknown": True, "outbox_id": record.id},
            )
        except VkApiError as exc:
            await self._storage.mark_outbox(record.id, "failed", error=_safe_api_error(exc))
            return _api_error_result(exc)
        except Exception as exc:  # noqa: BLE001 - the request may already have reached VK
            await self._storage.mark_outbox(record.id, "delivery_unknown", error=type(exc).__name__)
            return SendResult(
                success=False,
                error="VK delivery timed out after the request may have started",
                retryable=False,
                error_kind="unknown",
                raw_response={"delivery_unknown": True, "outbox_id": record.id},
            )

    async def _recover_prepared_outbox(self) -> None:
        if self._client is None or self._storage is None:
            return
        records = await self._storage.prepared_outbox()
        blocked_invocations: set[str] = set()
        for record_index, record in enumerate(records):
            if record.invocation_id in blocked_invocations:
                continue
            await self._storage.mark_outbox(record.id, "sending")
            try:
                response = await self._send_chunk(
                    {
                        "peer_id": record.peer_id,
                        "message": record.wire_content,
                        "random_id": record.random_id,
                        "reply_to": int(record.reply_target) if record.reply_target else None,
                    }
                )
                await self._storage.mark_outbox(record.id, "sent", message_id=_message_id(response))
            except VkDeliveryUnknownError:
                await self._storage.terminalize_outbox_failure(
                    record,
                    "delivery_unknown",
                    "recovery timed out",
                    [item for item in records[record_index + 1 :] if item.invocation_id == record.invocation_id],
                    "blocked after ambiguous recovery chunk",
                )
                blocked_invocations.add(record.invocation_id)
            except Exception as exc:  # noqa: BLE001 - recovery records terminal diagnostics
                await self._storage.terminalize_outbox_failure(
                    record,
                    "failed",
                    type(exc).__name__,
                    [item for item in records[record_index + 1 :] if item.invocation_id == record.invocation_id],
                    "blocked after failed recovery chunk",
                )
                blocked_invocations.add(record.invocation_id)

    async def edit_message(
        self,
        chat_id: str,
        message_id: str,
        content: str,
        *,
        finalize: bool = False,
    ) -> SendResult:
        if self._client is None:
            return SendResult(success=False, error="VK adapter is not connected", retryable=True)
        rendered = self._renderer.render_markdown(content)
        wire_chunks = split_message_with_spans(rendered.text, self._effective_limit)
        chunks = [chunk.text for chunk in wire_chunks]
        if len(chunks) != 1 and not finalize:
            chunks = [rendered.text[: max(1, self._effective_limit - 2)].rstrip() + " …"]
            wire_chunks = []
        try:
            await self._client.call(
                "messages.edit",
                {"peer_id": int(chat_id), "message_id": int(message_id), "message": chunks[0]},
            )
        except VkApiError as exc:
            if exc.code == VK_TOO_LONG_ERROR and self._effective_limit > MIN_MESSAGE_LIMIT:
                self._effective_limit = max(
                    MIN_MESSAGE_LIMIT,
                    min(self._effective_limit - 1, len(chunks[0]) // 2),
                )
                return await self.edit_message(
                    chat_id,
                    message_id,
                    content,
                    finalize=finalize,
                )
            return _api_error_result(exc)
        if len(chunks) == 1:
            return SendResult(success=True, message_id=message_id)
        continuation_ids: list[str] = []
        delivered_rendered_end = wire_chunks[0].end
        for chunk_index, chunk in enumerate(chunks[1:], start=1):
            result = await self.send(chat_id, chunk)
            if not result.success:
                delivered_prefix = _source_prefix_for_rendered(
                    content,
                    rendered.text,
                    delivered_rendered_end,
                )
                return SendResult(
                    success=False,
                    error="overflow_continuation_failed",
                    retryable=False,
                    message_id=continuation_ids[-1] if continuation_ids else message_id,
                    continuation_message_ids=tuple(continuation_ids),
                    raw_response={
                        "partial_overflow": True,
                        "delivered_chunks": 1 + len(continuation_ids),
                        "total_chunks": len(chunks),
                        "last_message_id": continuation_ids[-1] if continuation_ids else message_id,
                        "delivered_prefix": delivered_prefix,
                    },
                )
            raw_continuations = cast(
                "tuple[object, ...]",
                result.continuation_message_ids,  # pyright: ignore[reportUnknownMemberType]
            )
            continuation_ids.extend(str(value) for value in raw_continuations)
            if result.message_id:
                continuation_ids.append(result.message_id)
            delivered_rendered_end = wire_chunks[chunk_index].end
        return SendResult(
            success=True,
            message_id=continuation_ids[-1],
            continuation_message_ids=tuple(continuation_ids),
            retryable=False,
        )

    def supports_draft_streaming(self, chat_type: str | None = None, metadata: dict[str, Any] | None = None) -> bool:
        del chat_type, metadata
        return False

    def prefers_fresh_final_streaming(self, content: str, metadata: dict[str, Any] | None = None) -> bool:
        del content, metadata
        return False

    async def send_clarify(  # noqa: PLR0913 - exact Hermes compatibility contract
        self,
        chat_id: str,
        question: str,
        choices: list[object] | None,
        clarify_id: str,
        session_key: str,
        metadata: dict[str, Any] | None = None,
    ) -> SendResult:
        if not choices:
            mark_awaiting_text(clarify_id)
            return await self.send(
                chat_id,
                f"❓ {question}",
                reply_to=(metadata or {}).get("reply_to_message_id"),
            )
        values = [str(choice) for choice in choices]
        buttons = [(str(index + 1), "clarify", value, clarify_id) for index, value in enumerate(values)]
        buttons.append(("Другой ответ", "clarify_other", "", clarify_id))
        body = "❓ " + question + "\n\n" + "\n".join(f"{index + 1}. {value}" for index, value in enumerate(values))
        return await self._send_keyboard(chat_id, body, buttons, session_key, metadata)

    async def send_exec_approval(
        self,
        chat_id: str,
        command: str,
        session_key: str,
        description: str = "dangerous command",
        metadata: dict[str, Any] | None = None,
    ) -> SendResult:
        preview = command if len(command) <= MAX_APPROVAL_PREVIEW else command[:MAX_APPROVAL_PREVIEW] + "..."
        body = f"⚠️ Требуется подтверждение команды\n\n{preview}\n\nПричина: {description}"  # noqa: RUF001
        return await self._send_keyboard(
            chat_id,
            body,
            [("Разрешить", "approval", "approve", ""), ("Запретить", "approval", "deny", "")],
            session_key,
            metadata,
        )

    async def send_slash_confirm(  # noqa: PLR0913 - exact Hermes compatibility contract
        self,
        chat_id: str,
        title: str,
        message: str,
        session_key: str,
        confirm_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> SendResult:
        return await self._send_keyboard(
            chat_id,
            f"{title}\n\n{message}",
            [
                ("Один раз", "slash", "once", confirm_id),
                ("Всегда", "slash", "always", confirm_id),
                ("Отмена", "slash", "cancel", confirm_id),
            ],
            session_key,
            metadata,
        )

    async def _send_keyboard(
        self,
        chat_id: str,
        content: str,
        buttons: list[tuple[str, str, str, str]],
        session_key: str,
        metadata: dict[str, Any] | None,
    ) -> SendResult:
        if self._client is None:
            return SendResult(success=False, error="VK adapter is not connected", retryable=True)
        peer_id = int(chat_id)
        group = secrets.token_urlsafe(12)
        rows: list[list[KeyboardButton]] = []
        now = time.monotonic()
        self._interactions = {key: value for key, value in self._interactions.items() if value.expires_at >= now}
        for label, kind, value, target_id in buttons:
            nonce = secrets.token_urlsafe(24)
            payload = InteractionPayload.model_validate({"v": 1, "n": nonce}).model_dump_json(by_alias=True)
            self._interactions[nonce] = _Interaction(
                group, peer_id, peer_id, session_key, kind, value, target_id, now + INTERACTION_TTL_SECONDS
            )
            rows.append([KeyboardButton(action=KeyboardAction(label=label[:40], payload=payload))])
        keyboard = VkKeyboard(buttons=rows).model_dump_json(exclude_none=True)
        result = await self._send_direct(
            peer_id,
            content,
            reply_to=(metadata or {}).get("reply_to_message_id"),
            keyboard=keyboard,
        )
        if not result.success:
            self._interactions = {key: value for key, value in self._interactions.items() if value.group != group}
        return result

    async def delete_message(self, chat_id: str, message_id: str) -> bool:
        if self._client is None:
            return False
        try:
            await self._client.call(
                "messages.delete",
                {"peer_id": int(chat_id), "message_ids": int(message_id), "delete_for_all": True},
            )
        except Exception:  # noqa: BLE001
            return False
        return True

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: str | None = None,
        reply_to: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SendResult:
        if self._client is None:
            return SendResult(success=False, error="VK adapter is not connected", retryable=True)
        downloaded = None
        try:
            downloaded = await self._client.download_media(image_url)
            return await self.send_image_file(chat_id, str(downloaded.path), caption, reply_to)
        except Exception:  # noqa: BLE001 - arbitrary public URLs use the base text fallback
            return await BasePlatformAdapter.send_image(self, chat_id, image_url, caption, reply_to, metadata)
        finally:
            if downloaded is not None:
                downloaded.cleanup()

    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: str | None = None,
        reply_to: str | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,  # noqa: ANN401 - exact Hermes compatibility contract
    ) -> SendResult:
        del metadata, kwargs
        try:
            attachment = await self._upload_photo(int(chat_id), Path(image_path))
        except Exception as exc:  # noqa: BLE001
            return SendResult(success=False, error=f"VK photo upload failed: {type(exc).__name__}", retryable=False)
        return await self._send_direct(int(chat_id), caption or "", reply_to=reply_to, attachment=attachment)

    async def send_document(  # noqa: PLR0913 - exact Hermes compatibility contract
        self,
        chat_id: str,
        file_path: str,
        caption: str | None = None,
        file_name: str | None = None,
        reply_to: str | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,  # noqa: ANN401 - exact Hermes compatibility contract
    ) -> SendResult:
        del metadata, kwargs
        try:
            attachment = await self._upload_document(int(chat_id), Path(file_path), file_name=file_name)
        except Exception as exc:  # noqa: BLE001
            return SendResult(success=False, error=f"VK document upload failed: {type(exc).__name__}", retryable=False)
        return await self._send_direct(int(chat_id), caption or "", reply_to=reply_to, attachment=attachment)

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: str | None = None,
        reply_to: str | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,  # noqa: ANN401 - exact Hermes compatibility contract
    ) -> SendResult:
        del metadata, kwargs
        source = Path(audio_path)
        try:
            with tempfile.TemporaryDirectory(prefix="hermes-vk-voice-") as directory:
                voice = Path(directory) / "voice.ogg"
                await _convert_voice_to_ogg(source, voice, self.settings.media.total_timeout_seconds)
                attachment = await self._upload_document(
                    int(chat_id), voice, file_name="voice.ogg", upload_type="audio_message"
                )
                return await self._send_direct(int(chat_id), caption or "", reply_to=reply_to, attachment=attachment)
        except Exception as voice_error:  # noqa: BLE001 - certified normal-document fallback below
            logger.warning("[vk] audio_message upload failed: %s", type(voice_error).__name__)
            try:
                attachment = await self._upload_document(int(chat_id), source, file_name=source.name)
                return await self._send_direct(int(chat_id), caption or "", reply_to=reply_to, attachment=attachment)
            except Exception as fallback_error:  # noqa: BLE001
                return SendResult(
                    success=False,
                    error=f"VK voice upload failed: {type(fallback_error).__name__}",
                    retryable=False,
                )

    async def _upload_photo(self, peer_id: int, path: Path) -> str:
        if self._client is None:
            raise RuntimeError("VK client is not connected")
        server = UploadServer.model_validate(
            await self._safe_api_call("photos.getMessagesUploadServer", {"peer_id": peer_id})
        )
        mime = await asyncio.to_thread(_sniff_mime, path)
        if mime is None or not mime.startswith("image/"):
            raise ValueError("VK photo upload requires a sniffed image MIME type")
        extension = await asyncio.to_thread(_sniff_extension, path)
        try:
            uploaded = PhotoUploadResponse.model_validate(
                await self._client.upload_file(
                    server.upload_url,
                    "photo",
                    path,
                    content_type=mime,
                    filename=f"photo.{extension}" if extension else "photo",
                )
            )
        except Exception as exc:
            await self._record_orphan("photo-upload", peer_id, await asyncio.to_thread(_file_sha256, path), exc)
            raise
        try:
            saved = TypeAdapter(list[SavedPhoto]).validate_python(
                await self._client.call(
                    "photos.saveMessagesPhoto",
                    {"server": uploaded.server, "photo": uploaded.photo, "hash": uploaded.hash},
                )
            )
        except Exception as exc:
            await self._record_orphan("photo", peer_id, uploaded.model_dump_json(), exc)
            raise
        if not saved:
            raise ValueError("VK did not return a saved photo")
        return _attachment_id("photo", saved[0].owner_id, saved[0].id, saved[0].access_key)

    async def _upload_document(
        self,
        peer_id: int,
        path: Path,
        *,
        file_name: str | None = None,
        upload_type: str | None = None,
    ) -> str:
        if self._client is None:
            raise RuntimeError("VK client is not connected")
        server = UploadServer.model_validate(
            await self._safe_api_call(
                "docs.getMessagesUploadServer",
                {"peer_id": peer_id, "type": upload_type},
            )
        )
        mime = await asyncio.to_thread(_sniff_mime, path)
        try:
            uploaded = DocumentUploadResponse.model_validate(
                await self._client.upload_file(
                    server.upload_url,
                    "file",
                    path,
                    content_type=mime,
                    filename=file_name or path.name,
                )
            )
        except Exception as exc:
            await self._record_orphan("document-upload", peer_id, await asyncio.to_thread(_file_sha256, path), exc)
            raise
        try:
            saved = SaveDocumentResponse.model_validate(
                await self._client.call("docs.save", {"file": uploaded.file, "title": file_name or path.name})
            )
        except Exception as exc:
            await self._record_orphan("document", peer_id, uploaded.model_dump_json(), exc)
            raise
        document = saved.audio_message if upload_type == "audio_message" else saved.doc
        if document is None:
            raise ValueError("VK did not return the saved document object")
        return _attachment_id("doc", document.owner_id, document.id, document.access_key)

    async def _record_orphan(self, kind: str, peer_id: int, upload: str, error: Exception) -> None:
        if self._storage is None:
            return
        fingerprint = hashlib.sha256(upload.encode()).hexdigest()
        await self._storage.record_media_orphan(kind, peer_id, fingerprint, type(error).__name__)

    async def _safe_api_call(self, method: str, params: dict[str, object]) -> object:
        if self._client is None:
            raise RuntimeError("VK client is not connected")
        retrying = AsyncRetrying(
            retry=retry_if_exception(_is_retryable_safe_api_error),
            wait=wait_random_exponential(multiplier=1, max=5),
            stop=stop_after_attempt(3),
            reraise=True,
        )
        return await retrying(self._client.call, method, params)

    async def send_typing(self, chat_id: str, metadata: dict[str, Any] | None = None) -> None:
        del metadata
        if not self.settings.typing_indicator or self._client is None:
            return
        now = time.monotonic()
        if now < self._typing_cooldown.get(chat_id, 0):
            return
        if now - self._typing_last.get(chat_id, float("-inf")) < self.settings.typing.refresh_seconds:
            return
        try:
            await self._client.call(
                "messages.setActivity",
                {"peer_id": int(chat_id), "group_id": self.settings.group_id, "type": "typing"},
            )
            self._typing_last[chat_id] = now
        except Exception:  # noqa: BLE001
            self._typing_cooldown[chat_id] = now + self.settings.typing.failure_cooldown_seconds

    async def stop_typing(self, chat_id: str) -> None:
        del chat_id

    async def get_chat_info(self, chat_id: str) -> dict[str, Any]:
        if self._client is None:
            return {"name": chat_id, "type": "dm", "chat_id": chat_id}
        try:
            payload = await self._client.call("users.get", {"user_ids": int(chat_id)})
            users = TypeAdapter(list[User]).validate_python(payload)
            user = users[0]
            name = f"{user.first_name} {user.last_name}".strip()
        except Exception:  # noqa: BLE001
            name = chat_id
        return {"name": name, "type": "dm", "chat_id": chat_id}

    async def _verify_group(self) -> None:
        if self._client is None:
            raise RuntimeError("VK client is not connected")
        payload = await self._client.call("groups.getById", {"group_id": self.settings.group_id})
        if isinstance(payload, dict):
            groups = GroupsResponse.model_validate(payload).groups
        else:
            groups = TypeAdapter(list[Group]).validate_python(payload)
        if not groups or groups[0].id != self.settings.group_id:
            raise ValueError("VK token does not belong to the configured group_id")

    async def _poll_loop(self) -> None:
        while self._running:
            try:
                retrying = AsyncRetrying(
                    retry=retry_if_exception(_is_retryable_poll_error),
                    wait=wait_random_exponential(
                        multiplier=1,
                        min=self.settings.long_poll.retry_min_seconds,
                        max=self.settings.long_poll.retry_max_seconds,
                    ),
                    stop=stop_never,
                    before_sleep=_log_poll_retry,
                    reraise=True,
                )
                await retrying(self._poll_once)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._running = False
                self._set_fatal_error("vk_long_poll_terminal", f"VK Long Poll stopped: {exc}", retryable=False)
                logger.exception("[vk] Long Poll stopped after a non-retryable failure")
                await self._notify_fatal_error()

    async def _poll_once(self) -> None:
        client, lease, storage = self._polling_resources()
        ts = await storage.cursor(self.settings.group_id) or lease.ts
        response = await client.poll(
            lease,
            ts=ts,
            wait_seconds=self.settings.long_poll.wait_seconds,
        )
        if response.failed == 1 and response.ts is not None:
            await storage.admit_batch(self.settings.group_id, [], response.ts)
        elif response.failed == 2:  # noqa: PLR2004 - VK protocol failure code
            new_lease = await client.get_long_poll_lease(self.settings.group_id)
            self._lease = new_lease.model_copy(update={"ts": ts})
        elif response.failed == 3:  # noqa: PLR2004 - VK protocol failure code
            self._lease = await client.get_long_poll_lease(self.settings.group_id)
            await storage.admit_batch(self.settings.group_id, [], self._lease.ts)
            logger.warning("[vk] Long Poll history gap reported for group %s", self.settings.group_id)
        elif response.failed is not None:
            _raise_unsupported_failure(response.failed)
        elif response.ts is not None:
            await storage.admit_batch(self.settings.group_id, response.updates, response.ts)
            await self._dispatch_received()

    async def _send_chunk(self, params: dict[str, object]) -> object:
        if self._client is None:
            raise RuntimeError("VK client is not connected")
        retrying = AsyncRetrying(
            retry=retry_if_exception(_is_retryable_send_error),
            wait=wait_random_exponential(multiplier=1, max=5),
            stop=stop_after_attempt(3),
            reraise=True,
        )
        return await retrying(self._client.call, "messages.send", params)

    async def _dispatch_received(self) -> None:
        if self._storage is None:
            raise RuntimeError("VK storage is not open")
        for record in await self._storage.received():
            await self._dispatch_record(record)

    def _polling_resources(self) -> tuple[VkApiClient, LongPollLease, VkStorage]:
        if self._client is None or self._lease is None or self._storage is None:
            _raise_polling_resources_missing()
        return self._client, self._lease, self._storage

    async def _dispatch_record(self, record: InboxRecord) -> None:
        if self._storage is None:
            raise RuntimeError("VK storage is not open")
        try:
            update = VkUpdate.model_validate_json(record.normalized_json)
            if update.type != "message_new" or not isinstance(update.object, MessageNewObject):
                await self._storage.mark_inbox(record.id, "quarantined", "unsupported event type")
                return
            message = update.object.message
            sender = str(message.from_id)
            if message.from_id <= 0 or message.peer_id != message.from_id:
                await self._storage.mark_inbox(record.id, "quarantined", "sender is not authorized")
                return
            statically_allowed = sender in self._allow_from
            paired = self.settings.pairing.enabled and await self._storage.is_paired(message.from_id)
            if not statically_allowed and not paired:
                if (
                    self.settings.pairing.enabled
                    and message.text.strip()
                    and len(message.text) <= MAX_PAIRING_TEXT
                    and not message.attachments
                    and not message.fwd_messages
                    and message.payload is None
                    and await self._storage.consume_pairing_code(message.text, message.from_id)
                ):
                    await self._storage.mark_inbox(record.id, "dispatched")
                    await self.send(str(message.peer_id), "✅ Устройство привязано к Hermes.")
                    return
                await self._storage.mark_inbox(record.id, "quarantined", "sender is not authorized")
                return
            if message.payload is not None:
                if await self._consume_interaction(message):
                    await self._storage.mark_inbox(record.id, "dispatched")
                else:
                    await self._storage.mark_inbox(record.id, "quarantined", "invalid or expired keyboard payload")
                return
            format_data = cast("dict[str, object] | None", message.format_data)
            parsed = self._renderer.parse_incoming(message.text, format_data)
            media_urls, media_types, is_voice, attachment_text = await self._cache_attachments(message.attachments)
            context = _structured_context(message.reply_message, message.fwd_messages)
            geo_context = _geo_context(message.geo)
            text = "\n".join(part for part in (parsed.markdown, context, attachment_text, geo_context) if part).strip()
            if not text and not media_urls:
                await self._storage.mark_inbox(record.id, "quarantined", "empty message")
                return
            source = self.build_source(
                chat_id=str(message.peer_id),
                chat_type="dm",
                user_id=sender,
                message_id=str(message.id),
            )
            event = MessageEvent(
                text=text,
                message_type=MessageType.VOICE if is_voice else MessageType.TEXT,
                source=source,
                raw_message=message.model_dump(mode="json"),
                message_id=str(message.id),
                reply_to_message_id=str(message.reply_message.get("id")) if message.reply_message else None,
                media_urls=media_urls,
                media_types=media_types,
                metadata={
                    "vk_event_id": update.event_id,
                    "vk_format_data": message.format_data,
                    "vk_reply": message.reply_message,
                    "vk_forwards": message.fwd_messages,
                },
            )
            await self._storage.mark_inbox(record.id, "dispatched")
            await self.handle_message(event)
        except Exception as exc:  # noqa: BLE001
            await self._storage.mark_inbox(record.id, "quarantined", f"{type(exc).__name__}: {exc}")

    async def _consume_interaction(self, message: Any) -> bool:  # noqa: ANN401 - validated VkMessage at caller
        try:
            payload = InteractionPayload.model_validate_json(message.payload)
        except Exception:  # noqa: BLE001 - an untrusted payload must fail closed
            return False
        interaction = self._interactions.pop(payload.nonce, None)
        if interaction is None:
            return False
        if (
            interaction.expires_at < time.monotonic()
            or interaction.peer_id != message.peer_id
            or interaction.user_id != message.from_id
        ):
            return False
        self._interactions = {
            key: value for key, value in self._interactions.items() if value.group != interaction.group
        }
        if interaction.kind == "clarify":
            resolve_gateway_clarify(interaction.target_id, interaction.value)
        elif interaction.kind == "clarify_other":
            mark_awaiting_text(interaction.target_id)
        elif interaction.kind == "approval":
            resolve_gateway_approval(interaction.session_key, interaction.value)
        elif interaction.kind == "slash":
            response = await slash_confirm.resolve(interaction.session_key, interaction.target_id, interaction.value)
            if response:
                await self.send(str(message.peer_id), response)
        else:
            return False
        return True

    async def _cache_attachments(
        self,
        attachments: list[VkAttachment],
    ) -> tuple[list[str], list[str], bool, str]:
        if self._client is None:
            raise RuntimeError("VK client is not connected")
        paths: list[str] = []
        media_types: list[str] = []
        voice_flags: list[bool] = []
        descriptions: list[str] = []
        is_voice = False
        for attachment in attachments:
            candidate = _attachment_candidate(attachment)
            if candidate is None:
                descriptions.append(_attachment_description(attachment))
                continue
            url, filename, default_kind, voice = candidate
            try:
                downloaded = await self._client.download_media(url)
                try:
                    data = await asyncio.to_thread(downloaded.path.read_bytes)
                    cached = cache_media_bytes(
                        data,
                        filename=filename,
                        mime_type=downloaded.content_type,
                        default_kind=default_kind,
                    )
                finally:
                    downloaded.cleanup()
            except Exception as exc:  # noqa: BLE001 - attachment failure must not drop the admitted text
                logger.warning("[vk] attachment %s download failed: %s", attachment.type, type(exc).__name__)
                descriptions.append(f"[{_attachment_description(attachment)}: загрузка не удалась]")
                continue
            if cached is None:
                descriptions.append(f"[{_attachment_description(attachment)}: неподдерживаемый формат]")
                continue
            paths.append(cached.path)
            media_types.append(cached.media_type)
            voice_flags.append(voice)
            descriptions.append(cached.context_note())
            is_voice = is_voice or voice
        if is_voice:
            voice_media = [
                (path, media_type)
                for path, media_type, voice in zip(paths, media_types, voice_flags, strict=True)
                if voice
            ]
            paths = [item[0] for item in voice_media]
            media_types = [item[1] for item in voice_media]
        return paths, media_types, is_voice, "\n".join(descriptions)

    async def _stop_polling(self) -> None:
        task, self._poll_task = self._poll_task, None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def _close_resources(self, *, release_lock: bool) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None
        if self._storage is not None:
            await self._storage.close()
            self._storage = None
        if release_lock:
            self._release_platform_lock()


def _message_id(payload: object) -> str:
    if isinstance(payload, dict):
        payload = TypeAdapter(dict[str, object]).validate_python(payload).get("message_id")
    if not isinstance(payload, int):
        raise TypeError("VK messages.send returned no message_id")
    return str(payload)


def _attachment_id(kind: str, owner_id: int, object_id: int, access_key: str | None) -> str:
    value = f"{kind}{owner_id}_{object_id}"
    return f"{value}_{access_key}" if access_key else value


def _sniff_mime(path: Path) -> str | None:
    observed: object = filetype.guess_mime(str(path))  # pyright: ignore[reportUnknownMemberType]
    return observed if isinstance(observed, str) else None


def _sniff_extension(path: Path) -> str | None:
    observed: object = filetype.guess_extension(str(path))  # pyright: ignore[reportUnknownMemberType]
    return observed if isinstance(observed, str) else None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


async def _convert_voice_to_ogg(source: Path, destination: Path, timeout_seconds: float) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError("ffmpeg is required for VK voice messages")
    process = await asyncio.create_subprocess_exec(
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-protocol_whitelist",
        "file,pipe,crypto,data",
        "-i",
        str(source),
        "-map",
        "0:a:0",
        "-c:a",
        "libopus",
        str(destination),
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        return_code = await asyncio.wait_for(process.wait(), timeout=timeout_seconds)
    except TimeoutError:
        process.kill()
        await process.wait()
        raise
    if return_code != 0:
        raise ValueError("ffmpeg could not convert VK voice message")


def _api_error_result(exc: VkApiError) -> SendResult:
    mapping = {
        6: ("rate_limited", True),
        7: ("forbidden", False),
        10: ("transient", True),
        14: ("forbidden", False),
        900: ("forbidden", False),
        901: ("forbidden", False),
        914: ("too_long", False),
    }
    kind, retryable = mapping.get(exc.code, ("unknown", False))
    return SendResult(success=False, error=_safe_api_error(exc), retryable=retryable, error_kind=kind)


def _safe_api_error(exc: VkApiError) -> str:
    messages = {
        6: "VK rate limit was reached.",
        7: "VK denied this operation.",
        10: "VK reported a temporary internal error.",
        14: "VK requires an interactive validation that community bots cannot complete.",
        900: "VK denied access to this conversation.",
        901: "VK cannot deliver messages to this user.",
        914: "VK rejected the message because it is too long.",
    }
    return messages.get(exc.code, f"VK rejected the request (error {exc.code}).")


def _partial_result(delivered: list[str], chunks: list[str]) -> SendResult:
    return SendResult(
        success=True,
        message_id=delivered[-1],
        continuation_message_ids=tuple(delivered[:-1]),
        retryable=False,
        raw_response={
            "partial_delivery": {
                "delivered_chunks": len(delivered),
                "total_chunks": len(chunks),
                "missing_tail_sha256": hashlib.sha256("".join(chunks[len(delivered) :]).encode()).hexdigest(),
            }
        },
    )


def _source_prefix_for_rendered(source: str, rendered: str, rendered_end: int) -> str:
    """Map a delivered rendered offset to a conservative literal source prefix."""
    target = max(0, min(rendered_end, len(rendered)))
    source_end = 0
    for tag, source_start, source_stop, rendered_start, rendered_stop in difflib.SequenceMatcher(
        None,
        source,
        rendered,
        autojunk=False,
    ).get_opcodes():
        if rendered_start > target:
            break
        if tag == "delete":
            if rendered_start <= target:
                source_end = source_stop
            continue
        if target >= rendered_stop:
            source_end = source_stop
            continue
        if tag == "equal":
            source_end = source_start + max(0, target - rendered_start)
        elif tag == "replace" and rendered_stop > rendered_start:
            consumed = max(0, target - rendered_start)
            source_width = source_stop - source_start
            rendered_width = rendered_stop - rendered_start
            source_end = source_start + min(source_width, consumed * source_width // rendered_width)
        break
    while source_end < len(source) and source[source_end].isspace():
        source_end += 1
    return source[:source_end]


def _structured_context(reply: Mapping[str, object] | None, forwards: Sequence[Mapping[str, object]]) -> str:
    sections: list[str] = []
    if reply:
        text = str(reply.get("text") or "").strip()
        sender = reply.get("from_id")
        sections.append(f"[Ответ на сообщение от {sender}: {text[:1000]}]")
    for item in forwards[:20]:
        text = str(item.get("text") or "").strip()
        sender = item.get("from_id")
        sections.append(f"[Переслано от {sender}: {text[:1000]}]")
    return "\n".join(sections)


def _attachment_description(attachment: VkAttachment) -> str:
    labels = {
        "photo": "Фотография",
        "doc": "Документ",
        "audio_message": "Голосовое сообщение",
        "audio": "Аудиозапись",
        "video": "Видео",
        "sticker": "Стикер",
        "link": "Ссылка",
        "poll": "Опрос",
        "wall": "Запись на стене",
        "article": "Статья",
    }
    label = labels.get(attachment.type, f"Вложение: {attachment.type}")
    payload = getattr(attachment, attachment.type, None)
    if not isinstance(payload, dict):
        return label
    values = TypeAdapter(dict[str, object]).validate_python(payload)
    title = values.get("title") or values.get("question")
    text = values.get("text")
    url = values.get("url") or values.get("player")
    details = [str(value).strip()[:1000] for value in (title, text, url) if value]
    return f"{label}: {' — '.join(details)}" if details else label


def _geo_context(geo: Mapping[str, object] | None) -> str:
    if not geo:
        return ""
    coordinates = geo.get("coordinates")
    if isinstance(coordinates, dict):
        values = TypeAdapter(dict[str, object]).validate_python(coordinates)
        latitude = values.get("latitude")
        longitude = values.get("longitude")
        if isinstance(latitude, (int, float)) and isinstance(longitude, (int, float)):
            return f"[Геолокация: {latitude:.6f}, {longitude:.6f}]"
    if isinstance(coordinates, str) and len(coordinates) <= MAX_GEO_COORDINATES_LENGTH:
        return f"[Геолокация: {coordinates}]"
    return "[Геолокация без распознанных координат]"


def _attachment_candidate(attachment: VkAttachment) -> tuple[str, str, str, bool] | None:
    if attachment.type == "audio_message" and attachment.audio_message is not None:
        url = attachment.audio_message.link_ogg or attachment.audio_message.link_mp3
        if url:
            return url, "voice.ogg" if attachment.audio_message.link_ogg else "voice.mp3", "audio", True
    if attachment.type == "photo" and attachment.photo is not None and attachment.photo.sizes:
        largest = max(attachment.photo.sizes, key=lambda size: size.width * size.height)
        return largest.url, "photo.jpg", "image", False
    if attachment.type == "doc" and attachment.doc is not None and attachment.doc.url:
        filename = attachment.doc.title
        if attachment.doc.ext and not filename.lower().endswith(f".{attachment.doc.ext.lower()}"):
            filename = f"{filename}.{attachment.doc.ext}"
        return attachment.doc.url, filename, "document", False
    if attachment.type == "audio" and attachment.audio is not None and attachment.audio.url:
        return attachment.audio.url, f"{attachment.audio.title}.mp3", "audio", False
    return None


def _raise_unsupported_failure(code: int) -> Never:
    raise RuntimeError(f"unsupported Long Poll failure code {code}")


def _raise_polling_resources_missing() -> Never:
    raise RuntimeError("VK polling resources are not initialized")


def _is_retryable_send_error(exc: BaseException) -> bool:
    return isinstance(exc, VkApiError) and exc.code in {6, 10}


def _is_retryable_safe_api_error(exc: BaseException) -> bool:
    if isinstance(exc, VkApiError):
        return exc.code in {6, 10}
    if isinstance(exc, VkHttpError):
        return exc.status == HTTP_TOO_MANY_REQUESTS or exc.status >= HTTP_SERVER_ERROR_MIN
    return isinstance(exc, (TimeoutError, aiohttp.ClientConnectionError, aiohttp.ServerDisconnectedError, OSError))


def _is_retryable_poll_error(exc: BaseException) -> bool:
    if isinstance(exc, VkApiError):
        return exc.code in {6, 10}
    if isinstance(exc, VkHttpError):
        return exc.status == HTTP_TOO_MANY_REQUESTS or exc.status >= HTTP_SERVER_ERROR_MIN
    return isinstance(exc, (TimeoutError, aiohttp.ClientConnectionError, aiohttp.ServerDisconnectedError, OSError))


def _log_poll_retry(state: RetryCallState) -> None:
    exception = state.outcome.exception() if state.outcome is not None else None
    logger.warning(
        "[vk] transient Long Poll failure; retry %s in %.2fs (%s)",
        state.attempt_number,
        state.next_action.sleep if state.next_action is not None else 0,
        type(exception).__name__ if exception is not None else "unknown",
    )

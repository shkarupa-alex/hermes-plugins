from __future__ import annotations
import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Never

import aiohttp
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

from hermes_vk_community.client import VkApiClient
from hermes_vk_community.config import PolicyEnvironment, VkSettings, settings_from_platform_config
from hermes_vk_community.errors import VkApiError, VkDeliveryUnknownError
from hermes_vk_community.models import (
    Group,
    GroupsResponse,
    LongPollLease,
    MessageNewObject,
    User,
    VkAttachment,
    VkUpdate,
)
from hermes_vk_community.renderer import PlainVkRenderer, split_message
from hermes_vk_community.storage import InboxRecord, VkStorage

logger = logging.getLogger(__name__)
HTTP_TOO_MANY_REQUESTS = 429
HTTP_SERVER_ERROR_MIN = 500


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
            await self._dispatch_received()
            self._poll_task = asyncio.create_task(self._poll_loop(), name=f"vk-long-poll-{self.settings.group_id}")
            return True  # noqa: TRY300
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
        chunks = split_message(rendered.text, self.settings.max_message_length)
        outbox = await self._storage.prepare_outbox(int(chat_id), chunks, reply_to)
        delivered: list[str] = []
        for chunk, record in zip(chunks, outbox, strict=True):
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
                await self._storage.mark_outbox(record.id, "delivery_unknown", error="request timed out")
                if delivered:
                    return _partial_result(delivered, len(chunks))
                return SendResult(
                    success=False,
                    error="VK delivery timed out after the request may have succeeded",
                    retryable=False,
                    error_kind="unknown",
                    raw_response={"delivery_unknown": True, "outbox_id": record.id},
                )
            except VkApiError as exc:
                state = "partial_delivery" if delivered else "failed"
                await self._storage.mark_outbox(record.id, state, error=str(exc))
                if delivered:
                    return _partial_result(delivered, len(chunks))
                return _api_error_result(exc)
            except Exception as exc:  # noqa: BLE001
                await self._storage.mark_outbox(record.id, "delivery_unknown", error=type(exc).__name__)
                if delivered:
                    return _partial_result(delivered, len(chunks))
                return SendResult(
                    success=False,
                    error="VK delivery failed after the request may have started",
                    retryable=False,
                    error_kind="unknown",
                    raw_response={"delivery_unknown": True, "outbox_id": record.id},
                )
        return SendResult(
            success=True,
            message_id=delivered[-1] if delivered else None,
            continuation_message_ids=tuple(delivered[:-1]),
            retryable=False,
        )

    async def edit_message(
        self,
        chat_id: str,
        message_id: str,
        content: str,
        *,
        finalize: bool = False,
    ) -> SendResult:
        del finalize
        if self._client is None:
            return SendResult(success=False, error="VK adapter is not connected", retryable=True)
        rendered = self._renderer.render_markdown(content)
        chunks = split_message(rendered.text, self.settings.max_message_length)
        if len(chunks) != 1:
            return SendResult(success=False, error="VK edit overflow is not yet supported", error_kind="too_long")
        try:
            await self._client.call(
                "messages.edit",
                {"peer_id": int(chat_id), "message_id": int(message_id), "message": chunks[0]},
            )
        except VkApiError as exc:
            return _api_error_result(exc)
        return SendResult(success=True, message_id=message_id)

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
                        multiplier=self.settings.long_poll.retry_min_seconds,
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
            if message.from_id <= 0 or message.peer_id != message.from_id or sender not in self._allow_from:
                await self._storage.mark_inbox(record.id, "quarantined", "sender is not authorized")
                return
            parsed = self._renderer.parse_incoming(message.text, None)
            media_urls, media_types, is_voice, attachment_text = await self._cache_attachments(message.attachments)
            text = "\n".join(part for part in (parsed.markdown, attachment_text) if part).strip()
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
                metadata={"vk_event_id": update.event_id, "vk_format_data": None},
            )
            await self._storage.mark_inbox(record.id, "dispatched")
            await self.handle_message(event)
        except Exception as exc:  # noqa: BLE001
            await self._storage.mark_inbox(record.id, "quarantined", f"{type(exc).__name__}: {exc}")

    async def _cache_attachments(
        self,
        attachments: list[VkAttachment],
    ) -> tuple[list[str], list[str], bool, str]:
        if self._client is None:
            raise RuntimeError("VK client is not connected")
        paths: list[str] = []
        media_types: list[str] = []
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
                cached = cache_media_bytes(
                    downloaded.data,
                    filename=filename,
                    mime_type=downloaded.content_type,
                    default_kind=default_kind,
                )
            except Exception as exc:  # noqa: BLE001 - attachment failure must not drop the admitted text
                logger.warning("[vk] attachment %s download failed: %s", attachment.type, type(exc).__name__)
                descriptions.append(f"[{_attachment_description(attachment)}: загрузка не удалась]")
                continue
            if cached is None:
                descriptions.append(f"[{_attachment_description(attachment)}: неподдерживаемый формат]")
                continue
            paths.append(cached.path)
            media_types.append(cached.media_type)
            descriptions.append(cached.context_note())
            is_voice = is_voice or voice
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
    return SendResult(success=False, error=str(exc), retryable=retryable, error_kind=kind)


def _partial_result(delivered: list[str], total: int) -> SendResult:
    return SendResult(
        success=True,
        message_id=delivered[-1],
        continuation_message_ids=tuple(delivered[:-1]),
        retryable=False,
        raw_response={
            "partial_delivery": {
                "delivered_chunks": len(delivered),
                "total_chunks": total,
            }
        },
    )


def _attachment_description(attachment: VkAttachment) -> str:
    labels = {
        "photo": "Фотография",
        "doc": "Документ",
        "audio_message": "Голосовое сообщение",
        "audio": "Аудиозапись",
        "video": "Видео",
        "sticker": "Стикер",
        "link": "Ссылка",
    }
    return labels.get(attachment.type, f"Вложение: {attachment.type}")


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


def _is_retryable_poll_error(exc: BaseException) -> bool:
    if isinstance(exc, VkApiError):
        return exc.code in {6, 10}
    if isinstance(exc, aiohttp.ClientResponseError):
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

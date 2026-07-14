from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar, cast
from urllib.parse import urljoin

import aiohttp
from pydantic import TypeAdapter

from hermes_vk_community.config import API_VERSION, MediaSettings
from hermes_vk_community.errors import VkApiError, VkDeliveryUnknownError
from hermes_vk_community.models import LongPollLease, LongPollResponse, VkApiEnvelope
from hermes_vk_community.security import LONG_POLL_SUFFIXES, MEDIA_SUFFIXES, VkPinnedResolver, validate_https_url

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

logger = logging.getLogger(__name__)
T = TypeVar("T")
REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
MAX_MEDIA_REDIRECTS = 3


@dataclass(frozen=True, slots=True)
class DownloadedMedia:
    data: bytes
    content_type: str
    final_url: str


class VkApiClient:
    def __init__(
        self,
        token: str,
        *,
        api_version: str = API_VERSION,
        media: MediaSettings | None = None,
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        self._token = token
        self._api_version = api_version
        self._media = media or MediaSettings()
        self._session = session
        self._owns_session = session is None

    async def open(self) -> None:
        if self._session is not None:
            return
        timeout = aiohttp.ClientTimeout(
            total=self._media.total_timeout_seconds, connect=self._media.connect_timeout_seconds
        )
        self._session = aiohttp.ClientSession(timeout=timeout, trust_env=False)

    async def close(self) -> None:
        if self._session is not None and self._owns_session:
            await self._session.close()
        self._session = None

    async def call(self, method: str, params: Mapping[str, object] | None = None) -> object:
        await self.open()
        if self._session is None:
            raise RuntimeError("HTTP session did not open")
        form = {key: _form_value(value) for key, value in (params or {}).items() if value is not None}
        form.update({"access_token": self._token, "v": self._api_version})
        url = f"https://api.vk.com/method/{method}"
        try:
            async with self._session.post(url, data=form, allow_redirects=False) as response:
                response.raise_for_status()
                payload = await response.json(content_type=None)
        except TimeoutError as exc:
            raise VkDeliveryUnknownError("VK request timed out") from exc
        envelope = VkApiEnvelope[object].model_validate(payload)
        if envelope.error is not None:
            raise VkApiError(envelope.error.error_code, envelope.error.error_msg)
        return envelope.response

    async def get_long_poll_lease(self, group_id: int) -> LongPollLease:
        payload = await self.call("groups.getLongPollServer", {"group_id": group_id})
        return LongPollLease.model_validate(payload)

    async def poll(self, lease: LongPollLease, *, ts: str, wait_seconds: int) -> LongPollResponse:
        validate_https_url(lease.server, suffixes=LONG_POLL_SUFFIXES)
        timeout = aiohttp.ClientTimeout(total=wait_seconds + 10, connect=self._media.connect_timeout_seconds)
        resolver = VkPinnedResolver(LONG_POLL_SUFFIXES)
        connector = aiohttp.TCPConnector(resolver=resolver, use_dns_cache=False)
        try:
            async with (
                aiohttp.ClientSession(
                    connector=connector,
                    timeout=timeout,
                    trust_env=False,
                ) as session,
                session.get(
                    lease.server,
                    params={"act": "a_check", "key": lease.key, "ts": ts, "wait": wait_seconds},
                    allow_redirects=False,
                ) as response,
            ):
                response.raise_for_status()
                return LongPollResponse.model_validate(await response.json(content_type=None))
        finally:
            await resolver.close()

    async def download_media(self, url: str) -> DownloadedMedia:
        current_url = url
        for redirect_count in range(MAX_MEDIA_REDIRECTS + 1):
            validate_https_url(current_url, suffixes=MEDIA_SUFFIXES)
            resolver = VkPinnedResolver(MEDIA_SUFFIXES)
            connector = aiohttp.TCPConnector(resolver=resolver, use_dns_cache=False)
            timeout = aiohttp.ClientTimeout(
                total=self._media.total_timeout_seconds,
                connect=self._media.connect_timeout_seconds,
            )
            try:
                async with (
                    aiohttp.ClientSession(
                        connector=connector,
                        timeout=timeout,
                        trust_env=False,
                    ) as session,
                    session.get(current_url, allow_redirects=False) as response,
                ):
                    if response.status in REDIRECT_STATUSES:
                        if redirect_count == MAX_MEDIA_REDIRECTS:
                            raise ValueError("VK media redirect limit exceeded")
                        location = response.headers.get("Location")
                        if not location:
                            raise ValueError("VK media redirect has no Location")
                        current_url = urljoin(current_url, location)
                        continue
                    response.raise_for_status()
                    declared = response.content_length
                    if declared is not None and declared > self._media.max_download_bytes:
                        raise ValueError("VK media exceeds configured download limit")
                    data = await _read_limited(response, self._media.max_download_bytes)
                    content_type = response.headers.get("Content-Type", "application/octet-stream").split(";", 1)[0]
                    return DownloadedMedia(data=data, content_type=content_type, final_url=current_url)
            finally:
                await resolver.close()
        raise RuntimeError("unreachable VK media redirect state")


def _form_value(value: object) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (list, tuple, set)):
        return ",".join(str(item) for item in cast("Iterable[object]", value))
    return str(value)


def parse_response(payload: object, response_type: type[T]) -> T:
    return TypeAdapter(response_type).validate_python(payload)


async def _read_limited(response: aiohttp.ClientResponse, limit: int) -> bytes:
    content = bytearray()
    async for chunk in response.content.iter_chunked(64 * 1024):
        content.extend(chunk)
        if len(content) > limit:
            raise ValueError("VK media exceeds configured download limit")
    return bytes(content)

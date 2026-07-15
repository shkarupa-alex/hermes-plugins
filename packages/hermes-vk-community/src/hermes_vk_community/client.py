from __future__ import annotations
import asyncio
import logging
import mimetypes
import os
import ssl
import stat
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar, cast
from urllib.parse import urljoin

import aiohttp
from pydantic import TypeAdapter

from hermes_vk_community.config import API_VERSION, MediaSettings
from hermes_vk_community.errors import VkApiError, VkDeliveryUnknownError, VkHttpError
from hermes_vk_community.models import LongPollLease, LongPollResponse, VkApiEnvelope
from hermes_vk_community.security import LONG_POLL_SUFFIXES, MEDIA_SUFFIXES, VkPinnedResolver, validate_https_url

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from aiohttp.abc import AbstractResolver

logger = logging.getLogger(__name__)
T = TypeVar("T")
REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
MAX_MEDIA_REDIRECTS = 3
HTTP_REDIRECT_MIN = 300
HTTP_ERROR_MIN = 400


def _connector(resolver: AbstractResolver) -> aiohttp.TCPConnector:
    context = ssl.create_default_context()
    context.check_hostname = True
    context.verify_mode = ssl.CERT_REQUIRED
    return aiohttp.TCPConnector(resolver=resolver, use_dns_cache=False, ssl=context)


@dataclass(frozen=True, slots=True)
class DownloadedMedia:
    path: Path
    content_type: str
    final_url: str

    def cleanup(self) -> None:
        self.path.unlink(missing_ok=True)


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
                if response.status >= HTTP_ERROR_MIN:
                    raise VkHttpError(response.status, "API")
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
        connector = _connector(resolver)
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
                if response.status >= HTTP_ERROR_MIN:
                    # Never construct ClientResponseError: its URL contains the Long Poll key.
                    raise VkHttpError(response.status, "Long Poll")
                return LongPollResponse.model_validate(await response.json(content_type=None))
        finally:
            await resolver.close()

    async def download_media(self, url: str) -> DownloadedMedia:
        current_url = url
        for redirect_count in range(MAX_MEDIA_REDIRECTS + 1):
            validate_https_url(current_url, suffixes=MEDIA_SUFFIXES)
            resolver = VkPinnedResolver(MEDIA_SUFFIXES)
            connector = _connector(resolver)
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
                    if HTTP_REDIRECT_MIN <= response.status < HTTP_ERROR_MIN:
                        raise VkHttpError(response.status, "media redirect")
                    if response.status >= HTTP_ERROR_MIN:
                        # Access keys can be embedded in media URLs, so keep the URL out of errors.
                        raise VkHttpError(response.status, "media download")
                    declared = response.content_length
                    if declared is not None and declared > self._media.max_download_bytes:
                        raise ValueError("VK media exceeds configured download limit")
                    path = await _stream_limited_to_temp(response, self._media.max_download_bytes)
                    content_type = response.headers.get("Content-Type", "application/octet-stream").split(";", 1)[0]
                    return DownloadedMedia(path=path, content_type=content_type, final_url=current_url)
            finally:
                await resolver.close()
        raise RuntimeError("unreachable VK media redirect state")

    async def upload_file(
        self,
        url: str,
        field: str,
        path: Path,
        *,
        content_type: str | None = None,
        filename: str | None = None,
    ) -> object:
        """Upload one local file to a VK-issued, pinned HTTPS endpoint without redirects."""
        validate_https_url(url, suffixes=MEDIA_SUFFIXES)
        try:
            stat_result = await asyncio.to_thread(path.stat)
        except OSError as exc:
            raise ValueError("VK upload file is missing") from exc
        if not stat.S_ISREG(stat_result.st_mode) or stat_result.st_size > self._media.max_download_bytes:
            raise ValueError("VK upload file is missing or exceeds the configured byte limit")
        resolver = VkPinnedResolver(MEDIA_SUFFIXES)
        connector = _connector(resolver)
        timeout = aiohttp.ClientTimeout(
            total=self._media.total_timeout_seconds, connect=self._media.connect_timeout_seconds
        )
        mime = content_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        try:
            with path.open("rb") as handle:
                form = aiohttp.FormData()
                form.add_field(field, handle, filename=filename or path.name, content_type=mime)
                async with (
                    aiohttp.ClientSession(connector=connector, timeout=timeout, trust_env=False) as session,
                    session.post(url, data=form, allow_redirects=False) as response,
                ):
                    if HTTP_REDIRECT_MIN <= response.status < HTTP_ERROR_MIN:
                        raise VkHttpError(response.status, "upload redirect")
                    if response.status >= HTTP_ERROR_MIN:
                        raise VkHttpError(response.status, "upload")
                    return await response.json(content_type=None)
        except TimeoutError as exc:
            raise VkDeliveryUnknownError("VK upload timed out") from exc
        finally:
            await resolver.close()


def _form_value(value: object) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (list, tuple, set)):
        return ",".join(str(item) for item in cast("Iterable[object]", value))
    return str(value)


def parse_response(payload: object, response_type: type[T]) -> T:
    return TypeAdapter(response_type).validate_python(payload)


async def _stream_limited_to_temp(response: aiohttp.ClientResponse, limit: int) -> Path:
    descriptor, raw_path = tempfile.mkstemp(prefix="hermes-vk-download-")
    path = Path(raw_path)
    size = 0
    try:
        with os.fdopen(descriptor, "wb") as handle:
            async for chunk in response.content.iter_chunked(64 * 1024):
                size += len(chunk)
                if size > limit:
                    raise ValueError("VK media exceeds configured download limit")  # noqa: TRY301
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(raw_path, 0o600)  # noqa: PTH101 - avoids blocking Path operation in async code
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(raw_path)  # noqa: PTH108 - avoids blocking Path operation in async code
        raise
    else:
        return path

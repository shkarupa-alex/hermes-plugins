from __future__ import annotations
import logging
from typing import TYPE_CHECKING, TypeVar, cast

import aiohttp
from pydantic import TypeAdapter

from hermes_vk_community.config import API_VERSION, MediaSettings
from hermes_vk_community.errors import VkApiError, VkDeliveryUnknownError
from hermes_vk_community.models import LongPollLease, LongPollResponse, VkApiEnvelope
from hermes_vk_community.security import LONG_POLL_SUFFIXES, VkPinnedResolver, validate_https_url

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

logger = logging.getLogger(__name__)
T = TypeVar("T")


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


def _form_value(value: object) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (list, tuple, set)):
        return ",".join(str(item) for item in cast("Iterable[object]", value))
    return str(value)


def parse_response(payload: object, response_type: type[T]) -> T:
    return TypeAdapter(response_type).validate_python(payload)

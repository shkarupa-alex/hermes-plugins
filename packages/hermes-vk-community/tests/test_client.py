# pyright: reportPrivateUsage=false
from __future__ import annotations
import stat
from types import SimpleNamespace
from typing import TYPE_CHECKING, Self, cast

import pytest

from hermes_vk_community.client import DownloadedMedia, VkApiClient, _stream_limited_to_temp
from hermes_vk_community.errors import VkSecurityError

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    import aiohttp


class _Content:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def iter_chunked(self, _size: int) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk


@pytest.mark.asyncio
async def test_media_download_streams_to_private_temp_file() -> None:
    response = cast("aiohttp.ClientResponse", SimpleNamespace(content=_Content([b"hello", b" world"])))
    path = await _stream_limited_to_temp(response, 64)
    downloaded = DownloadedMedia(path, "text/plain", "https://example.invalid/media")
    try:
        assert path.read_bytes() == b"hello world"
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    finally:
        downloaded.cleanup()
    assert not path.exists()


@pytest.mark.asyncio
async def test_oversized_media_temp_file_is_removed(monkeypatch: pytest.MonkeyPatch) -> None:
    response = cast("aiohttp.ClientResponse", SimpleNamespace(content=_Content([b"1234", b"5678"])))
    created: list[str] = []

    import hermes_vk_community.client as client_module

    original = client_module.tempfile.mkstemp

    def capture(*, prefix: str = "") -> tuple[int, str]:
        descriptor, path = original(prefix=prefix)
        created.append(path)
        return descriptor, path

    monkeypatch.setattr(client_module.tempfile, "mkstemp", capture)
    with pytest.raises(ValueError, match="exceeds configured download limit"):
        await _stream_limited_to_temp(response, 6)
    assert created
    assert all(not client_module.Path(path).exists() for path in created)


class _RedirectResponse:
    def __init__(self, location: str) -> None:
        self.status = 302
        self.headers = {"Location": location}

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None


class _RedirectSession:
    def __init__(
        self,
        response: _RedirectResponse,
        calls: list[tuple[str, bool]],
    ) -> None:
        self._response = response
        self._calls = calls

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    def get(self, url: str, *, allow_redirects: bool) -> _RedirectResponse:
        self._calls.append((url, allow_redirects))
        return self._response


@pytest.mark.asyncio
async def test_media_redirects_are_manual_proxy_free_and_revalidated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hermes_vk_community.client as client_module

    responses = iter(
        [
            _RedirectResponse("https://next.userapi.com/media"),
            _RedirectResponse("https://attacker.invalid/private"),
        ]
    )
    session_options: list[dict[str, object]] = []
    calls: list[tuple[str, bool]] = []

    def session_factory(*_args: object, **kwargs: object) -> _RedirectSession:
        session_options.append(kwargs)
        return _RedirectSession(next(responses), calls)

    def inert_connector(_resolver: object) -> object:
        return object()

    monkeypatch.setattr(client_module.aiohttp, "ClientSession", session_factory)
    monkeypatch.setattr(client_module, "_connector", inert_connector)
    client = VkApiClient("secret")
    with pytest.raises(VkSecurityError, match="not approved"):
        await client.download_media("https://cdn.userapi.com/start")
    assert calls == [
        ("https://cdn.userapi.com/start", False),
        ("https://next.userapi.com/media", False),
    ]
    assert len(session_options) == 2
    assert all(options["trust_env"] is False for options in session_options)

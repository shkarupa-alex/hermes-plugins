# pyright: reportPrivateUsage=false
from __future__ import annotations
import stat
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from hermes_vk_community.client import DownloadedMedia, _stream_limited_to_temp

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

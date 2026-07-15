import asyncio
import socket

import pytest

from hermes_vk_community.errors import VkSecurityError
from hermes_vk_community.security import (
    MEDIA_SUFFIXES,
    VkPinnedResolver,
    canonical_host,
    host_matches,
    validate_global_address,
    validate_https_url,
)


def test_label_aware_suffix_matching() -> None:
    assert host_matches("lp.vk.com", ("vk.com",))
    assert not host_matches("evilvk.com", ("vk.com",))


def test_live_vk_voice_cdn_is_approved_without_suffix_confusion() -> None:
    assert validate_https_url("https://psv4.vkuserphoto.ru/voice.ogg", suffixes=MEDIA_SUFFIXES) == (
        "psv4.vkuserphoto.ru"
    )
    with pytest.raises(VkSecurityError, match="not approved"):
        validate_https_url("https://psv4.vkuserphoto.ru.attacker.invalid/voice.ogg", suffixes=MEDIA_SUFFIXES)


@pytest.mark.parametrize(
    "url",
    [
        "http://lp.vk.com/path",
        "https://user:pass@lp.vk.com/path",
        "https://127.0.0.1/path",
        "https://vk.com.evil.test/path",
        "https://lp.vk.com.:443/path",
    ],
)
def test_long_poll_url_rejections(url: str) -> None:
    with pytest.raises(VkSecurityError):
        validate_https_url(url, suffixes=("vk.com",))


@pytest.mark.parametrize("address", ["127.0.0.1", "10.0.0.1", "169.254.169.254", "::1", "100.64.0.1"])
def test_non_global_addresses_are_rejected(address: str) -> None:
    with pytest.raises(VkSecurityError):
        validate_global_address(address)


def test_host_canonicalization() -> None:
    assert canonical_host("LP.VK.COM") == "lp.vk.com"


@pytest.mark.parametrize("address", ["::ffff:127.0.0.1", "64:ff9b::7f00:1"])
def test_embedded_private_ipv4_addresses_are_rejected(address: str) -> None:
    with pytest.raises(VkSecurityError):
        validate_global_address(address)


@pytest.mark.asyncio
async def test_pinned_resolver_rejects_a_mixed_public_private_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    loop = asyncio.get_running_loop()

    async def mixed_answers(*_args: object, **_kwargs: object) -> list[tuple[object, ...]]:
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443)),
        ]

    monkeypatch.setattr(loop, "getaddrinfo", mixed_answers)
    with pytest.raises(OSError, match="prohibited address"):
        await VkPinnedResolver(("vk.com",)).resolve("api.vk.com", 443)


@pytest.mark.asyncio
async def test_pinned_resolver_resolves_again_for_each_new_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    loop = asyncio.get_running_loop()
    answers = iter(["8.8.8.8", "1.1.1.1"])
    calls = 0

    async def rotating_answer(*_args: object, **_kwargs: object) -> list[tuple[object, ...]]:
        nonlocal calls
        calls += 1
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (next(answers), 443))]

    monkeypatch.setattr(loop, "getaddrinfo", rotating_answer)
    resolver = VkPinnedResolver(("vk.com",))
    first = await resolver.resolve("api.vk.com", 443)
    second = await resolver.resolve("api.vk.com", 443)
    assert calls == 2
    assert first[0]["host"] == "8.8.8.8"
    assert second[0]["host"] == "1.1.1.1"

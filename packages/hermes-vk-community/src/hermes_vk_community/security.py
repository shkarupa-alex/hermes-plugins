from __future__ import annotations
import asyncio
import ipaddress
import socket
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from aiohttp.abc import AbstractResolver, ResolveResult

from hermes_vk_community.errors import VkSecurityError

if TYPE_CHECKING:
    from collections.abc import Iterable

LONG_POLL_SUFFIXES = ("vk.com", "userapi.com")
MEDIA_SUFFIXES = ("vk.com", "userapi.com", "vk-cdn.net", "vkuser.net")
NAT64_WELL_KNOWN = ipaddress.ip_network("64:ff9b::/96")
CGNAT = ipaddress.ip_network("100.64.0.0/10")
MIN_PRINTABLE_CODEPOINT = 32
MAX_HOST_LENGTH = 253
MAX_LABEL_LENGTH = 63


def canonical_host(host: str) -> str:
    if not host or host.endswith(".") or "%" in host or any(ord(char) < MIN_PRINTABLE_CODEPOINT for char in host):
        raise VkSecurityError("invalid host name")
    if not host.isascii():
        raise VkSecurityError("non-ASCII host names are not allowed")
    normalized = host.lower()
    try:
        ipaddress.ip_address(normalized)
    except ValueError:
        pass
    else:
        raise VkSecurityError("IP literals are not allowed")
    if len(normalized) > MAX_HOST_LENGTH:
        raise VkSecurityError("host name is too long")
    labels = normalized.split(".")
    if any(
        not label or len(label) > MAX_LABEL_LENGTH or label.startswith("-") or label.endswith("-") for label in labels
    ):
        raise VkSecurityError("invalid DNS label")
    if any(not all(char.isalnum() or char == "-" for char in label) for label in labels):
        raise VkSecurityError("invalid DNS character")
    return normalized


def host_matches(host: str, suffixes: Iterable[str]) -> bool:
    return any(host == suffix or host.endswith(f".{suffix}") for suffix in suffixes)


def validate_https_url(url: str, *, suffixes: Iterable[str]) -> str:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.username is not None or parsed.password is not None:
        raise VkSecurityError("URL must use HTTPS without userinfo")
    if parsed.port not in {None, 443}:
        raise VkSecurityError("non-standard HTTPS ports are not allowed")
    host = canonical_host(parsed.hostname or "")
    if not host_matches(host, suffixes):
        raise VkSecurityError(f"host {host!r} is not approved")
    return host


def validate_global_address(value: str) -> str:
    address = ipaddress.ip_address(value)
    embedded: ipaddress.IPv4Address | None = None
    if isinstance(address, ipaddress.IPv6Address):
        embedded = address.ipv4_mapped
        if address in NAT64_WELL_KNOWN:
            embedded = ipaddress.IPv4Address(int(address) & 0xFFFFFFFF)
    if (
        not address.is_global
        or (isinstance(address, ipaddress.IPv4Address) and address in CGNAT)
        or (embedded is not None and (not embedded.is_global or embedded in CGNAT))
    ):
        raise VkSecurityError(f"non-global address rejected: {address}")
    return str(address)


class VkPinnedResolver(AbstractResolver):
    def __init__(self, suffixes: Iterable[str]) -> None:
        self._suffixes = tuple(suffixes)

    async def resolve(
        self,
        host: str,
        port: int = 0,
        family: socket.AddressFamily = socket.AF_UNSPEC,
    ) -> list[ResolveResult]:
        try:
            canonical = canonical_host(host)
        except (ValueError, VkSecurityError) as exc:
            raise OSError("invalid resolver host") from exc
        if not host_matches(canonical, self._suffixes):
            raise OSError(f"host {canonical!r} is not approved")
        loop = asyncio.get_running_loop()
        records = await loop.getaddrinfo(canonical, port, family=family, type=socket.SOCK_STREAM)
        results: list[ResolveResult] = []
        for record_family, _socktype, _proto, _canonname, sockaddr in records:
            try:
                numeric = validate_global_address(str(sockaddr[0]))
            except (ValueError, VkSecurityError) as exc:
                raise OSError(f"host {canonical!r} resolved to a prohibited address") from exc
            results.append(
                ResolveResult(
                    hostname=canonical,
                    host=numeric,
                    port=port,
                    family=record_family,
                    proto=0,
                    flags=0,
                )
            )
        if not results:
            raise OSError(f"host {canonical!r} did not resolve")
        return results

    async def close(self) -> None:
        return None

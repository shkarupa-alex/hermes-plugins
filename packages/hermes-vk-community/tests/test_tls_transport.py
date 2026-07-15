# pyright: reportPrivateUsage=false
from __future__ import annotations
import asyncio
import socket
import ssl
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import aiohttp
import pytest
from aiohttp.abc import AbstractResolver, ResolveResult
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

from hermes_vk_community.client import _connector

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


TEST_HOST = "media.userapi.com"


class _LocalResolver(AbstractResolver):
    async def resolve(
        self,
        host: str,
        port: int = 0,
        family: socket.AddressFamily = socket.AF_UNSPEC,
    ) -> list[ResolveResult]:
        del family
        return [
            ResolveResult(
                hostname=host,
                host="127.0.0.1",
                port=port,
                family=socket.AF_INET,
                proto=0,
                flags=0,
            )
        ]

    async def close(self) -> None:
        return None


def _write_test_certificates(directory: Path) -> tuple[Path, Path, Path]:
    now = datetime.now(UTC)
    ca_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Hermes test CA")])
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), critical=False)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .sign(ca_key, hashes.SHA256())
    )
    server_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    server_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, TEST_HOST)])
    server_cert = (
        x509.CertificateBuilder()
        .subject_name(server_name)
        .issuer_name(ca_name)
        .public_key(server_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(days=1))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName(TEST_HOST)]), critical=False)
        .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(server_key.public_key()), critical=False)
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .sign(ca_key, hashes.SHA256())
    )
    ca_path = directory / "ca.pem"
    cert_path = directory / "server.pem"
    key_path = directory / "server-key.pem"
    ca_path.write_bytes(ca_cert.public_bytes(serialization.Encoding.PEM))
    cert_path.write_bytes(server_cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        server_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    return ca_path, cert_path, key_path


async def _serve_tls(
    cert_path: Path,
    key_path: Path,
    on_sni: Callable[[str | None], None],
) -> asyncio.Server:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert_path, key_path)
    context.set_servername_callback(lambda _socket, name, _context: on_sni(name))

    async def respond(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await reader.read(4096)
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok")
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    return await asyncio.start_server(respond, "127.0.0.1", 0, ssl=context)


@pytest.mark.asyncio
async def test_pinned_ip_preserves_original_host_sni_and_certificate_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ca_path, cert_path, key_path = _write_test_certificates(tmp_path)
    observed_sni: list[str | None] = []
    server = await _serve_tls(cert_path, key_path, observed_sni.append)
    port = server.sockets[0].getsockname()[1]
    create_default_context = ssl.create_default_context
    trusted = create_default_context(cafile=ca_path)
    monkeypatch.setattr("hermes_vk_community.client.ssl.create_default_context", lambda: trusted)
    try:
        async with (
            aiohttp.ClientSession(connector=_connector(_LocalResolver()), trust_env=False) as session,
            session.get(f"https://{TEST_HOST}:{port}/", allow_redirects=False) as response,
        ):
            assert await response.text() == "ok"
        assert observed_sni == [TEST_HOST]

        untrusted = create_default_context()
        monkeypatch.setattr("hermes_vk_community.client.ssl.create_default_context", lambda: untrusted)
        async with aiohttp.ClientSession(connector=_connector(_LocalResolver()), trust_env=False) as session:
            with pytest.raises(aiohttp.ClientConnectorCertificateError):
                await session.get(f"https://{TEST_HOST}:{port}/", allow_redirects=False)
    finally:
        server.close()
        await server.wait_closed()

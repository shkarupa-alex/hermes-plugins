from __future__ import annotations
import asyncio
import hashlib
import json
import secrets
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING

from agent.secret_scope import get_secret
from gateway.status import acquire_scoped_lock, release_scoped_lock
from hermes_cli.config import load_config
from hermes_constants import get_hermes_home
from pydantic import TypeAdapter

from hermes_vk_community.client import VkApiClient
from hermes_vk_community.compat import check_compatibility
from hermes_vk_community.config import API_VERSION, PolicyEnvironment, VkSettings
from hermes_vk_community.errors import VkApiError
from hermes_vk_community.models import (
    CommunityLongPollSettings,
    FormattingProbeArtifact,
    FormattingProbeCase,
    JsonObject,
    VkCapabilityProfile,
)
from hermes_vk_community.setup import (
    _load_group,  # pyright: ignore[reportPrivateUsage]
    interactive_setup,
    validate_long_poll_capabilities,
)
from hermes_vk_community.storage import VkStorage

MIN_PAIRING_TTL = 60
MAX_PAIRING_TTL = 86_400

if TYPE_CHECKING:
    import argparse


def setup_parser(parser: argparse.ArgumentParser) -> None:
    subparsers = parser.add_subparsers(dest="vk_command", required=True)
    subparsers.add_parser("setup", help="Run the interactive VK Community setup wizard")
    subparsers.add_parser("test-auth", help="Validate the scoped community token")
    doctor = subparsers.add_parser("doctor", help="Check Hermes and local VK state")
    doctor.add_argument("--inflight", action="store_true", help="Show ambiguous inbox work")
    doctor.add_argument("--delivery-unknown", action="store_true", help="Show ambiguous outbound sends")
    pair = subparsers.add_parser("pair", help="Create a one-time short-lived pairing code")
    pair.add_argument("--ttl", type=int, help="Override configured TTL in seconds")
    probe = subparsers.add_parser("probe-formatting", help="Run a private-chat formatting experiment")
    probe.add_argument("--peer-id", type=int, required=True)
    probe.add_argument("--output", type=Path, default=Path.cwd() / "vk-formatting-probe.json")


def handle_command(args: argparse.Namespace) -> int:
    command = args.vk_command
    if command == "setup":
        interactive_setup()
        return 0
    if command == "doctor":
        return asyncio.run(_doctor(show_inflight=args.inflight, show_delivery_unknown=args.delivery_unknown))
    if command == "test-auth":
        return asyncio.run(_test_auth())
    if command == "probe-formatting":
        return asyncio.run(_probe_formatting(args.peer_id, args.output))
    if command == "pair":
        return asyncio.run(_create_pairing_code(args.ttl))
    raise ValueError(f"unsupported vk command: {command}")


async def _test_auth() -> int:
    token = get_secret("VK_COMMUNITY_TOKEN")
    if not token:
        return 1
    client = VkApiClient(token)
    try:
        settings = _load_settings()
        await _load_group(client, settings.group_id)
        validate_long_poll_capabilities(
            CommunityLongPollSettings.model_validate(
                await client.call("groups.getLongPollSettings", {"group_id": settings.group_id})
            )
        )
        await client.get_long_poll_lease(settings.group_id)
    except Exception:  # noqa: BLE001
        return 1
    finally:
        await client.close()
    return 0


def _load_settings() -> VkSettings:
    raw = load_config().get("platforms", {}).get("vk", {})
    if not isinstance(raw, dict):
        raise TypeError("platforms.vk is missing")
    return VkSettings.model_validate(raw)


async def _doctor(  # noqa: PLR0915
    *, show_inflight: bool, show_delivery_unknown: bool
) -> int:
    ok = True
    compatible, message = check_compatibility()
    print(f"Hermes:       {message}")
    ok &= compatible
    token = get_secret("VK_COMMUNITY_TOKEN")
    print(f"Token:        {'present in active secret scope' if token else 'missing'}")
    ok &= bool(token)
    conflicts = PolicyEnvironment().conflicts()
    print(f"Authorization:{' conflict: ' + ', '.join(conflicts) if conflicts else ' YAML-only, no conflicts'}")
    ok &= not conflicts
    try:
        settings = _load_settings()
    except Exception as exc:  # noqa: BLE001
        print(f"Configuration: invalid ({exc})")
        return 1
    print(f"Configuration: group {settings.group_id}; {len(settings.allowed_user_ids)} allowed user(s)")
    enabled_plugins = set(load_config().get("plugins", {}).get("enabled", []))
    discovered = "vk-community" in enabled_plugins
    print(f"Discovery:    {'enabled' if discovered else 'installed but not enabled'}")
    ok &= discovered
    acquired, owner = acquire_scoped_lock("vk", str(settings.group_id), metadata={"platform": "vk", "diagnostic": True})
    if acquired:
        release_scoped_lock("vk", str(settings.group_id))
        print("Platform lock:available")
    else:
        owner_pid = owner.get("pid") if isinstance(owner, dict) else None
        print(f"Platform lock:held by live gateway{f' PID {owner_pid}' if owner_pid else ''}")
    storage = VkStorage(settings.resolve_storage_path(Path(get_hermes_home())))
    try:
        await storage.open()
        counts = await storage.counts()
        print("SQLite:      schema and durable state ready")
        print(f"Inbox:       {counts['inbox_dispatched']} ambiguous dispatched row(s)")
        print(f"Outbox:      {counts['outbox_delivery_unknown']} delivery-unknown row(s)")
        print(f"Media:       {counts['media_orphans']} ambiguous upload orphan(s)")
        if show_inflight:
            print(json.dumps(await storage.diagnostic_rows(inbox_state="dispatched"), ensure_ascii=False, indent=2))
        if show_delivery_unknown:
            print(
                json.dumps(
                    await storage.diagnostic_rows(outbox_state="delivery_unknown"),
                    ensure_ascii=False,
                    indent=2,
                )
            )
    except Exception as exc:  # noqa: BLE001
        print(f"SQLite:      failed ({exc})")
        ok = False
    finally:
        await storage.close()
    if token:
        client = VkApiClient(token, api_version=settings.api_version, media=settings.media)
        try:
            group = await _load_group(client, settings.group_id)
            lp = CommunityLongPollSettings.model_validate(
                await client.call("groups.getLongPollSettings", {"group_id": settings.group_id})
            )
            validate_long_poll_capabilities(lp)
            lease = await client.get_long_poll_lease(settings.group_id)
            print(f"VK API:      {group.name}; identity and token scope ready")
            print(f"Long Poll:   ready ({lease.server.split('://', 1)[-1].split('/', 1)[0]})")
        except Exception as exc:  # noqa: BLE001
            print(f"VK API:      failed ({type(exc).__name__}: {exc})")
            ok = False
        finally:
            await client.close()
    capability = files("hermes_vk_community").joinpath("data/vk-capabilities.json")
    try:
        capability_profile = VkCapabilityProfile.model_validate_json(capability.read_text(encoding="utf-8"))
        capability_ready = capability_profile.api_version == settings.api_version
    except (OSError, ValueError):
        capability_ready = False
    print(f"Formatting:  {'plain capability profile present' if capability_ready else 'profile missing'}")
    ok &= capability_ready
    print("Media:       pinned HTTPS download/upload flows available")
    return 0 if ok else 1


async def _create_pairing_code(ttl: int | None) -> int:
    settings = _load_settings()
    if not settings.pairing.enabled:
        print("Pairing is disabled in platforms.vk.pairing.enabled")
        return 1
    effective_ttl = ttl or settings.pairing.code_ttl_seconds
    if not MIN_PAIRING_TTL <= effective_ttl <= MAX_PAIRING_TTL:
        print("Pairing TTL must be between 60 and 86400 seconds")
        return 1
    code = "VK-" + secrets.token_hex(4).upper()
    storage = VkStorage(settings.resolve_storage_path(Path(get_hermes_home())))
    try:
        await storage.open()
        await storage.create_pairing_code(code, effective_ttl)
    finally:
        await storage.close()
    print(f"One-time pairing code (valid {effective_ttl}s): {code}")
    return 0


async def _probe_formatting(peer_id: int, output: Path) -> int:
    token = get_secret("VK_COMMUNITY_TOKEN")
    if not token:
        return 1
    client = VkApiClient(token)
    cases = [
        ("plain", "Обычный текст: Кириллица 😀 & < >"),
        ("html", "<b>Жирный</b> <i>курсив</i> <u>подчёркнутый</u>"),
        ("link", '<a href="https://example.com/">ссылка</a>'),
        ("markdown", "**Жирный Markdown** и *курсив*, `код`, ~~strike~~"),
        ("nested_malformed", "<b>жирный <i>вложенный</b> хвост</i> <broken>"),
        ("quote_list", "> цитата\n\n• первый пункт\n• второй пункт"),
        ("table", "<table><tr><th>A</th><th>Б</th></tr><tr><td>1</td><td>😀</td></tr></table>"),
        ("long_4096", "Я" * 4096),
    ]
    artifact = FormattingProbeArtifact(
        api_version=API_VERSION,
        generated_at=datetime.now(UTC).isoformat(),
        cases=[],
    )
    try:
        for name, message in cases:
            try:
                response = await client.call(
                    "messages.send",
                    {
                        "peer_id": peer_id,
                        "random_id": secrets.randbelow(2_147_483_647) + 1,
                        "message": message,
                        "disable_mentions": True,
                    },
                )
                message_id = _probe_message_id(response)
                readback = await client.call("messages.getById", {"message_ids": message_id})
                text, format_data = _probe_readback(readback)
                artifact.cases.append(
                    FormattingProbeCase(
                        name=name,
                        operation="send+readback",
                        request_message=_bounded_probe_text(message),
                        request_length=len(message),
                        request_sha256=hashlib.sha256(message.encode()).hexdigest(),
                        send_status="accepted",
                        readback_text=_bounded_probe_text(text) if text is not None else None,
                        readback_length=len(text) if text is not None else None,
                        readback_sha256=hashlib.sha256(text.encode()).hexdigest() if text is not None else None,
                        readback_format_data=format_data,
                    )
                )
                if name == "plain":
                    edited = "<b>Редактирование</b> **Markdown**"
                    await client.call(
                        "messages.edit",
                        {"peer_id": peer_id, "message_id": message_id, "message": edited},
                    )
                    edited_readback = await client.call("messages.getById", {"message_ids": message_id})
                    edit_text, edit_format = _probe_readback(edited_readback)
                    artifact.cases.append(
                        FormattingProbeCase(
                            name="edit",
                            operation="edit+readback",
                            request_message=edited,
                            request_length=len(edited),
                            request_sha256=hashlib.sha256(edited.encode()).hexdigest(),
                            send_status="accepted",
                            readback_text=edit_text,
                            readback_length=len(edit_text) if edit_text is not None else None,
                            readback_sha256=(
                                hashlib.sha256(edit_text.encode()).hexdigest() if edit_text is not None else None
                            ),
                            readback_format_data=edit_format,
                        )
                    )
            except VkApiError as exc:
                artifact.cases.append(
                    FormattingProbeCase(
                        name=name,
                        operation="send+readback",
                        request_message=_bounded_probe_text(message),
                        request_length=len(message),
                        request_sha256=hashlib.sha256(message.encode()).hexdigest(),
                        send_status="rejected",
                        error_code=exc.code,
                    )
                )
    finally:
        await client.close()
    await asyncio.to_thread(_write_probe, output, artifact)
    return 0


def _write_probe(output: Path, artifact: FormattingProbeArtifact) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(artifact.model_dump_json(indent=2) + "\n", encoding="utf-8")


def _bounded_probe_text(value: str, limit: int = 1000) -> str:
    return value if len(value) <= limit else f"<redacted generated payload: {len(value)} characters>"


def _probe_message_id(payload: object) -> int:
    if isinstance(payload, dict):
        payload = TypeAdapter(dict[str, object]).validate_python(payload).get("message_id")
    if not isinstance(payload, int):
        raise TypeError("VK formatting probe received no message id")
    return payload


def _probe_readback(payload: object) -> tuple[str | None, JsonObject | None]:
    if not isinstance(payload, dict):
        return None, None
    root = TypeAdapter(dict[str, object]).validate_python(payload)
    raw_items = root.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        return None, None
    items = TypeAdapter(list[dict[str, object]]).validate_python(raw_items)
    text = items[0].get("text")
    raw_format_data = items[0].get("format_data")
    format_data = (
        TypeAdapter(JsonObject).validate_python(raw_format_data) if isinstance(raw_format_data, dict) else None
    )
    return (
        text if isinstance(text, str) else None,
        format_data if isinstance(format_data, dict) else None,
    )

from __future__ import annotations
import asyncio
import hashlib
import json
import secrets
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from agent.secret_scope import get_secret
from gateway.status import acquire_scoped_lock, release_scoped_lock
from hermes_cli.config import load_config
from hermes_constants import get_hermes_home
from pydantic import TypeAdapter

from hermes_vk_community.capabilities import load_capability_profile, rich_capability_ready
from hermes_vk_community.client import VkApiClient
from hermes_vk_community.compat import check_compatibility
from hermes_vk_community.config import API_VERSION, PolicyEnvironment, VkSettings
from hermes_vk_community.errors import VkApiError
from hermes_vk_community.models import (
    CommunityLongPollSettings,
    FormattingProbeArtifact,
    FormattingProbeCase,
    JsonObject,
)
from hermes_vk_community.setup import (
    _load_group,  # pyright: ignore[reportPrivateUsage]
    interactive_setup,
    validate_long_poll_capabilities,
)
from hermes_vk_community.storage import VkStorage

MIN_PAIRING_TTL = 60
MAX_PAIRING_TTL = 86_400


@dataclass(frozen=True, slots=True)
class _FormattingProbeInput:
    name: str
    message: str
    format_data: JsonObject | None = None


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


async def _doctor(  # noqa: C901, PLR0912, PLR0915 - diagnostics report every independent release gate
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
    capability_profile = load_capability_profile()
    capability_ready = rich_capability_ready(
        settings.api_version,
        require_edit=settings.formatting.mode == "rich",
    )
    if settings.formatting.mode == "plain":
        formatting_status = "plain mode (rich profile available)" if capability_ready else "plain mode"
    elif capability_ready and capability_profile is not None:
        formatting_status = f"{capability_profile.profile} ready ({settings.formatting.mode})"
    else:
        formatting_status = f"rich profile unavailable ({settings.formatting.mode})"
    print(f"Formatting:  {formatting_status}")
    if settings.formatting.mode == "rich":
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
    code = "VK-" + secrets.token_hex(16).upper()
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
    cases = _formatting_probe_inputs()
    artifact = FormattingProbeArtifact(
        api_version=API_VERSION,
        generated_at=datetime.now(UTC).isoformat(),
        cases=[],
    )
    try:
        for case in cases:
            name = case.name
            message = case.message
            message_id: int | None = None
            try:
                send_params: dict[str, object] = {
                    "peer_id": peer_id,
                    "random_id": secrets.randbelow(2_147_483_647) + 1,
                    "message": message,
                    "disable_mentions": True,
                }
                if case.format_data is not None:
                    send_params["format_data"] = json.dumps(
                        case.format_data,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                response = await client.call(
                    "messages.send",
                    send_params,
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
                        request_format_data=case.format_data,
                        send_status="accepted",
                        readback_text=_bounded_probe_text(text) if text is not None else None,
                        readback_length=len(text) if text is not None else None,
                        readback_sha256=hashlib.sha256(text.encode()).hexdigest() if text is not None else None,
                        readback_format_data=format_data,
                    )
                )
                if name == "plain":
                    edited = "Жирное редактирование"
                    edit_format_data: JsonObject = {
                        "version": 1,
                        "items": [{"type": "bold", "offset": 0, "length": len("Жирное")}],
                    }
                    await client.call(
                        "messages.edit",
                        {
                            "peer_id": peer_id,
                            "message_id": message_id,
                            "message": edited,
                            "format_data": json.dumps(
                                edit_format_data,
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                        },
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
                            request_format_data=edit_format_data,
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
                        request_format_data=case.format_data,
                        send_status="rejected",
                        error_code=exc.code,
                    )
                )
            finally:
                if message_id is not None:
                    with suppress(Exception):  # best-effort cleanup of visible probe messages
                        await client.call(
                            "messages.delete",
                            {"peer_id": peer_id, "message_ids": message_id, "delete_for_all": True},
                        )
    finally:
        await client.close()
    await asyncio.to_thread(_write_probe, output, artifact)
    return 0


def _formatting_probe_inputs() -> list[_FormattingProbeInput]:
    rich = "Жирный курсив подчёркнутый ссылка"
    rich_items: list[dict[str, object]] = []
    for kind, label in (("bold", "Жирный"), ("italic", "курсив"), ("underline", "подчёркнутый")):
        rich_items.append({"type": kind, "offset": rich.index(label), "length": len(label)})
    link_label = "ссылка"
    rich_items.append(
        {
            "type": "url",
            "offset": rich.index(link_label),
            "length": len(link_label),
            "url": "https://example.com/",
        }
    )
    unicode_text = "😀 Жирный e\u0301"
    bold_offset = unicode_text.index("Жирный")
    underline_offset = unicode_text.index("e\u0301")
    unicode_items: list[dict[str, object]] = [
        {
            "type": "bold",
            "offset": bold_offset,
            "length": len("Жирный"),
        },
        {
            "type": "underline",
            "offset": underline_offset,
            "length": len("e\u0301"),
        },
    ]
    utf16_items: list[dict[str, object]] = [
        {
            "type": "bold",
            "offset": _utf16_units(unicode_text[:bold_offset]),
            "length": _utf16_units("Жирный"),
        },
        {
            "type": "underline",
            "offset": _utf16_units(unicode_text[:underline_offset]),
            "length": _utf16_units("e\u0301"),
        },
    ]
    nested = "Жирный курсив"
    overlap = "частичный overlap"
    return [
        _FormattingProbeInput("plain", "Обычный текст: Кириллица 😀 & < >"),
        _FormattingProbeInput(
            "format_data_basic",
            rich,
            _probe_format_data(rich_items),
        ),
        _FormattingProbeInput(
            "format_data_nested_same_boundary",
            nested,
            _probe_format_data(
                [
                    {"type": "bold", "offset": 0, "length": len(nested)},
                    {"type": "italic", "offset": nested.index("курсив"), "length": len("курсив")},
                ]
            ),
        ),
        _FormattingProbeInput(
            "format_data_partial_overlap",
            overlap,
            _probe_format_data(
                [
                    {"type": "bold", "offset": 0, "length": len("частичный")},
                    {"type": "italic", "offset": 5, "length": len("чный overlap")},
                ]
            ),
        ),
        _FormattingProbeInput(
            "format_data_unicode_codepoints",
            unicode_text,
            _probe_format_data(unicode_items),
        ),
        _FormattingProbeInput(
            "format_data_unicode_utf16",
            unicode_text,
            _probe_format_data(utf16_items),
        ),
        _FormattingProbeInput("html", "<b>Жирный</b> <i>курсив</i> <u>подчёркнутый</u>"),
        _FormattingProbeInput("link", '<a href="https://example.com/">ссылка</a>'),
        _FormattingProbeInput("markdown", "**Жирный Markdown** и *курсив*, `код`, ~~strike~~"),
        _FormattingProbeInput("nested_malformed", "<b>жирный <i>вложенный</b> хвост</i> <broken>"),
        _FormattingProbeInput("quote_list", "> цитата\n\n• первый пункт\n• второй пункт"),
        _FormattingProbeInput(
            "table",
            "<table><tr><th>A</th><th>Б</th></tr><tr><td>1</td><td>😀</td></tr></table>",
        ),
        _FormattingProbeInput("long_4096", "Я" * 4096),
    ]


def _utf16_units(value: str) -> int:
    return len(value.encode("utf-16-le")) // 2


def _probe_format_data(items: list[dict[str, object]]) -> JsonObject:
    return TypeAdapter(JsonObject).validate_python({"version": 1, "items": items})


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

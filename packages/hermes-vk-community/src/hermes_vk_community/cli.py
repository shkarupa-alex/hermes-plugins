from __future__ import annotations
import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agent.secret_scope import get_secret

from hermes_vk_community.client import VkApiClient
from hermes_vk_community.compat import check_compatibility
from hermes_vk_community.config import API_VERSION
from hermes_vk_community.setup import interactive_setup

if TYPE_CHECKING:
    import argparse


def setup_parser(parser: argparse.ArgumentParser) -> None:
    subparsers = parser.add_subparsers(dest="vk_command", required=True)
    subparsers.add_parser("setup", help="Run the interactive VK Community setup wizard")
    subparsers.add_parser("test-auth", help="Validate the scoped community token")
    doctor = subparsers.add_parser("doctor", help="Check Hermes and local VK state")
    doctor.add_argument("--inflight", action="store_true", help="Show ambiguous inbox work")
    doctor.add_argument("--delivery-unknown", action="store_true", help="Show ambiguous outbound sends")
    probe = subparsers.add_parser("probe-formatting", help="Run a private-chat formatting experiment")
    probe.add_argument("--peer-id", type=int, required=True)


def handle_command(args: argparse.Namespace) -> int:
    command = args.vk_command
    if command == "setup":
        interactive_setup()
        return 0
    if command == "doctor":
        compatible, _message = check_compatibility()
        if args.inflight or args.delivery_unknown:
            pass
        return 0 if compatible else 1
    if command == "test-auth":
        return asyncio.run(_test_auth())
    if command == "probe-formatting":
        return asyncio.run(_probe_formatting(args.peer_id))
    raise ValueError(f"unsupported vk command: {command}")


async def _test_auth() -> int:
    token = get_secret("VK_COMMUNITY_TOKEN")
    if not token:
        return 1
    client = VkApiClient(token)
    try:
        await client.call("groups.getById")
    except Exception:  # noqa: BLE001
        return 1
    finally:
        await client.close()
    return 0


async def _probe_formatting(peer_id: int) -> int:
    token = get_secret("VK_COMMUNITY_TOKEN")
    if not token:
        return 1
    client = VkApiClient(token)
    cases = [
        ("html", "<b>Жирный</b> <i>курсив</i> <u>подчёркнутый</u>"),
        ("link", '<a href="https://example.com/">ссылка</a>'),
        ("table", "<table><tr><th>A</th><th>Б</th></tr><tr><td>1</td><td>😀</td></tr></table>"),
    ]
    artifact: dict[str, Any] = {"api_version": API_VERSION, "peer_id": peer_id, "cases": []}
    try:
        for index, (name, message) in enumerate(cases, start=1):
            response = await client.call(
                "messages.send",
                {"peer_id": peer_id, "random_id": index, "message": message, "disable_mentions": True},
            )
            artifact["cases"].append({"name": name, "request_message": message, "send_response": response})
    except Exception:  # noqa: BLE001
        return 1
    finally:
        await client.close()
    target = Path.cwd() / "vk-formatting-probe.json"
    target.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0

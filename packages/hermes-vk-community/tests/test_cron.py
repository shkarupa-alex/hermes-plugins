# pyright: reportPrivateUsage=false
from __future__ import annotations
import asyncio
import importlib
import threading
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
from gateway.delivery import DeliveryRouter, DeliveryTarget
from gateway.platform_registry import PlatformEntry, platform_registry
from gateway.platforms.base import SendResult

from hermes_vk_community import adapter as adapter_module
from hermes_vk_community import plugin as plugin_module
from hermes_vk_community.adapter import VkCommunityAdapter
from hermes_vk_community.compat import supports_cron_delivery
from hermes_vk_community.errors import VkApiError, VkDeliveryUnknownError
from hermes_vk_community.plugin import build_adapter, register, send_standalone
from hermes_vk_community.storage import VkStorage

if TYPE_CHECKING:
    from pathlib import Path


class RegistryContext:
    def register_platform(self, **kwargs: Any) -> None:  # noqa: ANN401 - Hermes registration contract
        platform_registry.register(PlatformEntry(**kwargs))

    def register_cli_command(self, **_kwargs: object) -> None:
        pass


def _config(tmp_path: Path) -> PlatformConfig:
    register(RegistryContext())
    return PlatformConfig(
        enabled=True,
        extra={"group_id": 123, "allowed_user_ids": [456], "storage": {"path": str(tmp_path / "state.sqlite3")}},
    )


@pytest.fixture
def delivery_client(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    if not supports_cron_delivery():
        pytest.skip("VK cron requires the current Hermes no-resend contract")
    events: list[str] = []

    class Client:
        def __init__(self, _token: str, **_kwargs: object) -> None:
            events.append("created")

        async def open(self) -> None:
            events.append("opened")

        async def close(self) -> None:
            events.append("closed")

    async def verify_group(_self: VkCommunityAdapter) -> None:
        events.append("verified")

    def forbidden_lock(*_args: object, **_kwargs: object) -> None:
        pytest.fail("a cron sender must not acquire or release the Long Poll receiver lock")

    monkeypatch.setattr(adapter_module, "VkApiClient", Client)

    def profile_token(_name: str) -> str:
        return "profile-token"

    monkeypatch.setattr(adapter_module, "get_secret", profile_token)
    monkeypatch.setattr(VkCommunityAdapter, "_verify_group", verify_group)
    monkeypatch.setattr(VkCommunityAdapter, "_acquire_platform_lock", forbidden_lock)
    monkeypatch.setattr(VkCommunityAdapter, "_release_platform_lock", forbidden_lock)
    return events


@pytest.mark.asyncio
async def test_standalone_cron_reuses_adapter_send_and_closes_resources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    delivery_client: list[str],
) -> None:
    config = _config(tmp_path)
    sent: list[tuple[str, str]] = []

    async def send(self: VkCommunityAdapter, chat_id: str, content: str) -> SendResult:
        assert self._storage is not None
        assert self._poll_task is None
        sent.append((chat_id, content))
        return SendResult(success=True, message_id="42")

    monkeypatch.setattr(VkCommunityAdapter, "send", send)
    result = await send_standalone(config, "456", "**Отчёт**")
    assert result == {"success": True, "message_id": "42", "media_delivered": False}
    assert sent == [("456", "**Отчёт**")]
    assert delivery_client == ["created", "opened", "verified", "closed"]


@pytest.mark.asyncio
@pytest.mark.parametrize("chat_id", ["999", "2000000001", "-456", "vk.com/id456"])
async def test_standalone_rejects_non_allowlisted_target_before_io(
    tmp_path: Path,
    delivery_client: list[str],
    chat_id: str,
) -> None:
    assert "error" in await send_standalone(_config(tmp_path), chat_id, "Отчёт")
    assert delivery_client == []


@pytest.mark.asyncio
@pytest.mark.parametrize("chat_id", ["456", "999", "2000000001", "-456", "vk.com/id456"])
async def test_live_cron_delivery_enforces_private_allowlist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    chat_id: str,
) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = _config(tmp_path)
    instance = build_adapter(config)
    sent: list[object] = []

    async def transport(_self: VkCommunityAdapter, params: dict[str, object]) -> object:
        sent.append(params["peer_id"])
        return 42

    monkeypatch.setattr(VkCommunityAdapter, "_send_chunk", transport)
    # Use the genuine outbound pipeline and storage; replace only network I/O.
    instance._client = cast("Any", object())
    instance._storage = VkStorage(tmp_path / "state.sqlite3")
    await instance._storage.open()
    try:
        platform = Platform("vk")
        router = DeliveryRouter(GatewayConfig(platforms={platform: config}), adapters={platform: instance})
        target = DeliveryTarget(platform=platform, chat_id=chat_id, is_explicit=True)
        if chat_id == "456":
            await router._deliver_to_platform(target, "Отчёт", {"job_id": "test"})
            assert sent == [456]
        else:
            with pytest.raises(RuntimeError, match="allowed private-message user ID"):
                await router._deliver_to_platform(target, "Отчёт", {"job_id": "test"})
            assert sent == []
    finally:
        await instance._storage.close()


@pytest.mark.asyncio
async def test_live_router_rejects_vk_thread_target_before_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = _config(tmp_path)
    instance = build_adapter(config)

    async def forbidden(*_args: object, **_kwargs: object) -> object:
        pytest.fail("thread target must be rejected before transport")

    monkeypatch.setattr(VkCommunityAdapter, "_send_chunk", forbidden)
    platform = Platform("vk")
    router = DeliveryRouter(GatewayConfig(platforms={platform: config}), adapters={platform: instance})
    with pytest.raises(RuntimeError, match="do not support threads"):
        await router._deliver_to_platform(DeliveryTarget.parse("vk:456:123"), "Отчёт", {"job_id": "test"})


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["send_image", "send_image_file", "send_document", "send_voice"])
async def test_live_media_rejects_thread_routing_before_io(tmp_path: Path, method: str) -> None:
    instance = build_adapter(_config(tmp_path))
    sender = getattr(instance, method)
    result = await sender("456", "unused-media-path", metadata={"thread_id": "123"})
    assert not result.success
    assert result.error == "VK private messages do not support threads"


@pytest.mark.asyncio
async def test_pairing_confirmation_replies_media_and_approval_keep_working(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    config.extra["pairing"] = {"enabled": True}
    instance = build_adapter(config)
    sent: list[dict[str, object]] = []
    replies: list[SendResult] = []

    async def transport(_self: VkCommunityAdapter, params: dict[str, object]) -> object:
        sent.append(params)
        return len(sent)

    async def handle(self: VkCommunityAdapter, _event: object) -> None:
        replies.append(await self.send("789", "Ответ"))

    async def upload(*_args: object, **_kwargs: object) -> str:
        return "doc123_42"

    monkeypatch.setattr(VkCommunityAdapter, "_send_chunk", transport)
    monkeypatch.setattr(VkCommunityAdapter, "handle_message", handle)
    monkeypatch.setattr(VkCommunityAdapter, "_upload_document", upload)
    instance._client = cast("Any", object())
    instance._storage = VkStorage(tmp_path / "state.sqlite3")
    await instance._storage.open()
    try:
        await instance._storage.create_pairing_code("pair-code", 600)
        for message_id, text in enumerate(("pair-code", "Привет"), start=1):
            await instance._storage.admit_batch(
                123,
                [
                    {
                        "type": "message_new",
                        "group_id": 123,
                        "event_id": f"pair-{message_id}",
                        "object": {
                            "message": {
                                "id": message_id,
                                "date": 1,
                                "peer_id": 789,
                                "from_id": 789,
                                "text": text,
                            }
                        },
                    }
                ],
                str(message_id),
            )
        await instance._dispatch_received()
        assert await instance._storage.is_paired(789)
        assert "Устройство привязано" in str(sent[0]["message"])
        assert len(replies) == 1
        assert replies[0].success
        assert (await instance.send_document("789", "report.csv")).success
        assert (await instance.send_exec_approval("789", "command", "session")).success
        assert all(item["peer_id"] == 789 for item in sent)
        count = len(sent)
        assert not (await instance.send("999", "Отчёт")).success
        assert len(sent) == count
    finally:
        await instance._storage.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("enabled", "paired", "chat_id", "allowed"),
    [
        (True, True, "789", True),
        (True, False, "789", False),
        (False, True, "789", False),
        (True, True, "2000000001", False),
    ],
)
async def test_standalone_uses_only_active_private_pairing_grants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    delivery_client: list[str],
    enabled: bool,  # noqa: FBT001 - authorization fixture
    paired: bool,  # noqa: FBT001 - authorization fixture
    chat_id: str,
    allowed: bool,  # noqa: FBT001 - expected outcome
) -> None:
    config = _config(tmp_path)
    config.extra["pairing"] = {"enabled": enabled}
    storage = VkStorage(tmp_path / "state.sqlite3")
    await storage.open()
    try:
        if paired:
            await storage.create_pairing_code("pair-code", 600)
            assert await storage.consume_pairing_code("pair-code", int(chat_id))
    finally:
        await storage.close()

    async def send(_self: VkCommunityAdapter, _chat_id: str, _content: str) -> SendResult:
        return SendResult(success=True, message_id="42")

    monkeypatch.setattr(VkCommunityAdapter, "send", send)
    result = await send_standalone(config, chat_id, "Отчёт")
    assert bool(result.get("success")) == allowed
    assert bool(delivery_client) == allowed
    if allowed:
        assert delivery_client[-1] == "closed"


@pytest.mark.asyncio
@pytest.mark.parametrize("chat_id", ["999", "2000000001"])
@pytest.mark.parametrize("method", ["send_image", "send_image_file", "send_document", "send_voice"])
async def test_live_media_rejects_non_allowlisted_target_before_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    chat_id: str,
    method: str,
) -> None:
    instance = build_adapter(_config(tmp_path))

    async def forbidden(*_args: object, **_kwargs: object) -> object:
        pytest.fail("unauthorized outbound media reached file or network I/O")

    monkeypatch.setattr(VkCommunityAdapter, "_upload_photo", forbidden)
    monkeypatch.setattr(VkCommunityAdapter, "_upload_document", forbidden)
    monkeypatch.setattr(adapter_module, "_convert_voice_to_ogg", forbidden)

    class Client:
        download_media = staticmethod(forbidden)

    instance._client = cast("Any", Client())
    sender = getattr(instance, method)
    result = await sender(chat_id, "unused-media-path")
    assert not result.success
    assert "allowed private-message user ID" in result.error


@pytest.mark.asyncio
@pytest.mark.skipif(not supports_cron_delivery(), reason="standalone cron is not registered on this host")
async def test_hermes_standalone_media_requires_nonempty_report_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools import send_message_tool

    sent: list[str] = []

    async def sender(_config: PlatformConfig, _chat_id: str, message: str, **_kwargs: object) -> dict[str, object]:
        sent.append(message)
        return {"success": True, "media_delivered": True}

    monkeypatch.setattr(plugin_module, "send_standalone", sender)
    config = _config(tmp_path)
    report = tmp_path / "report.csv"
    report.write_text("value\n1\n", encoding="utf-8")
    route = cast("Any", send_message_tool)._send_to_platform
    media = [(str(report), False)]
    result = await route(Platform("vk"), config, "456", "", media_files=media)
    assert "target vk had only media attachments" in result["error"]
    assert sent == []
    result = await route(Platform("vk"), config, "456", "Отчёт", media_files=media)
    assert result["success"]
    assert sent == ["Отчёт"]


@pytest.mark.asyncio
async def test_standalone_cancellation_closes_resources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    delivery_client: list[str],
) -> None:
    async def cancelled(_self: VkCommunityAdapter, _chat_id: str, _content: str) -> SendResult:
        raise asyncio.CancelledError

    monkeypatch.setattr(VkCommunityAdapter, "send", cancelled)
    with pytest.raises(asyncio.CancelledError):
        await send_standalone(_config(tmp_path), "456", "Отчёт")
    assert delivery_client[-1] == "closed"


@pytest.mark.asyncio
@pytest.mark.parametrize(("kind", "failed_tail"), [("text", 2), ("document", 0)])
async def test_cancelled_standalone_cannot_recover_an_orphan_tail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    delivery_client: list[str],
    kind: str,
    failed_tail: int,
) -> None:
    config = _config(tmp_path)
    config.extra["max_message_length"] = 256
    started = asyncio.Event()

    async def interrupted(_self: VkCommunityAdapter, _params: dict[str, object]) -> object:
        started.set()
        await asyncio.Future()

    async def upload(*_args: object, **_kwargs: object) -> str:
        return "doc123_42"

    monkeypatch.setattr(VkCommunityAdapter, "_send_chunk", interrupted)
    monkeypatch.setattr(VkCommunityAdapter, "_upload_document", upload)
    report = tmp_path / "report.txt"
    report.write_text("report", encoding="utf-8")
    task = asyncio.create_task(
        send_standalone(config, "456", "x" * 600)
        if kind == "text"
        else send_standalone(config, "456", "", media_files=[(str(report), False)])
    )
    await asyncio.wait_for(started.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert delivery_client[-1] == "closed"

    recovered: list[dict[str, object]] = []

    async def transport(_self: VkCommunityAdapter, params: dict[str, object]) -> object:
        recovered.append(params)
        return 42

    monkeypatch.setattr(VkCommunityAdapter, "_send_chunk", transport)
    instance = build_adapter(config)
    instance._client = cast("Any", object())
    instance._storage = VkStorage(tmp_path / "state.sqlite3")
    await instance._storage.open()
    try:
        assert len(await instance._storage.diagnostic_rows(outbox_state="delivery_unknown")) == 1
        assert len(await instance._storage.diagnostic_rows(outbox_state="failed")) == failed_tail
        assert await instance._storage.prepared_outbox() == []
        await instance._recover_prepared_outbox()
        assert recovered == []
    finally:
        await instance._storage.close()


@pytest.mark.asyncio
async def test_standalone_reports_partial_delivery_without_scheduling_a_duplicate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    delivery_client: list[str],
) -> None:
    async def partial(_self: VkCommunityAdapter, _chat_id: str, _content: str) -> SendResult:
        return SendResult(success=True, message_id="42", raw_response={"partial_delivery": {"failed_segment": "text"}})

    monkeypatch.setattr(VkCommunityAdapter, "send", partial)
    result = await send_standalone(_config(tmp_path), "456", "Отчёт")
    assert result["success"] is True
    assert result["message_id"] == "42"
    assert result["warnings"]
    assert "error" not in result
    assert delivery_client[-1] == "closed"


@pytest.mark.skipif(not supports_cron_delivery(), reason="old-host VK cron is unsupported")
@pytest.mark.parametrize("failure", ["attachment_result", "attachment_exception", "unknown_text", "partial_text"])
def test_actual_scheduler_retains_standalone_partial_evidence_on_reconnect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, delivery_client: list[str], failure: str
) -> None:
    from cron import scheduler_delivery as delivery

    from tools import send_message_tool

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = _config(tmp_path)
    config.extra["max_message_length"] = 256
    report = tmp_path / "report.csv"
    report.write_text("name,value\na,1\n")
    wire_calls: list[dict[str, object]] = []

    async def transport(_self: VkCommunityAdapter, params: dict[str, object]) -> object:
        wire_calls.append(params)
        if failure == "unknown_text":
            raise VkDeliveryUnknownError("lost response")
        if failure == "partial_text" and len(wire_calls) == 2:
            raise VkApiError(7, "denied")
        return 42

    async def document(_self: VkCommunityAdapter, *_args: object, **_kwargs: object) -> SendResult:
        if failure == "attachment_exception":
            raise OSError("upload failed")
        return SendResult(success=False, error="upload failed", retryable=False)

    def no_live(_platform: object) -> tuple[None, None]:
        return None, None

    def forbidden_queue(*_args: object, **_kwargs: object) -> None:
        pytest.fail("partially visible standalone text was queued for whole-message redelivery")

    def no_mirror(*_args: object, **_kwargs: object) -> None:
        pass

    monkeypatch.setattr(VkCommunityAdapter, "_send_chunk", transport)
    monkeypatch.setattr(VkCommunityAdapter, "send_document", document)
    monkeypatch.setattr(send_message_tool, "_live_adapter", no_live)
    monkeypatch.setattr(delivery, "_queue_for_live_reconnect", forbidden_queue)
    monkeypatch.setattr(delivery, "_maybe_mirror_cron_delivery", no_mirror)
    target = SimpleNamespace(
        job={"id": "standalone-partial"},
        is_relay=False,
        where="vk:456",
        platform=Platform("vk"),
        platform_name="vk",
        pconfig=config,
        chat_id="456",
        thread_id=None,
        mirror_text="",
        live_error="send_path_degraded",
        origin_user_id=None,
        mirror_this_target=False,
    )
    errors: list[str] = []
    cast("Any", delivery)._deliver_standalone(
        cast("Any", target),
        "x" * 600 if failure == "partial_text" else "report",
        [(str(report), False)],
        [],
        errors,
    )
    assert any("delivery warning" in error for error in errors)
    assert len(wire_calls) == (2 if failure == "partial_text" else 1)
    assert delivery_client[-1] == "closed"


@pytest.mark.asyncio
async def test_cancelled_outbox_recovery_terminalizes_its_remaining_invocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = build_adapter(_config(tmp_path))
    instance._client = cast("Any", object())
    instance._storage = VkStorage(tmp_path / "state.sqlite3")
    await instance._storage.open()
    started = asyncio.Event()
    calls: list[dict[str, object]] = []

    async def interrupted(_self: VkCommunityAdapter, params: dict[str, object]) -> object:
        calls.append(params)
        started.set()
        await asyncio.Future()

    async def transport(_self: VkCommunityAdapter, params: dict[str, object]) -> object:
        calls.append(params)
        return 42

    monkeypatch.setattr(VkCommunityAdapter, "_send_chunk", interrupted)
    try:
        await instance._storage.prepare_outbox(456, ["head", "tail 1", "tail 2"], None)
        task = asyncio.create_task(instance._recover_prepared_outbox())
        await asyncio.wait_for(started.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await instance._storage.close()
        await instance._storage.open()
        assert len(await instance._storage.diagnostic_rows(outbox_state="delivery_unknown")) == 1
        assert len(await instance._storage.diagnostic_rows(outbox_state="failed")) == 2
        monkeypatch.setattr(VkCommunityAdapter, "_send_chunk", transport)
        await instance._recover_prepared_outbox()
        assert len(calls) == 1
    finally:
        await instance._storage.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("head_outcome", ["sent", "rejected"])
async def test_gateway_recovery_cannot_take_an_active_standalone_tail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    delivery_client: list[str],
    head_outcome: str,
) -> None:
    config = _config(tmp_path)
    config.extra["max_message_length"] = 256
    receiver = build_adapter(config)
    receiver._client = cast("Any", object())
    receiver._storage = VkStorage(tmp_path / "state.sqlite3")
    started, release = asyncio.Event(), asyncio.Event()
    writer_calls: list[dict[str, object]] = []
    recovery_calls: list[dict[str, object]] = []

    async def transport(self: VkCommunityAdapter, params: dict[str, object]) -> object:
        if self is receiver:
            recovery_calls.append(params)
            return 42
        writer_calls.append(params)
        started.set()
        await release.wait()
        if head_outcome == "rejected":
            raise VkApiError(15, "denied")
        return len(writer_calls)

    monkeypatch.setattr(VkCommunityAdapter, "_send_chunk", transport)
    task = asyncio.create_task(send_standalone(config, "456", "x" * 600))
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        # A gateway restart opens a second connection while the cron request
        # remains in flight. It must not send that invocation's unconfirmed tail.
        await receiver._storage.open()
        assert await receiver._storage.prepared_outbox() == []
        await receiver._recover_prepared_outbox()
        assert recovery_calls == []
        release.set()
        result = await task
        assert bool(result.get("success")) is (head_outcome == "sent")
        if head_outcome == "sent":
            assert len(writer_calls) == 3
            assert len(await receiver._storage.diagnostic_rows(outbox_state="sent")) == 3
        else:
            assert len(writer_calls) == 1
            assert len(await receiver._storage.diagnostic_rows(outbox_state="failed")) == 3
        assert delivery_client[-1] == "closed"
    finally:
        release.set()
        if not task.done():
            await task
        await receiver._storage.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("lane", ["native", "live_cron"])
async def test_concurrent_chat_reply_and_report_share_storage_safely(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lane: str
) -> None:
    if lane == "live_cron" and not supports_cron_delivery():
        pytest.skip("old-host VK cron is unsupported")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = _config(tmp_path)
    instance = build_adapter(config)
    calls: list[str] = []

    async def transport(_self: VkCommunityAdapter, params: dict[str, object]) -> object:
        calls.append(str(params["message"]))
        return len(calls)

    monkeypatch.setattr(VkCommunityAdapter, "_send_chunk", transport)
    instance._client = cast("Any", object())
    instance._storage = VkStorage(tmp_path / "state.sqlite3")
    await instance._storage.open()
    try:
        if lane == "live_cron":
            platform = Platform("vk")
            router = DeliveryRouter(GatewayConfig(platforms={platform: config}), adapters={platform: instance})
            target = DeliveryTarget(platform=platform, chat_id="456", is_explicit=True)
            report = router._deliver_to_platform(target, "scheduled report", {"job_id": "test"})
        else:
            report = instance.send("456", "scheduled report")
        results = await asyncio.gather(report, instance._send_with_retry("456", "ordinary chat reply"))
        for result in results:
            assert isinstance(result, SendResult)
            assert result.success
        assert sorted(calls) == ["ordinary chat reply", "scheduled report"]
        assert len(await instance._storage.diagnostic_rows(outbox_state="sent")) == 2
    finally:
        await instance._storage.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("lane", ["native", "live_router", "gateway_retry"])
async def test_partial_report_is_a_failure_without_duplicate_head(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lane: str,
) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = _config(tmp_path)
    instance = build_adapter(config)
    instance._effective_limit = 256
    chunks: list[str] = []

    async def transport(_self: VkCommunityAdapter, params: dict[str, object]) -> object:
        chunks.append(str(params["message"]))
        if len(chunks) == 1:
            return 42
        raise VkApiError(15, "denied")

    monkeypatch.setattr(VkCommunityAdapter, "_send_chunk", transport)
    instance._client = cast("Any", object())
    instance._storage = VkStorage(tmp_path / "state.sqlite3")
    await instance._storage.open()
    try:
        content = "x" * 600
        if lane == "live_router":
            if not supports_cron_delivery():
                pytest.skip("old-host VK cron is unsupported")
            platform = Platform("vk")
            router = DeliveryRouter(GatewayConfig(platforms={platform: config}), adapters={platform: instance})
            target = DeliveryTarget(platform=platform, chat_id="456", is_explicit=True)
            with pytest.raises(RuntimeError, match="partially delivered"):
                await router._deliver_to_platform(target, content, {"job_id": "test"})
        else:
            result = (
                await instance._send_with_retry("456", content, max_retries=0)
                if lane == "gateway_retry"
                else await instance.send("456", content)
            )
            assert result.success is (not supports_cron_delivery())
            assert not result.retryable
            assert result.message_id == "42"
            raw = cast("dict[str, Any]", result.raw_response)
            assert raw["partial_overflow"] is True
            assert raw["delivered_chunks"] == 1
            assert raw["partial_delivery"]["total_chunks"] == 3
            # Current Hermes' cron confirmation must reject the native result.
            try:
                delivery = importlib.import_module("cron.scheduler_delivery")
            except ImportError:
                delivery = importlib.import_module("cron.scheduler")
            confirm = getattr(delivery, "_confirm_adapter_delivery", None)
            if confirm is not None:
                assert bool(confirm(result)) is (not supports_cron_delivery())
        assert len(chunks) == 2  # no plain-text fallback or whole-report retry
        assert len(await instance._storage.prepared_outbox()) == 0
    finally:
        await instance._storage.close()


@pytest.mark.asyncio
async def test_standalone_rejects_host_without_no_resend_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(plugin_module, "supports_cron_delivery", lambda: False)

    def forbidden(_config: PlatformConfig) -> VkCommunityAdapter:
        pytest.fail("unsupported host must not create a cron sender")

    monkeypatch.setattr(plugin_module, "build_adapter", forbidden)
    result = await send_standalone(_config(tmp_path), "456", "Отчёт")
    assert "requires a current Hermes Git host" in str(result["error"])


@pytest.mark.skipif(not supports_cron_delivery(), reason="old-host VK cron is unsupported")
def test_live_cron_ambiguous_first_request_never_falls_back_to_new_random_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    delivery_client: list[str],
) -> None:
    from cron import scheduler_delivery as delivery

    from tools import send_message_tool

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = _config(tmp_path)
    gateway = GatewayConfig(platforms={Platform("vk"): config})
    instance = build_adapter(config)
    calls: list[dict[str, object]] = []

    async def transport(_self: VkCommunityAdapter, params: dict[str, object]) -> object:
        calls.append(params)
        if len(calls) == 1:
            raise VkDeliveryUnknownError("VK accepted the report, but its response was lost")
        return 42

    monkeypatch.setattr(VkCommunityAdapter, "_send_chunk", transport)
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: gateway)
    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda: None)

    # If a fallback were attempted, force it into the registered standalone
    # sender, so the genuine durable pipeline would allocate another random_id.
    def no_live_adapter(_platform: object) -> tuple[None, None]:
        return None, None

    monkeypatch.setattr(send_message_tool, "_live_adapter", no_live_adapter)
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()

    async def setup() -> None:
        instance._storage = VkStorage(tmp_path / "state.sqlite3")
        await instance._storage.open()
        instance._client = adapter_module.VkApiClient("profile-token")
        await instance._client.open()

    asyncio.run_coroutine_threadsafe(setup(), loop).result(timeout=5)
    try:
        result = cast("Any", delivery)._deliver_result(
            {"id": "test", "deliver": "vk:456"}, "Отчёт", adapters={Platform("vk"): instance}, loop=loop
        )
        assert result
        assert "may have succeeded" in result
        assert len(calls) == 1
        assert "random_id" in calls[0]
        # No second adapter/client was created by the standalone fallback.
        assert delivery_client == ["created", "opened"]
    finally:
        asyncio.run_coroutine_threadsafe(instance._close_resources(release_lock=False), loop).result(timeout=5)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()


def test_cron_lists_vk_and_resolves_home_and_explicit_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gateway import config as gateway_config

    # Hermes moved delivery helpers into a module but retains scheduler aliases.
    try:
        delivery = cast("Any", importlib.import_module("cron.scheduler_delivery"))
    except ImportError:
        delivery = cast("Any", importlib.import_module("cron.scheduler"))
    config = _config(tmp_path)
    home = HomeChannel(platform=Platform("vk"), chat_id="456", name="VK reports")

    class GatewayConfig:
        def get_connected_platforms(self) -> list[Platform]:
            return [Platform("vk")]

        def get_home_channel(self, platform: Platform) -> HomeChannel | None:
            return home if str(platform.value) == "vk" else None

    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda: None)
    monkeypatch.setattr(gateway_config, "load_gateway_config", GatewayConfig)

    def no_env(*_args: object, **_kwargs: object) -> str:
        return ""

    if hasattr(delivery, "_get_config_home_channel"):
        monkeypatch.setattr(delivery, "_home_env_lookup", no_env)
    else:
        # Old Hermes resolves bare platforms only through the env mirror.
        monkeypatch.setenv("VK_HOME_CHANNEL", "456")
    targets = delivery.cron_delivery_targets()
    if not supports_cron_delivery():
        assert all(target["id"] != "vk" for target in targets)
        assert isinstance(build_adapter(config), VkCommunityAdapter)
        return
    assert {"id": "vk", "name": "Vk", "home_target_set": True, "home_env_var": "VK_HOME_CHANNEL"} in targets
    for target in ("vk", "vk:456"):
        resolved = delivery._resolve_delivery_targets({"deliver": target})
        assert len(resolved) == 1
        assert resolved[0]["platform"] == "vk"
        assert resolved[0]["chat_id"] == "456"
        assert resolved[0]["thread_id"] is None
    assert isinstance(build_adapter(config), VkCommunityAdapter)


@pytest.mark.asyncio
async def test_exec_approval_accepts_current_hermes_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = build_adapter(_config(tmp_path))
    captured: list[tuple[str, str, str, str]] = []

    async def keyboard(
        _self: VkCommunityAdapter,
        _chat_id: str,
        _body: str,
        buttons: list[tuple[str, str, str, str]],
        _session: str,
        _metadata: object,
    ) -> SendResult:
        captured.extend(buttons)
        return SendResult(success=True, message_id="42")

    monkeypatch.setattr(VkCommunityAdapter, "_send_keyboard", keyboard)
    assert instance.supports_exec_approval_buttons()
    result = await instance.send_exec_approval(
        "456",
        "command",
        "session",
        None,
        allow_permanent=False,
        allow_session=False,
        smart_denied=True,
    )
    assert result.success
    assert [row[2] for row in captured] == ["once", "deny"]

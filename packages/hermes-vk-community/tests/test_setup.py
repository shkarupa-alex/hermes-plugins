from __future__ import annotations
from typing import Any

import pytest

from hermes_vk_community import setup
from hermes_vk_community.models import CommunityLongPollSettings
from hermes_vk_community.setup import (
    VkSetupResult,
    configure_private_community,
    interactive_setup,
    numeric_id_from_reference,
    screen_name_from_reference,
    validate_long_poll_capabilities,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("https://vk.com/club240186772", "club240186772"),
        ("https://m.vk.com/shkarupa.alex?from=groups", "shkarupa.alex"),
        ("@shkarupa.alex", "shkarupa.alex"),
        ("7750207", "7750207"),
    ],
)
def test_screen_name_accepts_vk_links_aliases_and_ids(value: str, expected: str) -> None:
    assert screen_name_from_reference(value) == expected


def test_screen_name_rejects_non_vk_links() -> None:
    with pytest.raises(ValueError, match=r"vk\.com"):
        screen_name_from_reference("https://example.com/id1")


def test_numeric_id_understands_vk_canonical_prefixes() -> None:
    assert numeric_id_from_reference("https://vk.com/club240186772", prefixes=("club", "public", "event")) == 240186772
    assert numeric_id_from_reference("https://vk.com/id7750207", prefixes=("id",)) == 7750207
    assert numeric_id_from_reference("https://vk.com/shkarupa.alex", prefixes=("id",)) is None


@pytest.mark.asyncio
async def test_custom_community_link_is_resolved_to_group_id() -> None:
    class ClientStub:
        async def call(self, method: str, params: dict[str, object]) -> object:
            assert method == "utils.resolveScreenName"
            assert params == {"screen_name": "batchio"}
            return {"object_id": 240186772, "type": "group"}

    assert (
        await setup._resolve_object(  # pyright: ignore[reportPrivateUsage]
            ClientStub(),  # type: ignore[arg-type]
            "https://vk.com/batchio",
            expected_type="group",
        )
        == 240186772
    )


@pytest.mark.parametrize(
    ("payload", "error"),
    [
        ({"is_enabled": False, "api_version": "5.199", "events": {"message_new": 1}}, "disabled"),
        ({"is_enabled": True, "api_version": "5.131", "events": {"message_new": 1}}, "version"),
        ({"is_enabled": True, "api_version": "5.199", "events": {"message_new": 0}}, "message_new"),
    ],
)
def test_long_poll_validation_requires_incoming_message_events(payload: dict[str, object], error: str) -> None:
    with pytest.raises(ValueError, match=error):
        validate_long_poll_capabilities(CommunityLongPollSettings.model_validate(payload))


def test_long_poll_validation_accepts_required_capabilities() -> None:
    validate_long_poll_capabilities(
        CommunityLongPollSettings.model_validate(
            {"is_enabled": True, "api_version": "5.199", "events": {"message_new": 1}},
        ),
    )


@pytest.mark.asyncio
async def test_private_community_configuration_uses_minimal_vk_settings() -> None:
    class ClientSpy:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, object]]] = []

        async def call(self, method: str, params: dict[str, object]) -> object:
            self.calls.append((method, params))
            return 1

    client = ClientSpy()
    await configure_private_community(client, 240186772)
    assert client.calls == [
        ("groups.edit", {"group_id": 240186772, "access": 2, "messages": True}),
        (
            "groups.setLongPollSettings",
            {"group_id": 240186772, "enabled": True, "api_version": "5.199", "message_new": True},
        ),
    ]


def test_write_setup_config_uses_non_secret_yaml_fields_only() -> None:
    writes: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def writer(*args: object, **kwargs: object) -> None:
        writes.append((args, kwargs))

    setup.write_setup_config(
        writer,
        VkSetupResult(group_id=240186772, group_name="BatchIO", allowed_user_ids=[7750207]),
    )

    assert writes == [
        (("vk", "enabled", True), {"raw": True}),
        (("vk", "group_id", 240186772), {"raw": True}),
        (("vk", "allowed_user_ids", [7750207]), {"raw": True}),
        (("vk", "typing_indicator", True), {"raw": True}),
    ]


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"platforms": {}},
        {"platforms": {"vk": {"enabled": True}}},
    ],
)
def test_saved_configuration_rejects_missing_required_fields(
    monkeypatch: pytest.MonkeyPatch,
    config: dict[str, object],
) -> None:
    monkeypatch.setattr(setup, "read_raw_config", lambda: config)
    assert setup.saved_configuration_is_valid() is False


def test_saved_configuration_accepts_complete_vk_block(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        setup,
        "read_raw_config",
        lambda: {
            "platforms": {
                "vk": {
                    "enabled": True,
                    "group_id": 240186772,
                    "allowed_user_ids": [7750207],
                },
            },
        },
    )
    assert setup.saved_configuration_is_valid() is True


def test_interactive_setup_saves_token_in_profile_env_and_ids_in_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    prompts = iter(["secret-token", "https://vk.com/club240186772", "https://vk.com/shkarupa.alex"])
    env_writes: list[tuple[str, str]] = []
    config_writes: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def get_env_value_stub(_name: str) -> None:
        return None

    def save_env_value_stub(key: str, value: str) -> None:
        env_writes.append((key, value))

    def write_config_stub(*args: Any, **kwargs: Any) -> None:  # noqa: ANN401 - mirrors generic Hermes writer
        config_writes.append((args, kwargs))

    def prompt_stub(*_args: Any, **_kwargs: Any) -> str:  # noqa: ANN401 - mirrors generic Hermes prompt
        return next(prompts)

    def output_stub(_message: str) -> None:
        return None

    def yes_stub(*_args: Any, **_kwargs: Any) -> bool:  # noqa: ANN401 - mirrors generic Hermes prompt
        return True

    monkeypatch.setattr(setup, "get_env_value", get_env_value_stub)
    monkeypatch.setattr(setup, "save_env_value", save_env_value_stub)
    monkeypatch.setattr(setup, "write_platform_config_field", write_config_stub)
    monkeypatch.setattr(setup, "prompt", prompt_stub)
    monkeypatch.setattr(setup, "print_error", output_stub)
    monkeypatch.setattr(setup, "print_header", output_stub)
    monkeypatch.setattr(setup, "print_info", output_stub)
    monkeypatch.setattr(setup, "print_success", output_stub)
    monkeypatch.setattr(setup, "prompt_yes_no", yes_stub)

    async def fake_inspect(
        token: str,
        group_ref: str,
        user_refs: list[str],
        *,
        configure_community: bool = False,
    ) -> VkSetupResult:
        assert token == "secret-token"  # noqa: S105 - inert test fixture value
        assert group_ref == "https://vk.com/club240186772"
        assert user_refs == ["https://vk.com/shkarupa.alex"]
        assert configure_community is True
        return VkSetupResult(group_id=240186772, group_name="BatchIO", allowed_user_ids=[7750207])

    monkeypatch.setattr(setup, "inspect_vk_setup", fake_inspect)

    interactive_setup()

    assert env_writes == [("VK_COMMUNITY_TOKEN", "secret-token")]
    assert all("secret-token" not in repr(write) for write in config_writes)
    assert (("vk", "group_id", 240186772), {"raw": True}) in config_writes
    assert (("vk", "allowed_user_ids", [7750207]), {"raw": True}) in config_writes


def test_interactive_setup_repairs_incomplete_config_even_when_token_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts = iter(["", "https://vk.com/club240186772", "https://vk.com/shkarupa.alex"])
    yes_no_prompts: list[str] = []
    config_writes: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def get_env_value_stub(_name: str) -> str:
        return "existing-token"

    def save_env_value_stub(_key: str, _value: str) -> None:
        return None

    def write_config_stub(*args: Any, **kwargs: Any) -> None:  # noqa: ANN401 - mirrors generic Hermes writer
        config_writes.append((args, kwargs))

    def prompt_stub(*_args: Any, **_kwargs: Any) -> str:  # noqa: ANN401 - mirrors generic Hermes prompt
        return next(prompts)

    def output_stub(_message: str) -> None:
        return None

    monkeypatch.setattr(setup, "get_env_value", get_env_value_stub)
    monkeypatch.setattr(setup, "saved_configuration_is_valid", lambda: False)
    monkeypatch.setattr(setup, "save_env_value", save_env_value_stub)
    monkeypatch.setattr(setup, "write_platform_config_field", write_config_stub)
    monkeypatch.setattr(setup, "prompt", prompt_stub)
    monkeypatch.setattr(setup, "print_error", output_stub)
    monkeypatch.setattr(setup, "print_header", output_stub)
    monkeypatch.setattr(setup, "print_info", output_stub)
    monkeypatch.setattr(setup, "print_success", output_stub)

    def yes_stub(message: str, *, default: bool) -> bool:
        yes_no_prompts.append(message)
        return default

    monkeypatch.setattr(setup, "prompt_yes_no", yes_stub)

    async def fake_inspect(
        token: str,
        group_ref: str,
        user_refs: list[str],
        *,
        configure_community: bool = False,
    ) -> VkSetupResult:
        assert token == "existing-token"  # noqa: S105 - inert test fixture value
        assert group_ref == "https://vk.com/club240186772"
        assert user_refs == ["https://vk.com/shkarupa.alex"]
        assert configure_community is True
        return VkSetupResult(group_id=240186772, group_name="BatchIO", allowed_user_ids=[7750207])

    monkeypatch.setattr(setup, "inspect_vk_setup", fake_inspect)

    interactive_setup()

    assert "Reconfigure VK Community?" not in yes_no_prompts
    assert (("vk", "group_id", 240186772), {"raw": True}) in config_writes

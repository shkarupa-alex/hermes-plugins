from __future__ import annotations
from typing import Any

import pytest

from hermes_vk_community import setup
from hermes_vk_community.setup import (
    VkSetupResult,
    interactive_setup,
    numeric_id_from_reference,
    screen_name_from_reference,
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

    monkeypatch.setattr(setup, "get_env_value", get_env_value_stub)
    monkeypatch.setattr(setup, "save_env_value", save_env_value_stub)
    monkeypatch.setattr(setup, "write_platform_config_field", write_config_stub)
    monkeypatch.setattr(setup, "prompt", prompt_stub)
    monkeypatch.setattr(setup, "print_error", output_stub)
    monkeypatch.setattr(setup, "print_header", output_stub)
    monkeypatch.setattr(setup, "print_info", output_stub)
    monkeypatch.setattr(setup, "print_success", output_stub)

    async def fake_inspect(token: str, group_ref: str, user_refs: list[str]) -> VkSetupResult:
        assert token == "secret-token"  # noqa: S105 - inert test fixture value
        assert group_ref == "https://vk.com/club240186772"
        assert user_refs == ["https://vk.com/shkarupa.alex"]
        return VkSetupResult(group_id=240186772, group_name="BatchIO", allowed_user_ids=[7750207])

    monkeypatch.setattr(setup, "inspect_vk_setup", fake_inspect)

    interactive_setup()

    assert env_writes == [("VK_COMMUNITY_TOKEN", "secret-token")]
    assert all("secret-token" not in repr(write) for write in config_writes)
    assert (("vk", "group_id", 240186772), {"raw": True}) in config_writes
    assert (("vk", "allowed_user_ids", [7750207]), {"raw": True}) in config_writes

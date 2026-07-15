from __future__ import annotations
import inspect
from dataclasses import dataclass
from typing import Any

import pytest

from hermes_vk_community import adapter
from hermes_vk_community.capabilities import rich_capability_ready
from hermes_vk_community.config import PolicyEnvironment, apply_yaml_config, settings_from_platform_config


@dataclass
class PlatformConfigStub:
    extra: dict[str, Any]


def test_minimal_config_round_trip() -> None:
    result = apply_yaml_config(
        {},
        {
            "enabled": True,
            "group_id": 123,
            "allowed_user_ids": [456],
            "typing_indicator": True,
        },
    )
    assert result["_vk_validation_errors"] == []
    assert result["allow_from"] == ["456"]
    settings = settings_from_platform_config(PlatformConfigStub(result))
    assert settings.group_id == 123
    assert settings.dm_policy == "allowlist"
    assert settings.group_policy == "deny"


def test_unknown_key_is_a_bounded_validation_error() -> None:
    result = apply_yaml_config({}, {"group_id": 1, "allowed_user_ids": [2], "mystery": True})
    assert len(result["_vk_validation_errors"]) == 1
    assert result["_vk_validation_errors"][0].startswith("mystery:")


@pytest.mark.parametrize("key", ["VK_COMMUNITY_TOKEN", "access_token_env"])
def test_token_is_rejected_in_yaml(key: str) -> None:
    result = apply_yaml_config({}, {"group_id": 1, "allowed_user_ids": [2], key: "secret"})
    assert result["_vk_validation_errors"]
    assert "only in the profile .env" in result["_vk_validation_errors"][0]


def test_different_allow_from_is_rejected() -> None:
    result = apply_yaml_config(
        {},
        {"group_id": 1, "allowed_user_ids": [2], "allow_from": ["3"]},
    )
    assert result["_vk_validation_errors"]


def test_policy_environment_reports_only_active_conflicts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GATEWAY_ALLOWED_USERS", "1")
    monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "false")
    monkeypatch.setenv("VK_ALLOW_ALL_USERS", "0")
    assert PolicyEnvironment().conflicts() == ["GATEWAY_ALLOWED_USERS", "VK_ALLOW_ALL_USERS"]


def test_live_certified_profile_enables_rich_mode() -> None:
    result = apply_yaml_config(
        {},
        {"group_id": 1, "allowed_user_ids": [2], "formatting": {"mode": "rich"}},
    )
    assert result["_vk_validation_errors"] == []
    assert result["formatting"]["mode"] == "rich"
    assert result["formatting"]["table_style"] == "jpeg"
    assert rich_capability_ready("5.199")


def test_runtime_never_reads_vk_token_with_os_getenv() -> None:
    assert 'os.getenv("VK_COMMUNITY_TOKEN")' not in inspect.getsource(adapter)

from __future__ import annotations
import copy
from pathlib import Path  # noqa: TC003 - Pydantic resolves this annotation at runtime
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, ValidationError, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

API_VERSION = "5.199"
MAX_VALIDATION_ERRORS = 20


class LongPollSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    wait_seconds: int = Field(default=25, ge=1, le=90)
    retry_min_seconds: float = Field(default=1, gt=0, le=60)
    retry_max_seconds: float = Field(default=60, gt=0, le=600)

    @model_validator(mode="after")
    def validate_retry_range(self) -> LongPollSettings:
        if self.retry_max_seconds < self.retry_min_seconds:
            raise ValueError("retry_max_seconds must be >= retry_min_seconds")
        return self


class StorageSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: Path | None = None


class MediaSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_download_bytes: int = Field(default=52_428_800, ge=1, le=536_870_912)
    connect_timeout_seconds: float = Field(default=10, gt=0, le=120)
    total_timeout_seconds: float = Field(default=120, gt=0, le=3600)


class FormattingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["auto", "rich", "plain"] = "auto"
    fallback: Literal["plain"] = "plain"
    disable_mentions: bool = True
    parse_link_previews: bool = True
    table_style: Literal["records"] = "records"


class StreamingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    update_interval_seconds: float = Field(default=1.5, ge=0.5, le=30)


class TypingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    refresh_seconds: float = Field(default=4, ge=1, le=10)
    failure_cooldown_seconds: float = Field(default=30, ge=1, le=600)


class PairingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    code_ttl_seconds: int = Field(default=600, ge=60, le=86_400)


class VkSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    group_id: PositiveInt
    api_version: str = API_VERSION
    allowed_user_ids: list[PositiveInt] = Field(min_length=1)
    allow_from: list[str] | None = None
    dm_policy: Literal["allowlist"] = "allowlist"
    group_policy: Literal["deny"] = "deny"
    pairing: PairingSettings = Field(default_factory=PairingSettings)
    long_poll: LongPollSettings = Field(default_factory=LongPollSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    media: MediaSettings = Field(default_factory=MediaSettings)
    formatting: FormattingSettings = Field(default_factory=FormattingSettings)
    streaming: StreamingSettings = Field(default_factory=StreamingSettings)
    typing_indicator: bool = True
    typing: TypingSettings = Field(default_factory=TypingSettings)
    max_message_length: int = Field(default=4096, ge=256, le=4096)

    @model_validator(mode="after")
    def normalize_allowlist(self) -> VkSettings:
        normalized = [str(value) for value in self.allowed_user_ids]
        if self.allow_from is not None and self.allow_from != normalized:
            raise ValueError("allow_from must exactly match allowed_user_ids")
        self.allow_from = normalized
        if self.api_version != API_VERSION:
            raise ValueError(f"api_version must be {API_VERSION}")
        if self.formatting.mode == "rich":
            raise ValueError("formatting.mode=rich requires a committed live VK capability profile")
        return self

    def resolve_storage_path(self, hermes_home: Path) -> Path:
        configured = self.storage.path
        if configured is None:
            return hermes_home / "vk-community" / str(self.group_id) / "state.sqlite3"
        return configured if configured.is_absolute() else hermes_home / configured


class PolicyEnvironment(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore", case_sensitive=True)

    GATEWAY_ALLOWED_USERS: str | None = None
    GATEWAY_ALLOW_ALL_USERS: str | None = None
    VK_ALLOWED_USERS: str | None = None
    VK_ALLOW_ALL_USERS: str | None = None

    def conflicts(self) -> list[str]:
        names: list[str] = []
        if self.GATEWAY_ALLOWED_USERS and self.GATEWAY_ALLOWED_USERS.strip():
            names.append("GATEWAY_ALLOWED_USERS")
        if self.GATEWAY_ALLOW_ALL_USERS and self.GATEWAY_ALLOW_ALL_USERS.strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            names.append("GATEWAY_ALLOW_ALL_USERS")
        if self.VK_ALLOWED_USERS and self.VK_ALLOWED_USERS.strip():
            names.append("VK_ALLOWED_USERS")
        if self.VK_ALLOW_ALL_USERS and self.VK_ALLOW_ALL_USERS.strip():
            names.append("VK_ALLOW_ALL_USERS")
        return names


def _errors(exc: ValidationError) -> list[str]:
    return [
        f"{'.'.join(str(item) for item in error['loc'])}: {error['msg']}"
        for error in exc.errors(include_url=False)[:MAX_VALIDATION_ERRORS]
    ]


def apply_yaml_config(_yaml_cfg: dict[str, Any], platform_cfg: dict[str, Any]) -> dict[str, Any]:
    raw = copy.deepcopy(platform_cfg)
    if "VK_COMMUNITY_TOKEN" in raw or "access_token_env" in raw:
        return {"_vk_validation_errors": ["VK token configuration is allowed only in the profile .env"]}
    try:
        settings = VkSettings.model_validate(raw)
    except ValidationError as exc:
        return {"_vk_validation_errors": _errors(exc)}
    result = settings.model_dump(mode="json")
    result["_vk_validation_errors"] = []
    return result


class PlatformConfigLike(Protocol):
    extra: dict[str, Any]


def settings_from_platform_config(config: PlatformConfigLike) -> VkSettings:
    extra = copy.deepcopy(getattr(config, "extra", {}) or {})
    errors = extra.pop("_vk_validation_errors", [])
    extra.pop("_enabled_explicit", None)
    if errors:
        raise ValueError("; ".join(str(error) for error in errors[:MAX_VALIDATION_ERRORS]))
    return VkSettings.model_validate(extra)


def validate_config(config: PlatformConfigLike) -> bool:
    try:
        settings_from_platform_config(config)
    except (ValidationError, ValueError):
        return False
    return not PolicyEnvironment().conflicts()

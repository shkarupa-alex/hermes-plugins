from __future__ import annotations
from importlib.resources import files

from pydantic import ValidationError

from hermes_vk_community.models import VkCapabilityProfile

RICH_PROFILE = "format-data-v1"
REQUIRED_FORMAT_TYPES = frozenset({"bold", "italic", "underline", "url"})


def load_capability_profile() -> VkCapabilityProfile | None:
    resource = files("hermes_vk_community").joinpath("data/vk-capabilities.json")
    try:
        return VkCapabilityProfile.model_validate_json(resource.read_text(encoding="utf-8"))
    except (OSError, ValidationError, ValueError):
        return None


def rich_capability_ready(api_version: str, *, require_edit: bool = False) -> bool:
    profile = load_capability_profile()
    if profile is None:
        return False
    return (
        profile.schema_version == 1
        and profile.api_version == api_version
        and profile.profile == RICH_PROFILE
        and profile.rich_send
        and (profile.rich_edit or not require_edit)
        and profile.format_data_observed
        and profile.offset_unit == "unicode_codepoints"
        and profile.nested_ranges == "properly_nested"
        and profile.overlapping_ranges == "accepted_transport_inbound_plain_fallback"
        and REQUIRED_FORMAT_TYPES.issubset(profile.supported_format_types)
    )

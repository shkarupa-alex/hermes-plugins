# pyright: reportPrivateUsage=false
from pydantic import TypeAdapter

from hermes_vk_community.cli import _formatting_probe_inputs


def test_formatting_probe_covers_rich_types_and_unicode_offset_hypotheses() -> None:
    cases = {case.name: case for case in _formatting_probe_inputs()}
    basic = cases["format_data_basic"]
    assert basic.format_data is not None
    items = TypeAdapter(list[dict[str, object]]).validate_python(basic.format_data["items"])
    assert {item["type"] for item in items} == {
        "bold",
        "italic",
        "underline",
        "url",
    }

    codepoints = cases["format_data_unicode_codepoints"].format_data
    utf16 = cases["format_data_unicode_utf16"].format_data
    assert codepoints is not None
    assert utf16 is not None
    codepoint_items = TypeAdapter(list[dict[str, object]]).validate_python(codepoints["items"])
    utf16_items = TypeAdapter(list[dict[str, object]]).validate_python(utf16["items"])
    assert codepoint_items[0]["offset"] == 2
    assert utf16_items[0]["offset"] == 3

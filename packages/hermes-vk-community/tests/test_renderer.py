from typing import cast

import pytest

from hermes_vk_community.renderer import (
    PlainVkRenderer,
    RenderedImageSegment,
    RenderedTableSegment,
    RenderedTextSegment,
    RichVkRenderer,
    format_data_for_chunk,
    sanitize_incoming_text,
    split_message,
    split_message_with_spans,
)


def test_plain_renderer_uses_ast_and_preserves_content() -> None:
    rendered = PlainVkRenderer().render_markdown(
        "# Заголовок\n\n**жирный** и [ссылка](https://example.com)\n\n```python\nprint('ok')\n```"
    )
    assert "Заголовок" in rendered.text
    assert "жирный" in rendered.text
    assert "ссылка — https://example.com" in rendered.text
    assert "Код:\nprint('ok')" in rendered.text
    assert "**" not in rendered.text


def test_inline_code_keeps_literal_backticks_in_all_modes() -> None:
    assert PlainVkRenderer().render_markdown("use `x = 1`").text == "use `x = 1`"
    assert RichVkRenderer().render_markdown("use `x = 1`").text == "use `x = 1`"


def test_incoming_mentions_and_controls() -> None:
    parsed = PlainVkRenderer().parse_incoming("Привет, [id123|Иван]\x00\r\n!", None)
    assert parsed.markdown == "Привет, Иван (@id123)\n!"
    assert sanitize_incoming_text("a\x00b") == "ab"


def test_incoming_format_data_reconstructs_canonical_markdown() -> None:
    text = "Жирный курсив подчёркнутый ссылка"
    parsed = RichVkRenderer().parse_incoming(
        text,
        {
            "version": "1",
            "items": [
                {"type": "bold", "offset": 0, "length": 6},
                {"type": "italic", "offset": 7, "length": 6},
                {"type": "underline", "offset": 14, "length": 12},
                {"type": "url", "offset": 27, "length": 6, "url": "https://example.com/"},
            ],
        },
    )
    assert parsed.markdown == ("**Жирный** *курсив* <u>подчёркнутый</u> [ссылка](https://example.com/)")


@pytest.mark.parametrize(
    "format_data",
    [
        {"version": 2, "items": []},
        {"version": 1, "items": [{"type": "unknown", "offset": 0, "length": 1}]},
        {
            "version": 1,
            "items": [
                {"type": "bold", "offset": 0, "length": 4},
                {"type": "italic", "offset": 2, "length": 4},
            ],
        },
        {"version": 1, "items": [{"type": "url", "offset": 0, "length": 1, "url": "javascript:x"}]},
    ],
)
def test_invalid_incoming_format_data_falls_back_to_plain(format_data: dict[str, object]) -> None:
    assert RichVkRenderer().parse_incoming("abcdef", format_data).markdown == "abcdef"


def test_markdown_list_has_no_blank_lines_between_items() -> None:
    rendered = PlainVkRenderer().render_markdown("- первый пункт\n- второй пункт")
    assert rendered.text == "• первый пункт\n• второй пункт"


def test_chunking_is_lossless_apart_from_boundary_whitespace() -> None:
    chunks = split_message("Первый абзац.\n\nВторой абзац длиннее.", 20)  # noqa: RUF001
    assert all(len(chunk) <= 20 for chunk in chunks)
    assert " ".join(" ".join(chunks).split()) == "Первый абзац. Второй абзац длиннее."


def test_chunk_spans_track_removed_boundary_whitespace() -> None:
    chunks = split_message_with_spans("first\n\nsecond", 7)
    assert [(chunk.text, chunk.start, chunk.end) for chunk in chunks] == [
        ("first", 0, 5),
        ("second", 7, 13),
    ]


def test_rich_renderer_compiles_inline_styles_quote_and_nested_lists() -> None:
    rendered = RichVkRenderer().render_markdown(
        "# Заголовок\n\n**жирный** *курсив* [ссылка](https://example.com)\n\n"
        "> важная цитата\n\n- первый\n  - вложенный\n\n1. один\n2. два"
    )
    assert len(rendered.segments) == 1
    segment = rendered.segments[0]
    assert isinstance(segment, RenderedTextSegment)
    assert "▎ важная цитата" in segment.text
    assert "• первый\n  • вложенный" in segment.text
    assert "1. один\n2. два" in segment.text
    assert segment.format_data is not None
    items = segment.format_data["items"]
    assert isinstance(items, list)
    typed_items = cast("list[dict[str, object]]", items)
    assert {item["type"] for item in typed_items} == {"bold", "italic", "url"}
    quote_start = segment.text.index("важная цитата")
    assert {"type": "italic", "offset": quote_start, "length": len("важная цитата")} in typed_items


def test_tables_are_ordered_media_segments_not_plain_text() -> None:
    rendered = RichVkRenderer().render_markdown(
        "До.\n\n| Поле | Значение |\n|---|---|\n| План | Pro |\n| Срок | 30 дней |\n\nПосле."  # noqa: RUF001
    )
    assert [type(segment) for segment in rendered.segments] == [
        RenderedTextSegment,
        RenderedTableSegment,
        RenderedTextSegment,
    ]
    table = rendered.segments[1]
    assert isinstance(table, RenderedTableSegment)
    assert table.headers == ("Поле", "Значение")
    assert table.rows == (("План", "Pro"), ("Срок", "30 дней"))
    assert "|---|" not in rendered.fallback_text


def test_standalone_markdown_image_is_an_ordered_attachment_segment() -> None:
    rendered = RichVkRenderer().render_markdown(
        "До.\n\n![Схема](https://example.com/diagram.jpg)\n\nПосле."  # noqa: RUF001
    )
    assert [type(segment) for segment in rendered.segments] == [
        RenderedTextSegment,
        RenderedImageSegment,
        RenderedTextSegment,
    ]
    image = rendered.segments[1]
    assert isinstance(image, RenderedImageSegment)
    assert image.url == "https://example.com/diagram.jpg"
    assert image.alt == "Схема"


def test_format_ranges_are_clipped_and_rebased_per_chunk() -> None:
    source = "abcdefghij"
    rich: dict[str, object] = {
        "version": 1,
        "items": [{"type": "bold", "offset": 2, "length": 7}],
    }
    chunks = split_message_with_spans(source, 5)
    assert format_data_for_chunk(rich, chunks[0]) == {
        "version": 1,
        "items": [{"type": "bold", "offset": 2, "length": 3}],
    }
    assert format_data_for_chunk(rich, chunks[1]) == {
        "version": 1,
        "items": [{"type": "bold", "offset": 0, "length": 4}],
    }


def test_ast_source_offsets_include_transport_markup_without_difflib_guessing() -> None:
    source = "**abc abc** and [abc](https://abc.example)"
    rendered = RichVkRenderer().render_markdown(source)
    segment = rendered.segments[0]
    assert isinstance(segment, RenderedTextSegment)
    assert segment.source_offsets is not None
    bold_end = len("abc abc")
    assert source[: segment.source_offsets[bold_end]] == "**abc abc**"
    assert source[: segment.source_offsets[len(segment.text)]] == source

# pyright: reportPrivateUsage=false
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from hermes_vk_community import table_image
from hermes_vk_community.renderer import RenderedTableSegment
from hermes_vk_community.table_image import _font_supports, _load_font_stack, _render_runs, render_table_jpegs


def test_table_is_rendered_as_readable_rgb_jpeg(tmp_path: Path) -> None:
    table = RenderedTableSegment(
        headers=("Поле", "Значение"),
        rows=(("План", "Pro"), ("Срок", "30 дней")),
    )
    paths = render_table_jpegs(table, tmp_path)
    assert len(paths) == 1
    assert paths[0].suffix == ".jpg"
    assert paths[0].read_bytes().startswith(b"\xff\xd8")
    with Image.open(paths[0]) as image:
        assert image.format == "JPEG"
        assert image.mode == "RGB"
        assert image.width >= 300
        assert image.height >= 150


def test_table_font_stack_never_emits_missing_glyphs_for_status_and_emoji() -> None:
    fonts = _load_font_stack(28)
    for value in ("✅ PASSED", "❌ FAILED", "⚠️ WARNING", "😀 Готово"):
        runs = _render_runs(value, fonts)
        assert runs
        assert all(_font_supports(run.font, run.text) for run in runs)


def test_table_uses_readable_text_when_no_emoji_font_is_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    fonts = table_image._FontStack(table_image._load_font(28))
    original_support = table_image._font_supports

    def no_emoji_support(font: table_image.Font, value: str) -> bool:
        return not table_image._is_emoji_cluster(value) and original_support(font, value)

    monkeypatch.setattr(
        table_image,
        "_font_supports",
        no_emoji_support,
    )
    rendered = "".join(run.text for run in table_image._render_runs("✅ ❌ ⚠️ 😀", fonts))
    assert rendered == "[OK] [X] [!] [emoji]"


def test_unicode_status_icons_change_rendered_table_geometry(tmp_path: Path) -> None:
    plain = RenderedTableSegment(
        headers=("Результат",),
        rows=(("   PASSED",), ("   FAILED",), ("   WARNING",), ("   Готово",)),
    )
    decorated = RenderedTableSegment(
        headers=("Результат",),
        rows=(("✅ PASSED",), ("❌ FAILED",), ("⚠️ WARNING",), ("😀 Готово",)),
    )
    plain_path = render_table_jpegs(plain, tmp_path / "plain")[0]
    decorated_path = render_table_jpegs(decorated, tmp_path / "decorated")[0]
    with Image.open(plain_path) as plain_image, Image.open(decorated_path) as decorated_image:
        assert decorated_image.width > plain_image.width
        fonts = _load_font_stack(28)
        uses_emoji_font = any(run.embedded_color for value in decorated.rows for run in _render_runs(value[0], fonts))
        if uses_emoji_font:
            assert decorated_image.height > plain_image.height
        else:
            assert decorated_image.height == plain_image.height


def test_text_only_rows_keep_primary_font_line_height() -> None:
    fonts = table_image._load_font_stack(28)
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    _left, top, _right, bottom = draw.textbbox((0, 0), "Ag", font=fonts.primary)
    assert table_image._line_height(draw, fonts, "Только текст") == bottom - top + 8


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("👍🏽", ["👍🏽"]),
        ("🇷🇺", ["🇷🇺"]),
        ("#⃣", ["#⃣"]),
        ("👩‍💻", ["👩‍💻"]),
    ],
)
def test_extended_emoji_sequences_are_single_clusters(value: str, expected: list[str]) -> None:
    assert table_image._grapheme_clusters(value) == expected


@pytest.mark.parametrize("value", ["👍🏽x", "🇷🇺x", "#⃣x", "👩‍💻x"])
def test_long_word_split_does_not_break_emoji_cluster(value: str) -> None:
    fonts = table_image._load_font_stack(28)
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    first_cluster = table_image._grapheme_clusters(value)[0]
    width = table_image._text_width(draw, first_cluster, fonts)
    assert table_image._fitting_prefix(draw, value, width, fonts) == len(first_cluster)

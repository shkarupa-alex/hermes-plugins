from __future__ import annotations
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from math import ceil
from pathlib import Path
from typing import TYPE_CHECKING, cast

from PIL import Image, ImageDraw, ImageFont

if TYPE_CHECKING:
    from collections.abc import Iterable

    from hermes_vk_community.renderer import RenderedTableSegment

TABLE_FONT_SIZE = 28
CELL_PADDING = 18
MIN_COLUMN_WIDTH = 150
MAX_COLUMN_WIDTH = 520
MAX_IMAGE_HEIGHT = 12_000
JPEG_QUALITY = 92

Font = ImageFont.FreeTypeFont | ImageFont.ImageFont


@dataclass(frozen=True, slots=True)
class _FontStack:
    primary: Font
    fallbacks: tuple[_FallbackFont, ...] = ()


@dataclass(frozen=True, slots=True)
class _FallbackFont:
    font: Font
    scale: float = 1.0


@dataclass(frozen=True, slots=True)
class _GlyphRun:
    text: str
    font: Font
    embedded_color: bool = False
    scale: float = 1.0


EMOJI_FONT_CANDIDATES = (
    Path("/System/Library/Fonts/Apple Color Emoji.ttc"),
    Path("C:/Windows/Fonts/seguiemj.ttf"),
    Path("/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"),
    Path("/usr/share/fonts/truetype/noto/NotoEmoji-Regular.ttf"),
    Path("/usr/share/fonts/opentype/noto/NotoColorEmoji.ttf"),
)
EMOJI_FONT_SIZES = (32, 28, 20, 48, 64, 109, 128, 137)
EMOJI_CODEPOINT_RANGES = ((0x1F000, 0x1FAFF), (0x2600, 0x27BF))
EMOJI_MODIFIER_RANGE = (0x1F3FB, 0x1F3FF)
REGIONAL_INDICATOR_RANGE = (0x1F1E6, 0x1F1FF)
TAG_CHARACTER_RANGE = (0xE0020, 0xE007F)
COMBINING_KEYCAP = "\u20e3"
VARIATION_SELECTORS = frozenset({"\ufe0e", "\ufe0f"})
ZWJ = "\u200d"
EMOJI_TEXT_FALLBACKS = {
    "✅": "[OK]",
    "❌": "[X]",
    "⚠": "[!]",
    "⚠️": "[!]",
}


def render_table_jpegs(table: RenderedTableSegment, directory: Path) -> list[Path]:
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    font = _load_font_stack(TABLE_FONT_SIZE)
    bold = _load_font_stack(TABLE_FONT_SIZE, bold=True)
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    column_count = max(len(table.headers), *(len(row) for row in table.rows), 1)
    headers = _normalize_row(table.headers, column_count, header=True)
    rows = [_normalize_row(row, column_count) for row in table.rows]
    widths = _column_widths(probe, headers, rows, bold, font)
    prepared_header = _prepare_row(probe, headers, widths, bold)
    prepared_rows = [_prepare_row(probe, row, widths, font) for row in rows]
    pages = _paginate(prepared_header[1], [height for _, height, _ in prepared_rows])
    paths: list[Path] = []
    for page_number, (start, end) in enumerate(pages, start=1):
        page_rows = prepared_rows[start:end]
        height = prepared_header[1] + sum(row_height for _, row_height, _ in page_rows) + 1
        image = Image.new("RGB", (sum(widths) + 1, height), "white")
        draw = ImageDraw.Draw(image)
        y = 0
        y = _draw_row(
            image,
            draw,
            prepared_header[0],
            widths,
            y,
            prepared_header[1],
            prepared_header[2],
            bold,
            "#E8EEF7",
        )
        for index, (cells, row_height, line_height) in enumerate(page_rows, start=start):
            fill = "#FFFFFF" if index % 2 == 0 else "#F7F9FC"
            y = _draw_row(image, draw, cells, widths, y, row_height, line_height, font, fill)
        path = directory / f"table-{page_number:03d}.jpg"
        image.save(path, format="JPEG", quality=JPEG_QUALITY, optimize=True, progressive=True)
        path.chmod(0o600)
        paths.append(path)
    return paths


def _load_font(size: int, *, bold: bool = False) -> Font:
    names = ["DejaVuSans-Bold.ttf", "Arial Bold.ttf"] if bold else ["DejaVuSans.ttf", "Arial.ttf"]
    candidates = [
        *(Path(name) for name in names),
        Path(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ),
        Path(
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
            if bold
            else "/System/Library/Fonts/Supplemental/Arial.ttf"
        ),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(str(candidate), size=size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def _load_font_stack(size: int, *, bold: bool = False) -> _FontStack:
    return _FontStack(_load_font(size, bold=bold), _load_emoji_fonts(size))


def _load_emoji_fonts(size: int) -> tuple[_FallbackFont, ...]:
    fonts: list[_FallbackFont] = []
    for candidate in EMOJI_FONT_CANDIDATES:
        if not candidate.is_file():
            continue
        for candidate_size in dict.fromkeys((size, *EMOJI_FONT_SIZES)):
            try:
                font = ImageFont.truetype(str(candidate), size=candidate_size)
            except OSError:
                continue
            fonts.append(_FallbackFont(font, size / candidate_size))
            break
    return tuple(fonts)


def _normalize_row(row: tuple[str, ...], count: int, *, header: bool = False) -> tuple[str, ...]:
    values = list(row[:count])
    while len(values) < count:
        values.append(f"Поле {len(values) + 1}" if header else "")
    return tuple(values)


def _column_widths(
    draw: ImageDraw.ImageDraw,
    headers: tuple[str, ...],
    rows: list[tuple[str, ...]],
    header_font: _FontStack,
    body_font: _FontStack,
) -> list[int]:
    widths: list[int] = []
    for column, header in enumerate(headers):
        values = [(header, header_font), *((row[column], body_font) for row in rows)]
        measured = max((_text_width(draw, value, font) for value, font in values), default=MIN_COLUMN_WIDTH)
        widths.append(max(MIN_COLUMN_WIDTH, min(MAX_COLUMN_WIDTH, measured + CELL_PADDING * 2)))
    return widths


def _prepare_row(
    draw: ImageDraw.ImageDraw,
    row: tuple[str, ...],
    widths: list[int],
    font: _FontStack,
) -> tuple[list[list[str]], int, int]:
    cells = [_wrap_text(draw, value, widths[index] - CELL_PADDING * 2, font) for index, value in enumerate(row)]
    line_height = max((_line_height(draw, font, line) for lines in cells for line in lines), default=1)
    height = max((len(lines) for lines in cells), default=1) * line_height + CELL_PADDING * 2
    return cells, height, line_height


def _paginate(header_height: int, row_heights: list[int]) -> list[tuple[int, int]]:
    if not row_heights:
        return [(0, 0)]
    pages: list[tuple[int, int]] = []
    start = 0
    height = header_height
    for index, row_height in enumerate(row_heights):
        if index > start and height + row_height > MAX_IMAGE_HEIGHT:
            pages.append((start, index))
            start = index
            height = header_height
        height += row_height
    pages.append((start, len(row_heights)))
    return pages


def _draw_row(  # noqa: PLR0913 - drawing requires explicit geometry and style inputs
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    cells: list[list[str]],
    widths: list[int],
    y: int,
    height: int,
    line_height: int,
    font: _FontStack,
    fill: str,
) -> int:
    x = 0
    for lines, width in zip(cells, widths, strict=True):
        draw.rectangle((x, y, x + width, y + height), fill=fill, outline="#AEB8C6", width=1)
        text_y = y + CELL_PADDING
        for line in lines:
            _draw_text(image, (x + CELL_PADDING, text_y), line, font, fill="#172033")
            text_y += line_height
        x += width
    return y + height


def _wrap_text(draw: ImageDraw.ImageDraw, value: str, width: int, font: _FontStack) -> list[str]:
    paragraphs = value.replace("\r", "").split("\n") or [""]
    result: list[str] = []
    for paragraph in paragraphs:
        words = paragraph.split() or [""]
        line = ""
        for word in words:
            remainder = word
            candidate = remainder if not line else f"{line} {remainder}"
            if _text_width(draw, candidate, font) <= width:
                line = candidate
                continue
            if line:
                result.append(line)
                line = ""
            while remainder and _text_width(draw, remainder, font) > width:
                split = _fitting_prefix(draw, remainder, width, font)
                result.append(remainder[:split])
                remainder = remainder[split:]
            line = remainder
        result.append(line)
    return result or [""]


def _fitting_prefix(draw: ImageDraw.ImageDraw, value: str, width: int, font: _FontStack) -> int:
    clusters = _grapheme_clusters(value)
    boundaries = [len("".join(clusters[:index])) for index in range(1, len(clusters) + 1)]
    if not boundaries:
        return 1
    low, high = 0, len(boundaries) - 1
    while low < high:
        middle = (low + high + 1) // 2
        if _text_width(draw, value[: boundaries[middle]], font) <= width:
            low = middle
        else:
            high = middle - 1
    return boundaries[low]


def _text_width(draw: ImageDraw.ImageDraw, value: str, font: _FontStack) -> int:
    return round(sum(draw.textlength(run.text, font=run.font) * run.scale for run in _render_runs(value or " ", font)))


def _line_height(draw: ImageDraw.ImageDraw, font: _FontStack, value: str) -> int:
    _left, top, _right, bottom = draw.textbbox((0, 0), "Ag", font=font.primary)
    heights: list[float] = [bottom - top]
    for run in _render_runs(value, font):
        if run.font is font.primary:
            continue
        _left, top, _right, bottom = draw.textbbox((0, 0), run.text, font=run.font)
        heights.append((bottom - top) * run.scale)
    return max(1, int(max(heights, default=1) + 8))


def _draw_text(
    image: Image.Image,
    position: tuple[int, int],
    value: str,
    fonts: _FontStack,
    *,
    fill: str,
) -> None:
    draw = ImageDraw.Draw(image)
    x, y = position
    for run in _render_runs(value, fonts):
        if run.scale == 1.0:
            draw.text((x, y), run.text, fill=fill, font=run.font, embedded_color=run.embedded_color)
        else:
            _draw_scaled_run(image, (x, y), run, fill)
        x += round(draw.textlength(run.text, font=run.font) * run.scale)


def _draw_scaled_run(image: Image.Image, position: tuple[int, int], run: _GlyphRun, fill: str) -> None:
    left, top, right, bottom = run.font.getbbox(run.text)
    width = max(1, ceil(right - left))
    height = max(1, ceil(bottom - top))
    glyph = Image.new("RGBA", (width, height))
    glyph_draw = ImageDraw.Draw(glyph)
    glyph_draw.text((-left, -top), run.text, fill=fill, font=run.font, embedded_color=run.embedded_color)
    scaled = glyph.resize(
        (max(1, round(width * run.scale)), max(1, round(height * run.scale))),
        Image.Resampling.LANCZOS,
    )
    x, y = position
    destination = (round(x + left * run.scale), round(y + top * run.scale))
    image.paste(scaled, destination, scaled)


def _render_runs(value: str, fonts: _FontStack) -> tuple[_GlyphRun, ...]:
    runs: list[_GlyphRun] = []
    for cluster in _grapheme_clusters(value):
        face = next(
            (
                candidate
                for candidate in (_FallbackFont(fonts.primary), *fonts.fallbacks)
                if _font_supports(candidate.font, cluster)
            ),
            None,
        )
        rendered = cluster
        if face is None:
            rendered = EMOJI_TEXT_FALLBACKS.get(cluster, "[emoji]" if _is_emoji_cluster(cluster) else cluster)
            face = _FallbackFont(fonts.primary)
        font = face.font
        embedded_color = font is not fonts.primary
        if not embedded_color:
            rendered = "".join(character for character in rendered if character not in VARIATION_SELECTORS)
        if (
            runs
            and runs[-1].font is font
            and runs[-1].embedded_color == embedded_color
            and runs[-1].scale == face.scale
        ):
            previous = runs[-1]
            runs[-1] = _GlyphRun(previous.text + rendered, font, embedded_color, face.scale)
        else:
            runs.append(_GlyphRun(rendered, font, embedded_color, face.scale))
    return tuple(runs)


def _grapheme_clusters(value: str) -> list[str]:
    clusters: list[str] = []
    for character in value:
        codepoint = ord(character)
        extends_previous = (
            character in VARIATION_SELECTORS
            or character in {ZWJ, COMBINING_KEYCAP}
            or bool(unicodedata.combining(character))
            or EMOJI_MODIFIER_RANGE[0] <= codepoint <= EMOJI_MODIFIER_RANGE[1]
            or TAG_CHARACTER_RANGE[0] <= codepoint <= TAG_CHARACTER_RANGE[1]
        )
        follows_joiner = bool(clusters and clusters[-1].endswith(ZWJ))
        pairs_regional_indicators = bool(
            clusters
            and REGIONAL_INDICATOR_RANGE[0] <= codepoint <= REGIONAL_INDICATOR_RANGE[1]
            and len(clusters[-1]) == 1
            and REGIONAL_INDICATOR_RANGE[0] <= ord(clusters[-1]) <= REGIONAL_INDICATOR_RANGE[1]
        )
        if clusters and (extends_previous or follows_joiner or pairs_regional_indicators):
            clusters[-1] += character
        else:
            clusters.append(character)
    return clusters


def _is_emoji_cluster(value: str) -> bool:
    return any(start <= ord(character) <= end for character in value for start, end in EMOJI_CODEPOINT_RANGES)


@lru_cache(maxsize=4096)
def _font_supports(font: Font, value: str) -> bool:
    meaningful = [
        character
        for character in value
        if character not in VARIATION_SELECTORS and character != ZWJ and not unicodedata.combining(character)
    ]
    if not meaningful:
        return True
    try:
        missing = _missing_glyph_signature(font)
        return all(_glyph_signature(font, character) != missing for character in meaningful)
    except (OSError, TypeError, UnicodeError, ValueError):
        return False


@lru_cache(maxsize=32)
def _missing_glyph_signature(font: Font) -> tuple[tuple[float, float, float, float] | None, bytes]:
    return _glyph_signature(font, "\U0010ffff")


def _glyph_signature(font: Font, value: str) -> tuple[tuple[float, float, float, float] | None, bytes]:
    mask = cast("Iterable[int]", font.getmask(value))
    return font.getbbox(value), bytes(mask)

from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image, ImageDraw, ImageFont

if TYPE_CHECKING:
    from hermes_vk_community.renderer import RenderedTableSegment

TABLE_FONT_SIZE = 28
CELL_PADDING = 18
MIN_COLUMN_WIDTH = 150
MAX_COLUMN_WIDTH = 520
MAX_IMAGE_HEIGHT = 12_000
JPEG_QUALITY = 92

Font = ImageFont.FreeTypeFont | ImageFont.ImageFont


def render_table_jpegs(table: RenderedTableSegment, directory: Path) -> list[Path]:
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    font = _load_font(TABLE_FONT_SIZE)
    bold = _load_font(TABLE_FONT_SIZE, bold=True)
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    column_count = max(len(table.headers), *(len(row) for row in table.rows), 1)
    headers = _normalize_row(table.headers, column_count, header=True)
    rows = [_normalize_row(row, column_count) for row in table.rows]
    widths = _column_widths(probe, headers, rows, bold, font)
    prepared_header = _prepare_row(probe, headers, widths, bold)
    prepared_rows = [_prepare_row(probe, row, widths, font) for row in rows]
    pages = _paginate(prepared_header[1], [height for _, height in prepared_rows])
    paths: list[Path] = []
    for page_number, (start, end) in enumerate(pages, start=1):
        page_rows = prepared_rows[start:end]
        height = prepared_header[1] + sum(row_height for _, row_height in page_rows) + 1
        image = Image.new("RGB", (sum(widths) + 1, height), "white")
        draw = ImageDraw.Draw(image)
        y = 0
        y = _draw_row(draw, prepared_header[0], widths, y, prepared_header[1], bold, "#E8EEF7")
        for index, (cells, row_height) in enumerate(page_rows, start=start):
            fill = "#FFFFFF" if index % 2 == 0 else "#F7F9FC"
            y = _draw_row(draw, cells, widths, y, row_height, font, fill)
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


def _normalize_row(row: tuple[str, ...], count: int, *, header: bool = False) -> tuple[str, ...]:
    values = list(row[:count])
    while len(values) < count:
        values.append(f"Поле {len(values) + 1}" if header else "")
    return tuple(values)


def _column_widths(
    draw: ImageDraw.ImageDraw,
    headers: tuple[str, ...],
    rows: list[tuple[str, ...]],
    header_font: Font,
    body_font: Font,
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
    font: Font,
) -> tuple[list[list[str]], int]:
    cells = [_wrap_text(draw, value, widths[index] - CELL_PADDING * 2, font) for index, value in enumerate(row)]
    line_height = _line_height(draw, font)
    height = max((len(lines) for lines in cells), default=1) * line_height + CELL_PADDING * 2
    return cells, height


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
    draw: ImageDraw.ImageDraw,
    cells: list[list[str]],
    widths: list[int],
    y: int,
    height: int,
    font: Font,
    fill: str,
) -> int:
    x = 0
    line_height = _line_height(draw, font)
    for lines, width in zip(cells, widths, strict=True):
        draw.rectangle((x, y, x + width, y + height), fill=fill, outline="#AEB8C6", width=1)
        text_y = y + CELL_PADDING
        for line in lines:
            draw.text((x + CELL_PADDING, text_y), line, fill="#172033", font=font)
            text_y += line_height
        x += width
    return y + height


def _wrap_text(draw: ImageDraw.ImageDraw, value: str, width: int, font: Font) -> list[str]:
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


def _fitting_prefix(draw: ImageDraw.ImageDraw, value: str, width: int, font: Font) -> int:
    low, high = 1, max(1, len(value))
    while low < high:
        middle = (low + high + 1) // 2
        if _text_width(draw, value[:middle], font) <= width:
            low = middle
        else:
            high = middle - 1
    return low


def _text_width(draw: ImageDraw.ImageDraw, value: str, font: Font) -> int:
    left, _top, right, _bottom = draw.textbbox((0, 0), value or " ", font=font)
    return int(right - left)


def _line_height(draw: ImageDraw.ImageDraw, font: Font) -> int:
    _left, top, _right, bottom = draw.textbbox((0, 0), "Ag", font=font)
    return max(1, int(bottom - top + 8))

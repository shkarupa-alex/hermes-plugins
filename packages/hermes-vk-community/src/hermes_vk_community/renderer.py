from __future__ import annotations
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypeAlias, cast
from urllib.parse import urlsplit

from markdown_it import MarkdownIt

if TYPE_CHECKING:
    from markdown_it.token import Token

VK_MENTION = re.compile(r"\[(id|club)(\d+)\|([^\]\r\n]+)]")
MIN_PRINTABLE_CODEPOINT = 32
MAX_INCOMING_URL_LENGTH = 2048
SUPPORTED_INCOMING_TYPES = frozenset({"bold", "italic", "underline", "url"})
MARKDOWN_SPECIAL = re.compile(r"([\\*_[\]`])")
ESCAPED_VK_MENTION = re.compile(r"\\\[(id|club)(\d+)\|([^\\\]\r\n]+)\\\]")


@dataclass(frozen=True, slots=True)
class RenderedTextSegment:
    text: str
    format_data: dict[str, object] | None = None
    source_offsets: tuple[int, ...] | None = None


@dataclass(frozen=True, slots=True)
class RenderedTableSegment:
    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]

    @property
    def fallback_text(self) -> str:
        lines: list[str] = []
        for row in self.rows:
            fields = [
                f"{self.headers[index] if index < len(self.headers) else f'Поле {index + 1}'}: {value}"
                for index, value in enumerate(row)
            ]
            lines.append("• " + "; ".join(fields))
        return "\n".join(lines) or "Таблица"


@dataclass(frozen=True, slots=True)
class RenderedImageSegment:
    url: str
    alt: str

    @property
    def fallback_text(self) -> str:
        return f"{self.alt or 'Изображение'} — {self.url}"


RenderedVkSegment: TypeAlias = RenderedTextSegment | RenderedTableSegment | RenderedImageSegment


@dataclass(frozen=True, slots=True)
class RenderedVkMessage:
    text: str
    format_data: dict[str, object] | None
    fallback_text: str
    capabilities_used: frozenset[str]
    segments: tuple[RenderedVkSegment, ...]


@dataclass(frozen=True, slots=True)
class ParsedIncomingMessage:
    markdown: str
    original_text: str
    format_data: dict[str, object] | None


@dataclass(frozen=True, slots=True)
class WireChunk:
    text: str
    start: int
    end: int


class VkMessageRenderer(Protocol):
    def render_markdown(self, markdown: str) -> RenderedVkMessage: ...

    def parse_incoming(self, text: str, format_data: dict[str, object] | None) -> ParsedIncomingMessage: ...


class RichVkRenderer:
    def __init__(self) -> None:
        self._parser = MarkdownIt("commonmark", {"html": False, "linkify": True}).enable("table")

    def render_markdown(self, markdown: str) -> RenderedVkMessage:
        segments = _render_segments(self._parser, markdown, rich=True)
        fallback = "\n\n".join(
            segment.text if isinstance(segment, RenderedTextSegment) else segment.fallback_text
            for segment in segments
            if not isinstance(segment, RenderedTextSegment) or segment.text
        )
        text_segments = [segment for segment in segments if isinstance(segment, RenderedTextSegment)]
        text = "\n\n".join(segment.text for segment in text_segments)
        format_data = text_segments[0].format_data if len(text_segments) == 1 else None
        capabilities = {"jpeg_table" for segment in segments if isinstance(segment, RenderedTableSegment)}
        capabilities.update("image_attachment" for segment in segments if isinstance(segment, RenderedImageSegment))
        for segment in text_segments:
            if segment.format_data:
                items = segment.format_data.get("items")
                if isinstance(items, list):
                    for value in cast("list[object]", items):
                        if isinstance(value, dict):
                            item = cast("dict[str, object]", value)
                            capabilities.add(str(item.get("type")))
        return RenderedVkMessage(
            text=text,
            format_data=format_data,
            fallback_text=fallback,
            capabilities_used=frozenset(capabilities),
            segments=segments,
        )

    def parse_incoming(self, text: str, format_data: dict[str, object] | None) -> ParsedIncomingMessage:
        clean = sanitize_incoming_text(text)
        plain = VK_MENTION.sub(lambda match: f"{match.group(3)} (@{match.group(1)}{match.group(2)})", clean)
        markdown = _incoming_markdown(clean, format_data)
        if markdown is None:
            markdown = plain
        return ParsedIncomingMessage(markdown=markdown, original_text=text, format_data=format_data)


class PlainVkRenderer(RichVkRenderer):
    def render_markdown(self, markdown: str) -> RenderedVkMessage:
        segments = _render_segments(self._parser, markdown, rich=False)
        fallback = "\n\n".join(
            segment.text if isinstance(segment, RenderedTextSegment) else segment.fallback_text
            for segment in segments
            if not isinstance(segment, RenderedTextSegment) or segment.text
        )
        plain_segments = tuple(
            RenderedTextSegment(segment.text, source_offsets=segment.source_offsets)
            if isinstance(segment, RenderedTextSegment)
            else segment
            for segment in segments
        )
        return RenderedVkMessage(
            text=fallback,
            format_data=None,
            fallback_text=fallback,
            capabilities_used=frozenset(
                {"jpeg_table" for segment in segments if isinstance(segment, RenderedTableSegment)}
                | {"image_attachment" for segment in segments if isinstance(segment, RenderedImageSegment)}
            ),
            segments=plain_segments,
        )


def sanitize_incoming_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return "".join(char for char in normalized if char in "\n\t" or ord(char) >= MIN_PRINTABLE_CODEPOINT)


@dataclass(frozen=True, slots=True)
class _IncomingRange:
    kind: str
    start: int
    end: int
    url: str | None = None


def _incoming_markdown(  # noqa: C901, PLR0911, PLR0912 - fail-closed range validation is intentionally explicit
    text: str,
    format_data: dict[str, object] | None,
) -> str | None:
    if not format_data or format_data.get("version") not in {1, "1"}:
        return None
    raw_items = format_data.get("items")
    if not isinstance(raw_items, list):
        return None
    ranges: list[_IncomingRange] = []
    for value in cast("list[object]", raw_items):
        if not isinstance(value, dict):
            return None
        item = cast("dict[str, object]", value)
        kind = item.get("type")
        offset = item.get("offset")
        length = item.get("length")
        if (
            not isinstance(kind, str)
            or kind not in SUPPORTED_INCOMING_TYPES
            or not isinstance(offset, int)
            or isinstance(offset, bool)
            or not isinstance(length, int)
            or isinstance(length, bool)
            or offset < 0
            or length <= 0
            or offset + length > len(text)
        ):
            return None
        url = item.get("url")
        if kind == "url":
            if not isinstance(url, str) or not _safe_incoming_url(url):
                return None
        elif url is not None:
            return None
        ranges.append(_IncomingRange(kind, offset, offset + length, url if isinstance(url, str) else None))
    ordered = sorted(ranges, key=lambda item: (item.start, -item.end, item.kind))
    active: list[_IncomingRange] = []
    for item in ordered:
        while active and item.start >= active[-1].end:
            active.pop()
        if active and item.end > active[-1].end:
            return None
        active.append(item)
    opens: dict[int, list[_IncomingRange]] = {}
    closes: dict[int, list[_IncomingRange]] = {}
    for item in ordered:
        opens.setdefault(item.start, []).append(item)
        closes.setdefault(item.end, []).append(item)
    output: list[str] = []
    for index in range(len(text) + 1):
        output.extend(_range_close(item) for item in reversed(closes.get(index, [])))
        output.extend(_range_open(item) for item in opens.get(index, []))
        if index < len(text):
            output.append(_escape_markdown(text[index]))
    markdown = "".join(output)
    return ESCAPED_VK_MENTION.sub(
        lambda match: f"{match.group(3)} (@{match.group(1)}{match.group(2)})",
        markdown,
    )


def _safe_incoming_url(value: str) -> bool:
    if len(value) > MAX_INCOMING_URL_LENGTH or any(ord(character) < MIN_PRINTABLE_CODEPOINT for character in value):
        return False
    parsed = urlsplit(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.hostname)


def _escape_markdown(value: str) -> str:
    return MARKDOWN_SPECIAL.sub(r"\\\1", value)


def _range_open(item: _IncomingRange) -> str:
    return {"bold": "**", "italic": "*", "underline": "<u>", "url": "["}[item.kind]


def _range_close(item: _IncomingRange) -> str:
    if item.kind == "url":
        url = (item.url or "").replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        return f"]({url})"
    return {"bold": "**", "italic": "*", "underline": "</u>"}[item.kind]


def split_message(text: str, limit: int) -> list[str]:
    return [chunk.text for chunk in split_message_with_spans(text, limit)]


def split_message_with_spans(text: str, limit: int) -> list[WireChunk]:
    if len(text) <= limit:
        return [WireChunk(text=text, start=0, end=len(text))]
    chunks: list[WireChunk] = []
    start = 0
    while start < len(text):
        remaining = text[start:]
        if len(remaining) <= limit:
            chunks.append(WireChunk(text=remaining, start=start, end=len(text)))
            break
        split_at = remaining.rfind("\n\n", 0, limit + 1)
        if split_at < limit // 3:
            split_at = remaining.rfind("\n", 0, limit + 1)
        if split_at < limit // 3:
            split_at = remaining.rfind(" ", 0, limit + 1)
        if split_at < limit // 3:
            split_at = limit
        raw_chunk = remaining[:split_at]
        chunk = raw_chunk.rstrip()
        if not chunk:
            chunk = remaining[:limit]
            split_at = limit
        chunk_end = start + len(chunk)
        chunks.append(WireChunk(text=chunk, start=start, end=chunk_end))
        start += split_at
        while start < len(text) and text[start].isspace():
            start += 1
    return chunks


def format_data_for_chunk(format_data: dict[str, object] | None, chunk: WireChunk) -> dict[str, object] | None:
    if not format_data or not isinstance(format_data.get("items"), list):
        return None
    rebased: list[dict[str, object]] = []
    raw_items = format_data["items"]
    if not isinstance(raw_items, list):
        return None
    for value in cast("list[object]", raw_items):
        if not isinstance(value, dict):
            continue
        raw = cast("dict[str, object]", value)
        offset = raw.get("offset")
        length = raw.get("length")
        if not isinstance(offset, int) or not isinstance(length, int) or length <= 0:
            continue
        start = max(offset, chunk.start)
        end = min(offset + length, chunk.end)
        if end <= start:
            continue
        item = dict(raw)
        item["offset"] = start - chunk.start
        item["length"] = end - start
        rebased.append(item)
    return {"version": 1, "items": rebased} if rebased else None


class _TextBuilder:
    def __init__(self) -> None:
        self.parts: list[str] = []
        self.length = 0
        self.items: list[dict[str, object]] = []
        self.source_offsets: list[int] = [0]

    @property
    def text(self) -> str:
        return "".join(self.parts)

    def append(self, value: str) -> None:
        if value:
            self.parts.append(value)
            self.length += len(value)
            self.source_offsets.extend([self.source_offsets[-1]] * len(value))

    def append_mapped(self, value: str, source_start: int) -> None:
        if value:
            self.parts.append(value)
            self.length += len(value)
            self.source_offsets.extend(source_start + index + 1 for index in range(len(value)))

    def advance_source(self, source_end: int) -> None:
        self.source_offsets[-1] = max(self.source_offsets[-1], source_end)

    def newlines(self, count: int) -> None:
        current = len(self.text) - len(self.text.rstrip("\n"))
        if current < count:
            self.append("\n" * (count - current))

    def mark(self, kind: str, start: int, *, url: str | None = None) -> None:
        if self.length <= start:
            return
        item: dict[str, object] = {"type": kind, "offset": start, "length": self.length - start}
        if url is not None:
            item["url"] = url
        self.items.append(item)

    def finish(self, *, rich: bool) -> RenderedTextSegment:
        raw = self.text
        left = len(raw) - len(raw.lstrip())
        text = raw.strip()
        right = left + len(text)
        items: list[dict[str, object]] = []
        if rich:
            for raw_item in self.items:
                raw_offset = raw_item["offset"]
                raw_length = raw_item["length"]
                if not isinstance(raw_offset, int) or not isinstance(raw_length, int):
                    continue
                start = raw_offset
                end = start + raw_length
                clipped_start = max(start, left)
                clipped_end = min(end, right)
                if clipped_end <= clipped_start:
                    continue
                item = dict(raw_item)
                item["offset"] = clipped_start - left
                item["length"] = clipped_end - clipped_start
                items.append(item)
        source_offsets = tuple(self.source_offsets[left : right + 1])
        return RenderedTextSegment(
            text,
            {"version": 1, "items": items} if items else None,
            source_offsets,
        )


def _render_segments(  # noqa: C901, PLR0912 - ordered AST block lowering has explicit media branches
    parser: MarkdownIt, markdown: str, *, rich: bool
) -> tuple[RenderedVkSegment, ...]:
    tokens = parser.parse(markdown)
    block_segments: list[tuple[int, int, RenderedVkSegment | None]] = [
        (token.map[0], token.map[1], None) for token in tokens if token.type == "table_open" and token.map is not None
    ]
    for index, token in enumerate(tokens):
        if token.type != "paragraph_open" or token.map is None or index + 1 >= len(tokens):
            continue
        inline = tokens[index + 1]
        children = inline.children or []
        if inline.type != "inline" or len(children) != 1 or children[0].type != "image":
            continue
        image = children[0]
        url = str(image.attrGet("src") or "")
        if url:
            block_segments.append((token.map[0], token.map[1], RenderedImageSegment(url, image.content.strip())))
    if not block_segments:
        segment = _render_text_tokens(tokens, source=markdown, rich=rich)
        return (segment,) if segment.text else ()
    lines = markdown.splitlines(keepends=True)
    segments: list[RenderedVkSegment] = []
    cursor = 0
    for start, end, block_segment in sorted(block_segments, key=lambda item: (item[0], item[1])):
        if start < cursor:
            continue
        before = "".join(lines[cursor:start])
        if before.strip():
            segment = _render_text_tokens(parser.parse(before), source=before, rich=rich)
            if segment.text:
                segments.append(segment)
        if block_segment is not None:
            segments.append(block_segment)
        else:
            table = _parse_table(parser.parse("".join(lines[start:end])))
            if table is not None:
                segments.append(table)
        cursor = end
    after = "".join(lines[cursor:])
    if after.strip():
        segment = _render_text_tokens(parser.parse(after), source=after, rich=rich)
        if segment.text:
            segments.append(segment)
    return tuple(segments)


def _render_text_tokens(  # noqa: C901, PLR0912, PLR0915 - token state machine mirrors Markdown block structure
    tokens: list[Token], *, source: str, rich: bool
) -> RenderedTextSegment:
    builder = _TextBuilder()
    line_offsets = _line_offsets(source)
    source_search_start = 0
    lists: list[dict[str, int | bool]] = []
    quote_depth = 0
    heading = False
    for token in tokens:
        kind = token.type
        if kind == "heading_open":
            heading = True
        elif kind == "heading_close":
            heading = False
            builder.newlines(2)
        elif kind == "blockquote_open":
            quote_depth += 1
        elif kind == "blockquote_close":
            quote_depth = max(0, quote_depth - 1)
            builder.newlines(2)
        elif kind == "paragraph_open" and quote_depth and not lists:
            if builder.length:
                builder.newlines(1)
            builder.append("▎ ")
        elif kind == "paragraph_close":
            builder.newlines(1 if lists else 2)
        elif kind == "bullet_list_open":
            lists.append({"ordered": False, "next": 1})
        elif kind == "ordered_list_open":
            lists.append({"ordered": True, "next": int(token.attrGet("start") or 1)})
        elif kind in {"bullet_list_close", "ordered_list_close"}:
            if lists:
                lists.pop()
            builder.newlines(1 if lists else 2)
        elif kind == "list_item_open":
            if builder.length:
                builder.newlines(1)
            depth = max(0, len(lists) - 1)
            current = lists[-1] if lists else {"ordered": False, "next": 1}
            if bool(current["ordered"]):
                index = int(current["next"])
                current["next"] = index + 1
                marker = f"{index}. "
            else:
                marker = "• "
            builder.append("  " * depth + marker)
        elif kind == "list_item_close":
            builder.newlines(1)
        elif kind == "inline":
            start = builder.length
            source_base = source.find(token.content, source_search_start)
            if source_base < 0 and token.map is not None and token.map[0] < len(line_offsets):
                source_base = line_offsets[token.map[0]]
            source_base = max(0, source_base)
            _render_inline_rich(
                token.children or [],
                builder,
                rich=rich,
                raw=token.content,
                source_base=source_base,
            )
            source_search_start = source_base + len(token.content)
            if rich and heading:
                builder.mark("bold", start)
            if rich and quote_depth:
                builder.mark("italic", start)
        elif kind in {"fence", "code_block"}:
            content = token.content.rstrip()
            builder.append("Код:\n")
            if token.map is not None and token.map[0] < len(line_offsets):
                block_start = line_offsets[token.map[0]]
                raw_block = source[block_start : line_offsets[min(token.map[1], len(line_offsets) - 1)]]
                content_start = raw_block.find(token.content)
                if content_start >= 0:
                    builder.append_mapped(content, block_start + content_start)
                else:
                    builder.append(content)
                builder.advance_source(block_start + len(raw_block))
            else:
                builder.append(content)
            builder.newlines(2)
    return builder.finish(rich=rich)


def _render_inline_rich(  # noqa: C901, PLR0912, PLR0915 - token state machine mirrors Markdown inline structure
    tokens: list[Token],
    builder: _TextBuilder,
    *,
    rich: bool,
    raw: str,
    source_base: int,
) -> None:
    styles: dict[str, list[int]] = {"bold": [], "italic": []}
    links: list[tuple[int, str]] = []
    raw_cursor = 0
    for token in tokens:
        kind = token.type
        if kind == "text":
            raw_cursor = _append_inline_literal(builder, raw, raw_cursor, token.content, source_base)
        elif kind == "code_inline":
            markup = token.markup or "`"
            opening = raw.find(markup, raw_cursor)
            content_start = raw.find(token.content, opening + len(markup)) if opening >= 0 else -1
            if content_start >= 0:
                closing = raw.find(markup, content_start + len(token.content))
                builder.append_mapped(markup, source_base + opening)
                builder.append_mapped(token.content, source_base + content_start)
                if closing >= 0:
                    builder.append_mapped(markup, source_base + closing)
                    raw_cursor = closing + len(markup)
                else:
                    builder.append(markup)
            else:
                builder.append(f"`{token.content}`")
        elif kind in {"softbreak", "hardbreak"}:
            newline = raw.find("\n", raw_cursor)
            if newline >= 0:
                builder.append_mapped("\n", source_base + newline)
                raw_cursor = newline + 1
            else:
                builder.append("\n")
        elif kind == "image":
            builder.append(token.content or "Изображение")
        elif kind == "strong_open":
            raw_cursor = _consume_inline_markup(builder, raw, raw_cursor, token.markup or "**", source_base)
            styles["bold"].append(builder.length)
        elif kind == "strong_close" and styles["bold"]:
            start = styles["bold"].pop()
            raw_cursor = _consume_inline_markup(builder, raw, raw_cursor, token.markup or "**", source_base)
            if rich:
                builder.mark("bold", start)
        elif kind == "em_open":
            raw_cursor = _consume_inline_markup(builder, raw, raw_cursor, token.markup or "*", source_base)
            styles["italic"].append(builder.length)
        elif kind == "em_close" and styles["italic"]:
            start = styles["italic"].pop()
            raw_cursor = _consume_inline_markup(builder, raw, raw_cursor, token.markup or "*", source_base)
            if rich:
                builder.mark("italic", start)
        elif kind == "link_open":
            opening = raw.find("[", raw_cursor)
            if opening >= 0:
                builder.advance_source(source_base + opening + 1)
                raw_cursor = opening + 1
            links.append((builder.length, str(token.attrGet("href") or "")))
        elif kind == "link_close" and links:
            start, url = links.pop()
            closing = raw.find(")", raw_cursor)
            if closing >= 0:
                builder.advance_source(source_base + closing + 1)
                raw_cursor = closing + 1
            if rich and url:
                builder.mark("url", start, url=url)
            elif url:
                builder.append(f" — {url}")
    builder.advance_source(source_base + len(raw))


def _line_offsets(source: str) -> list[int]:
    offsets = [0]
    offsets.extend(index + 1 for index, character in enumerate(source) if character == "\n")
    if offsets[-1] != len(source):
        offsets.append(len(source))
    return offsets


def _append_inline_literal(
    builder: _TextBuilder,
    raw: str,
    cursor: int,
    value: str,
    source_base: int,
) -> int:
    if not value:
        return cursor
    start = raw.find(value, cursor)
    if start < 0:
        builder.append(value)
        return cursor
    builder.append_mapped(value, source_base + start)
    return start + len(value)


def _consume_inline_markup(
    builder: _TextBuilder,
    raw: str,
    cursor: int,
    markup: str,
    source_base: int,
) -> int:
    start = raw.find(markup, cursor)
    if start < 0:
        return cursor
    end = start + len(markup)
    builder.advance_source(source_base + end)
    return end


def _parse_table(tokens: list[Token]) -> RenderedTableSegment | None:
    headers: tuple[str, ...] = ()
    rows: list[tuple[str, ...]] = []
    row: list[str] | None = None
    in_header = False
    for token in tokens:
        if token.type == "thead_open":
            in_header = True
        elif token.type == "thead_close":
            in_header = False
        elif token.type == "tr_open":
            row = []
        elif token.type == "inline" and row is not None:
            cell = _TextBuilder()
            _render_inline_rich(
                token.children or [],
                cell,
                rich=False,
                raw=token.content,
                source_base=0,
            )
            row.append(cell.text.strip())
        elif token.type == "tr_close" and row is not None:
            values = tuple(row)
            if in_header and not headers:
                headers = values
            else:
                rows.append(values)
            row = None
    return RenderedTableSegment(headers, tuple(rows)) if headers or rows else None

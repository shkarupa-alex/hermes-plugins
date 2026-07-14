from __future__ import annotations
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from markdown_it import MarkdownIt

if TYPE_CHECKING:
    from markdown_it.token import Token

VK_MENTION = re.compile(r"\[(id|club)(\d+)\|([^\]\r\n]+)]")
MIN_PRINTABLE_CODEPOINT = 32


@dataclass(frozen=True, slots=True)
class RenderedVkMessage:
    text: str
    format_data: dict[str, object] | None
    fallback_text: str
    capabilities_used: frozenset[str]


@dataclass(frozen=True, slots=True)
class ParsedIncomingMessage:
    markdown: str
    original_text: str
    format_data: dict[str, object] | None


class VkMessageRenderer(Protocol):
    def render_markdown(self, markdown: str) -> RenderedVkMessage: ...

    def parse_incoming(self, text: str, format_data: dict[str, object] | None) -> ParsedIncomingMessage: ...


class PlainVkRenderer:
    def __init__(self) -> None:
        self._parser = MarkdownIt("commonmark", {"html": False, "linkify": True})

    def render_markdown(self, markdown: str) -> RenderedVkMessage:
        text = _render_tokens(self._parser.parse(markdown)).strip()
        return RenderedVkMessage(
            text=text,
            format_data=None,
            fallback_text=text,
            capabilities_used=frozenset(),
        )

    def parse_incoming(self, text: str, format_data: dict[str, object] | None) -> ParsedIncomingMessage:
        clean = sanitize_incoming_text(text)
        clean = VK_MENTION.sub(lambda match: f"{match.group(3)} (@{match.group(1)}{match.group(2)})", clean)
        return ParsedIncomingMessage(markdown=clean, original_text=text, format_data=format_data)


def sanitize_incoming_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return "".join(char for char in normalized if char in "\n\t" or ord(char) >= MIN_PRINTABLE_CODEPOINT)


def split_message(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= limit:
            chunks.append(remaining)
            break
        split_at = remaining.rfind("\n\n", 0, limit + 1)
        if split_at < limit // 3:
            split_at = remaining.rfind("\n", 0, limit + 1)
        if split_at < limit // 3:
            split_at = remaining.rfind(" ", 0, limit + 1)
        if split_at < limit // 3:
            split_at = limit
        chunk = remaining[:split_at].rstrip()
        if not chunk:
            chunk = remaining[:limit]
            split_at = limit
        chunks.append(chunk)
        remaining = remaining[split_at:].lstrip()
    return chunks


def _render_tokens(tokens: list[Token]) -> str:  # noqa: C901, PLR0912
    output: list[str] = []
    link_stack: list[str] = []
    list_index: list[int] = []
    for token in tokens:
        kind = token.type
        if kind == "inline":
            output.append(_render_inline(token.children or []))
        elif kind in {"paragraph_close", "heading_close"}:
            output.append("\n\n")
        elif kind == "bullet_list_open":
            list_index.append(0)
        elif kind == "ordered_list_open":
            list_index.append(int(token.attrGet("start") or 1))
        elif kind in {"bullet_list_close", "ordered_list_close"}:
            if list_index:
                list_index.pop()
            output.append("\n")
        elif kind == "list_item_open":
            if list_index and list_index[-1] > 0:
                output.append(f"{list_index[-1]}. ")
                list_index[-1] += 1
            else:
                output.append("• ")
        elif kind == "list_item_close":
            output.append("\n")
        elif kind in {"fence", "code_block"}:
            output.append(f"Код:\n{token.content.rstrip()}\n\n")
        elif kind == "blockquote_open":
            output.append("> ")
        elif kind == "softbreak":
            output.append("\n")
        elif kind == "link_open":
            link_stack.append(str(token.attrGet("href") or ""))
        elif kind == "link_close" and link_stack:
            output.append(f" — {link_stack.pop()}")
    return _normalize_spacing("".join(output))


def _render_inline(tokens: list[Token]) -> str:
    output: list[str] = []
    links: list[tuple[str, int]] = []
    for token in tokens:
        if token.type in {"text", "code_inline"}:
            output.append(token.content)
        elif token.type in {"softbreak", "hardbreak"}:
            output.append("\n")
        elif token.type == "image":
            output.append(token.content or "Изображение")
        elif token.type == "link_open":
            links.append((str(token.attrGet("href") or ""), len(output)))
        elif token.type == "link_close" and links:
            url, start = links.pop()
            label = "".join(output[start:])
            if url and label != url:
                output.append(f" — {url}")
    return "".join(output)


def _normalize_spacing(text: str) -> str:
    text = re.sub(r"[ \t]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text)

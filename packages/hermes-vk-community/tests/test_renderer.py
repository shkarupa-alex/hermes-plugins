from hermes_vk_community.renderer import PlainVkRenderer, sanitize_incoming_text, split_message


def test_plain_renderer_uses_ast_and_preserves_content() -> None:
    rendered = PlainVkRenderer().render_markdown(
        "# Заголовок\n\n**жирный** и [ссылка](https://example.com)\n\n```python\nprint('ok')\n```"
    )
    assert "Заголовок" in rendered.text
    assert "жирный" in rendered.text
    assert "ссылка — https://example.com" in rendered.text
    assert "Код:\nprint('ok')" in rendered.text
    assert "**" not in rendered.text


def test_incoming_mentions_and_controls() -> None:
    parsed = PlainVkRenderer().parse_incoming("Привет, [id123|Иван]\x00\r\n!", None)
    assert parsed.markdown == "Привет, Иван (@id123)\n!"
    assert sanitize_incoming_text("a\x00b") == "ab"


def test_markdown_list_has_no_blank_lines_between_items() -> None:
    rendered = PlainVkRenderer().render_markdown("- первый пункт\n- второй пункт")
    assert rendered.text == "• первый пункт\n• второй пункт"


def test_chunking_is_lossless_apart_from_boundary_whitespace() -> None:
    chunks = split_message("Первый абзац.\n\nВторой абзац длиннее.", 20)  # noqa: RUF001
    assert all(len(chunk) <= 20 for chunk in chunks)
    assert " ".join(" ".join(chunks).split()) == "Первый абзац. Второй абзац длиннее."

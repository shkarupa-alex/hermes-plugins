# VK rich-text compatibility

Live capability result for the `hermes-vk-community` release gate.

## Tested profile

- Date: 2026-07-15
- VK API: `5.199`
- Transport: community token, `messages.send`
- Community: private group
- Client observation: VK Web manual observation; the UI did not expose an exact build number
- Machine fixture: `tests/fixtures/vk/formatting/live-probe-2026-07-15.json`

## Results

| Candidate | API result | Visual result |
| --- | --- | --- |
| `<b>`, `<i>`, `<u>` in `message` | accepted | tags displayed literally |
| `<a href="…">…</a>` in `message` | accepted | tag displayed literally |
| `<ul><li>…</li></ul>` in `message` | accepted | tags displayed literally |
| `<table>…</table>` in `message` | accepted | tags displayed literally |
| `messages.edit` with HTML and Markdown | accepted; exact readback | markup displayed literally |
| 4096 Cyrillic characters | accepted; exact length and SHA-256 readback | accepted |
| Bare HTTPS URL | accepted | rendered as a clickable link |
| Markdown list degraded to `•` lines | accepted | compact list without blank lines between items |
| `messages.setActivity(type="typing")` | response `1` | typing status visibly displayed |

Every send/edit readback returned the exact input text and no `format_data`.
The official VK API `5.199` schema exposes neither `format_data` nor another
parse-mode/rich-text parameter for `messages.send` or `messages.edit`. An
undocumented `format_data` payload was therefore not adopted or guessed.

## Release decision

- `formatting.mode: auto` resolves to deterministic plain rendering.
- `formatting.mode: rich` remains a validation error.
- Raw HTML must never be emitted: it is user-visible as literal markup.
- Preserve structure with Unicode bullets, whitespace, code labels, and bare
  URLs that VK auto-linkifies.
- Rich tables are unsupported; Markdown tables must use a readable plain
  fallback.

This profile can be revised only after a new live test demonstrates a supported,
documented representation on the target VK API and clients.

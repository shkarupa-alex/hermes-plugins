# VK rich-text compatibility

Live capability profile for the `hermes-vk-community` release gate.

## Tested profile

- Date: 2026-07-15
- VK API: `5.199`
- Transport: community token, `messages.send` and `messages.edit`
- Community: private group
- Client observation: VK Web; the UI did not expose an exact build number
- Machine artifact:
  `tests/fixtures/vk/formatting/live-probe-format-data-2026-07-15.json`
- Certified profile: `format-data-v1`, using Unicode codepoint offsets

## Live results

| Candidate | API/readback | VK Web |
| --- | --- | --- |
| `format_data` bold, italic, underline, URL | accepted and returned | styled correctly |
| nested ranges | accepted and returned | styled correctly |
| partially overlapping ranges | accepted and returned | styled correctly |
| `messages.edit` plus `format_data` | accepted and returned | styled correctly |
| raw HTML | accepted as text | tags displayed literally |
| raw Markdown | accepted as text | markers displayed literally |
| 4096 Cyrillic characters | exact length/hash readback | accepted |
| bare HTTPS URL | exact readback | clickable |
| `messages.setActivity(type="typing")` | response `1` | visibly displayed |
| Unicode-codepoint emoji-adjacent ranges | accepted and returned | intended bold/underline ranges |
| UTF-16 emoji-adjacent ranges | accepted and returned | same intended ranges in VK Web |

The live Unicode probe established that the renderer's Unicode-codepoint
payload styles the intended emoji-adjacent ranges in VK Web. The competing
UTF-16 payload was also normalized/displayed as intended by the tested client;
this does not change the wire contract emitted by the plugin. Python string
indices are therefore certified for API `5.199`, while both probe cases remain
in the regression artifact for future client/API reverification.

## Rendering profile

- Compile Markdown locally; never send raw HTML or Markdown markup.
- Map bold, italic and links to `format_data` version 1. Underline is supported
  by the wire profile even though standard Markdown has no underline node.
- Render headings as bold text.
- Render quotes as `▎ ` followed by italic text.
- Render unordered and ordered lists with Unicode bullets/numbers and preserve
  nesting with spaces.
- Render every Markdown table locally as one or more RGB JPEG images. JPEG is
  the only table wire representation, independent of formatting mode.
- Lower a standalone Markdown image to a VK photo attachment. Keep an inline
  image embedded in a text line as an alt-text fallback rather than guessing a
  disruptive message split.
- Preserve mixed-content order by sending a sequence such as text → photo →
  text. One logical Hermes reply is not required to fit one VK message.
- Rebase and clip formatting ranges after each 4096-character split.
- Persist each text chunk's exact `format_data` in the durable outbox so replay
  does not silently lose styling.

`formatting.mode: auto` uses this rich profile. Explicit `rich` is accepted for
API `5.199`; `plain` remains a deterministic fallback that strips native text
ranges but retains JPEG tables. Unknown API/profile combinations continue to
fail closed to plain mode, and explicit `rich` is rejected for them.

## Reverification

Run the private-peer probe after changing the VK API version or supported
clients:

```bash
hermes vk probe-formatting --peer-id <PRIVATE_TEST_PEER_ID> \
  --output vk-formatting-format-data-probe.json
```

The command tests send/edit, nesting, partial overlap and competing Unicode
offset hypotheses, then removes its visible probe messages best-effort. Update
both the redacted artifact and the capability profile only after manual client
observation. Never commit a community token or private peer identifier.

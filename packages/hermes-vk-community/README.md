# Hermes VK Community

External VK Community Long Poll platform plugin for Hermes Agent 0.18.x.

## Development install

```bash
uv sync --all-packages
uv pip install -e packages/hermes-vk-community
hermes plugins enable vk-community
```

Add `VK_COMMUNITY_TOKEN` to the active Hermes profile's `.env`, then add:

```yaml
platforms:
  vk:
    enabled: true
    group_id: 123456789
    allowed_user_ids: [987654321]
    typing_indicator: true
```

VK access is deliberately YAML-only and deny-by-default. The token is never
accepted in YAML. Until a live formatting capability profile has been recorded,
`formatting.mode: auto` safely converts Hermes Markdown to VK-friendly text.


# Hermes VK Community and ONNX ASR plugins

Status: implementation-ready specification, pending final user approval
Date: 2026-07-14
Hermes baseline: `0.18.2`, commit `226e8de827a669e8ffa7035b27d70c19e44b1208`
Target Python: the versions supported by the selected Hermes release

## 1. Objective

Build two independently installable external plugins for
[`NousResearch/hermes-agent`](https://github.com/NousResearch/hermes-agent):

1. `hermes-vk-community`: a VK community-bot gateway using Community Long Poll.
2. `hermes-onnx-asr`: a platform-independent Hermes transcription provider
   backed by [`istupakov/onnx-asr`](https://github.com/istupakov/onnx-asr), with
   `gigaam-v3-e2e-rnnt` as the default model and CPU-only ONNX Runtime on every
   operating system, including macOS.

The packages MUST NOT depend on one another. VK voice messages, Telegram voice
messages, and voice messages from every other Hermes platform MUST reach the
same ONNX ASR provider through Hermes' shared transcription pipeline.

The baseline release MUST work without modifying Hermes core. Two optional
upstream enhancements are specified separately and MUST NOT become hidden
runtime dependencies.

## 2. Confirmed Hermes extension contracts

### 2.1 Platform plugin

The VK package registers through:

```python
ctx.register_platform(
    name="vk",
    label="VK Community",
    adapter_factory=build_adapter,
    check_fn=check_requirements,
    validate_config=validate_config,
    apply_yaml_config_fn=apply_yaml_config,
    required_env=["VK_COMMUNITY_TOKEN"],
    max_message_length=4096,
    allow_update_command=False,
    pii_safe=True,
)
```

The adapter subclasses `BasePlatformAdapter`. It implements at minimum:

```python
async def connect(self, *, is_reconnect: bool = False) -> bool: ...
async def disconnect(self) -> None: ...
async def send(
    self,
    chat_id: str,
    content: str,
    reply_to: str | None = None,
    metadata: dict | None = None,
) -> SendResult: ...
async def send_typing(self, chat_id: str, metadata=None) -> None: ...
async def stop_typing(self, chat_id: str) -> None: ...
async def get_chat_info(self, chat_id: str) -> dict: ...
```

It also implements media methods needed by the v1 support matrix and the
interactive methods `send_clarify`, `send_exec_approval`, and
`send_slash_confirm`. `send_exec_approval` is not a base ABC method in Hermes
0.18.2; its signature is pinned against the optional `getattr` invocation in
`gateway/run.py`. The other method signatures are compared with their actual
base/call-site contracts rather than assuming a shared ABC.

The class sets `splits_long_messages = True` because `send()` owns VK chunking,
and overrides `enforces_own_access_policy` to return `True`. During
construction it sets `_dm_policy`, `_group_policy`, and the normalized
`allow_from` view from the validated YAML policy. V1 maps user-facing
`group_policy: deny` to internal `_group_policy="disabled"`; group-message
admission is not supported in v1, so `_group_policy="allowlist"` is forbidden.
This is an observed Hermes
convention, not an assumed base attribute: when no env allowlist exists,
`GatewayAuthorizationMixin._is_user_authorized()` calls
`_adapter_enforces_own_access_policy()`, then reads the live adapter through
`_adapter_dm_policy()` / `_adapter_group_policy()`. It returns true only when
the effective policy for that chat type is exactly `allowlist`. Therefore the
VK adapter sets `_dm_policy="allowlist"` and `_group_policy="disabled"`.
`allow_from` is consumed by the
adapter's own intake gate, not directly by Hermes core. Contract tests exercise
this exact no-env branch through the real mixin and prove an admitted listed
user is accepted. Unlisted-user denial is asserted at the adapter intake gate;
such a user MUST NOT reach the mixin at all.

V1 deliberately does not register `allowed_users_env`, `allow_all_env`,
`cron_deliver_env_var`, or `standalone_sender_fn`. Authorization is YAML-only,
and scheduled delivery from a process without a live adapter is out of scope.
These omissions avoid a second policy source and avoid advertising a cron path
that Hermes cannot execute without a standalone sender.

### 2.2 Transcription plugin

The ASR package registers an instance of:

```python
class OnnxAsrProvider(TranscriptionProvider):
    @property
    def name(self) -> str:
        return "onnx_asr"

    def transcribe(
        self,
        file_path: str,
        *,
        model: str | None = None,
        language: str | None = None,
        **extra,
    ) -> dict: ...
```

using:

```python
ctx.register_transcription_provider(OnnxAsrProvider())
```

Hermes forwards only `file_path`, `model`, and `language`. Plugin-specific
settings live under `stt.onnx_asr` in the current profile's `config.yaml`.
The provider obtains the current profile root through
`hermes_constants.get_hermes_home()` and reads the section with Hermes' safe
configuration loader.

`stt.onnx_asr.model` is forwarded by Hermes and is authoritative. A direct
`model=` argument passed to `transcribe_audio()` overrides the configured
model, consistent with Hermes' existing dispatcher.

### 2.3 Compatibility gate

Both packages MUST test:

- the minimum supported Hermes release;
- the latest Hermes release;
- current Hermes `main` as an allowed-to-fail early-warning job.

Startup MUST fail with an actionable compatibility error if a required ABC,
registration method, or method signature is absent. Signature drift MUST NOT
be handled by silently dropping arguments.

The harness compares `inspect.signature()` for `connect`, `send`, typing,
editing, media, and interactive call sites against every supported Hermes
version. In particular it tests the initial Hermes call and the reconnect call
with `connect(is_reconnect=...)`.

## 3. Package and repository structure

The source repository is a monorepo with independent wheels:

```text
hermes-plugins/
├── packages/
│   ├── hermes-vk-community/
│   │   ├── pyproject.toml
│   │   ├── plugin.yaml
│   │   ├── src/hermes_vk_community/
│   │   └── tests/
│   └── hermes-onnx-asr/
│       ├── pyproject.toml
│       ├── plugin.yaml
│       ├── src/hermes_onnx_asr/
│       └── tests/
├── integration-tests/
├── docs/
└── hermes-agent/          # inspected reference checkout; never included in wheels
```

Primary installation:

```shell
pip install hermes-vk-community
pip install hermes-onnx-asr
```

Editable installs and directory plugins under `~/.hermes/plugins` are supported
for development. Python package entry points are the production discovery
mechanism.

Each wheel declares the exact Hermes discovery group and a module-level
`register(ctx)` function:

```toml
[project.entry-points."hermes_agent.plugins"]
vk_community = "hermes_vk_community.plugin:register"
# in the other wheel:
onnx_asr = "hermes_onnx_asr.plugin:register"
```

The VK `plugin.yaml` uses `name: vk-community`, `kind: platform`; the ASR
manifest uses `name: onnx-asr`, `kind: backend`. Both declare version,
description, author, and dependency metadata; VK declares
`VK_COMMUNITY_TOKEN` in `requires_env` as `password: true`. Installation documentation
must include plugin discovery/listing and the Hermes enable step required by
the pinned release; `doctor` fails if the entry point is installed but not
enabled.

Subcommands are registered from `register(ctx)` with
`ctx.register_cli_command(...)`; no package patches Hermes' root CLI. The VK
wheel registers the `vk` command group and the ASR wheel the `onnx-asr` group,
with setup/handler functions whose signatures are pinned by the contract
harness.

The package metadata declares the minimum `hermes-agent>=0.18.2`. Runtime
compatibility remains fail-closed outside the contract-tested range, initially
`>=0.18.2,<0.19`; extending that range requires contract-suite success.

## 4. VK Community plugin

### 4.1 Scope

The v1 transport is VK Bots Community Long Poll. Callback API is out of scope.
The intended default deployment is a private community and a numeric allowlist
containing one human user.

The plugin uses a small direct async HTTP client rather than a third-party VK
SDK. This keeps Long Poll cursor handling, retries, URL validation, token
redaction, and new fields such as `format_data` under explicit control.

VK API version `5.199` is the baseline transport contract. Rich-text support is
additionally governed by the release's tested formatting capability profile,
because the public JSON Schema can lag the live API.

### 4.2 Configuration

Minimal `config.yaml`:

```yaml
platforms:
  vk:
    enabled: true
    group_id: 123456789
    allowed_user_ids:
      - 987654321
    typing_indicator: true
```

Secret:

```dotenv
VK_COMMUNITY_TOKEN=vk1.a...
```

Full schema:

```yaml
platforms:
  vk:
    enabled: true
    group_id: 123456789
    api_version: "5.199"

    allowed_user_ids: [987654321]
    dm_policy: allowlist
    group_policy: deny

    pairing:
      enabled: false
      code_ttl_seconds: 600

    long_poll:
      wait_seconds: 25
      retry_min_seconds: 1
      retry_max_seconds: 60

    storage:
      path: null  # default: get_hermes_home()/vk-community/<group_id>/state.sqlite3

    media:
      max_download_bytes: 52428800
      connect_timeout_seconds: 10
      total_timeout_seconds: 120

    formatting:
      mode: auto          # auto | rich | plain
      fallback: plain
      disable_mentions: true
      parse_link_previews: true
      table_style: jpeg

    streaming:
      enabled: true
      update_interval_seconds: 1.5

    typing_indicator: true
    typing:
      refresh_seconds: 4
      failure_cooldown_seconds: 30

    max_message_length: 4096
```

Hermes does not retain arbitrary platform keys automatically. The plugin
therefore registers `apply_yaml_config_fn=apply_yaml_config`. The hook receives
`(yaml_cfg, platform_cfg)`, deep-copies the complete VK configuration, and
returns a dict placed by Hermes under `PlatformConfig.extra`. Because pinned
Hermes catches hook exceptions and continues, the hook MUST NOT signal invalid
input by raising. It returns a bounded `_vk_validation_errors` list for unknown
or malformed plugin-owned keys; the registered `validate_config` treats any
entry as a fatal startup error. `adapter_factory` refuses configuration that
has not passed validation and reads VK-owned settings only from the validated
`extra` dict. A round-trip contract test proves every key in both examples
above reaches the adapter unchanged after normalization and every invalid key
fails startup rather than disappearing in Hermes' debug log.

Shared Hermes fields (`enabled`, `dm_policy`, `group_policy`, `allow_from`, and
`typing_indicator`) are also normalized into the effective adapter policy. The
canonical user-facing key is `allowed_user_ids`; the hook derives
`extra.allow_from` as decimal strings and rejects a simultaneously supplied,
different `allow_from` value.

Validation rules:

- `group_id` and allowlist entries are positive integers.
- Default access is deny.
- V1 requires `dm_policy: allowlist` and `group_policy: deny`; group events are
  discarded before dispatch. Group allowlists are a later separately specified
  feature, not an accepted hidden value.
- `VK_COMMUNITY_TOKEN` is never accepted in YAML.
- `access_token_env` is not supported; the secret name is fixed.
- Conflicting YAML and environment authorization policy is fatal at startup.
- `GATEWAY_ALLOW_ALL_USERS`, `VK_ALLOW_ALL_USERS`, and `VK_ALLOWED_USERS` MUST
  NOT silently weaken the YAML policy.
- `VK_ALLOW_ALL_USERS` and `VK_ALLOWED_USERS` are not registered with Hermes;
  if present, startup fails because VK authorization is YAML-only. Global
  gateway authorization cannot bypass the adapter's earlier own-policy gate.
- To keep the effective policy strictly YAML-only rather than an undocumented
  intersection/union, VK platform startup also fails when
  `GATEWAY_ALLOWED_USERS` is nonempty or `GATEWAY_ALLOW_ALL_USERS` is truthy.
  `doctor` names the conflicting variable without printing its value.
- `formatting.mode` is one of `auto`, `rich`, `plain`.
- Timeouts and byte limits are positive and bounded.
- A null `storage.path` resolves under the active profile returned by
  `get_hermes_home()` and the configured `group_id`; a relative override is
  resolved under that same root.
  Cross-profile shared SQLite is never the default.

### 4.3 Secret handling

`plugin.yaml` declares `VK_COMMUNITY_TOKEN` as a password field. Runtime code
resolves it only through:

```python
from agent.secret_scope import get_secret
token = get_secret("VK_COMMUNITY_TOKEN")
```

Direct runtime `os.getenv("VK_COMMUNITY_TOKEN")` is prohibited by a test. This
preserves Hermes single-profile `.env` behavior and its multiplexed secret
scope. The token MUST NOT be returned by `env_enablement_fn`, written to SQLite,
included in exceptions, or logged in URLs.

`connect()` is invoked inside Hermes' active profile secret scope and resolves
the token exactly once there. The adapter stores the resulting secret only in
its in-memory instance for the lifetime of that connection; background polling
tasks do not call `get_secret` after leaving the scoped construction path. A
multiplex integration test creates two profile scopes and proves each adapter
uses only its own token without `UnscopedSecretError` or cross-profile leakage.

### 4.4 Connection lifecycle and locking

`connect(*, is_reconnect=False)` performs, in order:

1. Validate configuration and authorization-policy consistency.
2. Resolve the scoped token.
3. Acquire the Hermes platform lock through the base helper.
4. Call an authenticated identity/group probe and verify `group_id`.
5. Open/migrate SQLite.
6. Recover the stored Long Poll cursor and ambiguous records.
7. Obtain a Long Poll lease.
8. Start the polling task and return success.

On `is_reconnect=True`, it first idempotently closes the previous polling HTTP
session and task, retains the durable SQLite state, reloads the committed
cursor, obtains a fresh Long Poll server/key, and resumes from that cursor. It
never drops pending updates and never replaces a valid committed `ts` merely
because the process reconnected. If VK returns `failed=3`, only the explicit
protocol rule below permits adopting a new lease `ts` and recording a history
gap. Cold connect and reconnect are covered separately by contract tests.

`disconnect()` cancels polling, waits for the active HTTP request, closes HTTP
and SQLite resources, and releases the base platform lock.

Lock contention is retryable and reported by `doctor`; it is not repaired by
deleting another live process's lock.

### 4.5 Long Poll state machine

On a normal response:

1. Normalize and validate every update.
2. For every update, durably insert, deduplicate, or quarantine it.
3. Commit the batch.
4. Persist the returned `ts` only after step 3 succeeds for every update.
5. Dispatch admitted events to Hermes.

A crash before step 4 intentionally causes VK to replay the batch.

Protocol errors:

- `failed=1`: adopt the returned `ts`.
- `failed=2`: refresh `server` and `key`, retaining the previous `ts`.
- `failed=3`: obtain a complete new lease including `ts`; emit a history-gap
  metric and warning.
- network errors and HTTP 5xx: exponential backoff with full jitter between the
  configured bounds.
- authentication, group mismatch, and invalid configuration: fail fast.

The Long Poll server is accepted only when HTTPS, without userinfo, and its
canonical hostname is `lp.vk.com`, a subdomain of `.vk.com`, or a subdomain of
`.userapi.com` present in the release fixture. Media downloads additionally
permit the release-tested suffixes `.userapi.com`, `.vk-cdn.net`, and `.vkuser.net`.
Suffix matching is label-aware (`host == suffix` or `host.endswith("."+suffix)`),
never substring-based. The fixture and update procedure are versioned; `doctor`
prints rejected new hosts without secrets so the allowlist can be deliberately
expanded in a patch release.

Redirects are disabled for Long Poll and enabled for media only as explicit,
bounded hops that repeat the full validation. DNS A/AAAA answers are resolved,
all private, loopback, link-local, multicast, reserved, unspecified, and
metadata-service addresses are rejected, and the HTTP transport connects to a
selected validated address while preserving the original hostname for TLS SNI
and certificate verification. Connection reuse is scoped to that hostname and
validated address set; DNS is re-resolved before a new connection. Tokens,
Long Poll keys, and media authorization never appear in logs or exception URLs.

The concrete direct dependency requires `aiohttp>=3.14.1`; the release lock
pins the tested version matching the Hermes
messaging extra. `ClientSession(trust_env=False)` forbids inherited HTTP(S)
proxies. `VkPinnedResolver` implements `aiohttp.abc.AbstractResolver`: it
canonicalizes and suffix-checks the URL host, resolves A/AAAA with the event
loop resolver, and returns aiohttp `ResolveResult` records containing exactly
`hostname`, validated numeric `host`, requested `port`, address `family`,
`proto=0`, and `flags=0`.

Host canonicalization rejects userinfo, IP literals, non-ASCII/IDNA names,
percent escapes, NUL/control characters, empty labels, labels outside DNS
length rules, and a trailing dot; ASCII is lowercased before label-aware suffix
matching. Every resolved address is parsed with `ipaddress.ip_address()` and is
accepted only when `is_global is True`. IPv4-mapped IPv6 and the well-known
NAT64 prefix are additionally decoded and their embedded IPv4 must also be
global. Thus loopback, private/ULA, link-local, unspecified, reserved,
multicast, documentation, benchmarking, CGNAT `100.64.0.0/10`, metadata
endpoints, and alternate textual encodings fail closed. If a DNS answer set
contains even one invalid address, the entire resolution is rejected rather
than silently selecting its public sibling.

The resolver returns only validated
addresses to `aiohttp.TCPConnector`. Requests keep the original hostname in the
URL, so aiohttp connects to the resolver-supplied IP while its normal TLS stack
uses the original hostname for SNI and `SSLContext(check_hostname=True,
CERT_REQUIRED)` certificate verification. The connector uses
`use_dns_cache=False`; a new TCP connection always re-enters the resolver, while
an already-authenticated pooled connection is reusable only for the same URL
host. Long Poll uses `allow_redirects=False`. Media follows at most three
redirects and only status 301, 302, 303, 307, or 308; relative `Location` is
resolved against the current validated URL, then the response is closed and
the complete URL+DNS policy is repeated before the next hop. Missing/invalid
`Location`, any other 3xx, downgrade, userinfo, or a fourth hop is terminal.
Tests use a fake resolver plus a local CA/TLS server to prove pinning,
original-host SNI, cert failure, mixed public/private answers, rebinding,
IPv4/IPv6/mapped forms, proxy non-use, and redirect rules.

### 4.6 Durable inbound inbox

SQLite inbox states:

```text
received -> dispatched -> completed
                   \----> quarantined
```

Primary deduplication order:

1. `group_id + event_id`;
2. `group_id + peer_id + conversation_message_id`;
3. SHA-256 of canonical normalized event JSON.

The row stores sanitized identifiers, canonical hash, state, timestamps, retry
metadata, and a bounded sanitized error. It does not store tokens or downloaded
media bytes.

SQLite opens with `PRAGMA journal_mode=WAL`, `foreign_keys=ON`,
`synchronous=FULL`, and a 5000 ms `busy_timeout`. Schema version 1 is created in
one `BEGIN IMMEDIATE` migration and contains, at minimum:

```sql
CREATE TABLE schema_meta (
  singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
  version INTEGER NOT NULL
);
CREATE TABLE long_poll_cursor (
  group_id INTEGER PRIMARY KEY,
  ts TEXT NOT NULL,
  updated_at_ms INTEGER NOT NULL
);
CREATE TABLE inbox (
  id INTEGER PRIMARY KEY,
  group_id INTEGER NOT NULL,
  event_id TEXT,
  peer_id INTEGER,
  conversation_message_id INTEGER,
  canonical_sha256 TEXT NOT NULL,
  state TEXT NOT NULL CHECK(state IN
    ('received','dispatched','completed','quarantined')),
  normalized_json TEXT NOT NULL CHECK(length(normalized_json) <= 262144),
  attempts INTEGER NOT NULL DEFAULT 0,
  error TEXT CHECK(error IS NULL OR length(error) <= 2048),
  created_at_ms INTEGER NOT NULL,
  updated_at_ms INTEGER NOT NULL
);
CREATE UNIQUE INDEX inbox_event_id_uq
  ON inbox(group_id, event_id) WHERE event_id IS NOT NULL;
CREATE UNIQUE INDEX inbox_cmid_uq
  ON inbox(group_id, peer_id, conversation_message_id)
  WHERE peer_id IS NOT NULL AND conversation_message_id IS NOT NULL;
CREATE UNIQUE INDEX inbox_hash_uq ON inbox(group_id, canonical_sha256);
```

Canonical JSON is UTF-8 RFC-style JSON with sorted object keys, no insignificant
whitespace, normalized integer values, preserved array order, and the volatile
Long Poll envelope fields removed by an explicit allowlist; CI freezes vectors
including Cyrillic and emoji. Each accepted response batch inserts/quarantines
events and updates `long_poll_cursor.ts` in the same `BEGIN IMMEDIATE`
transaction. `failed=1` adoption also updates the cursor in its own durable
transaction before polling resumes.

Current Hermes `handle_message()` schedules background processing and does not
provide a durable completion acknowledgement. Therefore v1 guarantees durable
transport admission, not exactly-once agent execution.

On startup, `received` rows are automatically dispatched in id order and moved
to `dispatched` in the same transaction that records the dispatch attempt.
`completed` is reserved for an operator reconciliation or a future completion
hook and is never written by automatic v1 processing.

After restart, ambiguous `dispatched` rows are not automatically redriven. They
are listed by:

```shell
hermes vk doctor --inflight
```

An operator may inspect and explicitly reconcile them. A future generic Hermes
completion hook can strengthen this behavior without changing the VK wire
contract.

### 4.7 Authorization before media I/O

For every inbound event, the adapter performs these checks before any URL fetch,
attachment cache write, or pairing-media processing:

1. Event type is supported.
2. Sender is a human user unless explicitly permitted otherwise.
3. `from_id`, `peer_id`, DM/group policy, and numeric allowlist pass.
4. Pairing state permits the message.

In pairing mode an unauthorized user may submit only the minimal text pairing
code. Attachments, forwards, payloads, and oversized text are ignored until the
user is paired. Pairing codes are one-time, short-lived, and stored hashed.

Hermes core authorization remains a second line of defense; adapter-side
authorization is authoritative for preventing media I/O.

### 4.8 Inbound message mapping

`message_new.object.message.text` is plain user text unless accompanied by a
validated `format_data` structure.

The adapter:

- converts CRLF to LF and removes forbidden control characters;
- preserves the original text and `format_data` in bounded platform metadata;
- converts VK mentions such as `[id123|Name]` to `Name (@id123)` for the agent;
- represents reply and forwarded messages as separate structured context, not
  as undifferentiated user text;
- maps supported media to Hermes media objects;
- describes unsupported attachments with safe deterministic text.

If validated `format_data` exists, the formatting parser reconstructs canonical
Markdown for supported styles. Unknown, overlapping, or invalid ranges fall back
to plain text. Offset interpretation (bytes, code points, or UTF-16 units) is a
release-tested capability and MUST NOT be guessed.

Inbound rich mapping, when proven by fixtures:

- bold -> `**text**`;
- italic -> `*text*`;
- underline -> `<u>text</u>`;
- link -> `[text](url)`;
- code, quote, lists, and table -> their canonical Markdown forms only when the
  capability profile proves their structure.

### 4.9 Attachment support matrix

| VK feature | Inbound v1 | Outbound v1 |
|---|---|---|
| Text | yes | yes |
| Photo | download | upload/attachment |
| Document | download | upload/attachment |
| `audio_message` | Hermes VOICE -> shared STT | upload for TTS/file |
| Ordinary audio | accessible object/URL | document fallback |
| Reply | structured | `reply_to` |
| Forwarded messages | structured | deferred |
| Video | metadata/link | deferred |
| Sticker | safe description | no |
| Poll/wall/article/link | safe description/URL | no |
| Geolocation | coordinates as text | no |
| Reactions | ignored as service events | no |
| Keyboard | validated input | limited one-time buttons |
| Carousel/template | no | deferred |

Media downloads require HTTPS and an approved host, reject redirects to an
unapproved host, enforce declared and actual byte limits, and stream to a
private temporary file. Oversized media is rejected, never truncated and passed
off as complete.

Outbound media uses explicit VK flows, each covered by redacted fixtures:

- photo: `photos.getMessagesUploadServer(peer_id)` -> validated multipart
  `photo` upload -> `photos.saveMessagesPhoto(server, photo, hash)` ->
  `photo{owner_id}_{id}` attachment (including `access_key` when returned);
- document: `docs.getMessagesUploadServer(peer_id)` -> validated multipart
  `file` upload -> `docs.save(file, title)` -> `doc{owner_id}_{id}` attachment;
- voice/TTS: the same document flow with the release-tested
  `type=audio_message` request shape; if the live spike does not prove it, v1
  sends a normal document rather than claiming a voice message.

Every API-returned upload URL passes the same HTTPS, suffix, DNS/IP pinning,
redirect, timeout, and byte-limit policy as downloads. Content is MIME-sniffed;
declared type, sniffed type, extension, and release-tested VK limits must agree.
Upload-server acquisition is safe to retry. An ambiguous multipart or save
response may leave an orphaned VK object but cannot be reused blindly; it is
recorded for diagnostics and a fresh upload flow is required. Only the final
`messages.send` makes media visible and is governed by the outbox/random-id
contract in §4.14.

### 4.10 Markdown and rich-text rendering

VK formatting has three modes:

- `auto`: use the capability profile tested for the pinned API/client matrix;
  use plain text for an unknown profile.
- `rich`: require the tested rich representation; `doctor` fails if unavailable.
- `plain`: deterministic Markdown-to-text degradation.

Renderer interface:

```python
@dataclass(frozen=True)
class RenderedVkMessage:
    text: str
    format_data: dict | None
    fallback_text: str
    capabilities_used: frozenset[str]

class VkMessageRenderer(Protocol):
    def render_markdown(self, markdown: str) -> RenderedVkMessage: ...
    def parse_incoming(
        self,
        text: str,
        format_data: dict | None,
    ) -> ParsedIncomingMessage: ...
```

The renderer uses a Markdown AST parser, not regex-only rewriting. It never
sends generated HTML or raw Markdown. The tested mappings are:

| Markdown | Rich candidate | Guaranteed plain fallback |
|---|---|---|
| bold | `format_data` bold range | text |
| italic | `format_data` italic range | text |
| underline | `format_data` underline range | text |
| link | `format_data` url range | `label — URL` |
| heading | `format_data` bold range | paragraph |
| inline code | `<code>` | literal backticks |
| fenced code | `<pre><code>` | `Code:` plus exact body |
| quote | `▎` prefix plus italic range | `▎ text` |
| strike | `<s>`/`<del>` | text |
| spoiler | tested special representation | `Spoiler: text` |
| list | Unicode bullets/numbers with indentation | bullets/numbers |
| standalone image | VK photo attachment | alt text plus URL |
| table | one or more RGB JPEG photos | row records |

Bold, italic, underline, and URL ranges are accepted by send and edit and were
visually confirmed in VK Web. Tables deliberately do not rely on native VK
markup: they are always rendered locally as JPEG photos. Text before and after
a table is delivered as separate messages in source order.

Formatting is performed before chunking. Chunking preserves paragraphs, list
items, and code bodies. The conservative baseline `messages.send` limit is 4096
characters; a committed live capability fixture may certify 9000 for a specific
request mode/API profile. Error 914 follows the progressive lower-limit
algorithm in §4.14. `disable_mentions=true` is sent by default. Link preview
parsing is a separate setting.

If a rich request is rejected before any visible send, the same logical chunk
may be rendered and sent once as plain text with a fresh `random_id`. A response
that may already be visible is never blindly retried merely to change formatting.

### 4.11 Mandatory live formatting spike

Before production renderer implementation, run:

```shell
hermes vk probe-formatting --peer-id <TEST_PEER_ID>
```

The command requires an explicit private test peer. It tests:

- HTML directly in `message`;
- `message` plus `format_data`;
- readback representation;
- bold, italic, underline, link, strike, code, pre, quote, lists, headings,
  spoiler candidates, nested styles, and tables;
- escaping, Cyrillic, emoji, malformed tags, long messages, attachments;
- `messages.edit` and streaming finalization;
- VK Web, one mobile client, and VK Desktop/macOS when available.

Results are committed as redacted fixtures and a capability document:

```text
docs/vk-rich-text-compatibility.md
tests/fixtures/vk/formatting/
```

The fixture records request shape, redacted API response, readback `text`,
`format_data`, tested client, and manual visual outcome. Raw HTML appearing in a
client is a failed capability.

This spike is a release gate for rich renderer/edit behavior. Its
machine-readable profile fixes the accepted request field,
schema/version, offset unit, nesting/overlap rules, supported tag/entity set,
send/edit limits, and client matrix. The committed 2026-07-15 artifact enables
`format-data-v1` for `auto` and `rich`; an unknown profile falls back to plain.
Updating the fixture changes the capability profile and tests; implementation
code never guesses an undocumented `format_data` shape.

### 4.12 Typing status

`send_typing()` calls `messages.setActivity` with:

```python
{
    "peer_id": int(chat_id),
    "group_id": group_id,
    "type": "typing",
}
```

Hermes' base `_keep_typing` lifecycle starts only after an admitted message and
pauses during user approval waits. The adapter rate-limits actual VK calls to
one per configured four seconds.

VK exposes no explicit stop activity in the inspected method contract, so
`stop_typing()` is a documented no-op. Activity expires naturally or disappears
after the response. Timeouts, 429, and 5xx are non-fatal and start a per-chat
cooldown.

### 4.13 Interactive buttons

The v1 adapter uses one-time VK text keyboards for clarify, dangerous-command
approval, and slash confirmation. A button contains an opaque random payload
bound server-side to:

- authorized user ID;
- peer ID;
- Hermes session;
- action type and allowed values;
- expiry;
- one-time consumption state.

Visible button text is not authorization. A manually typed copy is ordinary
user input. Payload processing repeats authorization before resolving the
Hermes action.

### 4.14 Outbound delivery, random IDs, and outbox

Long responses are split into logical chunks after rendering. Each chunk gets a
fresh non-zero signed 31-bit `random_id`, allocated once per live logical send
and retained only across retries of that same invocation.

The outbox stores:

```text
peer_id, content_digest, chunk_index, random_id, reply_target,
state, attempt_count, created_at, updated_at, returned_message_id
```

Its version-1 table is:

```sql
CREATE TABLE outbox (
  id INTEGER PRIMARY KEY,
  invocation_id TEXT NOT NULL,
  peer_id INTEGER NOT NULL,
  content_sha256 TEXT NOT NULL,
  chunk_index INTEGER NOT NULL,
  random_id INTEGER NOT NULL CHECK(random_id BETWEEN 1 AND 2147483647),
  reply_target TEXT,
  state TEXT NOT NULL CHECK(state IN
    ('prepared','sending','sent','partial_delivery','delivery_unknown','failed')),
  attempt_count INTEGER NOT NULL DEFAULT 0,
  returned_message_id TEXT,
  error TEXT CHECK(error IS NULL OR length(error) <= 2048),
  created_at_ms INTEGER NOT NULL,
  updated_at_ms INTEGER NOT NULL,
  UNIQUE(invocation_id, chunk_index),
  UNIQUE(peer_id, random_id)
);
```

`random_id` is drawn with rejection sampling under the write transaction until
the per-peer unique index accepts it. `prepared -> sending` commits immediately
before the HTTP request can begin; `sending -> sent` commits with the returned
message id. On restart, `prepared` is safe to send, while `sending` becomes
`delivery_unknown` and is never automatic. This distinction is tested with a
crash at every boundary.

States include `prepared`, `sending`, `sent`, `partial_delivery`,
`delivery_unknown`, and `failed`.

Rules:

- transient retries in the same live invocation reuse `random_id`;
- a later identical response gets a new `random_id`;
- content-derived permanent IDs are forbidden;
- after restart, an ambiguous `sending` row becomes `delivery_unknown` and is
  never automatically resent;
- after one or more visible chunks, a later permanent failure records terminal
  `partial_delivery` with delivered message IDs; the return shape differs for
  `send()` and `edit_message()` as specified below so the actual Hermes
  consumers suppress duplicates.

The adapter maps its outbox outcome to the pinned Hermes `SendResult` exactly:

| Outcome | `success` | `error` / IDs | Retry fields | Classification / metadata |
|---|---:|---|---|---|
| all chunks sent | `True` | `message_id` is the last visible VK id; `continuation_message_ids` contains preceding ids in send order | `retryable=False` | bounded redacted `raw_response` |
| definite failure before any request may have succeeded | `False` | bounded human-readable `error`; no IDs | transient: `retryable=True`, optional `retry_after`; permanent: `False` | `too_long`, `transient`, `rate_limited`, `bad_format`, `forbidden`, `not_found`, or `unknown` |
| request may have succeeded but response was lost | `False` | `error="VK delivery timed out after the request may have succeeded"`; no IDs | `retryable=False` | `error_kind="unknown"`, `raw_response={"delivery_unknown": true, "outbox_id": ...}` |
| `send()` failed after visible chunks | `True` | last delivered id plus preceding continuation ids | `retryable=False` | `raw_response.partial_delivery` with delivered/total counts and missing-tail digest |
| `edit_message()` failed after visible overflow chunks | `False` | `error="overflow_continuation_failed"`; last visible IDs | matches Hermes' existing edit convention | `raw_response.partial_overflow` with `delivered_chunks`, `total_chunks`, `last_message_id`, exact `delivered_prefix`, and continuation IDs |

These shapes intentionally follow observed Hermes branches. A partial ordinary
`send()` reports transport success because content is already visible; that
prevents `BasePlatformAdapter._send_with_retry()` from issuing its automatic
whole-message `content[:3500]` plain fallback. The durable outbox and `doctor`
retain the partial-failure diagnosis. A `delivery_unknown` failure contains the
exact `timed out` marker recognized by `_is_timeout_error()`, with
`retryable=False`, so `_send_with_retry()` returns it without retry or plain
fallback. `edit_message()` is not wrapped by `_send_with_retry()`; its
`partial_overflow` shape is the existing Telegram/`stream_consumer.py` contract,
which preserves the visible prefix and sends only the missing final tail.

For `edit_message()`, `delivered_prefix` is a literal slice of the original
unrendered `content` argument: `content[:source_span_end]`. The AST renderer
keeps source-span boundaries for every emitted wire chunk, so transport markup,
chunk counters, escaping, and preview cursors are never included in this field.
The invariant `content.startswith(delivered_prefix)` is checked before return.
`continuation_message_ids` contains only newly created continuation messages in
send order (not the original edited `message_id`); `last_message_id` is the last
visible continuation or the original id when none exists. The stream integration
test proves the fallback payload is exactly
`content[len(delivered_prefix):].lstrip()`.

No result after a possibly visible ordinary send is marked retryable.
Integration tests invoke the real `_send_with_retry()` and real stream consumer
and prove they never emit the whole response a second time after
`partial_delivery`, `partial_overflow`, or `delivery_unknown`.

The conservative release limit is 4096 characters. A formatting spike may
certify a higher per-mode limit (the inspected 5.199 schema says 9000), but the
larger value is enabled only by a committed capability fixture. Error 914
causes a definitely-rejected unsent chunk to be re-rendered with a progressively
smaller limit (bounded binary reduction, never below 256) and caches the lower
limit for the process/profile. Already delivered chunks are never resent; a
repeated 914 is terminal only at the floor.
When 914 is finally surfaced to Hermes, `error_kind="too_long"` and a bounded
`error` are set; all internally recoverable 914 responses remain inside
`send()` and never reach the base fallback layer.
The visible-delivery rule has precedence over this floor rule: if any earlier
chunk is visible, even a floor-level 914 returns the `success=True`
`partial_delivery` shape. A `success=False, error_kind="too_long"` result is
possible only when zero chunks were sent, so it cannot duplicate a visible
prefix.

Streaming previews are bounded, deduplicated, and plain-safe because partial
Markdown is often invalid. Intermediate overflow is truncated with a preview
indicator, never split. At finalization the completed Markdown is rendered and
the first message is edited; additional chunks are emitted only then. If rich
editing is not supported by the tested capability profile, the final rich
response is sent as a new message and the preview is safely finalized without a
whole-response resend.

This behavior is wired to Hermes rather than implemented as a parallel stream
loop. The adapter returns `False` from `supports_draft_streaming(...)`,
implements the exact base signature
`edit_message(chat_id, message_id, content, *, finalize=False)`, and implements
`prefers_fresh_final_streaming(content, metadata=None)`. The latter returns
`True` only when the live capability profile proves rich send but not equivalent
rich edit; otherwise finalization edits in place. `delete_message` is used
best-effort for a replaced preview when the tested VK method permits it.

### 4.15 VK diagnostics

Commands:

```shell
hermes vk setup
hermes vk test-auth
hermes vk doctor
hermes vk doctor --inflight
hermes vk doctor --delivery-unknown
hermes vk probe-formatting --peer-id <TEST_PEER_ID>
```

`doctor` checks discovery, token presence and scope, group identity, Long Poll,
API/capability profile, auth conflicts, SQLite schema, lock ownership, inflight
inbox rows, ambiguous outbox rows, and media/formatting dependencies. It never
prints token values or private message bodies.

## 5. ONNX ASR plugin

### 5.1 Scope and default

The provider name is `onnx_asr`. The default model is:

```text
gigaam-v3-e2e-rnnt
```

The package metadata requires `onnx-asr[cpu]>=0.12.0` and
`onnxruntime>=1.23.2`; per-platform lock files pin the concrete versions used
for each release. It never installs
`onnxruntime-gpu`. A platform for which those exact wheels are unavailable is
not released by silently loosening the pins; updating either pin requires the
full model, CPU-provider, codec, and long-file suite. This deliberately avoids
the upstream-incompatible ONNX Runtime 1.24.1 release.

All aliases advertised by the exact pinned `onnx-asr` release are generated
into a versioned catalog and load-tested. Initial Russian candidates include:

- `gigaam-v2-ctc`;
- `gigaam-v2-rnnt`;
- `gigaam-v3-ctc`;
- `gigaam-v3-rnnt`;
- `gigaam-v3-e2e-ctc`;
- `gigaam-v3-e2e-rnnt`;
- `nemo-fastconformer-ru-ctc`;
- `nemo-fastconformer-ru-rnnt`.

An alias is shown only if release CI can load it with the pinned package and
run a smoke transcription. Automatic model selection and automatic fallback to
a different model are forbidden.

### 5.2 Configuration

```yaml
stt:
  provider: onnx_asr

  onnx_asr:
    model: gigaam-v3-e2e-rnnt
    quantization: int8
    model_dir: "~/.hermes/models/onnx-asr"
    allow_runtime_download: false
    language: null

    vad:
      min_audio_seconds: 20
      engine: silero
      threshold: 0.5
      negative_threshold: 0.35
      min_speech_duration_ms: 250
      max_speech_duration_s: 20
      min_silence_duration_ms: 100
      speech_pad_ms: 30

    audio:
      max_duration_seconds: null
      temp_safety_margin_bytes: 1073741824

    runtime:
      concurrency: 1
      queue_depth: 4
      ffmpeg_timeout_seconds: 3600
      transcription_timeout_seconds: 21600
      intra_op_num_threads: 0
      inter_op_num_threads: 0
```

VAD threshold semantics:

- non-negative number: apply VAD at or above this duration;
- `0`: apply VAD to every recording;
- `null`: disable VAD and do not load Silero;
- negative or non-numeric: configuration error.

Validation also requires
`0 <= negative_threshold < threshold <= 1`, positive VAD durations, positive
timeouts, concurrency `>=1`, and queue depth `>=0`.

The public `negative_threshold` key maps exactly to upstream
`neg_threshold`. A unit test MUST prove the keyword passed to `.with_vad()`.

### 5.3 Thin-wrapper principle

The provider MUST rely on `onnx-asr` for WAV reading, channel selection,
resampling, preprocessing, VAD segmentation, and inference. It MUST NOT build a
parallel NumPy decoding or resampling pipeline.

Compatible PCM WAV input is passed directly:

```python
text = base_model.recognize(wav_path, channel="mean")
```

`onnx-asr` accepts PCM U8/16/24/32 WAV paths and supported sample rates. For
other containers/codecs the plugin performs only normalization to PCM WAV.

### 5.4 Model and VAD construction

Hard-coded runtime provider list:

```python
CPU_PROVIDERS = ["CPUExecutionProvider"]
```

Construction:

```python
cpu_session_config = {
    "sess_options": session_options,
    "providers": CPU_PROVIDERS,
}

base_model = onnx_asr.load_model(
    configured_model,
    path=installed_model_path,
    quantization=configured_quantization,
    providers=CPU_PROVIDERS,
    sess_options=session_options,
    asr_config=dict(cpu_session_config),
    preprocessor_config={
        "use_numpy_preprocessors": True,
        "max_concurrent_workers": 1,
    },
    resampler_config=dict(cpu_session_config),
)

if vad_min_audio_seconds is not None:
    vad = onnx_asr.load_vad(
        "silero",
        path=installed_vad_path,
        providers=CPU_PROVIDERS,
        sess_options=session_options,
    )
    vad_model = base_model.with_vad(
        vad,
        threshold=threshold,
        neg_threshold=negative_threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        max_speech_duration_s=max_speech_duration_s,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms,
    )
```

`base_model` and `vad_model` are two Python views over one GigaAM. Contract
tests require:

```python
vad_model.asr is base_model.asr
vad_model.resampler is base_model.resampler
```

Creating `vad_model` MUST NOT increase the GigaAM ONNX session count. Silero is
a separate small model and is loaded once when VAD is enabled.

If a future pinned `onnx-asr` version no longer shares GigaAM between wrappers,
the compatibility gate fails. It MUST NOT load a second GigaAM, silently force
always-on VAD, or violate `vad.min_audio_seconds: null`.

### 5.5 CPU-only assurance

CPU-only is a product invariant, not a default. User configuration cannot
override `providers`.

The CPU provider list is passed to ASR and VAD construction and repeated in the
ONNX-bearing `asr_config` and `resampler_config`. The preprocessor config selects
the NumPy CPU implementation and therefore contains no irrelevant ORT session
keys. For the pinned release, `cpu_preprocessing` is deprecated and ignored, so
it MUST NOT be used. The ONNX ASR and resampler sessions are explicitly
CPU-only.

The audit does not globally monkey-patch `onnxruntime.InferenceSession`. Under a
model-load lock it uses a pinned, version-specific introspector over stable
session-bearing fields in the locked `onnx-asr>=0.12.0`. For the default E2E RNNT profile,
the role manifest requires `asr.encoder`, `asr.decoder`, `asr.joiner`, optional
`vad.silero`, and, after an 8 kHz warm-up, `resampler.8000_to_16000`; the NumPy
preprocessor has no ORT session. Missing, duplicate, or unknown session roles
fail the audit. After smoke inference:

```python
session.get_providers() == ["CPUExecutionProvider"]
```

must hold for every observed session. Zero observed sessions is failure. The
observed count and component role are compared with a release-tested manifest.
An object-graph walk may supplement diagnostics but is not the primary proof.

This audit runs on Linux, Windows, Intel macOS, and Apple Silicon. Presence of
CoreML, CUDA, or another provider in any used session is fatal.

### 5.6 Audio normalization with ffmpeg

For a compatible PCM WAV, use the original path. Otherwise create a private
temporary directory and run:

```shell
ffmpeg -nostdin -hide_banner -loglevel error -y \
  -protocol_whitelist file,pipe,crypto,data \
  -i INPUT -map 0:a:0 -ac 1 -ar 16000 -c:a pcm_s16le OUTPUT.wav
```

Normalization and temporary storage are owned by the admitted worker job, not
by the caller waiting for its result:

```python
def run_admitted_job(job):
    with tempfile.TemporaryDirectory(prefix="hermes-onnx-asr-") as work_dir:
        normalized_wav = Path(work_dir) / "input.wav"
        run_ffmpeg(job.source, normalized_wav)
        return recognize_wav(normalized_wav)
```

Therefore a caller timeout detaches only the result waiter. It never exits the
temporary-directory context or deletes the WAV while non-cancellable ORT
inference is still using it; the worker cleans up only after recognition really
returns.

The subprocess uses an argument list, `shell=False`, `stdin=DEVNULL`,
`stdout=DEVNULL`, a capped asynchronous stderr reader retaining at most 64 KiB,
a process-group timeout, and process-tree termination. The protocol allowlist
prevents hostile containers/playlists from opening HTTP or other network
inputs. User filenames are not embedded into the temp basename or shell
command.

The temporary directory is removed after success or ordinary failure. `SIGKILL`
cannot guarantee cleanup; `doctor` removes only plugin-owned stale directories
older than a documented TTL.

### 5.7 Duration selection and recognition

After normalization, read WAV header frames and sample rate:

```python
duration_seconds = frames / sample_rate
```

Selection:

```python
selected_model = (
    vad_model
    if vad_min_audio_seconds is not None
    and duration_seconds >= vad_min_audio_seconds
    else base_model
)
```

Without VAD:

```python
transcript = base_model.recognize(wav_path, channel="mean").strip()
```

With VAD:

```python
segments = vad_model.recognize(wav_path, channel="mean")
transcript = " ".join(
    segment.text.strip()
    for segment in segments
    if segment.text.strip()
)
```

The provider does not add punctuation, change case, or perform LLM cleanup.
The E2E RNNT output is returned as produced, with only surrounding and repeated
inter-segment whitespace normalized.

No detected speech deterministically returns
`{"success": false, "transcript": "", "error": "No speech detected",
"provider": "onnx_asr", "error_code": "no_speech_detected"}`. A successful
empty transcript is forbidden because the pinned Hermes gateway would insert
an empty quoted utterance into the model prompt. The behavior is identical on
every platform.

### 5.8 Long recordings and temporary storage

The plugin's default `max_duration_seconds` is `null`: it imposes no artificial
duration cap and supports multi-hour direct transcription.

The implementation documents that current `onnx-asr` reads the PCM WAV into
memory before inference. VAD segments recognition but does not make file loading
streaming. A mono 16 kHz PCM16 temporary WAV requires approximately:

```text
duration_seconds * 16000 * 2 bytes
```

When input duration can be probed, the plugin checks the estimate plus
`temp_safety_margin_bytes` against free space before conversion. Otherwise it
monitors the produced file and treats ENOSPC as `insufficient_temp_space`.

The standard Hermes `transcribe_audio()` currently rejects files larger than
25 MiB before plugin dispatch. Therefore:

- the plugin CLI can process larger/multi-hour files;
- ordinary gateway delivery remains subject to the current Hermes 25 MiB gate;
- the plugin cannot bypass that gate without a Hermes API enhancement.

### 5.9 Model lifecycle and offline operation

Runtime model downloads are disabled by default. Commands:

```shell
hermes onnx-asr list-models
hermes onnx-asr fetch gigaam-v3-e2e-rnnt --quantization int8
hermes onnx-asr fetch-vad silero
hermes onnx-asr warmup
hermes onnx-asr doctor
hermes onnx-asr transcribe /path/to/audio
```

`fetch` resolves only catalog entries, pins repository revision, downloads to a
temporary directory, validates required files and available hashes, writes a
manifest, and atomically renames the bundle. Partially downloaded directories
are never considered installed.

The catalog is shipped, immutable JSON keyed by `(onnx_asr_version, alias,
quantization)`. Each entry names the upstream repository, immutable revision,
expected relative files, and download mechanism. The installed
`manifest.json` contains schema version, alias, quantization, source repository,
immutable revision, `onnx-asr` version, UTC creation time, and for every file
its relative POSIX path, byte length, and lowercase SHA-256. Absolute paths,
tokens, and mutable branch names are forbidden. The manifest itself is
canonical-JSON hashed.

Fetch holds a per-bundle interprocess lock, refuses an existing valid bundle,
quarantines an invalid destination rather than overwriting it, downloads into a
sibling mode-0700 staging directory, fsyncs files and directory, then performs
one atomic rename. A concurrent fetch either observes the completed valid
bundle or waits; it never merges trees.

At runtime the provider passes explicit local paths. Missing or mismatched model
and VAD bundles produce a clear not-ready result; they do not trigger network
access.

Before calling `load_model`/`load_vad`, runtime verifies the complete manifest
and required-file set. The offline release test denies socket creation and HTTP
client calls at process level, points Hugging Face/onnx-asr caches at empty
read-only directories, and proves warm-up succeeds only from the installed
bundle. Any attempted network access fails the test.

`warmup` loads the configured ASR/VAD, performs short non-speech and Russian
speech fixtures, validates wrapper sharing, and completes the CPU session audit.

### 5.10 Cache, concurrency, and timeouts

One immutable pipeline is resident per process. The cache key includes resolved
profile root, model alias/revision, quantization, VAD settings, and ORT thread
settings. Hermes' configured model normally matches this key. An explicit
`model=` override is resolved through the same catalog and may replace the
resident pipeline only under the construction lock when no call is running or
queued; otherwise the call fails with `model_switch_busy`. Replacement is
atomic and the old pipeline remains usable if construction fails. The direct
CLI runs in its own process and therefore does not contend with the gateway.

Default runtime behavior:

- inference concurrency: 1;
- waiting queue depth: 4 (one running plus four waiting, five admitted total);
- FIFO admission;
- the sixth simultaneous call returns `asr_queue_full`;
- model construction is protected by a global lock;
- failed construction does not publish a partial singleton.

An ORT inference cannot be assumed cancellable. When the caller timeout expires,
the provider returns a bounded timeout result, but the worker retains the model
slot until the actual inference exits. No second inference starts over it.

The transcription deadline starts at successful admission and includes queue,
normalization, and recognition time. A queued job whose deadline expires is
removed atomically from FIFO and never creates a temporary directory. A running
job whose deadline expires detaches its waiter but continues with sole ownership
of its source handle and temporary directory until recognition and cleanup
finish. `ffmpeg_timeout_seconds` is an inner phase deadline capped by the
remaining transcription deadline.

Shutdown rejects new jobs, fails queued jobs with `provider_shutting_down`, and
waits a documented 30-second grace period for the active worker. Because ORT is
not cancellable, the worker thread is process-scoped; after the grace period
Hermes may exit and the next `doctor` run removes only stale, mode-0700,
plugin-prefixed temp directories older than the configured TTL. A timeout never
releases the inference semaphore early.

### 5.11 Result envelope and errors

Success:

```json
{
  "success": true,
  "transcript": "Распознанный текст.",
  "provider": "onnx_asr",
  "model": "gigaam-v3-e2e-rnnt",
  "vad_applied": true,
  "audio_seconds": 37.2,
  "segments": 4
}
```

Failure always preserves Hermes' required `success`, `transcript`, and `error`
fields. `provider` is diagnostic; `error_code` is a stable plugin-internal field
for logs and CLI automation and is not interpreted by Hermes core:

```json
{
  "success": false,
  "transcript": "",
  "provider": "onnx_asr",
  "error": "Audio conversion failed.",
  "error_code": "decode_failed"
}
```

Stable error codes:

- `configuration_invalid`;
- `model_not_installed`;
- `model_not_in_catalog`;
- `vad_not_installed`;
- `ffmpeg_missing`;
- `no_audio_stream`;
- `decode_failed`;
- `insufficient_temp_space`;
- `audio_too_long` when the optional limit is set;
- `no_speech_detected`;
- `asr_queue_full`;
- `model_switch_busy`;
- `provider_shutting_down`;
- `ffmpeg_timeout`;
- `asr_timeout`;
- `cpu_provider_violation`;
- `model_load_failed`;
- `transcription_failed`.

Chat-facing errors never expose raw exception strings, local paths, model cache
paths, URLs, tokens, or audio content. Detailed redacted diagnostics remain in
logs.

### 5.12 ASR diagnostics

`hermes onnx-asr doctor` reports:

```text
Provider:       onnx_asr
Model:          gigaam-v3-e2e-rnnt / int8
VAD:            silero, threshold 20s
Execution:      CPUExecutionProvider only
GigaAM copies:  1
Model files:    ready, immutable revision
Runtime fetch:  disabled
ffmpeg:          ready
Warm-up:        passed
```

It checks discovery, configuration, minimum dependency versions, ffmpeg,
model/VAD manifests, wrapper sharing, all observed ORT providers, warm-up,
queue state, available temp space, stale temp directories, and the one-profile
constraint.

### 5.13 One-profile limitation

Hermes' transcription provider registry is process-global and its call does not
carry deterministic profile identity. Therefore v1 supports one active ONNX ASR
configuration per process. A multiplexed process with conflicting `onnx_asr`
profiles fails startup/doctor rather than mixing models or caches. Separate
Hermes processes remain isolated through `HERMES_HOME`.

Enforcement is lazy and does not depend on an unavailable startup hook. Under
the construction lock, the first warm-up or transcription resolves
`get_hermes_home()` and immutably binds the provider instance to that canonical
profile root. Every later call resolves it again before queue admission; a
different root returns `configuration_invalid` without reading that profile's
model files or mutating the cache. `doctor` also reports configured multiplex
profiles when Hermes exposes them, but the per-call root comparison is the
authoritative guard.

## 6. Operations and security

### 6.1 Logging and privacy

Logs use hashed or shortened peer/message identifiers. They never contain:

- VK tokens;
- full private message bodies;
- raw audio;
- pairing codes;
- unredacted media URLs containing access keys;
- local user paths in chat-facing errors.

Debug logging of payloads is opt-in, bounded, sanitized, and disabled by default.

### 6.2 Supply chain

- Declare minimum supported versions for `onnx-asr` and compatible CPU
  `onnxruntime`; pin concrete versions in the release lock.
- No CUDA, TensorRT, or CoreML extras are installed.
- Model manifests pin repository revisions and required files.
- CI runs dependency and secret scanning.
- Wheels are built from clean environments and exclude checkout, fixtures with
  private data, tokens, SQLite state, model files, and caches.

### 6.3 Supported operating systems

VK follows the OS/Python support of Hermes. ASR release CI covers:

- Linux x86_64 CPU;
- Windows x86_64 CPU;
- macOS Intel CPU;
- macOS Apple Silicon CPU.

macOS tests explicitly fail if a used session exposes CoreML.
If hosted Intel macOS CI is unavailable, release is blocked until the same
signed test artifact is run on a documented Intel Mac; “not run” is not a pass.

## 7. Testing and release gates

### 7.1 Contract harness first

Before feature implementation, tests pin the observed contracts for plugin
discovery, platform registration, adapter construction, secret scope, typing
lifecycle, streaming finalization, media dispatch, transcription registration,
`stt.onnx_asr` model/language forwarding, and result envelopes.

### 7.2 VK tests

Unit/fixture tests cover:

- `apply_yaml_config_fn` round-tripping minimal/full VK config into
  `PlatformConfig.extra`, rejecting unknown keys, and deriving effective policy;
- initial `connect(is_reconnect=False)` and cursor-preserving
  `connect(is_reconnect=True)` through the real Hermes call site;
- Long Poll `failed=1/2/3`;
- cursor commit boundaries and replay;
- duplicate and poison events;
- authorization before media I/O;
- fatal `GATEWAY_ALLOWED_USERS`/`GATEWAY_ALLOW_ALL_USERS` and VK env-policy
  conflicts, plus the real own-policy no-env mixin branch;
- pairing text-only restriction;
- SSRF/redirect/hostname validation;
- inbox/outbox migrations and crash recovery;
- fresh/stable `random_id` behavior;
- partial and ambiguous sends;
- every outbox outcome mapped through Hermes' real `SendResult` retry/fallback
  consumer with no duplicate whole-response fallback;
- error 914 re-splitting;
- Markdown AST plain rendering;
- rich fixture rendering and invalid `format_data` fallback;
- streaming preview/finalize;
- literal-source `partial_overflow.delivered_prefix` and exact missing-tail
  recovery through the real stream consumer;
- typing throttling and cooldown;
- button payload binding and expiry;
- all v1 attachment mappings.
- DNS validation-to-connect pinning, TLS hostname preservation, rotating allowed
  suffixes, and rejection of rebinding/private addresses;

Secret-gated live tests cover a private community: auth, Long Poll, DM reply,
typing, upload/download, formatting spike, edit, buttons, and restart behavior.

### 7.3 ASR tests

Tests cover:

- default `gigaam-v3-e2e-rnnt` selection;
- generated alias catalog and every advertised alias smoke load;
- exact `negative_threshold -> neg_threshold` mapping;
- `null`, `0`, `20`, and boundary duration VAD selection;
- base/VAD wrapper identity and unchanged GigaAM session count;
- CPU provider propagation and session audit on every OS;
- compatible PCM WAV direct path;
- stereo WAV `channel="mean"`;
- ffmpeg normalization for OGG/Opus, MP3, M4A/AAC, FLAC, and WebM;
- corrupt/no-audio input;
- temp cleanup on success/failure/timeout;
- a running inference timeout that proves the normalized WAV remains present
  until the worker exits;
- ffmpeg protocol fixtures that attempt HTTP/network input and are rejected;
- long recording without a default duration limit;
- insufficient disk estimate;
- segment text joining;
- no-speech behavior fixed against Hermes voice handling;
- concurrency 1, queue depth 4, timeout behavior, singleton load;
- runtime offline/no-download guarantee;
- direct CLI transcription above the Hermes 25 MiB limit;
- Telegram and VK fixtures reaching the same provider.

The redistributable Russian evaluation corpus contains at least one
user-provided voice recording with speech and silence. A versioned JSONL
manifest records clip SHA-256, license/source, speaker/noise tags, duration,
and verified UTF-8 ground truth. Scoring uses `jiwer==3.1.0`; hypothesis and reference are Unicode NFC,
lowercased, `ё`-preserving, punctuation-stripped by one frozen regex, and
whitespace-collapsed. Keyword recall uses a frozen case-folded keyword list.

Release comparison records WER, keyword recall, real-time factor, cold start,
warm latency, and peak RSS on the same declared CPU/hardware class. Cold means
a new process and model object; warm means five runs after one discarded
warm-up, reporting median and p95. A release MUST NOT regress WER by more than
2 absolute percentage points or keyword recall by more than 1 point versus the
previous released default without an explicit documented decision.

## 8. Implementation decomposition

Work packages are ordered but independently reviewable:

1. Contract harness and package discovery.
2. VK configuration, secret resolution, lock, and HTTP client.
3. VK SQLite schema, Long Poll cursor, inbox, quarantine.
4. VK authorization and event-to-Hermes mapping.
5. VK text send, chunking, outbox, random IDs, partial delivery.
6. VK attachments and upload/download security.
7. VK typing and interactive buttons.
8. VK live rich-text spike and committed capability profile.
9. VK AST renderer, incoming `format_data`, streaming finalize.
10. ASR configuration and generated model catalog.
11. ASR model/VAD construction, wrapper sharing, CPU audit.
12. ASR PCM direct path and ffmpeg temporary normalization.
13. ASR duration-based VAD selection and result aggregation.
14. ASR offline fetch, manifests, warm-up, doctor.
15. ASR queue, timeouts, long-file CLI, and cleanup.
16. Cross-platform integration, release matrix, documentation.

Each work package includes unit tests and updates its diagnostic command. No
package requires an architectural choice not specified above.

## 9. Optional Hermes upstream enhancements

These are separate PRs and are not required by the plugin wheels:

### 9.1 Durable completion callback

Add a generic completion acknowledgement to the adapter message lifecycle so a
transport inbox can distinguish background work completion from dispatch. This
would permit safe automatic reconciliation of `dispatched` VK rows.

### 9.2 Provider-specific STT file-size capability

Replace the global pre-dispatch 25 MiB limit with a provider capability:

```python
class TranscriptionProvider:
    def max_file_size(self) -> int | None:
        return 25 * 1024 * 1024
```

`OnnxAsrProvider` can then return a configured value or `None`. Existing cloud
providers retain 25 MiB by default.

## 10. Rejected and deferred alternatives

Rejected for v1:

- a single combined VK+ASR package;
- VK Callback API as the only transport;
- a third-party VK SDK dependency;
- tokens in YAML or arbitrary token environment names;
- authorization after media download;
- VK authorization union/intersection with global or VK-specific env lists;
- content-derived permanent VK `random_id`;
- automatic resend of ambiguous cross-restart VK sends;
- returning a possibly visible partial ordinary send through Hermes' automatic
  whole-message fallback branch;
- validating a hostname and then connecting through an unpinned system DNS or
  inherited proxy path;
- claims of exactly-once agent execution on current Hermes;
- regex-only Markdown conversion;
- plain text as the only VK formatting target;
- duplicate NumPy/resampling logic in the ASR plugin;
- PyAV when ffmpeg normalization plus `onnx-asr.recognize(path)` suffices;
- mandatory VAD for every recording;
- a 300-second default audio limit;
- automatic runtime model download;
- automatic model fallback/selection;
- GPU, CUDA, TensorRT, and CoreML;
- multiple conflicting ASR profiles in one process.

Deferred:

- VK Callback API for public-HTTPS/multi-replica deployments;
- VK video upload and carousel/template messages;
- native VK rich tables; tables are rendered as JPEG photos instead;
- ASR sidecar/subprocess isolation unless measurements justify it;
- speaker diarization and LLM transcript cleanup;
- realtime microphone streaming;
- automatic replay of ambiguous VK work until Hermes has completion IDs;
- the optional provider-specific Hermes file-size capability in §9.2.

## 11. Decision ledger

| ID | Decision | Status | Rationale |
|---|---|---|---|
| D01 | Two independent external pip plugins | adopted | Platform and ASR lifecycles remain decoupled |
| D02 | No required Hermes core modification | adopted | Existing platform and transcription extension points suffice |
| D03 | VK Community Long Poll | adopted | Fits private personal community without public HTTPS |
| D04 | Callback API | deferred | Useful for later multi-replica deployments |
| D05 | Direct async VK client | adopted | Small surface and explicit retry/security control |
| D06 | Third-party VK SDK | rejected | Adds coupling without needed behavior |
| D07 | Durable inbox/cursor/quarantine | adopted | Makes replay and poison events observable |
| D08 | Exactly-once agent execution | rejected | Hermes exposes no durable completion acknowledgement |
| D09 | Authorization before media I/O | adopted | Prevents unauthorized fetches and cache writes |
| D10 | Fixed scoped `VK_COMMUNITY_TOKEN` | adopted | Works with Hermes secret scope and avoids YAML secrets |
| D11 | VK typing via `messages.setActivity` | adopted | Uses Hermes' existing typing lifecycle |
| D12 | Rich formatting with capability profile | adopted | Preserve Markdown where live VK supports it |
| D13 | Live HTML/`format_data`/table spike | adopted | Public schema lags and client rendering must be observed |
| D14 | Plain-only VK renderer | rejected | Needlessly discards supported formatting |
| D15 | Conservative 4096-character baseline; fixture may certify 9000; progressive 914 fallback | adopted | Public schema and older live limits disagree, so default safely and measure |
| D16 | Fresh random ID per logical chunk, stable for live retries | adopted | Avoids duplicate retries and suppressed legitimate repeats |
| D17 | Auto-resend ambiguous restart sends | rejected | VK/Hermes provide no safe cross-restart delivery identity |
| D18 | Platform-independent `onnx_asr` provider | adopted | All Hermes voice-capable platforms share it |
| D19 | Default `gigaam-v3-e2e-rnnt` | adopted | User-tested Russian default with E2E output |
| D20 | CPUExecutionProvider only on every OS | adopted | Explicit product invariant, including macOS |
| D21 | Conditional Silero VAD from 20 seconds | adopted | Avoid overhead on short voice notes while segmenting longer audio |
| D22 | `0` always VAD, `null` no VAD | adopted | Simple complete threshold semantics |
| D23 | One shared GigaAM for base and VAD wrappers | adopted | Confirmed current `onnx-asr` wrapper contract |
| D24 | Mandatory VAD for all recordings | rejected | User chose duration-based optional VAD |
| D25 | Thin wrapper around `recognize(path)` | adopted | Avoid duplicating upstream WAV/resampler logic |
| D26 | ffmpeg to private temporary PCM WAV | adopted | Handles messenger codecs and follows Hermes practice |
| D27 | PyAV/NumPy custom decode pipeline | rejected | Unnecessary duplication for the selected design |
| D28 | No default duration limit | adopted | Preserve multi-hour `onnx-asr` use cases |
| D29 | Runtime model downloads disabled | adopted | Reproducible startup and no first-message download |
| D30 | One active ASR profile per process | adopted | Hermes provider call lacks deterministic profile identity |
| D31 | GPU/CoreML support | rejected | Conflicts with the explicit CPU-only product invariant |
| D32 | Optional provider-specific Hermes size capability | deferred | Needed only for gateway files above current 25 MiB limit |
| D33 | `apply_yaml_config_fn` owns VK config bridging | adopted | Hermes otherwise discards arbitrary platform keys |
| D34 | Exact `connect(is_reconnect=...)` and `send(reply_to, metadata)` signatures | adopted | Prevents startup/reconnect failure and silent argument loss |
| D35 | YAML-only VK authorization plus `enforces_own_access_policy` | adopted | Prevents environment bypass while satisfying Hermes' second-line gate |
| D36 | Worker owns normalized WAV after caller timeout | adopted | ORT inference is not cancellable and must not lose its active input |
| D37 | No speech is a deterministic failure | adopted | Hermes does not safely ignore a successful empty transcript |
| D38 | Minimum `onnx-asr`/ORT metadata versions, release-lock pins, and manifest-verified offline bundles | adopted | Allows compatible upgrades while keeping releases reproducible |
| D39 | Partial ordinary send reports success; ambiguous send uses Hermes timeout marker | adopted | Suppresses the actual `_send_with_retry()` whole-message fallback after possible visibility |
| D40 | `aiohttp>=3.14.1` custom resolver with global-only IP predicate; concrete release-lock pin | adopted | Pins DNS answers while preserving original-host TLS SNI and certificate checks |
| D41 | Global and VK env authorization conflicts fail VK startup | adopted | Keeps effective VK authorization strictly YAML-only |
| D42 | V1 groups disabled; SQLite default partitioned by profile/group | adopted | Avoids trusting an unspecified group gate and prevents state collisions |

## 12. Definition of Done

The specification is implemented when:

- both wheels install and are discovered independently;
- no required Hermes core file is modified;
- VK Long Poll survives lease rotation, replay, and network interruption;
- unauthorized VK users cannot trigger media I/O;
- VK typing is visible during processing;
- Markdown is rendered through a live-tested rich profile with safe plain
  fallback;
- Markdown tables are always rendered as JPEG photos and keep source order with
  surrounding text;
- VK partial/ambiguous delivery is observable and never blindly duplicated;
- `gigaam-v3-e2e-rnnt` is the tested default;
- short audio uses base GigaAM and audio at/above 20 seconds uses the shared
  Silero-wrapped view;
- exactly one GigaAM is loaded;
- every used ONNX session reports only `CPUExecutionProvider` on every OS;
- non-WAV messenger audio is normalized by ffmpeg and temporary files are
  cleaned;
- the plugin imposes no default duration limit and direct CLI handles a
  multi-hour fixture within documented resource constraints;
- VK and Telegram fixtures reach the same ASR provider;
- model/VAD downloads are explicit and immutable;
- both `doctor` commands provide actionable redacted diagnostics;
- contract, unit, integration, live-secret, and OS matrix tests pass.

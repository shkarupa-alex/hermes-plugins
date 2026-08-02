# Tag-derived PyPI release versions

Status: approved design, revised after clean-room review

## Architecture / Components

The Git release tag is the single source of truth for published package versions. Committed manifests may retain a previous development/release value; release jobs modify only disposable GitHub Actions checkouts before validation and build.

`tools/release_version.py` is standard-library-only and owns tag parsing, project discovery, manifest mutation, and artifact verification. Workflows must not duplicate prefix stripping in shell. Its operations are:

1. `resolve`: validate raw input and emit canonical tag, package version, and fully qualified Git ref.
2. `apply`: resolve input and rewrite every workspace `plugin.yaml`.
3. `verify-artifacts`: inspect every wheel and sdist, including embedded manifests.

The publish workflow adds a cheap `release-version` job before release gates. It resolves the tag once and exposes outputs. Every downstream job that consumes them declares a direct dependency. Every reusable release-gate job checks out the fully qualified tag. The supply-chain and build checkouts separately apply the resolved version before their metadata checks.

Normal branch/PR CI continues to validate committed manifests without deriving a release version.

SourceTree GitFlow creates tags before CI, so CI cannot prevent its configured prefix from being added. The supported convention for users who enter `vX.Y.Z` is:

```bash
git config gitflow.prefix.versiontag ""
```

The README documents the complete operator matrix: prefix `v` + input `X.Y.Z`, or empty prefix + input `vX.Y.Z`, both produce a canonical tag. Mixing prefix `v` with `vX.Y.Z` produces rejected `vvX.Y.Z`; empty prefix plus `X.Y.Z` creates a bare tag that does not trigger `tags: ["v*"]`.

## Interfaces & Schemas

```python
@dataclass(frozen=True, slots=True)
class ReleaseVersion:
    tag: str       # vX.Y.Z
    version: str   # X.Y.Z
    ref: str       # refs/tags/vX.Y.Z


@dataclass(frozen=True, slots=True)
class Project:
    name: str
    normalized_name: str
    directory: Path
    manifest: Path
    wheel_manifest: str


def resolve_release_version(raw: str) -> ReleaseVersion: ...
def discover_projects(root: Path) -> tuple[Project, ...]: ...
def apply_release_version(root: Path, release: ReleaseVersion) -> tuple[Path, ...]: ...
def verify_artifacts(root: Path, release: ReleaseVersion, dist_dir: Path) -> tuple[Path, ...]: ...
```

### Normalization

Accepted input is exactly stable SemVer `X.Y.Z` or `vX.Y.Z`, with canonical non-negative decimal components and no whitespace. Leading zeroes are rejected.

| Raw input | Result |
|---|---|
| `1.2.3` | tag `v1.2.3`, version `1.2.3`, ref `refs/tags/v1.2.3` |
| `v1.2.3` | same |
| `vv1.2.3` | validation error explaining SourceTree GitFlow prefix configuration |
| empty, whitespace, `1.2`, `v1.2.3rc1`, `01.2.3` | validation error |

CLI:

```text
python tools/release_version.py resolve RAW [--github-output PATH]
python tools/release_version.py apply RAW [--root PATH]
python tools/release_version.py verify-artifacts RAW DIST_DIR [--root PATH]
```

`resolve` prints newline-terminated `tag=`, `version=`, and `ref=` records. With `--github-output`, it appends the same records to that file. Invalid CLI input writes one actionable line of at most 500 characters to stderr and exits `2`; filesystem/archive validation failures exit `1`.

### Project discovery and manifests

`discover_projects` enumerates sorted `packages/*/pyproject.toml` files. Each must have exactly one sibling, non-symlink `plugin.yaml`, a unique string project name, dynamic Hatch version configuration pointing to that manifest, and exactly one wheel `force-include` destination for `plugin.yaml`. Orphan `packages/*/plugin.yaml` files are errors. `normalized_name` replaces each run of `-`, `_`, or `.` with `_`.

`apply` decodes every manifest as UTF-8 and requires exactly one unquoted top-level line matching `^version:\s*[^\s]+\s*$`. It preserves newline style, terminal newline, and every unrelated byte. It prevalidates and stages all sibling temporary files before replacement, then uses `Path.replace()` per target and cleans remaining temporaries. This is per-file atomic, not transactional across manifests: interruption can leave mixed versions, but rerunning is idempotent and repairs them.

### Artifact verification

For each project, `verify-artifacts` requires exactly:

- `{normalized_name}-{version}-py3-none-any.whl`;
- `{normalized_name}-{version}.tar.gz`.

Unrelated, duplicate, missing, symlink, or non-file entries in the dist directory fail verification. The sole permitted non-distribution file is the regular, non-symlink `.gitignore` created by `uv build --clear`. Using `zipfile`, `tarfile`, `email.parser`, and `tomllib`, the verifier requires:

- wheel `{normalized_name}-{version}.dist-info/METADATA` with exact project `Name` and exact `Version`;
- the wheel member named by Hatch `force-include` containing exactly the release manifest version;
- sdist `{normalized_name}-{version}/PKG-INFO` with the same exact metadata;
- sdist `{normalized_name}-{version}/plugin.yaml` containing exactly the release manifest version;
- all expected archive members to be unique regular files.

### GitHub Actions wiring

For a tag push, raw input is `github.ref_name`; for `workflow_dispatch`, it is `inputs.release_tag`. The resolver job checks out the trigger/default branch with `fetch-depth: 0`, runs a step `id: release`, and verifies `git show-ref --verify --quiet "$ref"` before exposing:

```yaml
outputs:
  tag: ${{ steps.release.outputs.tag }}
  version: ${{ steps.release.outputs.version }}
  ref: ${{ steps.release.outputs.ref }}
```

`release-gates` declares `needs: release-version` and receives `release_tag` and `release_ref`. `release-build` declares `needs: [release-version, release-gates]`, checks out `${{ needs.release-version.outputs.ref }}`, applies the version, and runs `check_versions.py --tag` before building.

Every reusable release-gate checkout uses `${{ inputs.release_ref || github.ref }}`. Direct `workflow_dispatch` of `release-gates.yml` without inputs remains a non-release diagnostic of its trigger ref and runs the committed-manifest check without mutation. Called release gates receive `refs/tags/vX.Y.Z`; supply-chain applies the tag version before checking metadata.

Manual recovery supports only tags whose source contains this helper and updated workflow. The canonical dispatch tag must already exist.

After build:

```text
python tools/release_version.py verify-artifacts "$RELEASE_VERSION" dist/
```

Any mismatch blocks artifact upload and PyPI publication.

## Requirements & Constraints

- No release version bump commit is required.
- Both plugins receive one version from one tag.
- Published wheel/sdist metadata and embedded `plugin.yaml` must agree.
- Invalid tags fail before model downloads, live VK tests, or builds.
- Preparation never pushes commits or moves tags.
- Existing `skip_existing` recovery remains supported for eligible tags.
- Only stable `X.Y.Z` is in scope.

## Operational Decisions

The top-level concurrency key may retain the raw input because it only deduplicates runs. All source checkout and package metadata decisions use normalized outputs. Already published PyPI versions remain immutable. Rollback is a workflow/tool revert; reruns use the existing canonical tag.

## Testing

- Unit-test every normalization example and CLI output/error contract.
- Test project discovery including missing, orphan, symlinked, and duplicate cases.
- Test LF/CRLF manifests, terminal-newline preservation, duplicate version lines, prevalidation, idempotence, and repair after simulated partial replacement.
- Build real distributions in a temporary copy and verify matching archives.
- Reject missing, duplicate, unrelated, wrong-name, wrong-version, wrong-metadata, and wrong embedded-manifest artifacts.
- Assert workflow job IDs, direct `needs`, outputs, reusable inputs, and checkout refs; run Actionlint.
- Run `check_versions.py --tag` after applying a test version, then Ruff, Pyright, full pytest, builds, and `git diff --check`.

## Decision Ledger

| Decision | Status | Rationale | Source |
|---|---|---|---|
| Tag is the published-version source | adopted | Eliminates manual bump commits | solo/r0 |
| Patch manifests only in disposable checkouts | adopted | Keeps package and plugin metadata aligned | solo/r0 |
| Central tested standard-library helper | adopted | Prevents workflow prefix drift and bootstraps before sync | solo/premortem, review |
| Accept `X.Y.Z` and `vX.Y.Z`; canonicalize one `v` | adopted | Unifies tag events and manual recovery | solo/r0 |
| Reject `vvX.Y.Z` | adopted | Does not legitimize malformed Git tags | solo/premortem |
| Empty SourceTree GitFlow tag prefix for `vX.Y.Z` input | adopted | SourceTree creates the extra prefix before CI | solo/r0 |
| Inspect filenames, metadata, and embedded manifests | adopted | Blocks stale/mixed publication | solo/premortem, review |
| Every release gate checks out fully qualified tag ref | adopted | Prevents testing default branch while publishing tag code | review |
| Per-file atomic and idempotent manifest replacement | adopted | States achievable filesystem guarantee | review |
| Use `hatch-vcs` | rejected | Does not update shipped plugin manifest | solo/r0 |
| Automatic release bump commit | rejected | Reintroduces local GitFlow coupling | solo/r0 |
| Support prerelease/postrelease now | deferred | Not requested | solo/premortem |

## Rejected / Deferred Alternatives

`hatch-vcs` leaves the shipped directory-plugin manifest stale unless mutation remains. A SourceTree hook requires unversioned local setup and produces mechanical commits. Prerelease syntax is deferred until a concrete need defines its Git-tag-to-PEP-440 mapping.

## Open Questions

None. Rebuilding tags created before this mechanism is intentionally unsupported.

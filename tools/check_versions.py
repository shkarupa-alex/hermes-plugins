"""Verify release-version consistency across plugin manifests and build metadata."""

from __future__ import annotations
import argparse
import re
import tomllib
from pathlib import Path
from typing import cast

VERSION_PATTERN = re.compile(r"^version:\s*(?P<version>[^\s]+)\s*$", re.MULTILINE)


def _toml(path: Path) -> dict[str, object]:
    return cast("dict[str, object]", tomllib.loads(path.read_text(encoding="utf-8")))


def _manifest_version(path: Path) -> str:
    match = VERSION_PATTERN.search(path.read_text(encoding="utf-8"))
    if match is None:
        raise ValueError(f"missing top-level version in {path}")
    return match.group("version")


def _project_metadata(path: Path) -> tuple[str, dict[str, object]]:
    data = _toml(path)
    project_value = data.get("project")
    if not isinstance(project_value, dict):
        raise TypeError(f"missing [project] in {path}")
    project = cast("dict[str, object]", project_value)
    name = project.get("name")
    dynamic = project.get("dynamic")
    if not isinstance(name, str):
        raise TypeError(f"missing project.name in {path}")
    if dynamic != ["version"] or "version" in project:
        raise ValueError(f"{path}: version must be dynamic and sourced from plugin.yaml")

    tool_value = data.get("tool")
    tool = cast("dict[str, object]", tool_value) if isinstance(tool_value, dict) else {}
    hatch_value = tool.get("hatch")
    hatch = cast("dict[str, object]", hatch_value) if isinstance(hatch_value, dict) else {}
    version_value = hatch.get("version")
    version = cast("dict[str, object]", version_value) if isinstance(version_value, dict) else {}
    if version.get("path") != "plugin.yaml" or version.get("pattern") != VERSION_PATTERN.pattern:
        raise ValueError(f"{path}: [tool.hatch.version] must read plugin.yaml")
    return name, project


def _locked_workspace_versions(path: Path) -> dict[str, str | None]:
    data = _toml(path)
    packages_value = data.get("package")
    if not isinstance(packages_value, list):
        raise TypeError(f"missing package list in {path}")
    versions: dict[str, str | None] = {}
    for item_value in cast("list[object]", packages_value):
        if not isinstance(item_value, dict):
            continue
        item = cast("dict[str, object]", item_value)
        source_value = item.get("source")
        source = cast("dict[str, object]", source_value) if isinstance(source_value, dict) else {}
        editable = source.get("editable")
        name = item.get("name")
        version = item.get("version")
        if isinstance(editable, str) and editable.startswith("packages/") and isinstance(name, str):
            versions[name] = version if isinstance(version, str) else None
    return versions


def verify(root: Path, release_tag: str | None = None) -> str:
    """Return the common version after validating every release metadata surface."""
    manifests: dict[str, str] = {}
    for project_path in sorted((root / "packages").glob("*/pyproject.toml")):
        name, _project = _project_metadata(project_path)
        manifests[name] = _manifest_version(project_path.with_name("plugin.yaml"))
    if not manifests:
        raise ValueError("no workspace packages found")
    versions = set(manifests.values())
    if len(versions) != 1:
        raise ValueError(f"plugin manifest versions differ: {manifests}")
    common_version = next(iter(versions))

    locked = _locked_workspace_versions(root / "uv.lock")
    missing_from_lock = sorted(set(manifests) - set(locked))
    stale_lock_versions = {name: locked[name] for name in manifests.keys() & locked.keys() if locked[name] is not None}
    if missing_from_lock or stale_lock_versions:
        raise ValueError(
            f"uv.lock dynamic workspace metadata is stale: missing={missing_from_lock}, versions={stale_lock_versions}"
        )

    if release_tag is not None and (not release_tag.startswith("v") or release_tag[1:] != common_version):
        raise ValueError(f"tag {release_tag!r} does not match plugin version {common_version}")
    return common_version


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", help="Expected release tag, including the v prefix")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        verify(args.root, args.tag)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()

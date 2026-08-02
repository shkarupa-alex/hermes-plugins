"""Resolve release tags and apply their versions to build metadata."""

from __future__ import annotations
import argparse
import os
import re
import sys
import tarfile
import tempfile
import tomllib
import zipfile
from dataclasses import dataclass
from email.parser import Parser
from pathlib import Path
from typing import NoReturn, cast

TAG_PATTERN = re.compile(r"v?(?P<version>(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*))\Z")
MANIFEST_VERSION_PATTERN = re.compile(rb"(?m)^(version:[ \t]*)([^ \t\r\n]+)([ \t]*)(?=\r?$)")
NORMALIZED_NAME_PATTERN = re.compile(r"[-_.]+")


class ReleaseVersionError(ValueError):
    """A release-version contract is invalid."""


@dataclass(frozen=True, slots=True)
class ReleaseVersion:
    """Canonical forms derived from a user-supplied release version."""

    tag: str
    version: str
    ref: str


@dataclass(frozen=True, slots=True)
class Project:
    """Build metadata surfaces for a workspace project."""

    name: str
    normalized_name: str
    directory: Path
    manifest: Path
    wheel_manifest: str


def resolve_release_version(raw: str) -> ReleaseVersion:
    """Normalize ``X.Y.Z`` or ``vX.Y.Z`` to canonical release forms."""
    match = TAG_PATTERN.fullmatch(raw)
    if match is None:
        if raw.startswith("vv"):
            raise ReleaseVersionError(
                "release has a duplicated v prefix; set gitflow.prefix.versiontag to empty and enter vX.Y.Z"
            )
        raise ReleaseVersionError("release must be X.Y.Z or vX.Y.Z with no whitespace, prerelease, or leading zeros")
    version = match.group("version")
    tag = f"v{version}"
    return ReleaseVersion(tag=tag, version=version, ref=f"refs/tags/{tag}")


def _mapping(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ReleaseVersionError(f"missing or invalid {context}")
    return cast("dict[str, object]", value)


def _project(project_file: Path) -> Project:
    try:
        data = cast("dict[str, object]", tomllib.loads(project_file.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise ReleaseVersionError(f"cannot read {project_file}: {exc}") from exc
    project_data = _mapping(data.get("project"), f"[project] in {project_file}")
    name = project_data.get("name")
    if not isinstance(name, str) or not name:
        raise ReleaseVersionError(f"missing project.name in {project_file}")
    if project_data.get("dynamic") != ["version"] or "version" in project_data:
        raise ReleaseVersionError(f"{project_file}: project version must be dynamic")

    tool = _mapping(data.get("tool"), f"[tool] in {project_file}")
    hatch = _mapping(tool.get("hatch"), f"[tool.hatch] in {project_file}")
    version = _mapping(hatch.get("version"), f"[tool.hatch.version] in {project_file}")
    if version.get("path") != "plugin.yaml":
        raise ReleaseVersionError(f"{project_file}: Hatch version must come from plugin.yaml")
    build = _mapping(hatch.get("build"), f"[tool.hatch.build] in {project_file}")
    targets = _mapping(build.get("targets"), f"[tool.hatch.build.targets] in {project_file}")
    wheel = _mapping(targets.get("wheel"), f"[tool.hatch.build.targets.wheel] in {project_file}")
    force_include = _mapping(
        wheel.get("force-include"), f"[tool.hatch.build.targets.wheel.force-include] in {project_file}"
    )
    wheel_manifest = force_include.get("plugin.yaml")
    if not isinstance(wheel_manifest, str) or not wheel_manifest.endswith("/plugin.yaml"):
        raise ReleaseVersionError(f"{project_file}: plugin.yaml must be force-included in the wheel")

    manifest = project_file.with_name("plugin.yaml")
    if not manifest.is_file() or manifest.is_symlink():
        raise ReleaseVersionError(f"missing {manifest}")
    return Project(
        name=name,
        normalized_name=NORMALIZED_NAME_PATTERN.sub("_", name),
        directory=project_file.parent,
        manifest=manifest,
        wheel_manifest=wheel_manifest,
    )


def discover_projects(root: Path) -> tuple[Project, ...]:
    """Discover and validate every buildable plugin under ``packages``."""
    project_files = sorted((root / "packages").glob("*/pyproject.toml"))
    if not project_files:
        raise ReleaseVersionError("no package pyproject.toml files found")
    projects = [_project(project_file) for project_file in project_files]

    names = [project.name for project in projects]
    if len(names) != len(set(names)):
        raise ReleaseVersionError(f"duplicate project names: {names}")
    known_manifests = {project.manifest for project in projects}
    all_manifests = set((root / "packages").glob("*/plugin.yaml"))
    if all_manifests != known_manifests:
        extras = sorted(str(path) for path in all_manifests - known_manifests)
        raise ReleaseVersionError(f"plugin manifests without a build project: {extras}")
    return tuple(projects)


def _replace_manifest_version(content: bytes, version: str, path: Path) -> bytes:
    matches = list(MANIFEST_VERSION_PATTERN.finditer(content))
    if len(matches) != 1:
        raise ReleaseVersionError(f"{path}: expected exactly one top-level version field")
    match = matches[0]
    replacement = match.group(1) + version.encode("ascii") + match.group(3)
    return content[: match.start()] + replacement + content[match.end() :]


def apply_release_version(root: Path, release: ReleaseVersion) -> tuple[Path, ...]:
    """Atomically rewrite each manifest after prevalidating the full workspace."""
    staged: list[tuple[Path, Path]] = []
    try:
        for project in discover_projects(root):
            try:
                content = project.manifest.read_bytes()
                content.decode("utf-8")
            except (OSError, UnicodeError) as exc:
                raise ReleaseVersionError(f"cannot read {project.manifest}: {exc}") from exc
            updated = _replace_manifest_version(content, release.version, project.manifest)
            if updated == content:
                continue
            with tempfile.NamedTemporaryFile(dir=project.manifest.parent, delete=False) as temporary:
                temporary.write(updated)
                temporary.flush()
                os.fsync(temporary.fileno())
                temporary_path = Path(temporary.name)
                temporary_path.chmod(project.manifest.stat().st_mode)
                staged.append((temporary_path, project.manifest))
        for temporary, manifest in staged:
            temporary.replace(manifest)
        return tuple(manifest for _temporary, manifest in staged)
    except OSError as exc:
        raise ReleaseVersionError(f"cannot update release manifests: {exc}") from exc
    finally:
        for temporary, _manifest in staged:
            temporary.unlink(missing_ok=True)


def _manifest_version(content: bytes, context: str) -> str:
    matches = list(MANIFEST_VERSION_PATTERN.finditer(content))
    if len(matches) != 1:
        raise ReleaseVersionError(f"{context}: expected exactly one top-level version field")
    try:
        return matches[0].group(2).decode("ascii")
    except UnicodeDecodeError as exc:
        raise ReleaseVersionError(f"{context}: version is not ASCII") from exc


def _metadata_version(content: bytes, context: str) -> tuple[str, str]:
    try:
        message = Parser().parsestr(content.decode("utf-8"))
    except UnicodeError as exc:
        raise ReleaseVersionError(f"{context}: metadata is not UTF-8") from exc
    name = message.get("Name")
    version = message.get("Version")
    if name is None or version is None:
        raise ReleaseVersionError(f"{context}: missing Name or Version")
    return name, version


def _read_unique_zip_member(archive: zipfile.ZipFile, member: str, context: Path) -> bytes:
    if archive.namelist().count(member) != 1:
        raise ReleaseVersionError(f"{context}: expected exactly one {member}")
    return archive.read(member)


def _read_unique_tar_member(archive: tarfile.TarFile, member: str, context: Path) -> bytes:
    matches = [item for item in archive.getmembers() if item.name == member and item.isfile()]
    if len(matches) != 1:
        raise ReleaseVersionError(f"{context}: expected exactly one {member}")
    extracted = archive.extractfile(matches[0])
    if extracted is None:
        raise ReleaseVersionError(f"{context}: cannot read {member}")
    return extracted.read()


def _validate_dist_entries(dist_dir: Path, expected: set[str]) -> None:
    try:
        entries = tuple(dist_dir.iterdir())
    except OSError as exc:
        raise ReleaseVersionError(f"cannot read artifact directory {dist_dir}: {exc}") from exc
    invalid_entries = sorted(
        path.name
        for path in entries
        if path.name != ".gitignore" and (path.name not in expected or not path.is_file() or path.is_symlink())
    )
    actual = {path.name for path in entries if path.name in expected and path.is_file() and not path.is_symlink()}
    bookkeeping = [path for path in entries if path.name == ".gitignore"]
    if len(bookkeeping) > 1 or any(not path.is_file() or path.is_symlink() for path in bookkeeping):
        invalid_entries.append(".gitignore")
    if actual != expected or invalid_entries:
        raise ReleaseVersionError(
            f"artifact set differs: missing={sorted(expected - actual)}, extra={invalid_entries}"
        )


def verify_artifacts(root: Path, release: ReleaseVersion, dist_dir: Path) -> tuple[Path, ...]:
    """Verify exact artifact names plus embedded package and plugin metadata."""
    projects = discover_projects(root)
    expected = {
        filename
        for project in projects
        for filename in (
            f"{project.normalized_name}-{release.version}-py3-none-any.whl",
            f"{project.normalized_name}-{release.version}.tar.gz",
        )
    }
    _validate_dist_entries(dist_dir, expected)

    for project in projects:
        wheel_path = dist_dir / f"{project.normalized_name}-{release.version}-py3-none-any.whl"
        dist_info = f"{project.normalized_name}-{release.version}.dist-info/METADATA"
        try:
            with zipfile.ZipFile(wheel_path) as archive:
                metadata = _read_unique_zip_member(archive, dist_info, wheel_path)
                manifest = _read_unique_zip_member(archive, project.wheel_manifest, wheel_path)
        except (OSError, zipfile.BadZipFile) as exc:
            raise ReleaseVersionError(f"cannot inspect {wheel_path}: {exc}") from exc
        if _metadata_version(metadata, str(wheel_path)) != (project.name, release.version):
            raise ReleaseVersionError(f"{wheel_path}: package metadata does not match {project.name} {release.version}")
        if _manifest_version(manifest, str(wheel_path)) != release.version:
            raise ReleaseVersionError(f"{wheel_path}: embedded plugin version does not match {release.version}")

        sdist_path = dist_dir / f"{project.normalized_name}-{release.version}.tar.gz"
        prefix = f"{project.normalized_name}-{release.version}"
        try:
            with tarfile.open(sdist_path, mode="r:gz") as archive:
                metadata = _read_unique_tar_member(archive, f"{prefix}/PKG-INFO", sdist_path)
                manifest = _read_unique_tar_member(archive, f"{prefix}/plugin.yaml", sdist_path)
        except (OSError, tarfile.TarError) as exc:
            raise ReleaseVersionError(f"cannot inspect {sdist_path}: {exc}") from exc
        if _metadata_version(metadata, str(sdist_path)) != (project.name, release.version):
            raise ReleaseVersionError(f"{sdist_path}: package metadata does not match {project.name} {release.version}")
        if _manifest_version(manifest, str(sdist_path)) != release.version:
            raise ReleaseVersionError(f"{sdist_path}: embedded plugin version does not match {release.version}")
    return tuple(dist_dir / filename for filename in sorted(expected))


def _write_github_output(path: Path, release: ReleaseVersion) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as output:
        output.write(_release_records(release))


def _release_records(release: ReleaseVersion) -> str:
    return f"tag={release.tag}\nversion={release.version}\nref={release.ref}\n"


def _exit(parser: argparse.ArgumentParser, status: int, error: Exception) -> NoReturn:
    message = f"error: {error}"[:499].rstrip()
    parser.exit(status, f"{message}\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    resolve = subparsers.add_parser("resolve")
    resolve.add_argument("raw")
    resolve.add_argument("--github-output", type=Path)
    apply = subparsers.add_parser("apply")
    apply.add_argument("raw")
    apply.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    verify = subparsers.add_parser("verify-artifacts")
    verify.add_argument("raw")
    verify.add_argument("dist_dir", type=Path)
    verify.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    try:
        release = resolve_release_version(cast("str", args.raw))
    except ReleaseVersionError as exc:
        _exit(parser, 2, exc)
    if args.command == "resolve":
        sys.stdout.write(_release_records(release))
        github_output = cast("Path | None", args.github_output)
        if github_output is not None:
            try:
                _write_github_output(github_output, release)
            except OSError as exc:
                _exit(parser, 1, exc)
        return
    try:
        if args.command == "apply":
            apply_release_version(cast("Path", args.root), release)
        else:
            verify_artifacts(cast("Path", args.root), release, cast("Path", args.dist_dir))
    except ReleaseVersionError as exc:
        _exit(parser, 1, exc)


if __name__ == "__main__":
    main()

from __future__ import annotations
import importlib.util
import io
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest

if TYPE_CHECKING:
    from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]


def _load_release_version() -> ModuleType:
    spec = importlib.util.spec_from_file_location("hermes_release_version_test", ROOT / "tools" / "release_version.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module.__name__] = module
    spec.loader.exec_module(module)
    return module


release_version = cast("Any", _load_release_version())
ReleaseVersionError = release_version.ReleaseVersionError
apply_release_version = release_version.apply_release_version
discover_projects = release_version.discover_projects
resolve_release_version = release_version.resolve_release_version
verify_artifacts = release_version.verify_artifacts


def _workspace(tmp_path: Path, *, newline: str = "\n") -> Path:
    root = tmp_path / "workspace"
    package = root / "packages" / "demo-plugin"
    package.mkdir(parents=True)
    (package / "pyproject.toml").write_text(
        """[project]
name = "demo-plugin"
dynamic = ["version"]

[tool.hatch.version]
path = "plugin.yaml"

[tool.hatch.build.targets.wheel.force-include]
"plugin.yaml" = "demo_plugin/plugin.yaml"
""",
        encoding="utf-8",
    )
    (package / "plugin.yaml").write_bytes(f"name: demo{newline}version: 0.4.0{newline}enabled: true{newline}".encode())
    return root


@pytest.mark.parametrize("raw", ["1.2.3", "v1.2.3"])
def test_resolve_accepts_plain_and_prefixed_versions(raw: str) -> None:
    release = resolve_release_version(raw)
    assert release.version == "1.2.3"
    assert release.tag == "v1.2.3"
    assert release.ref == "refs/tags/v1.2.3"


@pytest.mark.parametrize(
    "raw",
    ["vv1.2.3", " 1.2.3", "1.2.3 ", "1.2", "01.2.3", "1.02.3", "1.2.03", "1.2.3-rc1"],
)
def test_resolve_rejects_noncanonical_versions(raw: str) -> None:
    with pytest.raises(ReleaseVersionError):
        resolve_release_version(raw)


def test_resolve_cli_prints_records_and_appends_github_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "github-output"
    monkeypatch.setattr(sys, "argv", ["release_version.py", "resolve", "v1.2.3", "--github-output", str(output)])
    release_version.main()
    records = "tag=v1.2.3\nversion=1.2.3\nref=refs/tags/v1.2.3\n"
    assert capsys.readouterr().out == records
    assert output.read_text(encoding="utf-8") == records


def test_invalid_cli_release_exits_two(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["release_version.py", "resolve", "vv1.2.3"])
    with pytest.raises(SystemExit) as caught:
        release_version.main()
    assert caught.value.code == 2
    assert "gitflow.prefix.versiontag" in capsys.readouterr().err


def test_discovery_and_apply_preserve_unrelated_bytes_and_newlines(tmp_path: Path) -> None:
    root = _workspace(tmp_path, newline="\r\n")
    project = discover_projects(root)[0]
    changed = apply_release_version(root, resolve_release_version("v2.3.4"))
    assert changed == (project.manifest,)
    assert project.manifest.read_bytes() == b"name: demo\r\nversion: 2.3.4\r\nenabled: true\r\n"
    assert apply_release_version(root, resolve_release_version("2.3.4")) == ()


def test_apply_prevalidates_every_manifest_before_writing(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    valid_manifest = root / "packages" / "demo-plugin" / "plugin.yaml"
    second = root / "packages" / "broken-plugin"
    second.mkdir()
    (second / "pyproject.toml").write_text(
        """[project]
name = "broken-plugin"
dynamic = ["version"]
[tool.hatch.version]
path = "plugin.yaml"
[tool.hatch.build.targets.wheel.force-include]
"plugin.yaml" = "broken_plugin/plugin.yaml"
""",
        encoding="utf-8",
    )
    (second / "plugin.yaml").write_text("name: broken\n", encoding="utf-8")
    with pytest.raises(ReleaseVersionError):
        apply_release_version(root, resolve_release_version("1.0.0"))
    assert b"version: 0.4.0" in valid_manifest.read_bytes()


def _add_tar_bytes(archive: tarfile.TarFile, name: str, content: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(content)
    archive.addfile(info, io.BytesIO(content))


def _artifacts(root: Path, release_version: str) -> Path:
    dist = root / "dist"
    dist.mkdir()
    prefix = f"demo_plugin-{release_version}"
    metadata = f"Metadata-Version: 2.4\nName: demo-plugin\nVersion: {release_version}\n".encode()
    manifest = f"name: demo\nversion: {release_version}\n".encode()
    with zipfile.ZipFile(dist / f"{prefix}-py3-none-any.whl", mode="w") as archive:
        archive.writestr(f"{prefix}.dist-info/METADATA", metadata)
        archive.writestr("demo_plugin/plugin.yaml", manifest)
    with tarfile.open(dist / f"{prefix}.tar.gz", mode="w:gz") as archive:
        _add_tar_bytes(archive, f"{prefix}/PKG-INFO", metadata)
        _add_tar_bytes(archive, f"{prefix}/plugin.yaml", manifest)
    return dist


def test_verify_artifacts_checks_names_and_embedded_versions(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    release = resolve_release_version("1.2.3")
    dist = _artifacts(root, release.version)
    verify_artifacts(root, release, dist)

    wheel = dist / "demo_plugin-1.2.3-py3-none-any.whl"
    wheel.rename(dist / "demo_plugin-9.9.9-py3-none-any.whl")
    with pytest.raises(ReleaseVersionError, match="artifact set differs"):
        verify_artifacts(root, release, dist)

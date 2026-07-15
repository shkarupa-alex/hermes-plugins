from __future__ import annotations
import importlib.util
import sys
import tomllib
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
PLUGIN_CASES = (
    ("hermes-onnx-asr", "hermes_onnx_asr"),
    ("hermes-vk-community", "hermes_vk_community"),
)


def _load_directory_plugin(directory_name: str) -> ModuleType:
    plugin_dir = ROOT / "packages" / directory_name
    module_name = f"git_install_test_{directory_name.replace('-', '_')}"
    for loaded_name in tuple(sys.modules):
        if loaded_name == module_name or loaded_name.startswith(f"{module_name}."):
            sys.modules.pop(loaded_name)
    spec = importlib.util.spec_from_file_location(
        module_name,
        plugin_dir / "__init__.py",
        submodule_search_locations=[str(plugin_dir)],
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(("directory_name", "package_name"), PLUGIN_CASES)
def test_git_subdirectory_exports_register_and_sources_version_from_manifest(
    directory_name: str,
    package_name: str,
) -> None:
    plugin_dir = ROOT / "packages" / directory_name
    module = _load_directory_plugin(directory_name)
    assert callable(module.register)
    assert module.register.__module__.startswith(package_name)

    project = tomllib.loads((plugin_dir / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["dynamic"] == ["version"]
    assert "version" not in project["project"]
    assert project["tool"]["hatch"]["version"]["path"] == "plugin.yaml"


@pytest.mark.parametrize(("directory_name", "_package_name"), PLUGIN_CASES)
def test_git_subdirectory_installs_missing_dependencies_once(
    directory_name: str,
    _package_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_directory_plugin(directory_name)
    bootstrap = sys.modules[f"{module.__name__}._hermes_git_bootstrap"]
    requirement_states = iter((("missing-package>=1",), ("missing-package>=1",), ()))
    calls: list[list[str]] = []

    def missing_requirements(_requirements: tuple[str, ...]) -> tuple[str, ...]:
        return next(requirement_states)

    def install_command(_requirements: tuple[str, ...]) -> list[str]:
        return ["uv", "pip", "install"]

    monkeypatch.setattr(bootstrap, "_missing_requirements", missing_requirements)
    monkeypatch.setattr(bootstrap, "_lazy_installs_allowed", lambda: True)
    monkeypatch.setattr(bootstrap, "_install_command", install_command)
    monkeypatch.setattr(bootstrap, "_installation_lock", nullcontext)

    def fake_run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(bootstrap.subprocess, "run", fake_run)
    bootstrap._ensure_dependencies(ROOT / "packages" / directory_name)
    assert calls == [["uv", "pip", "install"]]


@pytest.mark.parametrize(("directory_name", "_package_name"), PLUGIN_CASES)
def test_git_subdirectory_respects_lazy_install_opt_out(
    directory_name: str,
    _package_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_directory_plugin(directory_name)
    bootstrap = sys.modules[f"{module.__name__}._hermes_git_bootstrap"]

    def missing_requirements(_requirements: tuple[str, ...]) -> tuple[str, ...]:
        return ("missing>=1",)

    def install_command(_requirements: tuple[str, ...]) -> list[str]:
        return ["uv", "pip", "install", "missing>=1"]

    monkeypatch.setattr(bootstrap, "_missing_requirements", missing_requirements)
    monkeypatch.setattr(bootstrap, "_lazy_installs_allowed", lambda: False)
    monkeypatch.setattr(bootstrap, "_install_command", install_command)
    with pytest.raises(RuntimeError, match="lazy installs are disabled; run: uv pip install"):
        bootstrap._ensure_dependencies(ROOT / "packages" / directory_name)

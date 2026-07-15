"""Bootstrap a src-layout plugin cloned by ``hermes plugins install``."""

from __future__ import annotations
import importlib
import importlib.metadata
import importlib.util
import os
import shlex
import shutil
import subprocess
import sys
import time
import tomllib
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING, cast

from hermes_cli.config import load_config
from packaging.requirements import Requirement
from packaging.version import Version

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

INSTALL_TIMEOUT_SECONDS = 1800
LOCK_STALE_SECONDS = 3600
MIN_HERMES_VENV_PARENT_COUNT = 4


def _requirements(plugin_dir: Path) -> tuple[str, ...]:
    data = cast("dict[str, object]", tomllib.loads((plugin_dir / "pyproject.toml").read_text(encoding="utf-8")))
    project_value = data.get("project")
    project = cast("dict[str, object]", project_value) if isinstance(project_value, dict) else {}
    values = project.get("dependencies", [])
    if not isinstance(values, list) or not all(isinstance(value, str) for value in cast("list[object]", values)):
        raise RuntimeError("plugin pyproject.toml has invalid project.dependencies")
    return tuple(cast("list[str]", values))


def _missing_requirements(requirements: tuple[str, ...]) -> tuple[str, ...]:
    missing: list[str] = []
    for value in requirements:
        requirement = Requirement(value)
        if requirement.marker is not None and not requirement.marker.evaluate():
            continue
        try:
            installed = Version(importlib.metadata.version(requirement.name))
        except importlib.metadata.PackageNotFoundError:
            missing.append(value)
            continue
        if requirement.specifier and installed not in requirement.specifier:
            missing.append(value)
    return tuple(missing)


def _lazy_installs_allowed() -> bool:
    try:
        config = load_config()
    except Exception:  # noqa: BLE001 - unreadable config follows Hermes' fail-open policy
        config = None
    if isinstance(config, dict):
        security_value = cast("dict[str, object]", config).get("security")
        security = cast("dict[str, object]", security_value) if isinstance(security_value, dict) else {}
        if not bool(security.get("allow_lazy_installs", True)):
            return False
    return os.environ.get("HERMES_DISABLE_LAZY_INSTALLS") != "1"


def _uv_executable() -> Path | None:
    executable_name = "uv.exe" if os.name == "nt" else "uv"
    found = shutil.which(executable_name)
    if found:
        return Path(found)
    candidates = [Path.home() / ".hermes" / "bin" / executable_name]
    resolved_python = Path(sys.executable).resolve()
    if len(resolved_python.parents) >= MIN_HERMES_VENV_PARENT_COUNT:
        candidates.append(resolved_python.parents[2].parent / "bin" / executable_name)
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def _install_command(requirements: tuple[str, ...]) -> list[str]:
    uv = _uv_executable()
    if uv is not None:
        return [str(uv), "pip", "install", "--python", sys.executable, *requirements]
    if importlib.util.find_spec("pip") is None:
        raise RuntimeError("plugin dependencies are missing and neither Hermes uv nor pip is available")
    return [sys.executable, "-m", "pip", "install", *requirements]


@contextmanager
def _installation_lock() -> Generator[None, None, None]:
    lock = Path(sys.executable).resolve().parent.parent / ".hermes-plugin-deps.lock"
    deadline = time.monotonic() + INSTALL_TIMEOUT_SECONDS
    while True:
        try:
            lock.mkdir()
            break
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime > LOCK_STALE_SECONDS:
                    lock.rmdir()
                    continue
            except (FileNotFoundError, OSError):
                pass
            if time.monotonic() >= deadline:
                raise RuntimeError(f"timed out waiting for dependency installer lock: {lock}") from None
            time.sleep(0.2)
    try:
        yield
    finally:
        with suppress(OSError):
            lock.rmdir()


def _ensure_dependencies(plugin_dir: Path) -> None:
    requirements = _requirements(plugin_dir)
    missing = _missing_requirements(requirements)
    if not missing:
        return
    command = _install_command(missing)
    if not _lazy_installs_allowed():
        raise RuntimeError(
            "plugin dependencies are missing and lazy installs are disabled; run: " + shlex.join(command)
        )
    with _installation_lock():
        missing = _missing_requirements(requirements)
        if not missing:
            return
        command = _install_command(missing)
        try:
            result = subprocess.run(  # noqa: S603 - argv is parsed from trusted plugin metadata
                command,
                capture_output=True,
                text=True,
                timeout=INSTALL_TIMEOUT_SECONDS,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"failed to install plugin dependencies: {exc}") from exc
        if result.returncode != 0:
            details = (result.stderr or result.stdout or "dependency installer failed").strip()
            raise RuntimeError(f"failed to install plugin dependencies: {details[-4000:]}")
        importlib.invalidate_caches()
        remaining = _missing_requirements(requirements)
        if remaining:
            raise RuntimeError(f"dependency installer completed but requirements remain missing: {remaining}")


def load_register(module_name: str, plugin_dir: Path) -> Callable[[object], object]:
    _ensure_dependencies(plugin_dir)
    source = str(plugin_dir / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    module = importlib.import_module(module_name)
    register = getattr(module, "register", None)
    if not callable(register):
        raise TypeError(f"{module_name} does not expose register(ctx)")
    return register

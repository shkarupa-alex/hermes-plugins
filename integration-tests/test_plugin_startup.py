from __future__ import annotations
import os
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


STARTUP_SCRIPT = """
import hermes_cli.config
from hermes_cli.plugins import discover_plugins
from agent.transcription_registry import get_provider
from gateway.platform_registry import platform_registry
discover_plugins()
assert get_provider('onnx_asr') is not None
assert platform_registry.get('vk') is not None
"""


def test_enabled_plugins_register_during_cold_hermes_startup(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text(
        "plugins:\n  enabled: [onnx-asr, vk-community]\n",
        encoding="utf-8",
    )
    result = subprocess.run(  # noqa: S603 - fixed test script in the active interpreter
        [sys.executable, "-c", STARTUP_SCRIPT],
        env={**os.environ, "HERMES_HOME": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Failed to load" not in result.stderr
    assert "partially initialized" not in result.stderr

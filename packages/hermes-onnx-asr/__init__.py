# ruff: noqa: N999
"""Hermes directory-plugin entry point for Git subdirectory installs."""

from pathlib import Path

from ._hermes_git_bootstrap import load_register

register = load_register("hermes_onnx_asr", Path(__file__).resolve().parent)

__all__ = ["register"]

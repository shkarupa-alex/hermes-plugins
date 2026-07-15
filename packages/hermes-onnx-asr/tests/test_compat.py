from importlib.metadata import PackageNotFoundError

import pytest

from hermes_onnx_asr import compat


def test_requirement_check_accepts_certified_version_window(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(compat, "check_compatibility", lambda: (True, "ok"))

    def installed(package: str) -> str:
        return {"onnx-asr": "0.12.4", "onnxruntime": "1.23.2"}[package]

    monkeypatch.setattr(compat, "version", installed)
    assert compat.check_requirements()


def test_requirement_check_rejects_uncertified_new_versions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(compat, "check_compatibility", lambda: (True, "ok"))

    def installed(package: str) -> str:
        return {"onnx-asr": "0.13.0", "onnxruntime": "1.24.1"}[package]

    monkeypatch.setattr(compat, "version", installed)
    assert not compat.check_requirements()


def test_requirement_check_rejects_missing_or_old_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(compat, "check_compatibility", lambda: (True, "ok"))

    def old(package: str) -> str:
        return "0.11.0" if package == "onnx-asr" else "1.23.2"

    monkeypatch.setattr(compat, "version", old)
    assert not compat.check_requirements()

    def missing(_package: str) -> str:
        raise PackageNotFoundError

    monkeypatch.setattr(compat, "version", missing)
    assert not compat.check_requirements()

import pytest

from hermes_vk_community import compat


def test_compatibility_accepts_newer_hermes_when_contract_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    def installed(_package: str) -> str:
        return "99.0.0"

    monkeypatch.setattr(compat, "version", installed)
    compatible, message = compat.check_compatibility()
    assert compatible
    assert message == "Hermes 99.0.0 contract is compatible"


def test_compatibility_rejects_hermes_below_minimum(monkeypatch: pytest.MonkeyPatch) -> None:
    def installed(_package: str) -> str:
        return "0.18.1"

    monkeypatch.setattr(compat, "version", installed)
    compatible, message = compat.check_compatibility()
    assert not compatible
    assert message == "hermes-agent 0.18.1 is below the minimum supported version >=0.18.2"

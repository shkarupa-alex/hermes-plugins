# pyright: reportPrivateUsage=false
from __future__ import annotations
from typing import TYPE_CHECKING

import pytest

from hermes_onnx_asr import catalog
from hermes_onnx_asr.catalog import Catalog, CatalogEntry, verify_bundle
from hermes_onnx_asr.errors import OnnxAsrError

if TYPE_CHECKING:
    from pathlib import Path


def fake_catalog() -> Catalog:
    return Catalog(
        schema_version=1,
        onnx_asr_version="0.12.0",
        models=[
            CatalogEntry(
                alias="test-model",
                repository="owner/repository",
                revision="a" * 40,
                quantizations=["int8"],
                files={"int8": ["model.onnx"]},
                download="huggingface_snapshot",
            ),
        ],
        vads=[],
    )


def test_manifest_round_trip_and_tamper_detection(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(catalog, "load_catalog", fake_catalog)

    def expected_patterns(_alias: str, _quantization: str | None, _kind: str) -> list[str]:
        return ["model.onnx"]

    monkeypatch.setattr(catalog, "_expected_patterns", expected_patterns)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "config.json").write_text("{}", encoding="utf-8")
    model = bundle / "model.onnx"
    model.write_bytes(b"model bytes")
    entry = fake_catalog().models[0]
    catalog._write_manifest(bundle, entry, "int8", "model")
    manifest = verify_bundle(bundle, "test-model", "int8")
    model_file = next(item for item in manifest.files if item.path == "model.onnx")
    assert model_file.sha256 == "9cb7487000bc86ac36ce83c4acfabe8878552be99572a6770f65ab1d048a5c48"
    model.write_bytes(b"tampered")
    with pytest.raises(OnnxAsrError) as caught:
        verify_bundle(bundle, "test-model", "int8")
    assert caught.value.code == "model_not_installed"


def test_bundle_without_upstream_config_is_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(catalog, "load_catalog", fake_catalog)

    def expected_patterns(_alias: str, _quantization: str | None, _kind: str) -> list[str]:
        return ["model.onnx"]

    monkeypatch.setattr(catalog, "_expected_patterns", expected_patterns)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "model.onnx").write_bytes(b"model bytes")
    catalog._write_manifest(bundle, fake_catalog().models[0], "int8", "model")
    with pytest.raises(OnnxAsrError, match="not installed"):
        verify_bundle(bundle, "test-model", "int8")


def test_catalog_contains_only_immutable_revisions() -> None:
    loaded = catalog.load_catalog()
    assert loaded.onnx_asr_version == "0.12.0"
    assert tuple(entry.alias for entry in loaded.models) == catalog.upstream_model_names()
    assert catalog.certified_model_names() == ("gigaam-v3-e2e-rnnt", "t-tech/t-one")
    assert {entry.alias for entry in loaded.models} >= {
        "gigaam-v3-e2e-rnnt",
        "nemo-fastconformer-ru-rnnt",
    }
    assert all(len(entry.revision) == 40 and set(entry.revision) <= set("0123456789abcdef") for entry in loaded.models)


def test_catalog_accepts_newer_upstream_with_additional_pending_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    current = catalog.upstream_model_names()

    def newer(_package: str) -> str:
        return "0.13.0"

    monkeypatch.setattr(catalog, "version", newer)
    monkeypatch.setattr(catalog, "upstream_model_names", lambda: (*current, "future-model"))
    loaded = catalog.load_catalog()
    assert loaded.onnx_asr_version == "0.12.0"
    assert "future-model" not in catalog.certified_model_names()


def test_catalog_rejects_dependency_below_minimum(monkeypatch: pytest.MonkeyPatch) -> None:
    def older(_package: str) -> str:
        return "0.11.0"

    monkeypatch.setattr(catalog, "version", older)
    with pytest.raises(RuntimeError, match="below the catalog minimum"):
        catalog.load_catalog()

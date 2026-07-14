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
        onnx_asr_version="0.11.0",
        models=[
            CatalogEntry(
                alias="test-model",
                repository="owner/repository",
                revision="a" * 40,
                quantizations=["int8"],
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
    model = bundle / "model.onnx"
    model.write_bytes(b"model bytes")
    entry = fake_catalog().models[0]
    catalog._write_manifest(bundle, entry, "int8", "model")
    manifest = verify_bundle(bundle, "test-model", "int8")
    assert manifest.files[0].sha256 == "9cb7487000bc86ac36ce83c4acfabe8878552be99572a6770f65ab1d048a5c48"
    model.write_bytes(b"tampered")
    with pytest.raises(OnnxAsrError) as caught:
        verify_bundle(bundle, "test-model", "int8")
    assert caught.value.code == "model_not_installed"


def test_catalog_contains_only_immutable_revisions() -> None:
    loaded = catalog.load_catalog()
    assert loaded.onnx_asr_version == "0.11.0"
    assert {entry.alias for entry in loaded.models} >= {
        "gigaam-v3-e2e-rnnt",
        "nemo-fastconformer-ru-rnnt",
    }
    assert all(len(entry.revision) == 40 and set(entry.revision) <= set("0123456789abcdef") for entry in loaded.models)

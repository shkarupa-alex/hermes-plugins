from __future__ import annotations
import hashlib
import json
import os
import shutil
import stat
import time
import uuid
from datetime import UTC, datetime
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path
from typing import Literal, get_args

from filelock import FileLock
from huggingface_hub import snapshot_download  # pyright: ignore[reportUnknownVariableType]
from packaging.version import Version
from pydantic import BaseModel, ConfigDict, field_validator

from hermes_onnx_asr.errors import OnnxAsrError, safe_error

MANIFEST_SCHEMA_VERSION = 1
GIT_COMMIT_LENGTH = 40


class CatalogEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    alias: str
    repository: str
    revision: str
    quantizations: list[str | None]
    files: dict[str, list[str]]
    download: Literal["huggingface_snapshot"]
    certified: bool = False
    requires_config: bool = True

    @field_validator("revision")
    @classmethod
    def immutable_revision(cls, value: str) -> str:
        if len(value) != GIT_COMMIT_LENGTH or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("catalog revision must be a lowercase 40-character Git commit")
        return value


class Catalog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int
    onnx_asr_version: str
    models: list[CatalogEntry]
    vads: list[CatalogEntry]


class ManifestFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    size: int
    sha256: str


class BundleManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int
    kind: Literal["model", "vad"]
    alias: str
    quantization: str | None
    repository: str
    revision: str
    onnx_asr_version: str
    created_at: datetime
    files: list[ManifestFile]
    manifest_sha256: str


def load_catalog() -> Catalog:
    resource = files("hermes_onnx_asr").joinpath("data/catalog.json")
    catalog = Catalog.model_validate_json(resource.read_text(encoding="utf-8"))
    installed = Version(version("onnx-asr"))
    minimum = Version(catalog.onnx_asr_version)
    if installed < minimum:
        raise RuntimeError(f"onnx-asr {installed} is below the catalog minimum {minimum}")
    upstream = set(upstream_model_names())
    unavailable_certified = {entry.alias for entry in catalog.models if entry.certified and entry.alias not in upstream}
    if unavailable_certified:
        names = ", ".join(sorted(unavailable_certified))
        raise RuntimeError(f"certified ONNX ASR models disappeared from the upstream registry: {names}")
    return catalog


def upstream_model_names() -> tuple[str, ...]:
    """Use the same typed model registry as the upstream onnx-asr CLI."""
    from onnx_asr.loader import AsrNames  # noqa: PLC0415

    names = get_args(AsrNames)
    if not names or not all(isinstance(name, str) and name for name in names):
        raise RuntimeError("onnx-asr exposed an invalid model registry")
    return names


def model_entry(alias: str) -> CatalogEntry:
    for entry in load_catalog().models:
        if entry.alias == alias:
            return entry
    raise safe_error("model_not_in_catalog")


def catalog_model_entries() -> tuple[CatalogEntry, ...]:
    upstream = set(upstream_model_names())
    return tuple(entry for entry in load_catalog().models if entry.alias in upstream)


def certified_model_names() -> tuple[str, ...]:
    return tuple(entry.alias for entry in catalog_model_entries() if entry.certified)


def model_languages(alias: str) -> list[str]:
    if alias in {
        "gigaam-multilingual-ctc",
        "gigaam-multilingual-large-ctc",
        "nemo-parakeet-tdt-0.6b-v3",
        "nemo-canary-1b-v2",
        "whisper-base",
    }:
        return ["multilingual"]
    if alias in {
        "nemo-parakeet-ctc-0.6b",
        "nemo-parakeet-rnnt-0.6b",
        "nemo-parakeet-tdt-0.6b-v2",
    }:
        return ["en"]
    return ["ru"]


def catalog_entry(alias: str, quantization: str | None, *, kind: Literal["model", "vad"] = "model") -> CatalogEntry:
    catalog = load_catalog()
    entries = catalog.models if kind == "model" else catalog.vads
    for entry in entries:
        if entry.alias == alias and quantization in entry.quantizations:
            return entry
    raise safe_error("model_not_in_catalog" if kind == "model" else "vad_not_installed")


def bundle_path(root: Path, alias: str, quantization: str | None, *, kind: Literal["model", "vad"] = "model") -> Path:
    suffix = quantization or "fp32"
    return root / ("models" if kind == "model" else "vad") / alias / suffix


def _expected_patterns(alias: str, quantization: str | None, kind: Literal["model", "vad"]) -> list[str]:
    entry = catalog_entry(alias, quantization, kind=kind)
    key = quantization or "fp32"
    try:
        patterns = entry.files[key]
    except KeyError as exc:
        raise safe_error("model_not_in_catalog" if kind == "model" else "vad_not_installed") from exc
    return [
        "config.json",
        "config.yaml",
        *patterns,
        *(str(Path(pattern).with_suffix(".onnx?data")) for pattern in patterns if Path(pattern).suffix == ".onnx"),
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _canonical_manifest_hash(data: dict[str, object]) -> str:
    unsigned = {key: value for key, value in data.items() if key != "manifest_sha256"}
    payload = json.dumps(unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def _inventory(directory: Path) -> list[ManifestFile]:
    result: list[ManifestFile] = []
    for path in sorted(directory.rglob("*")):
        if path.name == "manifest.json" or path.is_dir():
            continue
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            raise OnnxAsrError("model_load_failed", "Downloaded bundle contains an unsupported file type.")
        relative = path.relative_to(directory).as_posix()
        result.append(ManifestFile(path=relative, size=path.stat().st_size, sha256=_sha256(path)))
    return result


def _fsync_directory(directory: Path) -> None:
    try:
        directory_fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(directory_fd)
    except OSError:
        pass
    finally:
        os.close(directory_fd)


def _write_manifest(
    directory: Path,
    entry: CatalogEntry,
    quantization: str | None,
    kind: Literal["model", "vad"],
) -> BundleManifest:
    _fsync_files(directory)
    draft = BundleManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        kind=kind,
        alias=entry.alias,
        quantization=quantization,
        repository=entry.repository,
        revision=entry.revision,
        onnx_asr_version=version("onnx-asr"),
        created_at=datetime.now(UTC),
        files=_inventory(directory),
        manifest_sha256="",
    )
    manifest = draft.model_copy(update={"manifest_sha256": _canonical_manifest_hash(draft.model_dump(mode="json"))})
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    with manifest_path.open("rb") as handle:
        os.fsync(handle.fileno())
    return manifest


def _fsync_files(directory: Path) -> None:
    for path in directory.rglob("*"):
        if path.is_file() and not path.is_symlink():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())


def verify_bundle(
    directory: Path,
    alias: str,
    quantization: str | None,
    *,
    kind: Literal["model", "vad"] = "model",
) -> BundleManifest:
    entry = catalog_entry(alias, quantization, kind=kind)
    manifest_path = directory / "manifest.json"
    try:
        manifest = BundleManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise safe_error("model_not_installed" if kind == "model" else "vad_not_installed") from exc
    raw = manifest.model_dump(mode="json")
    if manifest.manifest_sha256 != _canonical_manifest_hash(raw):
        raise safe_error("model_not_installed" if kind == "model" else "vad_not_installed")
    identity = (manifest.kind, manifest.alias, manifest.quantization, manifest.repository, manifest.revision)
    expected = (kind, alias, quantization, entry.repository, entry.revision)
    if identity != expected or manifest.onnx_asr_version != version("onnx-asr"):
        raise safe_error("model_not_installed" if kind == "model" else "vad_not_installed")
    actual = _inventory(directory)
    if actual != manifest.files:
        raise safe_error("model_not_installed" if kind == "model" else "vad_not_installed")
    if (
        kind == "model"
        and entry.requires_config
        and not any((directory / name).is_file() for name in ("config.json", "config.yaml"))
    ):
        raise safe_error("model_not_installed" if kind == "model" else "vad_not_installed")
    for pattern in _expected_patterns(alias, quantization, kind):
        if pattern in {"config.json", "config.yaml"} or ".onnx?data" in pattern:
            continue
        matches = list(directory.glob(pattern))
        if not matches:
            raise safe_error("model_not_installed" if kind == "model" else "vad_not_installed")
    return manifest


def fetch_bundle(
    root: Path,
    alias: str,
    quantization: str | None,
    *,
    kind: Literal["model", "vad"] = "model",
) -> Path:
    entry = catalog_entry(alias, quantization, kind=kind)
    destination = bundle_path(root, alias, quantization, kind=kind)
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock = FileLock(str(destination) + ".lock")
    with lock:
        if destination.exists():
            try:
                verify_bundle(destination, alias, quantization, kind=kind)
            except OnnxAsrError:
                quarantine = destination.with_name(
                    f"{destination.name}.invalid-{int(time.time())}-{uuid.uuid4().hex[:8]}"
                )
                destination.rename(quarantine)
            else:
                return destination
        staging = destination.with_name(f".{destination.name}.staging-{uuid.uuid4().hex}")
        staging.mkdir(mode=0o700)
        try:
            snapshot_download(
                repo_id=entry.repository,
                revision=entry.revision,
                local_dir=staging,
                allow_patterns=_expected_patterns(alias, quantization, kind),
            )
            shutil.rmtree(staging / ".cache", ignore_errors=True)
            _write_manifest(staging, entry, quantization, kind)
            verify_bundle(staging, alias, quantization, kind=kind)
            _fsync_directory(staging)
            staging.rename(destination)
            _fsync_directory(destination.parent)
        except Exception as exc:
            shutil.rmtree(staging, ignore_errors=True)
            if isinstance(exc, OnnxAsrError):
                raise
            raise safe_error("model_load_failed") from exc
    return destination

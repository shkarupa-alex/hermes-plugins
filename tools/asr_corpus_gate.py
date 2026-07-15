from __future__ import annotations
import argparse
import hashlib
import re
import statistics
import time
import unicodedata
from pathlib import Path

import jiwer
import psutil
from pydantic import BaseModel, ConfigDict, Field

from hermes_onnx_asr.provider import provider

MIN_CLIPS = 30
MIN_SECONDS = 20 * 60
PUNCTUATION = re.compile(r"[^\w\sё]", re.IGNORECASE)


class CorpusRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    duration_seconds: float = Field(gt=0)
    reference: str = Field(min_length=1)
    keywords: list[str] = Field(default_factory=list[str])
    license: str = Field(min_length=1)
    source: str = Field(min_length=1)
    speaker_tags: list[str] = Field(default_factory=list[str])
    noise_tags: list[str] = Field(default_factory=list[str])


class Baseline(BaseModel):
    model_config = ConfigDict(extra="forbid")

    wer: float = Field(ge=0)
    keyword_recall: float = Field(ge=0, le=1)


def normalize(text: str) -> str:
    value = unicodedata.normalize("NFC", text).casefold()
    return " ".join(PUNCTUATION.sub(" ", value).split())


def load_manifest(path: Path) -> list[CorpusRow]:
    rows = [CorpusRow.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) < MIN_CLIPS or sum(row.duration_seconds for row in rows) < MIN_SECONDS:
        raise ValueError("Russian release corpus must contain at least 30 clips and 20 minutes")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()
    rows = load_manifest(args.manifest)
    references: list[str] = []
    hypotheses: list[str] = []
    keyword_hits = 0
    keyword_total = 0
    elapsed = 0.0
    latencies: list[float] = []
    peak_rss = 0
    for row in rows:
        audio = args.manifest.parent / row.path
        if hashlib.sha256(audio.read_bytes()).hexdigest() != row.sha256:
            raise ValueError(f"corpus hash mismatch: {row.path}")
        started = time.perf_counter()
        result = provider.transcribe(str(audio))
        latency = time.perf_counter() - started
        elapsed += latency
        latencies.append(latency)
        peak_rss = max(peak_rss, psutil.Process().memory_info().rss)
        if not result.get("success"):
            raise RuntimeError(f"transcription failed for {row.path}: {result.get('error_code')}")
        reference = normalize(row.reference)
        hypothesis = normalize(str(result["transcript"]))
        references.append(reference)
        hypotheses.append(hypothesis)
        for keyword in row.keywords:
            keyword_total += 1
            keyword_hits += int(normalize(keyword) in hypothesis)
    metrics = {
        "clips": len(rows),
        "seconds": sum(row.duration_seconds for row in rows),
        "wer": jiwer.wer(references, hypotheses),
        "keyword_recall": keyword_hits / keyword_total if keyword_total else 1.0,
        "rtf": elapsed / sum(row.duration_seconds for row in rows),
        "cold_start_seconds": latencies[0],
        "warm_median_seconds": statistics.median(latencies[1:]),
        "warm_p95_seconds": statistics.quantiles(latencies[1:], n=20)[18],
        "peak_rss_bytes": peak_rss,
    }
    baseline = Baseline.model_validate_json(args.baseline.read_text(encoding="utf-8"))
    if metrics["wer"] > baseline.wer + 0.02:
        raise RuntimeError("WER regressed by more than two absolute percentage points")
    if metrics["keyword_recall"] < baseline.keyword_recall - 0.01:
        raise RuntimeError("keyword recall regressed by more than one absolute percentage point")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

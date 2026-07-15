from __future__ import annotations
import argparse
import hashlib
import json
import platform
import re
import statistics
import sys
import time
import unicodedata
from pathlib import Path

import jiwer
import psutil
from pydantic import BaseModel, ConfigDict, Field

from hermes_onnx_asr.provider import provider

MIN_CLIPS = 30
MIN_SECONDS = 20 * 60
WARM_RUNS = 5
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


def transcribe_or_raise(audio: Path, phase: str) -> dict[str, object]:
    result = provider.transcribe(str(audio))
    if not result.get("success"):
        raise RuntimeError(f"{phase} transcription failed: {result.get('error_code')}")
    return result


def benchmark_warm(audio: Path) -> list[float]:
    transcribe_or_raise(audio, "warm-up")
    latencies: list[float] = []
    for _ in range(WARM_RUNS):
        started = time.perf_counter()
        transcribe_or_raise(audio, "warm")
        latencies.append(time.perf_counter() - started)
    return latencies


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = load_manifest(args.manifest)
    references: list[str] = []
    hypotheses: list[str] = []
    keyword_hits = 0
    keyword_total = 0
    elapsed = 0.0
    cold_start_seconds: float | None = None
    peak_rss = 0
    for row in rows:
        audio = args.manifest.parent / row.path
        if hashlib.sha256(audio.read_bytes()).hexdigest() != row.sha256:
            raise ValueError(f"corpus hash mismatch: {row.path}")
        started = time.perf_counter()
        result = transcribe_or_raise(audio, "corpus")
        latency = time.perf_counter() - started
        elapsed += latency
        if cold_start_seconds is None:
            cold_start_seconds = latency
        peak_rss = max(peak_rss, psutil.Process().memory_info().rss)
        reference = normalize(row.reference)
        hypothesis = normalize(str(result["transcript"]))
        references.append(reference)
        hypotheses.append(hypothesis)
        for keyword in row.keywords:
            keyword_total += 1
            keyword_hits += int(normalize(keyword) in hypothesis)
    benchmark_audio = args.manifest.parent / rows[0].path
    warm_latencies = benchmark_warm(benchmark_audio)
    peak_rss = max(peak_rss, psutil.Process().memory_info().rss)
    wer = jiwer.wer(references, hypotheses)
    keyword_recall = keyword_hits / keyword_total if keyword_total else 1.0
    metrics = {
        "clips": len(rows),
        "seconds": sum(row.duration_seconds for row in rows),
        "wer": wer,
        "keyword_recall": keyword_recall,
        "rtf": elapsed / sum(row.duration_seconds for row in rows),
        "cold_start_seconds": cold_start_seconds,
        "warm_median_seconds": statistics.median(warm_latencies),
        "warm_p95_seconds": statistics.quantiles(warm_latencies, n=20)[18],
        "peak_rss_bytes": peak_rss,
        "hardware": {
            "machine": platform.machine(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
    }
    baseline = Baseline.model_validate_json(args.baseline.read_text(encoding="utf-8"))
    rendered = json.dumps(metrics, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    sys.stdout.write(rendered)
    if wer > baseline.wer + 0.02:
        raise RuntimeError("WER regressed by more than two absolute percentage points")
    if keyword_recall < baseline.keyword_recall - 0.01:
        raise RuntimeError("keyword recall regressed by more than one absolute percentage point")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

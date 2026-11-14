"""Lab 1 measurements: streaming intake, lazy vs eager, and where the peak lives.

    uv run python bench/lab1.py [corpus_dir] [--limit N] [--out PATH]

What is measured, and how:

* **lazy** is ``findex.stats.lazy_stats`` - the generator pipeline Lab 1 ships;
* **eager** is ``findex.eager.eager_stats`` - the deliberately bad version that
  materialises every text and every token before counting;
* both run under ``tracemalloc`` (``findex.stats.measure``), so the timing
  includes the tracer. Compare the two rows with each other, never with a run
  outside the tracer;
* **single** is the same lazy pipeline over a corpus of exactly one document -
  a copy of the largest book - which answers the question "does the peak come
  from the corpus size or from one big document?";
* the answers are asserted equal between lazy and eager before anything is
  written; a mismatch is a bug, not a measurement.

Writes ``bench/lab1_results.json``; the numbers in the README come from that
file.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import time
import tracemalloc
import unicodedata
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:  # --no-project runs have no installed pkg
    sys.path.insert(0, str(ROOT / "src"))

from findex.corpus import iter_documents  # noqa: E402
from findex.eager import eager_stats  # noqa: E402
from findex.stats import Stats, lazy_stats, measure  # noqa: E402
from findex.tokenize import tokenize  # noqa: E402

TMP = ROOT / "bench" / "_tmp" / "lab1"


def corpus_files(root: Path) -> list[Path]:
    """Every document the lazy stream would visit, in the same order."""
    return [doc.path for doc in iter_documents(root)]


def largest(paths: list[Path]) -> Path:
    return max(paths, key=lambda p: p.stat().st_size)


def single_document_stats(path: Path) -> tuple[Stats, float, int, int]:
    """Run the shipped pipeline over a corpus of one file, traced.

    Returns ``(stats, elapsed, peak_bytes, file_bytes)``. The file is copied into
    ``bench/_tmp/lab1`` instead of being re-implemented: the point is to measure
    the *same* reader and tokenizer, not a hand-written stand-in.
    """
    shutil.rmtree(TMP, ignore_errors=True)
    TMP.mkdir(parents=True, exist_ok=True)
    copy = TMP / path.name
    shutil.copy2(path, copy)
    measurement = measure(lambda: lazy_stats(TMP))
    measured_paths = corpus_files(TMP)
    assert measured_paths == [copy], measured_paths
    return (
        measurement.result,
        measurement.elapsed,
        measurement.peak_bytes,
        path.stat().st_size,
    )


def stage_peaks(path: Path) -> dict[str, Any]:
    """Where the peak of the one-document run is spent.

    The same reader and the same tokenizer as the shipped pipeline; only the
    measurement is split into three phases, each with ``tracemalloc.reset_peak()``
    so the number belongs to that phase alone. ``count`` re-runs the whole
    tokenizer, so it includes the ``casefold`` buffer again - that is the point:
    it shows whether counting pushes the peak past the fold or not.
    """
    tracemalloc.start()
    try:
        doc = next(iter_documents(path.parent))
        assert doc.path == path, (doc.path, path)
        text = doc.text
        _, read_peak = tracemalloc.get_traced_memory()
        widest = ord(max(text))
        # CPython's compact representation: 1 byte for latin-1, 2 for the BMP,
        # 4 when the text needs astral code points.
        bytes_per_char = 1 if widest <= 0xFF else 4 if widest > 0xFFFF else 2

        tracemalloc.reset_peak()
        folded = unicodedata.normalize("NFC", text).casefold()
        folded_chars = len(folded)
        _, fold_peak = tracemalloc.get_traced_memory()
        del folded

        tracemalloc.reset_peak()
        counts: Counter[str] = Counter(tokenize(text))
        _, count_peak = tracemalloc.get_traced_memory()

        # The same fold over a pure-ASCII string of the same length: isolates
        # the price of the non-ASCII characters (typographic quotes, dashes)
        # the books are full of. The probe string itself is built before the
        # reset, so the peak is the fold, not the allocation of "a" * n.
        probe = "a" * len(text)
        tracemalloc.reset_peak()
        probe.casefold()
        _, ascii_fold_peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return {
        "chars": len(text),
        "bytes_per_char": bytes_per_char,
        "read_peak_bytes": read_peak,
        "casefold_peak_bytes": fold_peak,
        "ascii_casefold_peak_bytes": ascii_fold_peak,
        "folded_chars": folded_chars,
        "count_peak_bytes": count_peak,
        "terms": len(counts),
        "tokens": counts.total(),
    }


def summarise(measurements: list[Any], runs: int) -> dict[str, Any]:
    """Median elapsed and peak of ``runs`` measurements, plus the whole spread."""
    elapsed = sorted(m.elapsed for m in measurements)
    peaks = sorted(m.peak_bytes for m in measurements)
    middle = runs // 2
    if runs % 2:
        median_elapsed = elapsed[middle]
        median_peak = peaks[middle]
    else:
        median_elapsed = sum(elapsed[middle - 1 : middle + 1]) / 2
        median_peak = sum(peaks[middle - 1 : middle + 1]) / 2
    return {
        "result": measurements[0].result,
        "elapsed_s": median_elapsed,
        "peak_bytes": median_peak,
        "elapsed_all": [m.elapsed for m in measurements],
        "peak_all": [m.peak_bytes for m in measurements],
        "runs": runs,
    }


def warm_cache(paths: list[Path]) -> None:
    """Read every document once so all pipelines meet the same file cache.

    Without this the first pipeline measured pays for reading 75 MB from disk
    and the second one does not: on this machine that was 56 s against 46 s for
    the same work, which looks like a result and is not one.
    """
    for path in paths:
        path.read_bytes()


def paired_runs(
    pipelines: dict[str, Callable[[], Stats]], runs: int
) -> dict[str, dict[str, Any]]:
    """Measure every pipeline ``runs`` times, rotating the order each round.

    Rotation matters as much as the warm-up: a machine that heats up, throttles
    or gets busy with a Windows Defender scan drifts during the five minutes a
    full corpus takes, and always measuring lazy first would charge that drift
    to eager (or the other way round).
    """
    collected: dict[str, list[Any]] = {name: [] for name in pipelines}
    names = list(pipelines)
    for round_no in range(runs):
        order = names if round_no % 2 == 0 else list(reversed(names))
        for name in order:
            collected[name].append(measure(pipelines[name]))
    return {name: summarise(ms, runs) for name, ms in collected.items()}


def top_terms(stats: Stats) -> list[list[Any]]:
    """Top terms as plain lists, so the JSON stays portable."""
    return [[term, count] for term, count in stats.top]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bench/lab1.py", description=__doc__)
    parser.add_argument("root", type=Path, nargs="?", default=ROOT / "data")
    parser.add_argument("--limit", type=int, default=None, help="first N documents")
    parser.add_argument(
        "--out", type=Path, default=ROOT / "bench" / "lab1_results.json"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="repetitions of each traced run; the table reports the median",
    )
    parser.add_argument(
        "--skip-eager",
        action="store_true",
        help="lazy only (the eager run needs ~1 GB of RAM on the full corpus)",
    )
    args = parser.parse_args(argv)

    root = args.root.resolve()
    paths = corpus_files(root)
    if args.limit is not None:
        paths = paths[: args.limit]
    documents = len(paths)
    corpus_bytes = sum(p.stat().st_size for p in paths)
    biggest = largest(paths)

    print(f"corpus: {documents} documents, {corpus_bytes / 1e6:.1f} MB")
    print(f"largest: {biggest.name} ({biggest.stat().st_size / 1e6:.1f} MB)")

    pipelines: dict[str, Callable[[], Stats]] = {
        "lazy": lambda: lazy_stats(root, args.limit)
    }
    if not args.skip_eager:
        pipelines["eager"] = lambda: eager_stats(root, args.limit)

    print("warm-up: reading the corpus once ...", flush=True)
    warm_cache(paths)

    print(f"measuring {args.runs} rounds ...", flush=True)
    measured = paired_runs(pipelines, args.runs)
    lazy = measured["lazy"]
    eager = measured.get("eager")
    for name, row in measured.items():
        print(
            f"  {name}: median {row['elapsed_s']:.2f} s, "
            f"peak {row['peak_bytes'] / 2**20:.1f} MiB, "
            f"runs {[round(s, 1) for s in row['elapsed_all']]}"
        )
    if eager is not None:
        assert eager["result"] == lazy["result"], "eager and lazy results differ!"

    print(f"single document ({biggest.name}) ...", flush=True)
    single, single_elapsed, single_peak, single_bytes = single_document_stats(biggest)
    print(f"  {single_elapsed:.2f} s, peak {single_peak / 2**20:.1f} MiB")
    stages = stage_peaks(TMP / biggest.name)
    print(
        f"  stages: read {stages['read_peak_bytes'] / 2**20:.1f}, "
        f"casefold {stages['casefold_peak_bytes'] / 2**20:.1f}, "
        f"count {stages['count_peak_bytes'] / 2**20:.1f} MiB"
    )

    results: dict[str, Any] = {
        "machine": {
            "system": platform.system(),
            "release": platform.release(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "python": platform.python_version(),
        },
        "corpus": {
            "root": str(root.relative_to(ROOT))
            if root.is_relative_to(ROOT)
            else str(root),
            "documents": documents,
            "bytes": corpus_bytes,
            "largest": {"name": biggest.name, "bytes": single_bytes},
        },
        "lazy": {
            "documents": lazy["result"].documents,
            "tokens": lazy["result"].tokens,
            "vocabulary": lazy["result"].vocabulary,
            "elapsed_s": lazy["elapsed_s"],
            "peak_bytes": lazy["peak_bytes"],
            "runs": lazy["runs"],
            "elapsed_all": lazy["elapsed_all"],
            "peak_all": lazy["peak_all"],
        },
        "eager": None
        if eager is None
        else {
            "documents": eager["result"].documents,
            "tokens": eager["result"].tokens,
            "vocabulary": eager["result"].vocabulary,
            "elapsed_s": eager["elapsed_s"],
            "peak_bytes": eager["peak_bytes"],
            "runs": eager["runs"],
            "elapsed_all": eager["elapsed_all"],
            "peak_all": eager["peak_all"],
        },
        "single": {
            "name": biggest.name,
            "bytes": single_bytes,
            "tokens": single.tokens,
            "vocabulary": single.vocabulary,
            "elapsed_s": single_elapsed,
            "peak_bytes": single_peak,
            "documents": single.documents,
            "stages": stages,
        },
        "ratio": None
        if eager is None
        else eager["peak_bytes"] / max(lazy["peak_bytes"], 1),
        "top": top_terms(lazy["result"]),
        "method": {
            "measure": "tracemalloc peak around the whole run, perf_counter wall clock",
            "lazy": "findex.stats.lazy_stats",
            "eager": "findex.eager.eager_stats",
            "single": "lazy_stats over a one-file corpus copied to bench/_tmp/lab1",
            "runs": args.runs,
            "median": "median of --runs traced runs; the peak is identical across runs",
            "tokens_per_second": round(
                lazy["result"].tokens / max(lazy["elapsed_s"], 1e-9), 1
            ),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

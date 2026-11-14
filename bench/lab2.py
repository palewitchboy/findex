"""Lab 2 measurements: the memory study, serialization costs, merge vs set.

Run:  uv run python bench/lab2.py [corpus_dir] [--limit N]

Writes ``bench/lab2_results.json`` and prints the three README tables, so the
numbers in the report are exactly the numbers this script produced.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path

from findex.index import Index, build_index
from findex.search import search
from findex.stats import fmt_bytes
from findex.store import load, save_json, save_pickle

HERE = Path(__file__).resolve().parent
TMP = HERE / "_tmp"


@dataclass(frozen=True)
class PlainPosting:
    """A dataclass *without* slots - the 'before' row of the memory study."""

    doc_id: int
    tf: int


def chrono(fn):
    """Time a call without tracemalloc (it would slow the I/O path by 10x+)."""
    start = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - start, 0


def timed(fn):
    """Return (result, seconds, peak_bytes) with tracemalloc around the call."""
    tracemalloc.start()
    start = time.perf_counter()
    try:
        result = fn()
        elapsed = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, elapsed, peak


def file_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def build_variant(
    root: Path, variant: str, limit: int | None
) -> tuple[dict[str, object], Index]:
    kwargs: dict[str, object] = {"limit": limit}
    if variant == "plain":
        kwargs |= {"storage": "objects", "posting_type": PlainPosting}
    elif variant == "slots":
        kwargs |= {"storage": "objects"}
    else:
        kwargs |= {"storage": "arrays"}

    index, build_s, peak = timed(lambda: build_index(root, **kwargs))  # type: ignore[arg-type]
    assert isinstance(index, Index)

    TMP.mkdir(exist_ok=True)
    path = TMP / f"{variant}.pkl"
    save_s = chrono(lambda: save_pickle(index, path))[1]
    size_mb = file_mb(path)
    _, load_s, _ = chrono(lambda: load(path))

    return {
        "variant": variant,
        "terms": len(index.postings),
        "postings": index.n_postings(),
        "docs": len(index.doc_meta),
        "build_s": round(build_s, 2),
        "peak_mb": round(peak / (1024 * 1024), 1),
        "file_mb": round(size_mb, 1),
        "save_s": round(save_s, 2),
        "load_s": round(load_s, 2),
    }, index


def format_study(index: Index) -> list[dict[str, object]]:
    rows = []
    TMP.mkdir(exist_ok=True)
    for fmt, saver, loader in (
        ("pickle", save_pickle, load),
        ("json", save_json, load),
    ):
        path = TMP / f"formats.{'pkl' if fmt == 'pickle' else 'json'}"
        save_s = chrono(lambda saver=saver, path=path: saver(index, path))[1]
        size_mb = file_mb(path)
        _, load_s, _ = chrono(lambda loader=loader, path=path: loader(path))
        rows.append(
            {
                "format": fmt,
                "file_mb": round(size_mb, 1),
                "save_s": round(save_s, 2),
                "load_s": round(load_s, 2),
            }
        )
    return rows


def pick_terms(index: Index) -> tuple[list[str], list[str]]:
    by_size = sorted(index.postings, key=lambda t: len(index.postings[t]), reverse=True)
    common = by_size[:2]
    rare = [t for t in index.terms() if index.doc_freq(t) == 1][:2]
    return common, rare


def engine_bench(
    index: Index, terms: list[str], reps: int = 15
) -> list[dict[str, object]]:
    query = f"{terms[0]} {terms[1]}"
    rows = []
    for engine in ("merge", "set"):
        samples = []
        for _ in range(reps):
            start = time.perf_counter()
            hits = search(index, query, engine=engine)  # type: ignore[arg-type]
            samples.append(time.perf_counter() - start)
        rows.append(
            {
                "engine": engine,
                "query": query,
                "hits": len(hits),
                "median_ms": round(statistics.median(samples) * 1000, 2),
                "best_ms": round(min(samples) * 1000, 2),
                "df": [index.doc_freq(t) for t in terms],
            }
        )
    return rows


def positions_cost(root: Path, limit: int | None) -> dict[str, object]:
    plain, _, plain_peak = timed(lambda: build_index(root, limit=limit))
    with_pos, _, pos_peak = timed(
        lambda: build_index(root, limit=limit, positions=True)
    )
    TMP.mkdir(exist_ok=True)
    p1, p2 = TMP / "pos_off.pkl", TMP / "pos_on.pkl"
    save_pickle(plain, p1)
    save_pickle(with_pos, p2)
    return {
        "limit": limit,
        "peak_off_mb": round(plain_peak / (1024 * 1024), 1),
        "peak_on_mb": round(pos_peak / (1024 * 1024), 1),
        "file_off_mb": round(file_mb(p1), 1),
        "file_on_mb": round(file_mb(p2), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path("data"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--reps", type=int, default=15)
    parser.add_argument(
        "--skip-plain", action="store_true", help="skip the slowest variant"
    )
    args = parser.parse_args()

    variants = ["slots", "arrays"] if args.skip_plain else ["plain", "slots", "arrays"]
    print(f"corpus: {args.root} (limit={args.limit})\n")

    memory_rows = []
    slots_index: Index | None = None
    for variant in variants:
        row, index = build_variant(args.root, variant, args.limit)
        memory_rows.append(row)
        if variant == "slots":
            slots_index = index
        print(
            f"{variant:<7} peak {row['peak_mb']:>7} MB | file {row['file_mb']:>6} MB | "
            f"build {row['build_s']:>6} s | load {row['load_s']:>5} s | "
            f"{row['postings']:,} postings"
        )

    formats = format_study(slots_index)
    print()
    for row in formats:
        print(
            f"{row['format']:<7} file {row['file_mb']:>6} MB | "
            f"save {row['save_s']:>6} s | load {row['load_s']:>5} s"
        )

    common, rare = pick_terms(index)
    mer = engine_bench(index, common, args.reps) + engine_bench(index, rare, args.reps)
    print()
    for row in mer:
        print(
            f"{row['engine']:<6} {row['query']!r:<28} df={row['df']} "
            f"hits={row['hits']:<6} median {row['median_ms']:>7} ms"
        )

    pos = positions_cost(args.root, args.limit if args.limit else 20)
    print()
    print(
        f"positions: peak {pos['peak_off_mb']} -> {pos['peak_on_mb']} MB, "
        f"file {pos['file_off_mb']} -> {pos['file_on_mb']} MB"
    )

    results = {
        "corpus": str(args.root),
        "limit": args.limit,
        "docs": len(index.doc_meta),
        "tokens": index.n_tokens,
        "terms": len(index.postings),
        "postings": index.n_postings(),
        "memory_study": memory_rows,
        "formats": formats,
        "engines": mer,
        "positions": pos,
    }
    out = HERE / "lab2_results.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n== markdown ==")
    print("| Postings representation | Peak memory (build) | File size | Load time |")
    print("|---|---|---|---|")
    label = {
        "plain": "`list[Posting]`, plain `@dataclass`",
        "slots": "`list[Posting]`, `slots=True`",
        "arrays": "`array('I')` pairs, no objects",
    }
    for row in memory_rows:
        print(
            f"| {label[row['variant']]} | {row['peak_mb']} MB | {row['file_mb']} MB | "
            f"{row['load_s']} s |"
        )
    print(f"\nsaved -> {out} ({fmt_bytes(out.stat().st_size)})")


if __name__ == "__main__":
    main()

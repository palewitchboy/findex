"""M3 - the lazy stats pipeline.

    python -m findex.stats data/ [--limit N] [--top 50]

documents -> tokens -> Counter, in one pass. The Counter is the only structure
that grows (it holds the answer); everything upstream is a stream.
"""

from __future__ import annotations

import argparse
import logging
import time
import tracemalloc
from collections import Counter
from collections.abc import Callable
from itertools import islice
from pathlib import Path
from typing import NamedTuple

from findex.corpus import iter_documents
from findex.tokenize import tokenize


class Stats(NamedTuple):
    documents: int
    tokens: int
    vocabulary: int
    top: list[tuple[str, int]]


class Measurement(NamedTuple):
    result: Stats
    elapsed: float  # seconds
    peak_bytes: int


def lazy_stats(root: Path, limit: int | None = None, top: int = 50) -> Stats:
    """Single pass, generators all the way down."""
    docs = iter_documents(root)
    if limit is not None:
        docs = islice(docs, limit)

    counts: Counter[str] = Counter()
    n_docs = 0
    for doc in docs:
        n_docs += 1
        counts.update(tokenize(doc.text))  # consumes the generator, one token at a time
    return Stats(n_docs, counts.total(), len(counts), counts.most_common(top))


def measure(fn: Callable[[], Stats]) -> Measurement:
    """Run ``fn`` and report wall-clock time and tracemalloc peak memory."""
    tracemalloc.start()
    start = time.perf_counter()
    try:
        result = fn()
        elapsed = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return Measurement(result, elapsed, peak)


def fmt_bytes(n: int) -> str:
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def print_report(m: Measurement, title: str = "lazy (generators)") -> None:
    s = m.result
    print(f"== {title} ==")
    print(f"documents  : {s.documents}")
    print(f"tokens     : {s.tokens}")
    print(f"vocabulary : {s.vocabulary}")
    print(f"elapsed    : {m.elapsed:.2f} s")
    print(f"peak memory: {fmt_bytes(m.peak_bytes)}")
    print(f"\ntop {len(s.top)} terms:")
    for rank, (term, count) in enumerate(s.top, start=1):
        print(f"{rank:>3}. {term:<20} {count}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m findex.stats", description=__doc__)
    parser.add_argument(
        "root", type=Path, help="directory of .txt files or a .jsonl file"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="only the first N documents"
    )
    parser.add_argument(
        "--top", type=int, default=50, help="how many top terms to print"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    m = measure(lambda: lazy_stats(args.root, args.limit, args.top))
    print_report(m)


if __name__ == "__main__":
    main()

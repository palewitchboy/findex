"""M3 - the lazy stats pipeline.

    python -m findex.stats data/ [--limit N] [--top 50]

documents -> tokens -> Counter, in one pass. The Counter is the only structure
that grows (it holds the answer); everything upstream is a stream.
"""

from __future__ import annotations

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


#: ``python -m findex.stats`` stays a thin shim over the typer app; the CLI
#: owns printing, logging and exit codes (Lab 4).
if __name__ == "__main__":  # pragma: no cover - see findex.cli
    from findex.cli import stats_main

    stats_main()

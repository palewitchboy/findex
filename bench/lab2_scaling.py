"""Where does merge actually win? Scaling experiment for the Lab 2 README.

The corpus has 120 documents, so postings lists are at most 120 ids long and
both engines finish in microseconds. This script scales the lists up to a
million ids to see the asymptotic behaviour - and it produces a result that
contradicts the textbook: CPython's ``set`` intersection is C code, so it beats
a Python-level two-pointer loop on *time*, while the merge wins on *memory*
because it allocates nothing.

Run:  uv run python bench/lab2_scaling.py   (writes bench/_scaling.json)
"""

from __future__ import annotations

import json
import statistics
import time
import tracemalloc
from pathlib import Path

from findex.search import intersect

HERE = Path(__file__).resolve().parent
SIZES = (1_000, 10_000, 100_000, 1_000_000)
REPS = 5


def make_lists(n: int) -> tuple[list[int], list[int]]:
    """Two ascending id lists that overlap by ~2/3."""
    a = list(range(0, n))
    b = list(range(n // 3, n + n // 3))
    return a, b


def timed(fn, reps: int = REPS) -> tuple[float, int]:
    samples = []
    tracemalloc.start()
    try:
        for _ in range(reps):
            start = time.perf_counter()
            fn()
            samples.append(time.perf_counter() - start)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return statistics.median(samples) * 1000, peak


def main() -> None:
    rows = []
    header = f"{'n':>9} {'merge, ms':>10} {'set, ms':>9}"
    print(header + f" {'merge mem, MB':>14} {'set mem, MB':>12}")
    for n in SIZES:
        a, b = make_lists(n)
        m_ms, m_peak = timed(lambda a=a, b=b: list(intersect(a, b)))
        s_ms, s_peak = timed(lambda a=a, b=b: set(a) & set(b))
        rows.append(
            {
                "n": n,
                "merge_ms": round(m_ms, 2),
                "set_ms": round(s_ms, 2),
                "merge_mb": round(m_peak / 1024 / 1024, 1),
                "set_mb": round(s_peak / 1024 / 1024, 1),
            }
        )
        print(
            f"{n:>9,} {m_ms:>10.2f} {s_ms:>9.2f} "
            f"{m_peak / 1024 / 1024:>14.1f} {s_peak / 1024 / 1024:>12.1f}"
        )

    out = HERE / "_scaling.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()

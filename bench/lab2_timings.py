"""Re-measure the timing columns of the Lab 2 tables, in isolation.

The first benchmark run overlapped with other builds, so its save/load and query
timings were noisy while the memory peaks were stable. This script re-measures
only the timings (median of several runs, nothing else running) and writes them
back into bench/lab2_results.json, leaving the measured peaks untouched.

Run:  uv run python bench/lab2_timings.py
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

from findex.search import search
from findex.store import load, save_json, save_pickle

# plain.pkl was written while bench/lab2.py ran as __main__, so pickle recorded
# the class as ``__main__.PlainPosting``. Register the same name here, or the
# file is unloadable - a small demonstration of why pickle ties data to code.
from lab2 import PlainPosting  # noqa: E402

sys.modules["__main__"].PlainPosting = PlainPosting

HERE = Path(__file__).resolve().parent
TMP = HERE / "_tmp"
REPS_LOAD = 3
REPS_SAVE = 2
REPS_QUERY = 15


def timeit(fn, reps: int) -> float:
    samples = []
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def main() -> None:
    data = json.loads((HERE / "lab2_results.json").read_text(encoding="utf-8"))

    print(f"load times (median of {REPS_LOAD}):")
    for row in data["memory_study"]:
        path = TMP / f"{row['variant']}.pkl"
        row["load_s"] = round(timeit(lambda p=path: load(p), REPS_LOAD), 2)
        print(f"  {row['variant']:<7} {row['load_s']:>6} s")

    slots = load(TMP / "slots.pkl")
    for row in data["formats"]:
        path = TMP / ("formats.pkl" if row["format"] == "pickle" else "formats.json")
        saver = save_pickle if row["format"] == "pickle" else save_json
        row["save_s"] = round(timeit(lambda s=saver, p=path: s(slots, p), REPS_SAVE), 2)
        row["load_s"] = round(timeit(lambda p=path: load(p), REPS_LOAD), 2)
        print(
            f"  {row['format']:<7} save {row['save_s']:>6} s  load {row['load_s']:>6} s"
        )

    print(f"queries (median of {REPS_QUERY}):")
    for row in data["engines"]:
        row["hits"] = len(search(slots, row["query"], engine=row["engine"]))
        ms = timeit(
            lambda r=row: search(slots, r["query"], engine=r["engine"]),
            REPS_QUERY,
        )
        row["median_ms"] = round(ms * 1000, 3)
        print(
            f"  {row['engine']:<6} {row['query']!r:<24} "
            f"{row['median_ms']:>7} ms  {row['hits']} hits"
        )

    (HERE / "lab2_results.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nupdated bench/lab2_results.json")


if __name__ == "__main__":
    main()

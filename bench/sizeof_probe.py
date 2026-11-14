"""What does one posting cost? Measured two ways.

1. ``sys.getsizeof`` - the object header, and the *materialised* ``__dict__``.
2. ``tracemalloc`` over one million instances - the number that matters: what
   the process actually pays per posting, allocator overhead included.

The second one is the honest one, and on Python 3.13 it disagrees with the
classic "slots saves 3x" claim, because 3.11+ keeps instance attributes in a
compact managed dict instead of a full hash table.

Run:  uv run python bench/sizeof_probe.py   (writes bench/_sizeof.json)
"""

from __future__ import annotations

import json
import sys
import tracemalloc
from array import array
from dataclasses import dataclass
from pathlib import Path

from findex.index import Posting

HERE = Path(__file__).resolve().parent
N = 1_000_000


@dataclass(frozen=True)
class PlainPosting:
    doc_id: int
    tf: int


def per_object(cls: type, n: int = N) -> float:
    """Bytes per instance, allocator overhead included, via tracemalloc."""
    tracemalloc.start()
    try:
        rows = [cls(i, 1) for i in range(n)]
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    del rows
    return peak / n


def main() -> None:
    data = {
        "n": N,
        "plain_b_per_object": round(per_object(PlainPosting), 1),
        "slots_b_per_object": round(per_object(Posting), 1),
        "array_pair_bytes": sys.getsizeof(array("I", [1, 2]))
        - sys.getsizeof(array("I")),
        "plain_header": sys.getsizeof(PlainPosting(1, 2)),
        "plain_dict": sys.getsizeof(PlainPosting(1, 2).__dict__),
        "slots_header": sys.getsizeof(Posting(1, 2)),
        "list_pointer": sys.getsizeof(0),
    }
    out = HERE / "_sizeof.json"
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    for key, value in data.items():
        print(f"{key:<22} {value}")
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()

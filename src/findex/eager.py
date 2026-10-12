"""M4 - the deliberately EAGER version, and the eager-vs-lazy comparison.

    python -m findex.eager data/ [--limit N]

The eager version does exactly what the lab warns about: read every document
into a list, tokenize every document into a list of lists, then count.
Never use it for real work - it exists only to be measured against the lazy one.
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path

from findex.corpus import iter_documents
from findex.stats import Measurement, Stats, fmt_bytes, lazy_stats, measure
from findex.tokenize import tokenize


def eager_stats(root: Path, limit: int | None = None, top: int = 50) -> Stats:
    documents = list(iter_documents(root))  # 1. ALL texts in RAM
    if limit is not None:
        documents = documents[:limit]

    token_lists = [list(tokenize(d.text)) for d in documents]  # 2. ALL tokens in RAM

    counts: Counter[str] = Counter()
    for tokens in token_lists:  # 3. only now count
        counts.update(tokens)
    return Stats(len(documents), counts.total(), len(counts), counts.most_common(top))


def markdown_table(eager: Measurement, lazy: Measurement) -> str:
    rows = [
        ("eager (lists)", eager),
        ("lazy (generators)", lazy),
    ]
    lines = [
        "| Version | Documents | Peak memory | Elapsed |",
        "|---|---|---|---|",
    ]
    for name, m in rows:
        lines.append(
            f"| {name} | {m.result.documents} | {fmt_bytes(m.peak_bytes)} "
            f"| {m.elapsed:.2f} s |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m findex.eager", description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    eager = measure(lambda: eager_stats(args.root, args.limit))
    lazy = measure(lambda: lazy_stats(args.root, args.limit))

    # Sanity check: both versions must give the same answer.
    assert eager.result == lazy.result, "eager and lazy results differ!"

    print(markdown_table(eager, lazy))
    ratio = eager.peak_bytes / max(lazy.peak_bytes, 1)
    print(f"\neager used {ratio:.1f}x more peak memory than lazy.")


if __name__ == "__main__":
    main()

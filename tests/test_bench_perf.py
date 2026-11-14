"""Lab 8 - benchmarks as tests: ``pytest-benchmark`` keeps the hot path honest.

Three measurements, each a real code path rather than a micro-loop:

* ``search`` with snippets - the number a user feels (Lab 7 measured it at
  540 ms on 48 books before Lab 8 touched it);
* ``search`` without snippets - the ranking alone, so a regression can be
  attributed to scoring or to snippets;
* ``build_index`` - the number that decides how long a container takes to
  start after a corpus change.

The corpus here is six tiny files on purpose: this is a **regression gate for
CI**, not the corpus study (``bench/lab8.py`` is that). It runs in a second or
so, which is what makes it survive in a pipeline.

Run them explicitly:

    pytest tests/test_bench_perf.py --benchmark-only
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pytest_benchmark", reason="pytest-benchmark is a dev dependency")

from findex.index import Index, build_index  # noqa: E402
from findex.ranking import Searcher  # noqa: E402

pytestmark = [pytest.mark.benchmark, pytest.mark.slow]

DOCUMENTS = 6
BODY = " ".join(f"word{i}" for i in range(400))
QUERY = "fox and dog"


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Six books of ~1 200 tokens each: long enough that scoring is visible."""
    root = tmp_path_factory.mktemp("bench")
    for number in range(DOCUMENTS):
        text = (
            f"Title: Book {number}\n\n"
            f"The quick brown fox jumps over the lazy dog. {BODY} "
            f"A fox is quick and a dog is lazy, and the river runs past. {BODY}\n"
        )
        (root / f"book{number}.txt").write_text(text, encoding="utf-8")
    return root


@pytest.fixture(scope="module")
def index(corpus: Path) -> Index:
    """The same layout the server ships: ``arrays`` postings, no positions."""
    return build_index(corpus, positions=False, storage="arrays")


def test_bench_search_with_snippets(benchmark: object, index: Index) -> None:
    searcher = Searcher(index, snippets=True)
    hits = benchmark(searcher.search, QUERY, k=DOCUMENTS)  # type: ignore[operator]
    assert len(hits) == DOCUMENTS
    assert all(hit.snippet for hit in hits)


def test_bench_search_without_snippets(benchmark: object, index: Index) -> None:
    searcher = Searcher(index, snippets=False)
    hits = benchmark(searcher.search, QUERY, k=DOCUMENTS)  # type: ignore[operator]
    assert len(hits) == DOCUMENTS


def test_bench_build_index(benchmark: object, corpus: Path) -> None:
    built = benchmark(build_index, corpus, positions=False, storage="arrays")  # type: ignore[operator]
    assert built.num_docs == DOCUMENTS
    assert len(built) > 0

"""Lab 8 - the NumPy scorer must rank exactly like the Python one.

The contract has two halves and this file tests both:

* **the ranking is identical** - same doc ids, same order, tie-breaks included;
* **the scores agree to 1e-9** - ``math.log`` and ``np.log`` may differ in the
  last bit, and the accumulation order is the same, so 1e-9 is generous and
  still catches a wrong formula.

Anything that would let the two drift - a different idf, a swapped length
normalisation, a tie broken the other way - shows up as a doc id out of place.
"""

from __future__ import annotations

from array import array
from pathlib import Path

import numpy as np
import pytest

from findex.index import ArrayPostings, Index, ObjectPostings, Posting, build_index
from findex.ranking import BM25, Searcher, TfIdf
from findex.vector import TermArrays, VectorIndex, term_arrays, top_k, top_k_of

QUERIES = (
    "fox",
    "dog",
    "quick",
    "rabbit",
    "fox AND dog",
    "fox OR rabbit",
    "quick AND fox AND dog",
    "nothing",
    "zzz",
    "the",
    "a",
)


def _results(searcher: Searcher, query: str, k: int) -> list[tuple[int, float]]:
    return [(hit.doc_id, hit.score) for hit in searcher.search(query, k=k)]


@pytest.mark.parametrize("scorer", [BM25(), TfIdf(), BM25(k1=0.0), TfIdf(log_tf=False)])
@pytest.mark.parametrize("k", [1, 2, 3, 10, 100])
def test_ranking_is_identical(index: Index, scorer: BM25 | TfIdf, k: int) -> None:
    reference = Searcher(index, scorer, snippets=False, engine="python")
    vectorised = Searcher(index, scorer, snippets=False, engine="numpy")
    for query in QUERIES:
        old = _results(reference, query, k)
        new = _results(vectorised, query, k)
        assert [doc_id for doc_id, _ in new] == [doc_id for doc_id, _ in old], query
        for (_, left), (_, right) in zip(old, new, strict=True):
            assert abs(left - right) <= 1e-9


def test_scores_match_on_a_full_ranking(index: Index) -> None:
    """Every matched document, not just the top-k: the formula, not the cut."""
    reference = Searcher(index, BM25(), snippets=False, engine="python")
    vectorised = Searcher(index, BM25(), snippets=False, engine="numpy")
    for query in ("fox", "dog", "rabbit", "quick"):
        assert _results(reference, query, 100) == pytest.approx(
            _results(vectorised, query, 100), abs=1e-9
        )


def test_custom_scorer_falls_back_to_the_python_loop(index: Index) -> None:
    """``Scorer`` is a protocol; a closure cannot be vectorised, and must not be."""

    def constant(term: str, doc_id: int, tf: int, source: Index) -> float:
        return float(tf)

    searcher = Searcher(index, constant, snippets=False, engine="numpy")
    hits = searcher.search("fox", k=3)
    assert [hit.doc_id for hit in hits] == [
        hit.doc_id
        for hit in Searcher(index, constant, snippets=False, engine="python").search(
            "fox", k=3
        )
    ]


def test_unknown_engine_is_a_usage_error(index: Index) -> None:
    with pytest.raises(ValueError, match="unknown engine"):
        Searcher(index, engine="numba")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- top_k
def test_top_k_breaks_ties_on_the_lowest_doc_id() -> None:
    doc_ids = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    scores = np.array([5.0, 5.0, 5.0, 1.0, 9.0])
    assert top_k(doc_ids, scores, 3) == [(4, 9.0), (0, 5.0), (1, 5.0)]


def test_top_k_matches_heapq_over_many_random_ties() -> None:
    """The tie-break is the whole risk of argpartition; fuzz it."""
    import heapq
    import random

    rng = random.Random(20261112)
    for _ in range(200):
        size = rng.randint(0, 40)
        doc_ids = np.arange(size, dtype=np.int32)
        # Few distinct values -> plenty of ties at the cut.
        scores = np.array([float(rng.randint(0, 5)) for _ in range(size)])
        k = rng.randint(1, 10)
        pairs = list(zip(doc_ids.tolist(), scores.tolist(), strict=True))
        expected = heapq.nlargest(k, pairs, key=lambda pair: (pair[1], -pair[0]))
        assert top_k(doc_ids, scores, k) == [(int(i), float(s)) for i, s in expected]


def test_top_k_edge_cases() -> None:
    doc_ids = np.array([3, 7], dtype=np.int32)
    scores = np.array([1.0, 2.0])
    assert top_k(doc_ids, scores, 0) == []
    assert top_k(doc_ids, scores, 5) == [(7, 2.0), (3, 1.0)]
    assert top_k(np.array([], dtype=np.int32), np.zeros(0), 3) == []


# --------------------------------------------------------------------------- arrays
def test_term_arrays_reads_array_postings_without_copying() -> None:
    flat = array("I", [0, 5, 3, 1])
    arrays = term_arrays(ArrayPostings(flat))
    assert isinstance(arrays, TermArrays)
    assert arrays.doc_ids.tolist() == [0, 3]
    assert arrays.tfs.tolist() == [5, 1]
    # A view shares memory with the buffer: mutating the array is visible.
    flat[0] = 2
    assert arrays.doc_ids.tolist() == [2, 3]


def test_term_arrays_falls_back_for_object_postings() -> None:
    postings = ObjectPostings([Posting(1, 4), Posting(6, 2)])
    arrays = term_arrays(postings)
    assert arrays.doc_ids.tolist() == [1, 6]
    assert arrays.tfs.tolist() == [4, 2]
    assert len(arrays) == 2


def test_vector_index_handles_objects_storage(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("Title: A\n\nalpha beta alpha\n", encoding="utf-8")
    built = build_index(tmp_path, storage="objects")
    vectors = VectorIndex(built)
    arrays = vectors.term("alpha")
    assert arrays is not None
    assert arrays.tfs.tolist() == [2]
    assert vectors.term("missing") is None
    # Same ranking either way, which is what "storage is an implementation
    # detail" meant in Lab 2.
    python = Searcher(built, snippets=False, engine="python").search("alpha", k=1)
    numpy = Searcher(built, snippets=False, engine="numpy").search("alpha", k=1)
    assert [h.doc_id for h in python] == [h.doc_id for h in numpy]


def test_top_k_of_empty_match_set() -> None:
    assert top_k_of([], np.zeros(3), 5) == []


def test_vector_index_is_lazy(index: Index) -> None:
    vectors = VectorIndex(index)
    assert vectors._terms == {}
    vectors.bm25(["fox"])
    assert list(vectors._terms) == ["fox"]

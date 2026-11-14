"""Lab 3 - ranking: TF-IDF, BM25, the Scorer protocol, top-k and the query cache."""

from __future__ import annotations

import logging
import math
from pathlib import Path

import pytest

from findex.index import DocMeta, Index, ObjectPostings, Posting, build_index
from findex.query import parse
from findex.ranking import BM25, Scorer, Searcher, SearchResult, TfIdf, explain, rank


def synthetic(
    postings: dict[str, dict[int, int]], lengths: dict[int, int]
) -> Index:
    """An index from ``{term: {doc_id: tf}}`` plus ``{doc_id: length}``."""
    return Index(
        postings={
            term: ObjectPostings(
                [Posting(doc_id, tf) for doc_id, tf in sorted(per_doc.items())]
            )
            for term, per_doc in postings.items()
        },
        doc_lengths=dict(lengths),
        doc_meta={
            doc_id: DocMeta(doc_id, f"doc{doc_id}.txt", f"Doc {doc_id}", length)
            for doc_id, length in lengths.items()
        },
        n_tokens=sum(lengths.values()),
    )


# --------------------------------------------------------------------------- tf-idf
def test_tfidf_is_sublinear_tf_times_idf() -> None:
    index = synthetic({"a": {0: 3}, "b": {0: 1, 1: 1}}, {0: 10, 1: 10})
    scorer = TfIdf()
    expected = (1 + math.log(3)) * math.log(2 / 1)
    assert scorer("a", 0, 3, index) == pytest.approx(expected)
    # df == N -> idf == log(1) == 0: a term in every document weighs nothing.
    assert scorer("b", 0, 1, index) == 0.0


def test_tfidf_can_use_raw_term_frequency() -> None:
    index = synthetic({"a": {0: 4}, "b": {1: 1}}, {0: 10, 1: 10})
    raw = TfIdf(log_tf=False)("a", 0, 4, index)
    logged = TfIdf()("a", 0, 4, index)
    assert raw == pytest.approx(4 * math.log(2))
    assert logged == pytest.approx((1 + math.log(4)) * math.log(2))
    assert raw > logged  # log compresses the four occurrences


def test_unknown_terms_score_zero() -> None:
    index = synthetic({"a": {0: 1}}, {0: 10})
    assert TfIdf()("zzz", 0, 1, index) == 0.0
    assert BM25()("zzz", 0, 1, index) == 0.0
    assert TfIdf()("a", 0, 0, index) == 0.0


# --------------------------------------------------------------------------- bm25
@pytest.mark.parametrize(("k1", "b"), [(-0.1, 0.75), (1.5, -0.1), (1.5, 1.1)])
def test_bm25_validates_its_parameters(k1: float, b: float) -> None:
    with pytest.raises(ValueError):
        BM25(k1=k1, b=b)


def test_bm25_defaults_are_the_textbook_ones() -> None:
    scorer = BM25()
    assert (scorer.k1, scorer.b) == (1.5, 0.75)


def test_bm25_b_zero_ignores_document_length() -> None:
    index = synthetic({"t": {0: 1, 1: 1}}, {0: 10, 1: 2_000})
    scorer = BM25(b=0.0)
    assert scorer("t", 0, 1, index) == pytest.approx(scorer("t", 1, 1, index))


def test_bm25_b_one_prefers_the_short_document() -> None:
    index = synthetic({"t": {0: 1, 1: 1}}, {0: 10, 1: 2_000})
    scorer = BM25(b=1.0)
    short, long = scorer("t", 0, 1, index), scorer("t", 1, 1, index)
    assert short > long
    assert short / long > 2.0


def test_bm25_saturates_on_repetition() -> None:
    index = synthetic({"t": {0: 1}}, {0: 100})
    scorer = BM25()
    first = scorer("t", 0, 1, index)
    nineteenth = scorer("t", 0, 19, index)
    twentieth = scorer("t", 0, 20, index)
    assert first > 0
    assert twentieth - nineteenth < 0.05 * first  # the 20th adds almost nothing


def test_bm25_nineteen_beats_one_by_less_than_two_times() -> None:
    index = synthetic({"t": {0: 1}}, {0: 100})
    scorer = BM25()
    assert scorer("t", 0, 19, index) < 2.4 * scorer("t", 0, 1, index)


# --------------------------------------------------------------------------- protocol
class LengthScorer:
    """A fake scorer: five lines, no base class, accepted by Searcher."""

    def __call__(self, term: str, doc_id: int, tf: int, index: Index) -> float:
        return float(index.doc_length(doc_id))


def test_scorer_is_structural() -> None:
    assert isinstance(BM25(), Scorer)
    assert isinstance(TfIdf(), Scorer)
    assert isinstance(LengthScorer(), Scorer)
    assert not isinstance("not a scorer", Scorer)


def test_a_custom_scorer_plugs_into_rank() -> None:
    index = synthetic({"t": {0: 1, 1: 1, 2: 1}}, {0: 30, 1: 10, 2: 20})
    hits = rank(index, parse("t"), LengthScorer(), k=3)
    assert [hit.doc_id for hit in hits] == [0, 2, 1]


# --------------------------------------------------------------------------- top-k
def test_rank_orders_by_score_descending() -> None:
    index = synthetic({"t": {0: 1, 1: 1, 2: 1}}, {0: 10, 1: 10, 2: 10})

    class ByDocId:
        def __call__(self, term: str, doc_id: int, tf: int, index: Index) -> float:
            return float(doc_id)

    hits = rank(index, parse("t"), ByDocId(), k=3)
    assert [hit.doc_id for hit in hits] == [2, 1, 0]


def test_rank_breaks_ties_by_ascending_doc_id() -> None:
    index = synthetic({"t": {0: 1, 1: 1, 2: 1}}, {0: 10, 1: 10, 2: 10})
    hits = rank(index, parse("t"), TfIdf(), k=3)  # df == N, every score is 0.0
    assert [hit.doc_id for hit in hits] == [0, 1, 2]


def test_rank_cuts_to_k() -> None:
    index = synthetic({"t": {0: 1, 1: 1, 2: 1}}, {0: 10, 1: 10, 2: 10})
    assert len(rank(index, parse("t"), TfIdf(), k=2)) == 2
    assert len(rank(index, parse("t"), TfIdf(), k=10)) == 3


def test_rank_scores_each_document_once_per_term() -> None:
    index = synthetic({"a": {0: 2}, "b": {0: 3}, "c": {1: 1}}, {0: 10, 1: 10})
    hits = rank(index, parse("a OR b"), TfIdf(), k=1)
    assert hits[0].score == pytest.approx(
        TfIdf()("a", 0, 2, index) + TfIdf()("b", 0, 3, index)
    )
    assert hits[0].score > 0


def test_rank_returns_titles_from_doc_meta(corpus: Path) -> None:
    index = build_index(corpus)
    hits = rank(index, parse("fox"), BM25(), k=3)
    assert hits[0].title in {"Alpha", "Beta", "Gamma"}
    assert len(hits) == 3


def test_pure_negation_ranks_everything_else() -> None:
    index = synthetic({"a": {0: 1}}, {0: 10, 1: 10, 2: 10})
    hits = rank(index, parse("NOT a"), BM25(), k=5)
    assert [hit.doc_id for hit in hits] == [1, 2]
    assert all(hit.score == 0.0 for hit in hits)


# --------------------------------------------------------------------------- results
def test_sorted_results_put_the_best_first() -> None:
    results = [
        SearchResult(1, 0.5, "b"),
        SearchResult(2, 0.9, "a"),
        SearchResult(0, 0.5, "c"),
    ]
    assert [r.doc_id for r in sorted(results)] == [2, 0, 1]
    # reversed() flips "better than", so the worst result comes first
    assert sorted(results, reverse=True)[0].doc_id == 1
    assert sorted(results, reverse=True)[-1].doc_id == 2


def test_result_str_is_one_line() -> None:
    text = str(SearchResult(7, 1.23456, "A Title"))
    assert "7" in text and "1.235" in text and "A Title" in text


# --------------------------------------------------------------------------- searcher
def test_searcher_returns_ranked_results(index: Index) -> None:
    searcher = Searcher(index, BM25(), snippets=False)
    hits = searcher.search("fox", k=3)
    assert {hit.doc_id for hit in hits} == {0, 1, 2}
    assert [hit.score for hit in hits] == sorted(
        (hit.score for hit in hits), reverse=True
    )


def test_searcher_respects_k(index: Index) -> None:
    assert len(Searcher(index, BM25(), snippets=False).search("fox", k=1)) == 1


def test_searcher_adds_snippets(index: Index) -> None:
    hits = Searcher(index, BM25()).search("fox", k=1)
    assert "[fox]" in hits[0].snippet


def test_searcher_can_skip_snippets(index: Index) -> None:
    assert Searcher(index, BM25(), snippets=False).search("fox", k=1)[0].snippet == ""


def test_repeated_query_is_a_cache_hit(
    index: Index, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger="findex.ranking")
    searcher = Searcher(index, BM25(), snippets=False)
    searcher.search("fox", k=3)
    assert searcher.cache_info().hits == 0
    searcher.search("fox", k=3)
    assert searcher.cache_info().hits == 1
    assert "cache hit" in caplog.text
    assert "'fox'" in caplog.text


def test_cache_size_zero_disables_the_cache(index: Index) -> None:
    searcher = Searcher(index, BM25(), snippets=False, cache_size=0)
    searcher.search("fox")
    searcher.search("fox")
    assert searcher.cache_info().hits == 0
    assert searcher.cache_info().maxsize == 0


def test_one_cache_entry_serves_both_scorers(index: Index) -> None:
    searcher = Searcher(index, BM25(), snippets=False)
    searcher.search("fox")
    searcher.scorer = TfIdf()
    searcher.search("fox")
    assert searcher.cache_info().hits == 1
    assert searcher.cache_info().misses == 1


def test_clear_cache_forgets_everything(index: Index) -> None:
    searcher = Searcher(index, BM25(), snippets=False)
    searcher.search("fox")
    searcher.clear_cache()
    assert searcher.cache_info().currsize == 0


# --------------------------------------------------------------------------- explain
def test_explain_prints_the_tree_and_document_frequencies(index: Index) -> None:
    text = explain(index, parse("fox NOT dog"))
    assert "tree" in text
    assert "(fox AND NOT (dog))" in text
    assert "df=    3" in text
    assert "matched" in text

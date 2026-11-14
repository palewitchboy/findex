"""Lab 3 - ``Index`` as a ``Mapping`` and ``open_index`` as a context manager."""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from findex.index import DocMeta, Index, ObjectPostings, Posting, build_index
from findex.store import load, open_index, save, save_json, save_pickle


# --------------------------------------------------------------------------- mapping
def test_len_counts_distinct_terms(index: Index) -> None:
    assert len(index) == len(index.postings) == len(list(index))
    assert len(index) > 0


def test_contains_answers_the_membership_question(index: Index) -> None:
    assert "fox" in index
    assert "nonexistent" not in index
    assert "Fox" not in index  # terms are casefolded, the mapping is exact


def test_getitem_returns_postings_and_raises_keyerror(index: Index) -> None:
    postings = index["fox"]
    assert list(postings.items()) == [(0, 2), (1, 1), (2, 1)]
    with pytest.raises(KeyError):
        index["nonexistent"]


def test_iteration_yields_every_term(index: Index) -> None:
    assert set(iter(index)) == set(index.postings)
    assert list(dict(index)["dog"].items()) == [(0, 1), (1, 2)]
    assert index.get("fox") is index["fox"]
    assert index.get("nonexistent") is None
    assert set(index.keys()) == set(index.postings)


def test_repr_shows_the_three_counts(index: Index) -> None:
    text = repr(index)
    assert text.startswith("Index(terms=")
    assert f"docs={len(index.doc_meta)}" in text
    assert f"postings={index.n_postings():,}" in text


def test_num_docs_and_average_length(index: Index) -> None:
    assert index.num_docs == len(index.doc_meta) == 3
    expected = sum(index.doc_lengths.values()) / len(index.doc_lengths)
    assert index.avg_doc_length == pytest.approx(expected)


def test_avg_doc_length_is_computed_once_then_cached(index: Index) -> None:
    assert "avg_doc_length" not in index.__dict__
    first = index.avg_doc_length
    assert "avg_doc_length" in index.__dict__  # cached_property writes here
    index.doc_lengths[0] = 10_000  # a cache does not notice mutations
    assert index.avg_doc_length == first


def test_df_and_doc_freq_are_the_same_function(index: Index) -> None:
    assert Index.doc_freq is Index.df
    assert index.df("fox") == 3
    assert index.doc_freq("fox") == 3
    assert index.df("nonexistent") == 0


def test_doc_length_is_strict(index: Index) -> None:
    assert index.doc_length(0) == index.doc_lengths[0]
    with pytest.raises(KeyError):
        index.doc_length(99)


def test_empty_index_is_falsy_and_safe() -> None:
    empty = Index(postings={}, doc_lengths={}, doc_meta={}, n_tokens=0)
    assert not empty
    assert len(empty) == 0
    assert empty.avg_doc_length == 0.0  # no ZeroDivisionError
    assert "terms=0" in repr(empty)


def test_two_indexes_compare_by_content(corpus: Path) -> None:
    left, right = build_index(corpus), build_index(corpus)
    assert left == right
    assert left.postings["fox"] == right.postings["fox"]
    assert ObjectPostings([Posting(0, 1)]) != ObjectPostings([Posting(0, 2)])
    assert ObjectPostings([Posting(0, 1)]) != "not postings"


# --------------------------------------------------------------------------- lifecycle
def test_close_empties_the_index_but_keeps_it_valid(index: Index) -> None:
    _ = index.avg_doc_length  # populate the cached property
    index.close()
    assert index.closed
    assert len(index) == 0
    assert index.doc_meta == {}
    assert index.positions is None
    assert not index
    assert repr(index).startswith("Index(terms=0")


def test_pickle_skips_transient_state(corpus: Path, tmp_path: Path) -> None:
    index = build_index(corpus)
    assert index.avg_doc_length > 0
    path = tmp_path / "cached.pkl"
    save_pickle(index, path)
    restored = load(path)
    assert "avg_doc_length" not in restored.__dict__
    assert "avg_doc_length" not in pickle.loads(path.read_bytes()).__dict__
    assert restored.avg_doc_length == pytest.approx(index.avg_doc_length)


def test_open_index_yields_a_working_index(corpus: Path, tmp_path: Path) -> None:
    path = tmp_path / "index.pkl"
    save_pickle(build_index(corpus), path)
    with open_index(path) as index:
        assert not index.closed
        assert len(index) > 0
        assert "fox" in index


def test_open_index_releases_the_index_after_the_block(
    corpus: Path, tmp_path: Path
) -> None:
    path = tmp_path / "index.pkl"
    save_pickle(build_index(corpus), path)
    with open_index(path) as index:
        assert index.doc_meta
    assert index.closed
    assert len(index) == 0


def test_open_index_closes_even_when_the_body_raises(
    corpus: Path, tmp_path: Path
) -> None:
    path = tmp_path / "index.pkl"
    save_pickle(build_index(corpus), path)
    with pytest.raises(RuntimeError, match="boom"):
        with open_index(path) as index:
            assert not index.closed
            raise RuntimeError("boom")
    assert index.closed, "the finally block must run on the exception path"


def test_open_index_accepts_json(index: Index, tmp_path: Path) -> None:
    path = tmp_path / "index.json"
    save_json(index, path)
    with open_index(path) as restored:
        assert len(restored) == len(index)
        assert restored.df("fox") == 3


def test_open_index_propagates_load_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        with open_index(tmp_path / "missing.pkl"):
            pass


def test_save_and_load_still_sniff_the_format(index: Index, tmp_path: Path) -> None:
    for fmt in ("pickle", "json"):
        path = tmp_path / f"round.{fmt}"
        save(index, path, fmt=fmt)
        assert load(path).doc_meta[0].title == "Alpha"


def test_doc_meta_is_the_same_dataclass(index: Index) -> None:
    meta = index.doc_meta[0]
    assert isinstance(meta, DocMeta)
    assert meta.n_tokens == index.doc_lengths[0]

"""Lab 2 - the inverted index: structure, invariants, storage layouts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, fields
from pathlib import Path

import pytest

from findex.index import (
    ArrayPostings,
    DocMeta,
    Index,
    ObjectPostings,
    Posting,
    build_index,
)
from findex.tokenize import tokenize


# --------------------------------------------------------------------------- records
def test_posting_is_frozen_and_hashable() -> None:
    a, b = Posting(1, 2), Posting(1, 2)
    assert a == b
    assert hash(a) == hash(b)
    assert len({a, b}) == 1  # legal inside a set
    with pytest.raises(FrozenInstanceError):
        a.tf = 3  # type: ignore[misc]


def test_records_use_slots_not_instance_dicts() -> None:
    assert not hasattr(Posting(2, 1), "__dict__")
    assert not hasattr(DocMeta(1, "p", "t", 10), "__dict__")
    assert not hasattr(ObjectPostings([]), "__dict__")
    assert isinstance(Posting.__slots__, tuple)
    assert {f.name for f in fields(DocMeta)} == {"doc_id", "path", "title", "n_tokens"}


# --------------------------------------------------------------------------- build
def test_one_pass_build_fills_the_three_dicts(index: Index) -> None:
    assert len(index.doc_meta) == 3
    assert len(index) == len(index.postings) > 0
    assert index.n_tokens == sum(index.doc_lengths.values())
    assert index.doc_meta[0].title == "Alpha"  # parsed from the Gutenberg header
    assert index.doc_meta[2].title == "Gamma"


def test_postings_are_sorted_by_doc_id(index: Index) -> None:
    for term in index.terms():
        ids = list(index.doc_ids(term))
        assert ids == sorted(ids)
        assert len(ids) == len(set(ids))


def test_term_frequency_is_per_document(index: Index) -> None:
    # "fox" occurs twice in Alpha ("fox jumps", "A fox"), once in Beta, once in Gamma
    assert dict(index.tf_items("fox")) == {0: 2, 1: 1, 2: 1}
    assert index.doc_freq("fox") == 3
    assert index.doc_freq("quick") == 2
    assert index.doc_freq("nonexistent") == 0
    assert list(index.doc_ids("nonexistent")) == []


def test_doc_lengths_count_tokens(index: Index) -> None:
    assert index.doc_lengths[0] == 15
    assert index.doc_meta[1].n_tokens == index.doc_lengths[1]


def test_limit_streams_only_the_first_documents(corpus: Path) -> None:
    limited = build_index(corpus, limit=1)
    assert len(limited.doc_meta) == 1
    assert limited.doc_meta[0].title == "Alpha"


# --------------------------------------------------------------------------- layouts
@pytest.mark.parametrize("storage", ["objects", "arrays"])
def test_both_layouts_expose_the_same_postings(corpus: Path, storage: str) -> None:
    index = build_index(corpus, storage=storage)
    as_dict = {term: dict(index.tf_items(term)) for term in index.terms()}
    assert as_dict["fox"] == {0: 2, 1: 1, 2: 1}
    assert as_dict["dog"] == {0: 1, 1: 2}
    if storage == "arrays":
        assert isinstance(index.postings["fox"], ArrayPostings)
        assert index.postings["fox"]._flat.typecode == "I"
    else:
        assert isinstance(index.postings["fox"], ObjectPostings)


def test_plain_dataclass_can_be_swapped_in(corpus: Path) -> None:
    @dataclass(frozen=True)
    class PlainPosting:
        doc_id: int
        tf: int

    index = build_index(corpus, posting_type=PlainPosting)
    assert dict(index.tf_items("dog")) == {0: 1, 1: 2}


def test_positions_are_optional_and_correct(corpus: Path) -> None:
    plain = build_index(corpus)
    assert plain.positions is None

    with_positions = build_index(corpus, positions=True)
    assert with_positions.positions is not None
    assert with_positions.positions["fox"][0] == (5, 12)  # token offsets in alpha.txt
    tokens = list(tokenize((corpus / "alpha.txt").read_text(encoding="utf-8")))
    assert [tokens[i] for i in with_positions.positions["fox"][0]] == ["fox", "fox"]

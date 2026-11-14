"""Lab 2 - Boolean search: parsing, merge semantics, and merge/set equivalence."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from findex.index import Index, ObjectPostings, Posting, build_index
from findex.search import (
    Clause,
    difference,
    explain,
    intersect,
    parse,
    search,
    union,
)


# --------------------------------------------------------------------------- parsing
def test_implicit_and_is_a_single_group() -> None:
    assert parse("quick fox") == [[Clause("quick"), Clause("fox")]]


def test_or_splits_groups_and_not_negates() -> None:
    assert parse("quick NOT lazy OR fox") == [
        [Clause("quick"), Clause("lazy", True)],
        [Clause("fox")],
    ]


def test_lowercase_and_is_a_term_not_an_operator() -> None:
    assert parse("the and") == [[Clause("the"), Clause("and")]]


def test_redundant_and_is_ignored_and_case_folds() -> None:
    assert parse("Quick AND Fox") == [[Clause("quick"), Clause("fox")]]


@pytest.mark.parametrize(
    "query", ["", "   ", "OR fox", "quick OR", "NOT", "NOT NOT fox", "(fox)"]
)
def test_invalid_queries_raise(query: str) -> None:
    with pytest.raises(ValueError):
        parse(query)


# --------------------------------------------------------------------------- semantics
def test_and_requires_both_terms(index: Index) -> None:
    assert search(index, "quick fox") == [0, 2]


def test_or_unions_the_groups(index: Index) -> None:
    assert search(index, "rabbits OR brown") == [0, 1, 2]


def test_not_subtracts_from_the_current_group(index: Index) -> None:
    assert search(index, "fox NOT quick") == [1]
    assert search(index, "quick NOT fox") == []


def test_leading_not_means_everything_except(index: Index) -> None:
    assert search(index, "NOT fox") == []
    assert search(index, "NOT rabbits") == [0]


def test_unknown_term_matches_nothing(index: Index) -> None:
    assert search(index, "zzz") == []
    assert search(index, "zzz OR fox") == [0, 1, 2]


def test_k_caps_the_number_of_hits(index: Index) -> None:
    assert search(index, "the", k=1) == [0]


def test_explain_lists_document_frequencies(index: Index) -> None:
    text = explain(index, "fox NOT dog")
    assert "fox(3 dfs)" in text
    assert "-dog(2 dfs)" in text


# --------------------------------------------------------------------------- merge ops
@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ([], [], []),
        ([1, 2, 3], [], []),
        ([1, 3, 5], [2, 3, 6], [3]),
        ([1, 2, 3], [1, 2, 3], [1, 2, 3]),
        ([1], [2], []),
    ],
)
def test_intersect(a: list[int], b: list[int], expected: list[int]) -> None:
    assert list(intersect(a, b)) == expected


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ([], [], []),
        ([1, 2, 3], [], [1, 2, 3]),
        ([1, 3, 5], [2, 3, 6], [1, 2, 3, 5, 6]),
        ([1], [2], [1, 2]),
    ],
)
def test_union(a: list[int], b: list[int], expected: list[int]) -> None:
    assert list(union(a, b)) == expected


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ([], [], []),
        ([1, 2, 3], [], [1, 2, 3]),
        ([1, 2, 3], [2], [1, 3]),
        ([1, 2, 3], [4, 5], [1, 2, 3]),
        ([1, 2, 3], [1, 2, 3], []),
    ],
)
def test_difference(a: list[int], b: list[int], expected: list[int]) -> None:
    assert list(difference(a, b)) == expected


def _synthetic(postings: dict[str, list[int]]) -> Index:
    """An index whose postings are arbitrary (already sorted) doc-id lists."""
    docs = sorted({d for ids in postings.values() for d in ids})
    return Index(
        postings={
            term: ObjectPostings([Posting(d, 1) for d in ids])
            for term, ids in postings.items()
        },
        doc_lengths={d: 10 for d in docs},
        doc_meta={d: DocMetaStub(d) for d in docs},  # type: ignore[dict-item]
        n_tokens=10 * len(docs),
    )


class DocMetaStub:
    def __init__(self, doc_id: int) -> None:
        self.doc_id = doc_id
        self.path = f"doc{doc_id}.txt"
        self.title = f"doc {doc_id}"
        self.n_tokens = 10


def test_merge_and_set_engines_agree_on_random_data() -> None:
    rng = random.Random(1234)
    for _ in range(200):
        universe = list(range(rng.randint(0, 40)))
        terms = {}
        for name in ("a", "b", "c", "d"):
            terms[name] = sorted(rng.sample(universe, rng.randint(0, len(universe))))
        index = _synthetic(terms)
        for query in (
            "a",
            "a b",
            "a OR b",
            "a NOT b",
            "a b NOT c",
            "a NOT b OR c",
            "NOT a",
        ):
            assert search(index, query, engine="merge") == search(
                index, query, engine="set"
            ), query


def test_arrays_and_objects_layouts_search_identically(corpus: Path) -> None:
    objects = build_index(corpus, storage="objects")
    arrays = build_index(corpus, storage="arrays")
    for query in ("fox", "fox NOT quick", "rabbits OR brown", "NOT fox"):
        assert search(objects, query) == search(arrays, query)

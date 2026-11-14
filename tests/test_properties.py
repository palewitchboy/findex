"""Lab 4 - Hypothesis properties: the invariants the examples can't enumerate."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from findex.index import Index, build_index
from findex.query import parse
from findex.search import difference, intersect, union
from findex.store import load, save, to_dict
from findex.tokenize import tokenize

#: Ascending, duplicate-free sequences - the contract every merge function has.
sorted_ids = st.lists(
    st.integers(min_value=0, max_value=200), max_size=25, unique=True
).map(sorted)

#: Lowercase words and digits only: no operator is uppercase, no character is
#: special to the lexer, so every such string must parse.
plain_words = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789 ", min_size=1
).filter(lambda text: bool(text.strip()))


@given(a=sorted_ids, b=sorted_ids)
def test_intersect_agrees_with_sets(a: list[int], b: list[int]) -> None:
    assert list(intersect(a, b)) == sorted(set(a) & set(b))


@given(a=sorted_ids, b=sorted_ids)
def test_union_agrees_with_sets(a: list[int], b: list[int]) -> None:
    merged = list(union(a, b))
    assert merged == sorted(set(a) | set(b))
    assert merged == sorted(merged)


@given(a=sorted_ids, b=sorted_ids)
def test_difference_agrees_with_sets(a: list[int], b: list[int]) -> None:
    assert list(difference(a, b)) == sorted(set(a) - set(b))


@given(ids=sorted_ids)
def test_intersect_with_self_is_identity(ids: list[int]) -> None:
    assert list(intersect(ids, ids)) == ids


@given(text=st.text(max_size=200))
def test_tokenize_never_yields_an_empty_or_whitespace_token(text: str) -> None:
    for token in tokenize(text):
        assert token
        assert not any(ch.isspace() for ch in token)


@given(text=st.text(max_size=200))
def test_tokenize_is_idempotent(text: str) -> None:
    once = list(tokenize(text))
    assert list(tokenize(" ".join(once))) == once


@given(query=plain_words)
def test_parse_accepts_any_sequence_of_plain_words(query: str) -> None:
    tree = parse(query)
    assert tree.terms()


@given(query=plain_words)
def test_parse_round_trips_through_str(query: str) -> None:
    tree = parse(query)
    assert parse(str(tree)) == tree


@pytest.mark.slow
@given(
    texts=st.lists(
        st.text(alphabet="abcdefghijklmnopqrstuvwxyz .,'\n", max_size=120),
        max_size=3,
    )
)
@settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_save_load_round_trip_is_the_identity(texts: list[str], tmp_path: Path) -> None:
    """For any tiny corpus: load(save(index)) has the same content and shape."""
    case = tmp_path / "corpus"
    shutil.rmtree(case, ignore_errors=True)
    case.mkdir()
    for number, text in enumerate(texts):
        (case / f"doc{number}.txt").write_text(
            f"Title: Doc {number}\n\n{text}", encoding="utf-8"
        )

    original = build_index(case)
    for term in original.terms():
        ids = list(original.doc_ids(term))
        assert ids == sorted(ids)
        assert all(tf > 0 for _, tf in original.tf_items(term))

    restored = load(_saved(original, tmp_path / "index.pkl"))
    assert to_dict(restored) == to_dict(original)
    assert restored == original


def _saved(index: Index, path: Path) -> Path:
    save(index, path)
    return path

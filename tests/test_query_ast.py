"""Lab 3 - the query language: parser trees, precedence, and evaluation."""

from __future__ import annotations

from pathlib import Path

import pytest

from findex.index import Index, build_index
from findex.query import (
    And,
    MissingPositions,
    Not,
    Or,
    Phrase,
    QuerySyntaxError,
    Term,
    parse,
)


# --------------------------------------------------------------------------- trees
def test_a_single_word_is_a_term() -> None:
    assert parse("fox") == Term("fox")


def test_adjacent_words_are_an_implicit_and() -> None:
    assert parse("quick fox") == And(Term("quick"), Term("fox"))


def test_explicit_and_is_the_same_tree() -> None:
    assert parse("quick AND fox") == parse("quick fox")


def test_and_is_casefolded_but_operators_are_not() -> None:
    assert parse("Quick AND Fox") == And(Term("quick"), Term("fox"))


def test_or_has_lower_precedence_than_and() -> None:
    # The whole point of the grammar: "a OR b c" is NOT "(a OR b) c".
    assert parse("a OR b c") == Or(Term("a"), And(Term("b"), Term("c")))


def test_parentheses_override_the_precedence() -> None:
    assert parse("(a OR b) c") == And(Or(Term("a"), Term("b")), Term("c"))


def test_not_binds_tighter_than_and() -> None:
    assert parse("a NOT b") == And(Term("a"), Not(Term("b")))
    assert parse("NOT a") == Not(Term("a"))


def test_double_negation_is_allowed_by_the_recursive_grammar() -> None:
    assert parse("NOT NOT a") == Not(Not(Term("a")))


def test_operator_symbols_parse_like_the_keywords() -> None:
    assert parse("a & b | ~c") == Or(And(Term("a"), Term("b")), Not(Term("c")))
    assert parse("a & b") == parse("a AND b")
    assert parse("~a") == parse("NOT a")


def test_operators_need_no_spaces() -> None:
    assert parse("(a|b)&c") == And(Or(Term("a"), Term("b")), Term("c"))


def test_lowercase_operators_are_ordinary_terms() -> None:
    assert parse("the and") == And(Term("the"), Term("and"))
    assert parse("or") == Term("or")
    assert parse("not") == Term("not")
    assert parse("war or peace") == And(And(Term("war"), Term("or")), Term("peace"))


def test_quoted_text_is_a_phrase() -> None:
    assert parse('"war and peace"') == Phrase(("war", "and", "peace"))
    assert parse('"War AND Peace"') == Phrase(("war", "and", "peace"))


def test_phrase_and_terms_combine() -> None:
    assert parse('"of the" AND war') == And(Phrase(("of", "the")), Term("war"))


def test_python_operators_build_the_same_tree_as_the_parser() -> None:
    assert Term("a") & Term("b") == parse("a b")
    assert Term("a") | Term("b") & Term("c") == parse("a OR b c")
    assert ~Term("a") == parse("NOT a")
    assert (Term("a") | Term("b")) & Term("c") == parse("(a OR b) c")


def test_terms_reports_only_positive_terms() -> None:
    assert parse("a b").terms() == frozenset({"a", "b"})
    assert parse('"of the"').terms() == frozenset({"of", "the"})
    assert parse("NOT a").terms() == frozenset()
    assert parse("a OR NOT b").terms() == frozenset({"a"})


# --------------------------------------------------------------------------- printing
@pytest.mark.parametrize(
    "query",
    [
        "fox",
        "a b",
        "a OR b c",
        "(a OR b) c",
        "NOT a",
        "a NOT b",
        'NOT "of the"',
        'war AND ("of the" OR peace)',
        "a & b | ~c",
    ],
)
def test_str_is_fully_parenthesised_and_reparses(query: str) -> None:
    tree = parse(query)
    assert parse(str(tree)) == tree


def test_str_of_the_three_shapes() -> None:
    assert str(Term("fox")) == "fox"
    assert str(Phrase(("of", "the"))) == '"of the"'
    assert str(And(Term("a"), Term("b"))) == "(a AND b)"
    assert str(Not(Term("a"))) == "NOT (a)"


# --------------------------------------------------------------------------- errors
@pytest.mark.parametrize(
    "query",
    [
        "",
        "   ",
        "(a",
        "a)",
        "a OR",
        "AND a",
        "a AND AND b",
        "a & & b",
        "()",
        "(",
        ")",
        '"unterminated',
        '""',
        "OR",
        "NOT",
    ],
)
def test_bad_queries_raise_query_syntax_error(query: str) -> None:
    with pytest.raises(QuerySyntaxError):
        parse(query)


def test_query_syntax_error_is_a_value_error() -> None:
    assert issubclass(QuerySyntaxError, ValueError)


# --------------------------------------------------------------------------- evaluate
def test_and_or_not_evaluate_as_sets(index: Index) -> None:
    assert parse("quick fox").evaluate(index) == {0, 2}
    assert parse("rabbits OR brown").evaluate(index) == {0, 1, 2}
    assert parse("brown OR (rabbits AND quick)").evaluate(index) == {0, 2}
    assert parse("quick NOT fox").evaluate(index) == set()
    assert parse("NOT rabbits").evaluate(index) == {0}
    assert parse("NOT zzz").evaluate(index) == {0, 1, 2}


def test_unknown_terms_match_nothing(index: Index) -> None:
    assert parse("zzz").evaluate(index) == set()
    assert parse("zzz OR fox").evaluate(index) == {0, 1, 2}


def test_phrase_needs_positions(index: Index) -> None:
    assert index.positions is None
    with pytest.raises(MissingPositions, match="positions=True"):
        parse('"lazy dog"').evaluate(index)


def test_phrase_matches_only_adjacent_words(corpus: Path) -> None:
    index = build_index(corpus, positions=True)
    assert parse('"lazy dog"').evaluate(index) == {0, 1}
    assert parse('"a fox"').evaluate(index) == {0, 1, 2}
    assert parse('"quick brown fox"').evaluate(index) == {0}
    assert parse('"fox the"').evaluate(index) == set()


def test_phrase_combines_with_boolean_operators(corpus: Path) -> None:
    index = build_index(corpus, positions=True)
    assert parse('"lazy dog" AND rabbits').evaluate(index) == {1}
    assert parse('"lazy dog" OR brown').evaluate(index) == {0, 1}
    assert parse('NOT "lazy dog"').evaluate(index) == {2}


def test_nodes_are_hashable_and_frozen() -> None:
    # Frozen dataclasses hash by field, which is what makes the trees usable
    # as cache keys and safe to share between threads.
    assert len({Term("a"), Term("a"), Term("b")}) == 2
    assert hash(parse("a OR b")) == hash(Or(Term("a"), Term("b")))

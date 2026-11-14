"""M5 - Boolean search over the inverted index (Lab 2 semantics).

Lab 3 supersedes this module for the CLI: ``findex.query`` parses the full
language (parentheses, phrases, ``&``/``|``/``~``) into an AST and
``findex.ranking`` scores what it matches. What stays here is the Boolean
engine itself, because ``bench/lab2.py`` compares ``merge`` against ``set`` and
because "which documents match" is the primitive that ranking is built on.

Query language (deliberately small - Lab 3 grows a real parser and ranking):

* ``term``            - one term;
* ``a b``             - implicit AND (both terms must occur);
* ``a OR b``          - either term (uppercase keywords only: ``and``/``or``/
  ``not`` in lower case are ordinary terms);
* ``a NOT b``         - ``a`` and not ``b``;
* ``NOT b``           - every document except those containing ``b``.

Evaluation is left to right, ``OR`` splitting the query into disjunctive groups:

    a b NOT c OR d   ==  ((a AND b) AND NOT c) OR (d)

Two engines implement the same semantics:

``merge``
    Two-pointer merge over the postings lists, which are sorted by ``doc_id``.
    O(n + m) time, O(1) extra memory, no hashing - the reason postings are kept
    sorted in the first place.

``set``
    ``set(postings)`` and ``&`` / ``|`` / ``-``. Costs one hash table per term
    (~60 B per element) but is shorter to write. ``bench/lab2.py`` measures where
    each one wins.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Literal

from findex.index import Index

Engine = Literal["merge", "set"]


@dataclass(frozen=True, slots=True)
class Clause:
    term: str
    negated: bool = False


# --------------------------------------------------------------------------- parsing
def parse(query: str) -> list[list[Clause]]:
    """Split a query into OR-groups of AND-ed clauses.

    ``AND`` is accepted but redundant: two adjacent terms mean the same thing.
    """
    if any(ch in query for ch in "()[]{}"):
        raise ValueError("parentheses are not supported until Lab 3")
    groups: list[list[Clause]] = [[]]
    negate_next = False
    for raw in query.split():
        word = raw.strip()
        if not word:
            continue
        # Operators are the *uppercase* words only: "and" is the second most
        # frequent term in an English corpus and must stay searchable.
        if word == "OR":
            if not groups[-1]:
                raise ValueError("empty operand before OR")
            groups.append([])
            negate_next = False
            continue
        if word == "AND":
            continue
        if word == "NOT":
            if negate_next:
                raise ValueError("NOT NOT is not supported")
            negate_next = True
            continue
        groups[-1].append(Clause(word.casefold(), negate_next))
        negate_next = False
    if not groups[-1]:
        raise ValueError("empty query")
    return groups


# --------------------------------------------------------------------------- merge ops
def intersect(a: Iterable[int], b: Iterable[int]) -> Iterator[int]:
    """Two-pointer intersection of two ascending sequences."""
    it_a, it_b = iter(a), iter(b)
    xa, xb = next(it_a, None), next(it_b, None)
    while xa is not None and xb is not None:
        if xa == xb:
            yield xa
            xa, xb = next(it_a, None), next(it_b, None)
        elif xa < xb:
            xa = next(it_a, None)
        else:
            xb = next(it_b, None)


def union(a: Iterable[int], b: Iterable[int]) -> Iterator[int]:
    """Two-pointer union of two ascending sequences."""
    it_a, it_b = iter(a), iter(b)
    xa, xb = next(it_a, None), next(it_b, None)
    while xa is not None and xb is not None:
        if xa == xb:
            yield xa
            xa, xb = next(it_a, None), next(it_b, None)
        elif xa < xb:
            yield xa
            xa = next(it_a, None)
        else:
            yield xb
            xb = next(it_b, None)
    if xa is not None:
        yield xa
        yield from it_a
    if xb is not None:
        yield xb
        yield from it_b


def difference(a: Iterable[int], b: Iterable[int]) -> Iterator[int]:
    """Two-pointer ``a - b`` for ascending sequences."""
    it_b = iter(b)
    xb = next(it_b, None)
    for xa in a:
        while xb is not None and xb < xa:
            xb = next(it_b, None)
        if xb != xa:
            yield xa


# --------------------------------------------------------------------------- engines
def _evaluate_merge(index: Index, groups: list[list[Clause]]) -> list[int]:
    result: Iterator[int] | None = None
    for group in groups:
        positives = [c for c in group if not c.negated]
        negatives = [c for c in group if c.negated]

        # Intersect the shortest lists first: the result shrinks, so later
        # merges touch fewer postings.
        positives.sort(key=lambda c: index.doc_freq(c.term))
        if positives:
            acc: Iterator[int] = index.doc_ids(positives[0].term)
            for clause in positives[1:]:
                acc = intersect(acc, index.doc_ids(clause.term))
        else:
            acc = iter(index.all_doc_ids())
        for clause in negatives:
            acc = difference(acc, index.doc_ids(clause.term))

        result = acc if result is None else union(result, acc)
    return sorted(result) if result is not None else []


def _evaluate_set(index: Index, groups: list[list[Clause]]) -> list[int]:
    result: set[int] = set()
    universe = set(index.all_doc_ids())
    for group in groups:
        positives = [c for c in group if not c.negated]
        negatives = [c for c in group if c.negated]
        acc = universe if not positives else set(index.doc_ids(positives[0].term))
        for clause in positives[1:]:
            acc &= set(index.doc_ids(clause.term))
        for clause in negatives:
            acc -= set(index.doc_ids(clause.term))
        result |= acc
    return sorted(result)


def search(
    index: Index, query: str, *, engine: Engine = "merge", k: int | None = None
) -> list[int]:
    """Return matching doc ids, ascending; ``k`` caps the number of hits."""
    groups = parse(query)
    ids = (
        _evaluate_merge(index, groups)
        if engine == "merge"
        else _evaluate_set(index, groups)
    )
    return ids if k is None else ids[:k]


def explain(index: Index, query: str) -> str:
    """Human-readable trace: which terms, how many documents each one has."""
    lines: list[str] = []
    for n, group in enumerate(parse(query), start=1):
        parts = [
            f"{'-' if c.negated else ''}{c.term}({index.doc_freq(c.term)} dfs)"
            for c in group
        ]
        lines.append(f"group {n}: " + " AND ".join(parts))
    return "\n".join(lines)


# ------------------------------------------------------------------ entry point
if __name__ == "__main__":  # pragma: no cover - see findex.cli
    from findex.cli import search_main

    search_main()

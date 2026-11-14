"""Lab 3 - the query language: a recursive-descent parser and an AST.

Grammar (one production per method in ``_Parser`` - that is the whole trick):

    or_expr  := and_expr (("OR" | "|") and_expr)*
    and_expr := not_expr (("AND" | "&")? not_expr)*        # adjacency = AND
    not_expr := ("NOT" | "~") not_expr | atom
    atom     := "(" or_expr ")" | '"' words '"' | term

Precedence, loosest first: ``OR``, ``AND`` (implicit or explicit), ``NOT``,
parentheses. That is also Python's own precedence for ``|``, ``&`` and ``~``,
so a hand-built tree and a parsed one mean the same thing:

    parse("a OR b c") == Or(Term("a"), And(Term("b"), Term("c")))
    Term("a") | Term("b") & Term("c") == parse("a OR b c")

``AND``/``OR``/``NOT`` are operators **only in uppercase**; ``and`` is the
second most frequent word in an English corpus and stays searchable. The same
holds inside a quoted phrase: ``"war and peace"`` is a three-word phrase.

Nodes are frozen dataclasses, which buys structural ``__eq__`` - that is why
the tests assert on *trees*, not on strings. They also overload ``&``, ``|``
and ``~`` (via :class:`Node`), so a query can be composed in Python without
touching the parser at all.

Evaluating a node returns a ``set[int]`` of doc ids. ``NOT`` is the complement
against the whole collection, so every node evaluates to a well-defined set and
``~Term("x")`` means "everything except x", not "nothing".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from findex.tokenize import tokenize

if TYPE_CHECKING:
    from findex.index import Index


class QuerySyntaxError(ValueError):
    """The query string does not parse (bad operator placement, stray quote...)."""


class MissingPositions(ValueError):
    """A phrase query needs token positions; rebuild with ``positions=True``."""


# --------------------------------------------------------------------------- protocol
class Query(Protocol):
    """What every node can do: evaluate, list its positive terms, compose."""

    def evaluate(self, index: Index) -> set[int]: ...

    def terms(self) -> frozenset[str]: ...

    def __and__(self, other: Query) -> Query: ...

    def __or__(self, other: Query) -> Query: ...

    def __invert__(self) -> Query: ...


class Node:
    """Mixin: every query node supports ``&``, ``|`` and ``~``.

    ``__slots__ = ()`` keeps subclasses slotted - a mixin without it would hand
    every node an instance ``__dict__`` and silently undo ``slots=True``.
    """

    __slots__ = ()

    def __and__(self: Query, other: Query) -> Query:
        return And(self, other)

    def __or__(self: Query, other: Query) -> Query:
        return Or(self, other)

    def __invert__(self: Query) -> Query:
        return Not(self)


# --------------------------------------------------------------------------- leaves
@dataclass(frozen=True, slots=True)
class Term(Node):
    """One normalised (casefolded, tokenised) term."""

    text: str

    def evaluate(self, index: Index) -> set[int]:
        return set(index.doc_ids(self.text))

    def terms(self) -> frozenset[str]:
        return frozenset((self.text,))

    def __str__(self) -> str:
        return self.text


@dataclass(frozen=True, slots=True)
class Phrase(Node):
    """Consecutive words that must appear adjacently, in order.

    The field is ``words``, not ``terms``: ``terms()`` is the method that hands
    the scorer and the snippet highlighter their vocabulary.
    """

    words: tuple[str, ...]

    def evaluate(self, index: Index) -> set[int]:
        """Documents where the words occur at consecutive token offsets."""
        if index.positions is None:
            raise MissingPositions(
                "phrase queries need token positions: rebuild the index with "
                "positions=True (python -m findex.index data/ --positions)"
            )
        positions = index.positions
        candidates = set(index.doc_ids(self.words[0]))
        for word in self.words[1:]:
            candidates &= set(index.doc_ids(word))
            if not candidates:
                return set()

        found: set[int] = set()
        for doc_id in candidates:
            # starts = offsets of the first word that are also offsets of the
            # i-th word shifted back by i. Intersecting sets beats a nested
            # loop over positions, and the sets are tiny (one per document).
            starts = set(positions[self.words[0]].get(doc_id, ()))
            for offset, word in enumerate(self.words[1:], start=1):
                starts &= {pos - offset for pos in positions[word].get(doc_id, ())}
                if not starts:
                    break
            if starts:
                found.add(doc_id)
        return found

    def terms(self) -> frozenset[str]:
        return frozenset(self.words)

    def __str__(self) -> str:
        return '"' + " ".join(self.words) + '"'


# --------------------------------------------------------------------------- operators
@dataclass(frozen=True, slots=True)
class And(Node):
    """Conjunction; ``evaluate`` is a set intersection."""

    left: Query
    right: Query

    def evaluate(self, index: Index) -> set[int]:
        return self.left.evaluate(index) & self.right.evaluate(index)

    def terms(self) -> frozenset[str]:
        return self.left.terms() | self.right.terms()

    def __str__(self) -> str:
        return f"({self.left} AND {self.right})"


@dataclass(frozen=True, slots=True)
class Or(Node):
    """Disjunction; ``evaluate`` is a set union."""

    left: Query
    right: Query

    def evaluate(self, index: Index) -> set[int]:
        return self.left.evaluate(index) | self.right.evaluate(index)

    def terms(self) -> frozenset[str]:
        return self.left.terms() | self.right.terms()

    def __str__(self) -> str:
        return f"({self.left} OR {self.right})"


@dataclass(frozen=True, slots=True)
class Not(Node):
    """Complement against the whole collection."""

    child: Query

    def evaluate(self, index: Index) -> set[int]:
        return set(index.all_doc_ids()) - self.child.evaluate(index)

    def terms(self) -> frozenset[str]:
        """Empty: a term the user excluded must not be highlighted in a snippet."""
        return frozenset()

    def __str__(self) -> str:
        return f"NOT ({self.child})"


# --------------------------------------------------------------------------- parser
#: A quoted phrase, a single operator character, or a run of everything else.
_TOKEN = re.compile(r'"[^"]*"|[()&|~]|[^\s()&|~"]+')

#: Tokens that end the current ``and_expr``.
_CLOSERS = frozenset({")", "OR", "|"})

#: Tokens that can never start an atom.
_NOT_AN_ATOM = frozenset({"AND", "OR", "&", "|", "~", ")"})


def _lex(query: str) -> list[str]:
    """Split a query into phrases, operator characters and bare words.

    Works character by character instead of ``str.split()`` so that ``a&b`` and
    ``(a)`` need no spaces - and so an unbalanced quote is an error instead of
    being silently dropped between two matches.
    """
    tokens: list[str] = []
    end = 0
    for match in _TOKEN.finditer(query):
        gap = query[end : match.start()].strip()
        if gap:
            raise QuerySyntaxError(f"cannot parse {gap!r} (unbalanced quote?)")
        tokens.append(match.group())
        end = match.end()
    tail = query[end:].strip()
    if tail:
        raise QuerySyntaxError(f"cannot parse {tail!r} (unbalanced quote?)")
    return tokens


def _term(raw: str) -> Query:
    """A bare word: tokenised, folded, and AND-ed back together if it splits."""
    parts = list(tokenize(raw))
    if not parts:
        raise QuerySyntaxError(f"{raw!r} contains no searchable term")
    node: Query = Term(parts[0])
    for part in parts[1:]:
        node = And(node, Term(part))
    return node


def _phrase(raw: str) -> Phrase:
    """The inside of a quoted string: one word per token, in order."""
    words = tuple(tokenize(raw))
    if not words:
        raise QuerySyntaxError("empty phrase ''")
    return Phrase(words)


class _Parser:
    """Recursive descent over the token list; one method per grammar rule."""

    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def take(self) -> str:
        token = self.tokens[self.pos]
        self.pos += 1
        return token

    def or_expr(self) -> Query:
        node = self.and_expr()
        while self.peek() in {"OR", "|"}:
            self.take()
            node = Or(node, self.and_expr())
        return node

    def and_expr(self) -> Query:
        node = self.not_expr()
        while True:
            token = self.peek()
            if token is None or token in _CLOSERS:
                return node
            if token in {"AND", "&"}:
                self.take()
            node = And(node, self.not_expr())

    def not_expr(self) -> Query:
        if self.peek() in {"NOT", "~"}:
            self.take()
            return Not(self.not_expr())
        return self.atom()

    def atom(self) -> Query:
        token = self.peek()
        if token is None:
            raise QuerySyntaxError("unexpected end of query")
        if token in _NOT_AN_ATOM:
            raise QuerySyntaxError(f"unexpected {token!r}")
        if token == "(":
            self.take()
            node = self.or_expr()
            if self.peek() != ")":
                raise QuerySyntaxError("missing ')'")
            self.take()
            return node
        if token.startswith('"'):
            self.take()
            return _phrase(token[1:-1])
        self.take()
        return _term(token)


def parse(query: str) -> Query:
    """Parse a query string into a tree of :class:`Query` nodes.

    Idempotent in the sense that matters for tests: ``parse(str(tree))`` gives
    back a tree equal to the original, because ``__str__`` is fully
    parenthesised.
    """
    tokens = _lex(query)
    if not tokens:
        raise QuerySyntaxError("empty query")
    parser = _Parser(tokens)
    node = parser.or_expr()
    leftover = parser.peek()
    if leftover is not None:
        raise QuerySyntaxError(f"unexpected {leftover!r}")
    return node

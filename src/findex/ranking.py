"""Lab 3 - ranking: TF-IDF, BM25 and the top-k heap.

Boolean search answers *which* documents match. Ranking answers *in what order*.
Both scorers here walk the same postings and accumulate one float per document;
the only difference is the formula.

TF-IDF (the 1970s baseline, still the control group)::

    idf(t)      = log(N / df(t))
    score(q, d) = sum over t in q of (1 + log tf(t, d)) * idf(t)

``1 + log tf`` is the cheap fix for raw ``tf``: ten occurrences should count
more than one, but not ten times more. Note what the formula says when
``df == N``: ``idf = log(1) = 0``. A term that occurs in every document carries
no information at all - that is the arithmetic behind stop-word lists.

BM25 (the default in Lucene, Elasticsearch and most of the pre-neural web)::

    idf(t)      = log(1 + (N - df + 0.5) / (df + 0.5))
    score(q, d) = sum over t in q of idf(t) * tf * (k1 + 1)
                                       / (tf + k1 * (1 - b + b * dl / avgdl))

Two fixes over TF-IDF, each with a named knob:

* ``k1`` (1.2-2.0, default 1.5) sets **saturation**. The numerator grows with
  ``tf`` and the denominator grows with it too, so the score approaches an
  asymptote: the 20th occurrence of a word adds almost nothing. ``k1 = 0``
  throws the term frequency away entirely.
* ``b`` (0..1, default 0.75) sets **length normalisation**. ``b = 0`` ignores
  document length; ``b = 1`` divides the frequency by ``dl / avgdl`` in full,
  so a short document that mentions a term once beats a long one that also
  mentions it once. On this corpus that is the difference between a 20 kB poem
  and a 5 MB Shakespeare folio.

Top-k uses ``heapq.nlargest``: O(n log k) time and O(k) memory, versus
O(n log n) and O(n) for ``sorted(scores)[:k]``. With n = 1M candidates and
k = 10 that is a full sort of a million tuples replaced by a ten-element heap;
``bench/lab3.py`` times both.

The ``Scorer`` protocol is structural: anything with a matching ``__call__``
is a scorer. No base class, no registration - the fakes in the tests are five
lines long and the Searcher cannot tell them from BM25.
"""

from __future__ import annotations

import functools
import heapq
import logging
import math
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from findex import vector
from findex.decorators import timed
from findex.index import Index
from findex.query import Query, parse
from findex.snippet import snippet

log = logging.getLogger(__name__)

#: Which scorer implementation a :class:`Searcher` uses. ``numpy`` is Lab 8's
#: vectorised path; ``python`` is Lab 3's reference loop, kept because the two
#: are compared in the tests and because a custom ``Scorer`` can only run there.
Engine = Literal["python", "numpy"]
ENGINES: tuple[Engine, ...] = ("python", "numpy")


# --------------------------------------------------------------------------- protocol
@runtime_checkable
class Scorer(Protocol):
    """A scoring function: term + posting + index in, relevance out.

    The protocol only records the *shape* of the call. ``TfIdf`` and ``BM25``
    below are frozen dataclasses, the tests use closures and plain classes -
    none of them inherits from anything, and all of them satisfy ``Scorer``.
    """

    def __call__(self, term: str, doc_id: int, tf: int, index: Index) -> float: ...


# --------------------------------------------------------------------------- scorers
@dataclass(frozen=True, slots=True)
class TfIdf:
    """``(1 + log tf) * log(N / df)`` summed over the query terms.

    ``log_tf=False`` switches to raw term frequency, which is what the original
    formula used and what the sanity checks in ``bench/lab3.py`` compare against.
    """

    log_tf: bool = True

    def __call__(self, term: str, doc_id: int, tf: int, index: Index) -> float:
        df = index.df(term)
        if tf <= 0 or df <= 0 or index.num_docs <= 0:
            return 0.0
        idf = math.log(index.num_docs / df)
        weight = 1.0 + math.log(tf) if self.log_tf else float(tf)
        return weight * idf


@dataclass(frozen=True, slots=True)
class BM25:
    """Okapi BM25 with saturation (``k1``) and length normalisation (``b``)."""

    k1: float = 1.5
    b: float = 0.75

    def __post_init__(self) -> None:
        if self.k1 < 0:
            raise ValueError(f"k1 must be >= 0, got {self.k1}")
        if not 0.0 <= self.b <= 1.0:
            raise ValueError(f"b must be in [0, 1], got {self.b}")

    def __call__(self, term: str, doc_id: int, tf: int, index: Index) -> float:
        df = index.df(term)
        n_docs = index.num_docs
        if tf <= 0 or df <= 0 or n_docs <= 0:
            return 0.0
        idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
        avgdl = index.avg_doc_length or 1.0
        dl = index.doc_length(doc_id)
        normalisation = self.k1 * (1.0 - self.b + self.b * dl / avgdl)
        return idf * tf * (self.k1 + 1.0) / (tf + normalisation)


# --------------------------------------------------------------------------- results
@functools.total_ordering
@dataclass(frozen=True, slots=True)
class SearchResult:
    """One ranked hit.

    ``__lt__`` is **reversed on purpose**: it compares "better than", so
    ``sorted(results)`` puts the best hit first and ``results[:5]`` is the top
    five. ``total_ordering`` derives ``>``, ``<=`` and ``>=`` from it, and
    ``__eq__`` (from the dataclass) stays field equality - which is why ``==``
    and ``<=`` can disagree about two results with the same score.

    Ties break on ascending ``doc_id``, so a result list is deterministic and a
    diff between two runs is a real change.
    """

    doc_id: int
    score: float
    title: str
    snippet: str = ""

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, SearchResult):
            return NotImplemented
        return (self.score, -self.doc_id) > (other.score, -other.doc_id)

    def __str__(self) -> str:
        return f"[{self.doc_id:>4}] {self.score:8.3f}  {self.title}"


def _rank_key(item: tuple[int, float]) -> tuple[float, int]:
    """Heap key: highest score first, lowest doc id on a tie."""
    return (item[1], -item[0])


# --------------------------------------------------------------------------- ranking
def rank(
    index: Index, node: Query, scorer: Scorer, *, k: int = 10
) -> list[SearchResult]:
    """Score every document matching ``node`` and return the best ``k``.

    This is the **reference implementation** - Lab 3's per-posting loop with
    ``heapq.nlargest``. :class:`Searcher` no longer calls it (Lab 8 routes
    ``BM25``/``TfIdf`` through :mod:`findex.vector`), but it stays public and
    exercised by the tests: it defines what "the same ranking" means.
    """
    matched = node.evaluate(index)
    scores = dict.fromkeys(matched, 0.0)
    for term in node.terms():
        for doc_id, tf in index.tf_items(term):
            if doc_id in scores:
                scores[doc_id] += scorer(term, doc_id, tf, index)
    top = heapq.nlargest(k, scores.items(), key=_rank_key)
    return [
        SearchResult(doc_id=doc_id, score=score, title=index.doc_meta[doc_id].title)
        for doc_id, score in top
    ]


@dataclass(frozen=True, slots=True)
class _Match:
    """A cached query result: the matched doc ids and the terms to highlight."""

    docs: tuple[int, ...]
    terms: frozenset[str]


class Searcher:
    """An index + a scorer + an LRU cache of query → matched docs.

    The cache deliberately stores the *match set*, not the ranked list: the set
    of documents matching a query does not depend on the scorer or on ``k``, so
    one cached entry serves both scorers and every page size.

    ``functools.lru_cache`` is applied to the **bound** method and kept on the
    instance, so the key is the query string alone. A module-level
    ``@lru_cache`` on ``search(index, query)`` would have to key on the index -
    which is unhashable, and which a global cache would then pin in memory
    forever. ``cache_size=0`` disables caching, which is how the benchmark
    measures the cost of not having it.

    ``engine`` picks the scorer implementation (Lab 8):

    ``"numpy"`` (default)
        :mod:`findex.vector` scores whole posting lists as ``int32`` arrays and
        selects the top-k with ``np.argpartition``. Identical ranking, verified
        by ``tests/test_vector.py``.
    ``"python"``
        Lab 3's per-posting loop. Kept because it is the reference the NumPy
        path is tested against, and because a custom :class:`Scorer` - a test
        closure, a plugin - can only run here; the vector path recognises
        ``BM25`` and ``TfIdf`` by type and falls back for anything else.
    """

    def __init__(
        self,
        index: Index,
        scorer: Scorer | None = None,
        *,
        cache_size: int = 256,
        snippets: bool = True,
        mark: tuple[str, str] = ("[", "]"),
        engine: Engine = "numpy",
    ) -> None:
        if engine not in ENGINES:
            raise ValueError(f"unknown engine {engine!r}; expected one of {ENGINES}")
        self.index = index
        self.scorer: Scorer = BM25() if scorer is None else scorer
        self.snippets = snippets
        #: How ``snippet()`` brackets a match. ``("[", "]")`` is the terminal
        #: and JSON contract; the web layer passes control characters instead,
        #: escapes the text and turns them into ``<mark>`` - without escaping,
        #: a book containing ``[war]`` would forge a highlight.
        self.mark = mark
        self.engine: Engine = engine
        #: Built on first search, and only when the vector engine is asked for.
        self._vectors = vector.VectorIndex(index) if engine == "numpy" else None
        self._match = functools.lru_cache(maxsize=cache_size)(self._match_uncached)

    def _match_uncached(self, query: str) -> _Match:
        node = parse(query)
        return _Match(tuple(sorted(node.evaluate(self.index))), node.terms())

    def count(self, query: str) -> int:
        """How many documents match ``query`` - the denominator for pagination.

        Reads the same LRU-cached match set that :meth:`search` uses, so
        "12 hits in total" and "the first 10 of them" cost one evaluation
        rather than two. The set is scorer-independent, which is why one entry
        serves BM25, TF-IDF and every page size.
        """
        return len(self._match(query).docs)

    def cache_info(self) -> Any:
        """Hits, misses, maxsize and currsize of the query cache.

        The real type is ``functools._CacheInfo``, a private name; it is a
        ``NamedTuple`` of four ints and ``Any`` keeps the annotation honest
        without importing from a private module.
        """
        return self._match.cache_info()

    def clear_cache(self) -> None:
        self._match.cache_clear()

    def _scores_python(self, match: _Match) -> dict[int, float]:
        """Lab 3's loop: one Python call per posting, kept as the reference."""
        scores = dict.fromkeys(match.docs, 0.0)
        for term in match.terms:
            for doc_id, tf in self.index.tf_items(term):
                if doc_id in scores:
                    scores[doc_id] += self.scorer(term, doc_id, tf, self.index)
        return scores

    def _top_k(self, match: _Match, k: int) -> list[tuple[int, float]]:
        """Top ``k`` ``(doc_id, score)`` pairs, vectorised when possible.

        The vector path is used only for the two scorers this project ships:
        ``Scorer`` is a protocol, so anything else (a lambda in a test, a
        plugin) has to go through the per-posting loop - which is exactly why
        that loop is still here and still tested.
        """
        vectors = self._vectors
        if vectors is not None and match.docs:
            scores = None
            if isinstance(self.scorer, BM25):
                scores = vectors.bm25(
                    match.terms, k1=self.scorer.k1, b=self.scorer.b
                )
            elif isinstance(self.scorer, TfIdf):
                scores = vectors.tfidf(match.terms, log_tf=self.scorer.log_tf)
            if scores is not None:
                return vector.top_k_of(match.docs, scores, k)
        return heapq.nlargest(k, self._scores_python(match).items(), key=_rank_key)

    @timed
    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        """Parse, match, score and cut to ``k``; ``@timed`` logs the milliseconds."""
        hits_before = self._match.cache_info().hits
        match = self._match(query)
        if self._match.cache_info().hits > hits_before:
            log.debug("query cache hit for %r", query)

        top = self._top_k(match, k)
        results: list[SearchResult] = []
        for doc_id, score in top:
            text = (
                snippet(self.index, doc_id, match.terms, mark=self.mark)
                if self.snippets
                else ""
            )
            results.append(
                SearchResult(
                    doc_id=doc_id,
                    score=score,
                    title=self.index.doc_meta[doc_id].title,
                    snippet=text,
                )
            )
        return results


# --------------------------------------------------------------------------- explain
def explain(index: Index, node: Query) -> str:
    """Pretty-print the tree and the document frequency behind every term."""
    lines = [f"tree    : {node}", f"matched : {len(node.evaluate(index))} docs"]
    for term in sorted(node.terms()):
        df = index.df(term)
        idf = math.log(index.num_docs / df) if df else 0.0
        lines.append(f"  {term:<16} df={df:>5}  idf={idf:5.2f}")
    return "\n".join(lines)


def terms_of(node: Query) -> frozenset[str]:
    """Convenience re-export so callers need not know about ``Node``."""
    return node.terms()


__all__ = [
    "BM25",
    "ENGINES",
    "Engine",
    "Scorer",
    "SearchResult",
    "Searcher",
    "TfIdf",
    "explain",
    "rank",
    "terms_of",
]

"""Lab 8 - vectorised scoring: move the loop into NumPy.

Lab 3's scorer is a Python call per posting::

    for doc_id, tf in index.tf_items(term):
        if doc_id in scores:
            scores[doc_id] += scorer(term, doc_id, tf, index)

Every iteration boxes an int, looks up a dict, calls a function, allocates a
float and updates a reference count - dozens of machine instructions per
posting, and 446 575 postings in the 48-document corpus. The formula is three
lines of arithmetic; the interpreter is the cost.

This module computes the same three lines over whole arrays at once:

* **postings as `np.int32`** - ``np.frombuffer`` over Lab 2's ``array('I')``
  gives a *view*, not a copy: the same bytes the index already holds, read as
  two strided integer arrays. No new memory, no per-posting object.
* **ufuncs and broadcasting** - ``idf * tfs * (k1 + 1) / (tfs + norm[doc_ids])``
  is one compiled loop per term, with ``doc_len[doc_ids]`` a fancy-index gather.
* **`np.argpartition` for top-k** - O(n) selection instead of a full sort.

Correctness is not negotiable: the ranking must be *identical* to the Python
scorer, tie-breaks included (highest score first, lowest ``doc_id`` on a tie).
``tests/test_vector.py`` compares both on every query of the fixture corpus, and
``bench/lab8.py`` compares them on real postings; the score tolerance is 1e-9,
because ``math.log`` and ``np.log`` may differ in the last bit.

Where the win is small, and why
-------------------------------
Vectorising pays for itself only when a posting list is long. On this corpus
the commonest term occurs in all 48 documents, so the biggest array has 48
elements - and NumPy's per-call overhead (allocate, dispatch, validate dtype)
is a few microseconds, which is the same order as the Python loop it replaces.
The measurement in ``bench/lab8.py`` reports exactly that: the speedup is real
but modest on 48 documents, and grows with ``df``. The honest summary is that
this is an architecture change that scales, not a 40x win on a toy corpus.
"""

# NumPy's bundled stubs are generated from its C signatures, so almost every
# ufunc call comes back as "partially unknown" - `reportUnknownMemberType`
# fires on `np.frombuffer`, `np.partition`, `np.lexsort` and friends no matter
# how precisely the arguments are typed. Turning the rule off *for this file*
# is the honest alternative to eight `# type: ignore` comments; everything else
# in the file is still checked in strict mode.
# pyright: reportUnknownMemberType=false

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from findex.index import ArrayPostings, Index, Postings

#: int32 is deliberate: 48 documents do not need int64, and halving the width
#: halves the bytes the gather has to stream. Posting ids are non-negative and
#: bounded by the document count, so 2^31 documents is the (absurd) limit.
ID = np.int32


@dataclass(frozen=True, slots=True)
class TermArrays:
    """One term's postings as two integer arrays: doc ids and term frequencies."""

    doc_ids: npt.NDArray[ID]
    tfs: npt.NDArray[ID]

    def __len__(self) -> int:
        return int(self.doc_ids.size)


def term_arrays(postings: Postings) -> TermArrays:
    """Turn a ``Postings`` object into ``int32`` arrays, without copying if possible.

    ``np.frombuffer`` over ``array('I')`` is the whole point of Lab 2's flat
    layout: the bytes are already ``[doc_id, tf, doc_id, tf, ...]``, so reading
    them as ``uint32`` and taking every other element is a *view*. ``view``
    reinterprets the same memory as ``int32`` (identical bit patterns for the
    non-negative values postings hold), so no copy happens at all.

    Anything that is not an ``ArrayPostings`` (the ``objects`` storage, a test
    double) goes through ``np.fromiter`` - slower, but correct for every
    ``Postings`` implementation.
    """
    if isinstance(postings, ArrayPostings) and postings.itemsize == 4:
        flat = np.frombuffer(postings.flat, dtype=np.uint32)
        return TermArrays(doc_ids=flat[0::2].view(ID), tfs=flat[1::2].view(ID))
    length = len(postings)
    doc_ids = np.fromiter(postings.doc_ids(), dtype=ID, count=length)
    tfs = np.fromiter(
        (tf for _, tf in postings.items()), dtype=ID, count=length
    )
    return TermArrays(doc_ids=doc_ids, tfs=tfs)


class VectorIndex:
    """A lazily built NumPy view of one :class:`Index`.

    Nothing is converted until a query asks for a term, so an index that is
    loaded and never searched costs nothing. The cache is per ``VectorIndex``,
    i.e. per :class:`~findex.ranking.Searcher`, which keeps the lifetime obvious:
    drop the searcher, drop the arrays. Two searchers over one index duplicate
    the arrays - on this corpus that is a few hundred kilobytes, on a real one
    it would argue for a shared cache keyed by index identity.
    """

    def __init__(self, index: Index) -> None:
        self.index = index
        self._terms: dict[str, TermArrays] = {}
        self._doc_lengths: npt.NDArray[ID] | None = None
        self._norm: dict[tuple[float, float], npt.NDArray[np.float64]] = {}

    def normalisation(
        self, k1: float, b: float
    ) -> npt.NDArray[np.float64]:
        """``k1 * (1 - b + b * dl / avgdl)`` for every document, cached per (k1, b).

        One array per parameter pair. Without the cache every query rebuilt it,
        which on a 48-document index is a measurable share of a 0.2 ms score -
        the same "compute it once" argument as ``avg_doc_length`` in Lab 3.
        """
        key = (k1, b)
        cached = self._norm.get(key)
        if cached is None:
            avgdl = self.index.avg_doc_length or 1.0
            lengths = self.doc_lengths().astype(np.float64)
            cached = np.asarray(k1 * (1.0 - b + b * lengths / avgdl), np.float64)
            self._norm[key] = cached
        return cached

    def term(self, term: str) -> TermArrays | None:
        """The term's arrays, or ``None`` when the term is not indexed."""
        cached = self._terms.get(term)
        if cached is None:
            postings = self.index.postings.get(term)
            if postings is None:
                return None
            cached = term_arrays(postings)
            self._terms[term] = cached
        return cached

    def doc_lengths(self) -> npt.NDArray[ID]:
        """``doc_len`` as one ``int32[N]`` array - the ``dl`` of BM25."""
        if self._doc_lengths is None:
            n_docs = self.index.num_docs
            lengths = np.zeros(n_docs, dtype=ID)
            for doc_id, length in self.index.doc_lengths.items():
                if 0 <= doc_id < n_docs:
                    lengths[doc_id] = length
            self._doc_lengths = lengths
        return self._doc_lengths

    # -- scorers ------------------------------------------------------------
    def bm25(
        self, terms: Iterable[str], *, k1: float = 1.5, b: float = 0.75
    ) -> npt.NDArray[np.float64]:
        """BM25 for every document, in one pass per term.

        The expression is Lab 3's, character for character::

            idf = log(1 + (N - df + 0.5) / (df + 0.5))
            norm = k1 * (1 - b + b * dl / avgdl)
            score += idf * tf * (k1 + 1) / (tf + norm)
        """
        n_docs = self.index.num_docs
        scores = np.zeros(n_docs, dtype=np.float64)
        if n_docs == 0:
            return scores
        normalisation = self.normalisation(k1, b)
        for term in terms:
            arrays = self.term(term)
            if arrays is None or len(arrays) == 0:
                continue
            df = len(arrays)
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            tfs = arrays.tfs.astype(np.float64)
            # doc ids are unique inside one term's postings, so a fancy-index
            # scatter-add is safe; np.add.at would be the slow, general form.
            scores[arrays.doc_ids] += idf * tfs * (k1 + 1.0) / (
                tfs + normalisation[arrays.doc_ids]
            )
        return scores

    def tfidf(
        self, terms: Iterable[str], *, log_tf: bool = True
    ) -> npt.NDArray[np.float64]:
        """TF-IDF for every document: ``(1 + log tf) * log(N / df)`` per term."""
        n_docs = self.index.num_docs
        scores = np.zeros(n_docs, dtype=np.float64)
        if n_docs == 0:
            return scores
        for term in terms:
            arrays = self.term(term)
            if arrays is None or len(arrays) == 0:
                continue
            df = len(arrays)
            idf = math.log(n_docs / df)
            tfs = arrays.tfs.astype(np.float64)
            weight = 1.0 + np.log(tfs) if log_tf else tfs
            scores[arrays.doc_ids] += weight * idf
        return scores


def top_k(
    doc_ids: npt.NDArray[ID], scores: npt.NDArray[np.float64], k: int
) -> list[tuple[int, float]]:
    """The best ``k`` ``(doc_id, score)`` pairs, in the order Lab 3 produced them.

    ``heapq.nlargest`` breaks ties on **ascending doc id**, and a plain
    ``argpartition`` cannot promise that: among equal scores at the cut it may
    keep whichever index it likes. The fix is the standard two-stage selection -

    1. ``np.partition`` to find the k-th largest score (O(n), no sort);
    2. everything strictly above it is in; fill the remainder from the ties,
       taking the smallest doc ids first.

    ``doc_ids`` arrives ascending (the match set is sorted), so the ties are
    already in the order the tie-break wants. Then only the k chosen entries are
    sorted - O(k log k) - and the result is identical to ``heapq``'s.
    """
    size = int(doc_ids.size)
    if k <= 0 or size == 0:
        return []
    if k >= size:
        chosen = np.arange(size)
    else:
        threshold = np.partition(scores, size - k)[size - k]
        above = np.flatnonzero(scores > threshold)
        if above.size < k:
            ties = np.flatnonzero(scores == threshold)
            chosen = np.concatenate((above, ties[: k - above.size]))
        else:
            chosen = above[:k]
    order = np.lexsort((doc_ids[chosen], -scores[chosen]))
    picked = chosen[order]
    return [(int(doc_ids[i]), float(scores[i])) for i in picked]


def top_k_of(
    docs: Sequence[int], scores: npt.NDArray[np.float64], k: int
) -> list[tuple[int, float]]:
    """``top_k`` for a plain Python sequence of ids and a dense score array.

    The gather ``scores[ids]`` is what turns a per-document array into a
    per-*match* one; keeping it here means ``findex.ranking`` never has to
    import NumPy itself.
    """
    if not docs:
        return []
    ids = np.asarray(docs, dtype=ID)
    return top_k(ids, scores[ids], k)


__all__ = ["TermArrays", "VectorIndex", "term_arrays", "top_k", "top_k_of"]

"""M4/M6 - the inverted index: build it in one streaming pass, then *be* a Mapping.

The index is three dictionaries:

* ``postings[term] -> Postings`` - for every term, the documents that contain it
  and the term frequency (``tf``) in each of them, **sorted by doc_id**;
* ``doc_lengths[doc_id] -> int`` - tokens per document (BM25 needs it);
* ``doc_meta[doc_id] -> DocMeta`` - where the document came from, and its title.

Lab 3 turned that bag of dicts into an object that behaves like Python:

    len(index)         # how many distinct terms
    "python" in index  # is the term indexed?
    index["python"]    # the Postings object, KeyError if absent
    for term in index  # iterate the vocabulary
    index.get(term)    # Mapping.get(), free from collections.abc

``Index`` inherits ``collections.abc.Mapping[str, Postings]`` and implements only
the three abstract methods (``__getitem__``, ``__len__``, ``__iter__``); ``keys``,
``values``, ``items``, ``get`` and value-based ``__eq__`` come from the ABC. The
derived attributes are properties: ``num_docs`` is cheap, ``avg_doc_length`` is a
``functools.cached_property`` because BM25 asks for it once per scored document
and the sum must be computed once, not a million times.

Two storage representations are implemented, and both are measured in
``bench/lab2.py``:

* ``"objects"`` - ``list[Posting]``, where ``Posting`` is a frozen slotted
  dataclass (the default);
* ``"arrays"``  - one flat ``array("I")`` per term, as
  ``[doc_id, tf, doc_id, tf, ...]`` - no Python objects per posting at all.

Search code never sees the difference: it talks to the ``Postings`` protocol
(``doc_ids()`` / ``items()``), so the two-pointer merge, the set engine and the
scorers are written once.

The pipeline stays lazy: ``iter_documents`` streams the corpus, and only the
index grows. The per-document ``Counter`` is flushed into postings and dropped
before the next document is read.

Lab 5 splits the build in two so that it can run on a pool:

* :func:`build_partial` - one document in, a compact :class:`PartialIndex` out;
  a module-level function, therefore picklable and spawn-safe;
* :func:`merge` - partials in, one :class:`Index` out; a per-term concatenation
  in worker order, which is already doc_id order because doc ids are assigned
  from the **sorted** path list before any worker starts.

Determinism is the whole trick. :func:`document_paths` sorts the corpus, the doc
id is the position in that list, each worker gets a contiguous range, and every
postings list is therefore ascending and disjoint. That is why serial, threaded
and process builds compare equal with ``==`` - and pickle to identical bytes.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import re
from array import array
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from functools import cached_property
from functools import partial as partial_call
from itertools import islice
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from findex.corpus import Document, iter_documents, load_document, read_document
from findex.decorators import timed
from findex.tokenize import tokenize

log = logging.getLogger(__name__)

_TITLE = re.compile(r"^[ \t]*Title:[ \t]*(.+?)[ \t]*$", re.MULTILINE)

#: How the work is executed. ``serial`` is a plain loop, ``threads`` one
#: ``ThreadPoolExecutor``, ``processes`` one ``ProcessPoolExecutor``.
ExecutorName = Literal["serial", "threads", "processes"]
EXECUTORS: tuple[ExecutorName, ...] = ("serial", "threads", "processes")

#: The CLI default - the executor that won on the full corpus (see the Lab 5
#: section of the README). A *library* call still defaults to ``serial``: a
#: function should not spawn interpreters behind the caller's back.
DEFAULT_EXECUTOR: ExecutorName = "processes"

#: Upper bound for the *default* worker count. Eight is where the benchmark
#: flattens: on the measured machine `os.cpu_count()` reports 24 logical CPUs
#: (12 physical cores), and 24 workers were *slower* than 8 - the extra
#: processes only add spawn, pickling and memory-bandwidth contention.
#: ``--workers`` is the escape hatch, and the benchmark keeps a column for
#: ``os.cpu_count()``.
MAX_DEFAULT_WORKERS = 8

StorageName = Literal["objects", "arrays"]
STORAGES: tuple[StorageName, ...] = ("objects", "arrays")


# --------------------------------------------------------------------------- records
@dataclass(frozen=True, slots=True)
class Posting:
    """One (document, term frequency) pair.

    Frozen + slotted: 56 bytes, hashable (frozen dataclasses get ``__hash__``),
    and legal inside a ``set`` - the attribute table lives in the class, not in
    every instance.
    """

    doc_id: int
    tf: int


@dataclass(frozen=True, slots=True)
class DocMeta:
    """Everything needed to display a hit without re-reading the file."""

    doc_id: int
    path: str
    title: str
    n_tokens: int


# --------------------------------------------------------------------------- postings
@runtime_checkable
class Postings(Protocol):
    """Read-only view of one term's postings list, ascending by ``doc_id``."""

    def __len__(self) -> int: ...

    def doc_ids(self) -> Iterator[int]: ...

    def items(self) -> Iterator[tuple[int, int]]: ...


class ObjectPostings:
    """Postings stored as a list of ``Posting``-like objects."""

    __slots__ = ("_postings",)

    def __init__(self, postings: list[Any]) -> None:
        self._postings = postings

    def __len__(self) -> int:
        return len(self._postings)

    def doc_ids(self) -> Iterator[int]:
        return (p.doc_id for p in self._postings)

    def items(self) -> Iterator[tuple[int, int]]:
        return ((p.doc_id, p.tf) for p in self._postings)

    def __eq__(self, other: object) -> bool:
        """Compare by content, so two indexes loaded from disk compare equal.

        Defining ``__eq__`` sets ``__hash__`` to ``None``: a container that
        compares by value is mutable as far as Python is concerned, and nothing
        in findex hashes a postings list.
        """
        if not isinstance(other, ObjectPostings):
            return NotImplemented
        return list(self.items()) == list(other.items())

    def __repr__(self) -> str:
        return f"ObjectPostings({len(self._postings)} postings)"


class ArrayPostings:
    """Postings stored as one flat ``array('I')``: ``[doc_id, tf, doc_id, tf, ...]``.

    A C array of 4-byte unsigned ints - no per-posting Python object, no
    pointer chasing, and it serialises as raw bytes.

    Lab 8 added :attr:`flat` and :attr:`itemsize` so ``findex.vector`` can hand
    the very same bytes to NumPy through ``np.frombuffer`` - a view, not a copy.
    That is the whole reason the layout exists, and it stays an implementation
    detail: search code still talks to ``doc_ids()`` / ``items()``.
    """

    __slots__ = ("_flat",)

    def __init__(self, flat: array[int]) -> None:
        self._flat = flat

    @property
    def flat(self) -> array[int]:
        """The interleaved buffer itself, for zero-copy readers (Lab 8)."""
        return self._flat

    @property
    def itemsize(self) -> int:
        """Bytes per value - 4 on every platform this project supports."""
        return self._flat.itemsize

    def __len__(self) -> int:
        return len(self._flat) // 2

    def doc_ids(self) -> Iterator[int]:
        flat = self._flat
        return (flat[i] for i in range(0, len(flat), 2))

    def items(self) -> Iterator[tuple[int, int]]:
        flat = self._flat
        return ((flat[i], flat[i + 1]) for i in range(0, len(flat), 2))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ArrayPostings):
            return NotImplemented
        return self._flat == other._flat

    def __repr__(self) -> str:
        return f"ArrayPostings({len(self)} postings, {self._flat.itemsize * 2} B each)"


# --------------------------------------------------------------------------- index
class Index(Mapping[str, Postings]):
    """The inverted index, as a read-only mapping from term to ``Postings``.

    ``Mapping`` gives ``keys()``, ``values()``, ``items()``, ``get()``, a
    value-based ``__eq__`` and ``__contains__`` (overridden here for speed);
    this class supplies the three abstract methods and the corpus statistics.
    ``bool(index)`` falls back to ``__len__``, so an empty index is falsy.

    ``Index`` is **unhashable**, as any value-comparing mutable container
    should be. Lab 3's query cache therefore keys on the query string, not on
    the index.
    """

    def __init__(
        self,
        postings: dict[str, Postings],
        doc_lengths: dict[int, int],
        doc_meta: dict[int, DocMeta],
        n_tokens: int,
        positions: dict[str, dict[int, tuple[int, ...]]] | None = None,
    ) -> None:
        self.postings = postings
        self.doc_lengths = doc_lengths
        self.doc_meta = doc_meta
        self.n_tokens = n_tokens
        self.positions = positions
        self._closed = False

    # -- Mapping protocol ---------------------------------------------------
    def __getitem__(self, term: str) -> Postings:
        """The postings list for ``term``; ``KeyError`` if it is not indexed."""
        return self.postings[term]

    def __len__(self) -> int:
        """Number of distinct terms."""
        return len(self.postings)

    def __iter__(self) -> Iterator[str]:
        return iter(self.postings)

    def __contains__(self, term: object) -> bool:
        return term in self.postings

    def __repr__(self) -> str:
        return (
            f"Index(terms={len(self.postings):,}, docs={len(self.doc_meta):,}, "
            f"postings={self.n_postings():,})"
        )

    # -- derived attributes -------------------------------------------------
    @property
    def num_docs(self) -> int:
        """How many documents were indexed (``N`` in every IDF)."""
        return len(self.doc_meta)

    @cached_property
    def avg_doc_length(self) -> float:
        """Mean document length in tokens - the ``avgdl`` of BM25.

        ``cached_property`` computes it on first access and stores the result in
        the instance ``__dict__``, where the next ``index.avg_doc_length`` finds
        it with a plain dict lookup. BM25 asks for it once per scored document;
        without the cache that is a 12-million-integer sum per query.
        """
        if not self.doc_lengths:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)

    # -- read API -----------------------------------------------------------
    def df(self, term: str) -> int:
        """Document frequency: in how many documents ``term`` occurs."""
        postings = self.postings.get(term)
        return 0 if postings is None else len(postings)

    #: Lab 2's name for :meth:`df`; the same function object, kept so old code
    #: and old notes keep working.
    doc_freq = df

    def doc_length(self, doc_id: int) -> int:
        """Tokens in ``doc_id``; ``KeyError`` for a document we never saw."""
        return self.doc_lengths[doc_id]

    def doc_ids(self, term: str) -> Iterator[int]:
        """Ascending doc ids for ``term``; empty iterator for an unknown term."""
        postings = self.postings.get(term)
        return iter(()) if postings is None else postings.doc_ids()

    def tf_items(self, term: str) -> Iterator[tuple[int, int]]:
        """``(doc_id, tf)`` pairs for ``term``; empty for an unknown term.

        Lab 2 called this ``items(term)``, which had to go: on a ``Mapping``
        ``items`` is the view over *all* keys, and one method cannot be both.
        """
        postings = self.postings.get(term)
        return iter(()) if postings is None else postings.items()

    def all_doc_ids(self) -> list[int]:
        return sorted(self.doc_meta)

    def terms(self) -> Iterator[str]:
        """Same as ``iter(index)``; kept because Lab 2 code reads better with it."""
        return iter(self.postings)

    def n_postings(self) -> int:
        return sum(len(p) for p in self.postings.values())

    def to_dict(self) -> dict[str, Any]:
        """A plain, JSON-shaped snapshot - for tests, diffs and the benchmark.

        ``to_dict`` is deliberately *lossy in types only*: terms stay strings,
        doc ids stay ints (so ``json.dumps`` needs ``sort_keys`` and a reader
        that turns them back), and every postings layout becomes a list of
        ``[doc_id, tf]`` pairs. Two indexes built by different executors must
        produce **equal** snapshots - that is the Lab 5 identity test.
        """
        return {
            "n_tokens": self.n_tokens,
            "doc_lengths": dict(sorted(self.doc_lengths.items())),
            "doc_meta": {
                doc_id: {
                    "path": meta.path,
                    "title": meta.title,
                    "n_tokens": meta.n_tokens,
                }
                for doc_id, meta in sorted(self.doc_meta.items())
            },
            "postings": {
                term: [list(pair) for pair in postings.items()]
                for term, postings in self.postings.items()
            },
            "positions": None
            if self.positions is None
            else {
                term: {doc_id: list(offsets) for doc_id, offsets in by_doc.items()}
                for term, by_doc in self.positions.items()
            },
        }

    # -- lifecycle ----------------------------------------------------------
    @property
    def closed(self) -> bool:
        """True once :meth:`close` (or ``open_index`` on the way out) ran."""
        return self._closed

    def close(self) -> None:
        """Release the heavy structures. The index stays a valid, empty mapping.

        There is no file handle to close - ``load`` reads the file and returns -
        but a deserialized index *is* the resource: hundreds of megabytes of
        dicts. Dropping them makes ``open_index``'s ``finally`` observable, and
        the object stays safe to ``len()``, ``repr()`` or iterate afterwards.
        """
        self.postings.clear()
        self.doc_lengths.clear()
        self.doc_meta.clear()
        self.positions = None
        self.n_tokens = 0
        self._closed = True

    def __getstate__(self) -> dict[str, Any]:
        """Pickle only the real data: not the closed flag, not the cached mean.

        ``cached_property`` writes into ``self.__dict__`` under its own name, so
        a pickle taken after a query would carry a stale ``avg_doc_length``.
        Dropping both keys makes the round-trip produce a pristine object.
        """
        state = self.__dict__.copy()
        state.pop("_closed", None)
        state.pop("avg_doc_length", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._closed = False


# --------------------------------------------------------------------------- partials
@dataclass(frozen=True, slots=True)
class PartialIndex:
    """The index of *one* document, in the cheapest shape that pickles well.

    Every field is a plain built-in (``dict``, ``int``, ``tuple``) - no
    ``ObjectPostings`` wrappers, no ``Posting`` instances - because this object
    is what crosses the process boundary. Postings are ``(doc_id, tf)`` tuples,
    so each term's list holds exactly one pair and the merge is one
    ``list.extend`` per term.

    ``frozen`` is not about immutability of the dicts (they are mutable) but
    about the contract: a partial is a value produced by a worker and consumed
    once by the parent.
    """

    doc_lengths: dict[int, int]
    doc_meta: dict[int, DocMeta]
    postings: dict[str, list[tuple[int, int]]]
    n_tokens: int
    positions: dict[str, dict[int, tuple[int, ...]]] | None = None


def build_partial(
    path: str | Path,
    doc_id: int,
    *,
    positions: bool = False,
) -> PartialIndex:
    """Tokenize one document into a :class:`PartialIndex`.

    Module level, two scalar arguments, keyword-only options: exactly what
    ``ProcessPoolExecutor`` needs. Under ``spawn`` the parent pickles the
    *function reference* and ``(path, doc_id)``, not the document; the worker
    reads its own file, so the parent's memory is never copied.

    ``positions=True`` is supported in every executor: the offsets travel inside
    the partial. It is the most expensive thing to pickle (one tuple per term
    per document), so the benchmark measures the tax with and without it.

    Raises ``OSError`` if the file cannot be read - a worker that silently
    skipped a document would produce a *smaller* index, and a partial index that
    looks successful is worse than a loud failure.
    """
    file = Path(path)
    text = read_document(file)
    counts: Counter[str] = Counter()
    offsets_by_term: dict[str, list[int]] | None = {} if positions else None
    for offset, token in enumerate(tokenize(text)):
        counts[token] += 1
        if offsets_by_term is not None:
            offsets_by_term.setdefault(token, []).append(offset)

    n_tokens = counts.total()
    return PartialIndex(
        doc_lengths={doc_id: n_tokens},
        doc_meta={
            doc_id: DocMeta(
                doc_id=doc_id,
                path=str(file),
                title=_title_of(text, file),
                n_tokens=n_tokens,
            )
        },
        postings={term: [(doc_id, tf)] for term, tf in counts.items()},
        n_tokens=n_tokens,
        positions=None
        if offsets_by_term is None
        else {
            term: {doc_id: tuple(offsets)}
            for term, offsets in offsets_by_term.items()
        },
    )


def _as_postings(
    merged: dict[str, list[tuple[int, int]]],
    storage: StorageName,
    posting_type: Callable[[int, int], object],
) -> dict[str, Postings]:
    """Turn merged ``(doc_id, tf)`` lists into the requested layout."""
    if storage == "objects":
        return {
            term: ObjectPostings([posting_type(doc_id, tf) for doc_id, tf in pairs])
            for term, pairs in merged.items()
        }
    flat: dict[str, Postings] = {}
    for term, pairs in merged.items():
        values = array("I")
        for doc_id, tf in pairs:
            values.append(doc_id)
            values.append(tf)
        flat[term] = ArrayPostings(values)
    return flat


def merge(
    partials: Iterable[PartialIndex],
    *,
    storage: StorageName = "objects",
    posting_type: Callable[[int, int], object] = Posting,
) -> Index:
    """Reduce partial indexes into one :class:`Index`.

    The reduce step is a **concatenation per term in the order the partials are
    given**. That is correct - and cheap - only because of the doc-id contract:
    ids are positions in the *sorted* path list and each worker owns a
    contiguous range, so partials arrive in ascending doc-id order and every
    concatenated postings list stays sorted. The function does not re-sort; it
    *checks* that no doc id appears twice and fails with ``ValueError`` when the
    contract is broken (two workers given overlapping ranges), which is far
    easier to debug than a corrupt index.

    Zero partials produce a valid empty index - ``merge([])`` is what a corpus
    with no ``.txt`` files reduces to.
    """
    if storage not in STORAGES:
        raise ValueError(f"unknown storage {storage!r}; expected 'objects' or 'arrays'")

    merged: dict[str, list[tuple[int, int]]] = {}
    doc_lengths: dict[int, int] = {}
    doc_meta: dict[int, DocMeta] = {}
    all_positions: dict[str, dict[int, tuple[int, ...]]] | None = None
    n_tokens = 0

    for partial in partials:
        for doc_id in partial.doc_lengths:
            if doc_id in doc_lengths:
                raise ValueError(f"duplicate doc_id {doc_id}: partial ranges overlap")
        if partial.positions is not None:
            if all_positions is None:
                all_positions = {}
            for term, by_doc in partial.positions.items():
                all_positions.setdefault(term, {}).update(by_doc)
        doc_lengths.update(partial.doc_lengths)
        doc_meta.update(partial.doc_meta)
        n_tokens += partial.n_tokens
        for term, pairs in partial.postings.items():
            merged.setdefault(term, []).extend(pairs)

    return Index(
        postings=_as_postings(merged, storage, posting_type),
        doc_lengths=doc_lengths,
        doc_meta=doc_meta,
        n_tokens=n_tokens,
        positions=all_positions,
    )


# --------------------------------------------------------------------------- executors
def default_workers() -> int:
    """The CLI's default pool size: ``min(8, os.cpu_count())``, at least 1.

    The cap is empirical: on the measured machine ``os.cpu_count()`` reports 24
    logical CPUs (12 physical cores) and eight workers were the fastest
    configuration - 24 were slower. More processes than the work can keep busy
    only buy spawn, pickling and memory-bandwidth contention. ``--workers``
    overrides, and the benchmark's ``all`` column is exactly this trade-off.
    """
    return max(1, min(MAX_DEFAULT_WORKERS, os.cpu_count() or 1))


def validate_workers(workers: int | None, executor: ExecutorName) -> int:
    """Resolve and check ``--workers`` for ``executor``.

    ``None`` means "pick a sensible default". Serial has no pool, so a worker
    count other than 1 is a *usage error*, not something to ignore silently.
    """
    if executor not in EXECUTORS:
        raise ValueError(f"unknown executor {executor!r}; expected one of {EXECUTORS}")
    if workers is None:
        return 1 if executor == "serial" else default_workers()
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    if executor == "serial" and workers != 1:
        raise ValueError("--workers has no effect with --executor serial")
    return workers


def document_paths(root: Path, limit: int | None = None) -> list[Path]:
    """The corpus as a **sorted** list of ``.txt`` paths - the doc-id source.

    Sorting is the determinism guarantee of Lab 5: ``Path.rglob`` returns
    whatever the filesystem hands over, and parallel workers must agree on which
    document is number 7. Sorting also makes ``--limit`` mean "the first N in
    name order" instead of "the first N the disk felt like returning".
    """
    root = Path(root)
    if not root.is_dir():
        raise ValueError(
            f"{root}: parallel indexing needs a directory of .txt files "
            "(a .jsonl corpus is one stream and has no per-file chunks)"
        )
    paths = sorted(path for path in root.rglob("*.txt") if path.is_file())
    return paths if limit is None else paths[:limit]


def _make_pool(executor: ExecutorName, workers: int) -> Executor:
    """One pool class, two backends - and an explicit ``spawn`` context.

    ``spawn`` is the only start method available on Windows and the safest one
    everywhere: no inherited locks or threads, no ``fork``-after-threads
    warnings. The price is a fresh interpreter per worker, which the benchmark
    measures rather than hides.
    """
    if executor == "threads":
        return ThreadPoolExecutor(max_workers=workers, thread_name_prefix="findex")
    context = multiprocessing.get_context("spawn")
    return ProcessPoolExecutor(max_workers=workers, mp_context=context)


def build_partials(
    tasks: Sequence[tuple[str, int]],
    *,
    executor: ExecutorName = "serial",
    workers: int | None = None,
    positions: bool = False,
    on_partial: Callable[[int], None] | None = None,
) -> list[PartialIndex]:
    """Build one partial per ``(path, doc_id)`` task, in task order.

    ``executor.map`` preserves input order, so the returned list is in doc-id
    order even though workers finish out of order - the merge relies on it.
    An exception raised inside a worker is re-raised here by ``map`` with the
    worker's traceback attached as ``__cause__`` by ``concurrent.futures``;
    nothing is swallowed, and the ``with`` block still shuts the pool down.
    """
    workers = validate_workers(workers, executor)
    if executor == "serial":
        built: list[PartialIndex] = []
        for count, (path, doc_id) in enumerate(tasks, start=1):
            built.append(build_partial(path, doc_id, positions=positions))
            if on_partial is not None:
                on_partial(count)
        return built

    if not tasks:
        return []

    # One bound function for every worker: functools.partial pickles as
    # (module-qualified function, kwargs), so this stays spawn-safe.
    task = partial_call(build_partial, positions=positions)
    paths = [path for path, _ in tasks]
    ids = [doc_id for _, doc_id in tasks]
    results: list[PartialIndex] = []
    with _make_pool(executor, min(workers, len(tasks))) as pool:
        for count, item in enumerate(pool.map(task, paths, ids), start=1):
            results.append(item)
            if on_partial is not None:
                on_partial(count)
    return results


# --------------------------------------------------------------------------- building
def _title_of(text: str, path: Path) -> str:
    """Gutenberg header carries ``Title:``; fall back to the file name."""
    match = _TITLE.search(text[:4000])
    return match.group(1) if match else path.stem


def _build_serial(
    root: Path,
    *,
    limit: int | None,
    positions: bool,
    storage: StorageName,
    posting_type: Callable[[int, int], object],
    progress_every: int,
    on_document: Callable[[int], None] | None,
) -> Index:
    """The Lab 2 pipeline, unchanged except that it walks a sorted path list.

    A directory corpus is enumerated with :func:`document_paths`, so doc ids
    match what the parallel executors assign; a ``.jsonl`` corpus keeps the
    streaming ``iter_documents`` path, because one stream has no per-file chunks.
    """
    if root.is_dir():
        docs: Iterable[Document] = (
            doc
            for doc_id, path in enumerate(document_paths(root, limit))
            if (doc := load_document(path, doc_id)) is not None
        )
    else:
        streamed = iter_documents(root)
        docs = streamed if limit is None else islice(streamed, limit)

    object_postings: dict[str, list[Any]] = defaultdict(list)
    array_postings: dict[str, array[int]] = {}
    doc_lengths: dict[int, int] = {}
    doc_meta: dict[int, DocMeta] = {}
    all_positions: dict[str, dict[int, tuple[int, ...]]] | None = (
        {} if positions else None
    )
    n_tokens = 0
    n_docs = 0

    for doc in docs:
        counts: Counter[str] = Counter()
        pos: dict[str, list[int]] | None = {} if positions else None
        for offset, token in enumerate(tokenize(doc.text)):
            counts[token] += 1
            if pos is not None:
                pos.setdefault(token, []).append(offset)

        n_tokens_doc = counts.total()
        doc_lengths[doc.doc_id] = n_tokens_doc
        doc_meta[doc.doc_id] = DocMeta(
            doc_id=doc.doc_id,
            path=str(doc.path),
            title=_title_of(doc.text, doc.path),
            n_tokens=n_tokens_doc,
        )
        n_tokens += n_tokens_doc
        n_docs += 1

        # Flush this document's Counter into the postings and drop it.
        if storage == "objects":
            for term, tf in counts.items():
                object_postings[term].append(posting_type(doc.doc_id, tf))
        else:
            for term, tf in counts.items():
                flat: array[int] | None = array_postings.get(term)
                if flat is None:
                    flat = array_postings[term] = array("I")
                flat.append(doc.doc_id)
                flat.append(tf)

        if pos is not None and all_positions is not None:
            for term, offsets in pos.items():
                all_positions.setdefault(term, {})[doc.doc_id] = tuple(offsets)

        del counts
        if on_document is not None:
            on_document(n_docs)
        if progress_every and n_docs % progress_every == 0:
            log.info("%d documents, %s tokens", n_docs, f"{n_tokens:,}")

    if storage == "objects":
        postings: dict[str, Postings] = {
            term: ObjectPostings(lst) for term, lst in object_postings.items()
        }
    else:
        postings = {term: ArrayPostings(flat) for term, flat in array_postings.items()}

    # Documents arrive in ascending order, so every postings list is already
    # sorted by doc_id - the merge algorithms depend on that invariant.
    return Index(
        postings=postings,
        doc_lengths=doc_lengths,
        doc_meta=doc_meta,
        n_tokens=n_tokens,
        positions=all_positions,
    )


@timed
def build_index(
    root: Path,
    *,
    limit: int | None = None,
    positions: bool = False,
    storage: StorageName = "objects",
    posting_type: Callable[[int, int], object] = Posting,
    progress_every: int = 0,
    on_document: Callable[[int], None] | None = None,
    executor: ExecutorName = "serial",
    workers: int | None = None,
) -> Index:
    """Build the index - in one streaming pass, or as map-reduce over a pool.

    ``posting_type`` lets the Lab 2 memory study swap in a *plain* (non-slotted)
    dataclass without touching this function. ``on_document`` is called once per
    indexed document with the running count - it exists so the CLI can drive a
    ``rich`` progress bar without this module importing a UI library.

    ``executor="serial"`` (the default) keeps Lab 2's single pass: no partial
    objects, no merge, the lowest memory of all three. ``threads`` and
    ``processes`` map :func:`build_partial` over a pool and then call
    :func:`merge`; both produce an index that is ``==`` to the serial one - and
    byte-identical when pickled - because doc ids come from the sorted path
    list, not from completion order.
    """
    if storage not in STORAGES:
        raise ValueError(f"unknown storage {storage!r}; expected 'objects' or 'arrays'")
    if executor not in EXECUTORS:
        raise ValueError(f"unknown executor {executor!r}; expected one of {EXECUTORS}")
    root = Path(root)
    if executor == "serial":
        return _build_serial(
            root,
            limit=limit,
            positions=positions,
            storage=storage,
            posting_type=posting_type,
            progress_every=progress_every,
            on_document=on_document,
        )

    validate_workers(workers, executor)
    tasks = [
        (str(path), doc_id)
        for doc_id, path in enumerate(document_paths(root, limit))
    ]
    partials = build_partials(
        tasks,
        executor=executor,
        workers=workers,
        positions=positions,
        on_partial=on_document,
    )
    return merge(partials, storage=storage, posting_type=posting_type)


# ------------------------------------------------------------------ entry point
if __name__ == "__main__":  # pragma: no cover - see findex.cli
    from findex.cli import index_main

    index_main()

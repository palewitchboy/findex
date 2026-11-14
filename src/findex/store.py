"""M4 - persist the index and load it back.

Two formats, deliberately different in spirit:

``pickle``
    One call, fast, handles the object graph as-is. **Security:** ``pickle.load``
    executes arbitrary code - it is safe *only* for a file this process wrote
    itself. Never load an index received over the network or from a stranger;
    that is remote code execution by design, not a bug you can patch around.

``json``
    Safe to eyeball and to ship between processes, but it only understands
    dicts/lists/strings/numbers, so postings become ``[[doc_id, tf], ...]`` and
    the compact ``array`` layout is lost. Bigger and slower, and it proves the
    numbers in ``bench/lab2.py``.

Both formats round-trip the same ``Index`` (postings, doc lengths, doc meta and,
when present, token positions).
"""

from __future__ import annotations

import json
import pickle
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from findex.decorators import timed
from findex.index import ArrayPostings, DocMeta, Index, ObjectPostings, Postings

_PICKLE_MAGIC = b"\x80"  # every pickle protocol >= 2 starts with PROTO
_JSON_MAGIC = b"{"


# --------------------------------------------------------------------------- dicts
def to_dict(index: Index) -> dict[str, Any]:
    """The portable half of an ``Index``: only lists, dicts, strings, numbers."""
    return {
        "version": 2,
        "n_tokens": index.n_tokens,
        "doc_lengths": {str(k): v for k, v in index.doc_lengths.items()},
        "doc_meta": [
            {
                "doc_id": m.doc_id,
                "path": m.path,
                "title": m.title,
                "n_tokens": m.n_tokens,
            }
            for m in (index.doc_meta[k] for k in sorted(index.doc_meta))
        ],
        "postings": {
            term: [[d, tf] for d, tf in p.items()] for term, p in index.postings.items()
        },
        "positions": (
            None
            if index.positions is None
            else {
                term: {str(doc): list(offs) for doc, offs in per_doc.items()}
                for term, per_doc in index.positions.items()
            }
        ),
    }


def from_dict(data: dict[str, Any]) -> Index:
    postings: dict[str, Postings] = {
        term: ObjectPostings([_Pair(d, tf) for d, tf in pairs])
        for term, pairs in data["postings"].items()
    }
    doc_meta = {
        int(m["doc_id"]): DocMeta(
            doc_id=int(m["doc_id"]),
            path=m["path"],
            title=m["title"],
            n_tokens=int(m["n_tokens"]),
        )
        for m in data["doc_meta"]
    }
    positions = data.get("positions")
    return Index(
        postings=postings,
        doc_lengths={int(k): int(v) for k, v in data["doc_lengths"].items()},
        doc_meta=doc_meta,
        n_tokens=int(data["n_tokens"]),
        positions=(
            None
            if positions is None
            else {
                term: {int(doc): tuple(offs) for doc, offs in per_doc.items()}
                for term, per_doc in positions.items()
            }
        ),
    )


class _Pair:
    """Minimal stand-in for ``Posting`` when rebuilding an index from JSON."""

    __slots__ = ("doc_id", "tf")

    def __init__(self, doc_id: int, tf: int) -> None:
        self.doc_id = doc_id
        self.tf = tf


# --------------------------------------------------------------------------- pickle
def save_pickle(index: Index, path: Path) -> None:
    """Serialize with pickle. Only ever load files you produced yourself."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(index, fh, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: Path) -> Index:
    """Load a trusted pickle. Do **not** call this on untrusted input."""
    with open(path, "rb") as fh:
        index = pickle.load(fh)  # noqa: S301 - trusted, locally produced file
    if not isinstance(index, Index):
        raise TypeError(f"{path}: not a findex index")
    return index


# --------------------------------------------------------------------------- json
def save_json(index: Index, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(to_dict(index), fh, ensure_ascii=False, separators=(",", ":"))


def load_json(path: Path) -> Index:
    with open(path, encoding="utf-8") as fh:
        return from_dict(json.load(fh))


# --------------------------------------------------------------------------- facade
def save(index: Index, path: Path, fmt: str = "pickle") -> None:
    if fmt == "pickle":
        save_pickle(index, path)
    elif fmt == "json":
        save_json(index, path)
    else:
        raise ValueError(f"unknown format {fmt!r}; expected 'pickle' or 'json'")


@timed
def load(path: Path, fmt: str | None = None) -> Index:
    """Load an index, sniffing the format from the first bytes when not given."""
    path = Path(path)
    if fmt is None:
        with open(path, "rb") as fh:
            head = fh.read(1)
        if head == _PICKLE_MAGIC:
            fmt = "pickle"
        elif head == _JSON_MAGIC:
            fmt = "json"
        else:
            raise ValueError(f"{path}: unknown index format (first byte {head!r})")
    return load_pickle(path) if fmt == "pickle" else load_json(path)


@contextmanager
def open_index(path: Path, fmt: str | None = None) -> Generator[Index]:
    """Load an index and guarantee it is released, even if the body raises.

    The generator form of a context manager: everything before ``yield`` is
    ``__enter__``, everything in ``finally`` is ``__exit__``. ``return``,
    ``break`` or an exception inside the ``with`` body all run the ``finally``,
    which is the whole point of ``with`` over a bare ``load`` call.

    There is no file handle to leak - ``load_pickle`` closes its own - so what
    is released is the index itself: hundreds of megabytes of dicts. Read what
    you need inside the block; after it, the index is a valid empty mapping::

        with open_index("index.pkl") as index:
            hits = Searcher(index).search("war AND peace", k=5)
        # index.closed is True, index.doc_meta is empty
    """
    index = load(path, fmt)
    try:
        yield index
    finally:
        index.close()


def storage_of(index: Index) -> str:
    """Which postings layout an index uses - handy for benchmarks and reprs."""
    for postings in index.postings.values():
        if isinstance(postings, ArrayPostings):
            return "arrays"
        if isinstance(postings, ObjectPostings):
            return "objects"
        break
    return "empty"

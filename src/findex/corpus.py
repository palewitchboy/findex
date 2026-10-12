"""M1 - lazy document stream.

``iter_documents`` is a generator: it opens one file (or reads one JSON line)
at a time, yields it and moves on, so the corpus never sits in memory.

Supported inputs
----------------
* a directory tree of ``.txt`` files (found lazily with ``Path.rglob``);
* a JSON-lines file (``.jsonl`` / ``.ndjson``, optionally ``.gz``): one document
  per line, text taken from the ``"text"`` field.

Encoding policy: everything is read as UTF-8. If a file is not valid UTF-8 we
log a warning and re-read it with ``errors="replace"`` (bad bytes become U+FFFD)
instead of crashing the whole run.
"""

from __future__ import annotations

import gzip
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

log = logging.getLogger(__name__)

JSONL_SUFFIXES = {".jsonl", ".ndjson"}


class Document(NamedTuple):
    doc_id: int
    path: Path
    text: str


def _read_text_file(path: Path) -> str | None:
    """Read one file as UTF-8; fall back to ``errors="replace"``; None on I/O error."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        log.warning("bad UTF-8 in %s - re-reading with errors='replace'", path)
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        log.warning("cannot read %s: %s", path, exc)
        return None


def _is_jsonl(path: Path) -> bool:
    suffixes = [s.lower() for s in path.suffixes]
    if suffixes and suffixes[-1] == ".gz":
        suffixes = suffixes[:-1]
    return bool(suffixes) and suffixes[-1] in JSONL_SUFFIXES


def _iter_txt_tree(root: Path) -> Iterator[Document]:
    doc_id = 0
    for path in root.rglob("*.txt"):  # lazy generator over the tree
        if not path.is_file():
            continue
        text = _read_text_file(path)
        if text is None:
            continue
        yield Document(doc_id, path, text)
        doc_id += 1


def _iter_jsonl(path: Path, field: str = "text") -> Iterator[Document]:
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    doc_id = 0
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, start=1):  # the file is an iterator of lines
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                log.warning("%s:%d is not valid JSON - skipped", path, lineno)
                continue
            text = record.get(field) if isinstance(record, dict) else None
            if not isinstance(text, str):
                log.warning(
                    "%s:%d has no string field %r - skipped", path, lineno, field
                )
                continue
            yield Document(doc_id, path, text)
            doc_id += 1


def iter_documents(root: Path) -> Iterator[Document]:
    """Yield documents one at a time from a directory of .txt files or a .jsonl file."""
    root = Path(root)
    if root.is_dir():
        yield from _iter_txt_tree(root)
    elif root.is_file() and _is_jsonl(root):
        yield from _iter_jsonl(root)
    else:
        raise FileNotFoundError(
            f"{root}: expected a directory with .txt files or a .jsonl/.jsonl.gz file"
        )

"""Lab 8 A/B control: the *verbatim* Lab 3 snippet implementation.

This module is ``src/findex/snippet.py`` as it stood at the end of Lab 7, kept
unchanged so ``bench/lab8.py`` can measure old and new in one process, on the
same corpus, with the same warm-up - the only honest way to attribute a
speedup. Nothing imports it; ``tests/test_bench_perf.py`` does not either. It
exists to be the "before" column and to prove byte-identical output:

    legacy.snippet(index, doc_id, terms) == findex.snippet.snippet(...)

Its cache is separate from the shipped one (``lru_cache`` is per function), so
warming one does not warm the other.
"""

from __future__ import annotations

import functools
import re
from collections.abc import Iterable
from pathlib import Path

from findex.index import Index

#: Stop scanning for a better window after this many matches - "the" occurs
#: hundreds of thousands of times in this corpus and the tenth window is not
#: meaningfully better than the two-hundredth.
_MAX_CANDIDATES = 200


@functools.lru_cache(maxsize=8)
def _read_document(path: str) -> str:
    """Read and whitespace-normalise a document; ``""`` if it vanished.

    ``lru_cache`` pays off when ten hits come from the same book (a 5 MB file
    re-read for every result is the slowest part of a query); the tiny maxsize
    keeps the memory bill bounded.

    The size is deliberately small: lab 3 answers *one* query per process, and
    eight normalised books is already tens of megabytes. A server answers
    thousands of queries over the same corpus, so the web layer calls
    :func:`configure_text_cache` once at startup and raises it to the number of
    documents it actually serves.
    """
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return " ".join(text.split())


def configure_text_cache(maxsize: int) -> int:
    """Replace the document-text cache with one of ``maxsize`` entries.

    Returns the previous size so a caller can restore it (the tests do). The
    old cache is dropped whole rather than resized - ``lru_cache`` cannot change
    its bound in place - which is exactly what a server wants at startup when
    the cache is empty anyway.

    Why this exists: normalising a 5.3 MB book costs ~217 ms and scanning it for
    matches another ~46 ms, so re-reading the same ten books for every request
    is the single largest cost of ``GET /search`` (see the Lab 7 README section).
    Raising the bound from 8 to the corpus size removes the first half.
    """
    global _read_document
    previous = _read_document.cache_info().maxsize or 0
    inner = getattr(_read_document, "__wrapped__", None)
    if inner is None:  # pragma: no cover - only if someone replaced the function
        return previous
    _read_document = functools.lru_cache(maxsize=maxsize)(inner)  # type: ignore[assignment]
    return previous


def _pattern(terms: Iterable[str]) -> re.Pattern[str] | None:
    """One alternation matching any query term on word boundaries.

    ``(?<![^\\W_])`` / ``(?![^\\W_])`` are "not a letter or digit before/after":
    ``war`` then does not match inside ``warning``, while ``don't`` and
    ``well-known`` still match as whole tokens, because those characters are
    inside the term, not at its edge.
    """
    unique = sorted({term for term in terms if term}, key=len, reverse=True)
    if not unique:
        return None
    alternation = "|".join(re.escape(term) for term in unique)
    return re.compile(rf"(?<![^\W_])(?:{alternation})(?![^\W_])", re.IGNORECASE)


def _window(
    text: str, pattern: re.Pattern[str], width: int
) -> tuple[int, int] | None:
    """Pick the window covering the most *distinct* query terms, or None."""
    best_score = (-1, 0)
    best: tuple[int, int] | None = None
    for count, match in enumerate(pattern.finditer(text)):
        if count >= _MAX_CANDIDATES:
            break
        start = max(0, match.start() - width)
        end = min(len(text), match.end() + width)
        inside = text[start:end]
        distinct = len({m.group(0).casefold() for m in pattern.finditer(inside)})
        score = (distinct, -start)
        if score > best_score:
            best_score, best = score, (start, end)
    return best


def _cut_on_words(text: str, start: int, end: int) -> tuple[int, int]:
    """Move the edges outwards to the nearest space so no word is chopped."""
    if start > 0:
        space = text.find(" ", start)
        if space != -1:
            start = space + 1
    if end < len(text):
        space = text.rfind(" ", start, end)
        if space != -1:
            end = space
    return start, end


def mark_terms(
    text: str, terms: Iterable[str], *, mark: tuple[str, str] = ("[", "]")
) -> str:
    """Bracket every whole-word occurrence of ``terms`` in an arbitrary text.

    ``snippet()`` marks the ±80-character window it chose; the web layer needs
    the same substitution over a **whole document** (the ``/docs/{id}`` page),
    so the marking rule lives here once instead of being re-implemented with a
    different regex at the edge. Same word boundaries, same case-insensitivity,
    same "no match means the text is returned unchanged".
    """
    pattern = _pattern(terms)
    if pattern is None:
        return text
    return pattern.sub(lambda m: f"{mark[0]}{m.group(0)}{mark[1]}", text)


def snippet(
    index: Index,
    doc_id: int,
    terms: Iterable[str],
    *,
    width: int = 80,
    mark: tuple[str, str] = ("[", "]"),
) -> str:
    """Return ``doc_id``'s best ±``width``-character window with terms marked.

    Returns ``""`` when the document cannot be read, and an unmarked opening of
    the document when nothing matches (an empty body is still a body).
    """
    text = _read_document(index.doc_meta[doc_id].path)
    if not text:
        return ""

    opening = text[: 2 * width] + ("…" if len(text) > 2 * width else "")
    pattern = _pattern(terms)
    if pattern is None:
        return opening
    found = _window(text, pattern, width)
    if found is None:
        return opening

    start, end = _cut_on_words(text, *found)
    window = text[start:end]
    highlighted = pattern.sub(
        lambda m: f"{mark[0]}{m.group(0)}{mark[1]}", window
    )
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    return f"{prefix}{highlighted}{suffix}"

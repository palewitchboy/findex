"""Lab 8 - snippets: the window is found by offsets, not by re-scanning the book.

A ranked list without context is a list of titles; users read the snippet to
decide whether a hit is worth opening. The rule here is the classic one: find
where the query matches, take ±``width`` characters around it, cut on word
boundaries and mark every query term inside the window.

Highlighting happens **in the text**, with plain ASCII markers, not in HTML:
the snippet is a string that a terminal, a test or a JSON file can all use, and
nothing has to be escaped. Swap ``mark=("[", "]")`` for ``("**", "**")`` if it
is going into markdown.

What Lab 3 did, and what Lab 8 measured
---------------------------------------
Lab 3 scanned the whole book with ``re`` for every hit: a 5.4 MB novel and a
three-term query cost ~345 ms of ``re.finditer`` because ``IGNORECASE`` case
folds every character it walks past, plus ~196 ms to normalise the whitespace,
plus the regex run again over each of 200 candidate windows. Ten hits therefore
cost ~540 ms while the ranking itself cost 0.46 ms - 99.9% of the query was the
snippet (Lab 7's README section is the measurement that found this).

Lab 8 keeps the *result* identical and changes the mechanism, three times:

1. **One casefolded twin per document.** ``str.find`` is a two-way substring
   search in C: 2.6 ms for a full pass over the same 5.4 MB, ~130x faster than
   ``re.finditer`` with ``IGNORECASE``. The twin is built on the document's
   *second* snippet request and lives next to the normalised text in the cache,
   so a one-shot CLI query pays nothing for it and a server pays it once.
   Folding is per character, so when it does not change the length the offsets
   are the same in both strings; when it does (``ß`` -> ``ss``) the twin is
   dropped and the module falls back to the regex - correctness before speed.
2. **An offsets memo per (document, term).** The first question about a term
   costs one ``find`` pass; every later question about the same term costs a
   dict lookup. That is option (b) of the lab brief - cache the split with
   offsets at first read - without paying for a token table the query never
   touches.
3. **Word boundaries as a predicate, not as a lookaround.** ``(?<![^\\W_])``
   says "the neighbour is not a letter or a digit", which is exactly
   ``str.isalnum()``; two character checks are ~40x cheaper than letting the
   regex engine build a match object for each of 200 candidate windows.

Why not store positions in the index (option a of the brief)? ``bench/lab8.py``
measures it: token positions cost +47% of the index file on this corpus, and
they are *token* offsets, not character offsets - a snippet would still have to
map token -> character, i.e. read and normalise the book anyway. The cache
solves the cheap case (a server answering many queries over a static corpus);
positions solve the cold case. The lab reports both numbers.
"""

from __future__ import annotations

import functools
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from findex.index import Index

#: Stop scanning for a better window after this many matches - "the" occurs
#: hundreds of thousands of times in this corpus and the tenth window is not
#: meaningfully better than the two-hundredth.
_MAX_CANDIDATES = 200

#: How many documents the text cache holds by default. Lab 3 used 8 and the
#: measurement found the bug: ``k=10`` snippets from ten different books
#: thrash an 8-slot cache, so every query re-read and re-normalised two books
#: (~390 ms). 32 is a server-sized default; ``configure_text_cache()`` and
#: ``TEXT_CACHE`` still override it.
_DEFAULT_CACHE = 32

#: How many terms per document the offsets memo keeps. A query touches a
#: handful of terms; a long-lived server over a fixed corpus would otherwise
#: grow one tuple per (document, term) it ever saw.
_OFFSET_MEMO = 64


@dataclass(slots=True)
class _Document:
    """A cached document: normalised text, its casefolded twin, the offset memo.

    ``folded`` is built **on the second snippet request** for this document, not
    on the first: a CLI answers one query per process and asks each hit's book
    exactly once, so folding 5.4 MB to save a 7 ms regex would be a 70 ms loss.
    A server asks the same book again, and from then on the twin pays for
    itself. ``None`` means "not built yet", ``""`` means "folding moved the
    offsets" (``ß`` -> ``ss``, the Kelvin sign, ligatures) - then the module
    falls back to the regex rather than returning a window that is off by
    however many characters the fold added.
    """

    text: str
    folded: str | None = None
    uses: int = 0
    offsets: dict[str, tuple[int, ...]] = field(
        default_factory=dict[str, tuple[int, ...]]
    )

    def twin(self) -> str | None:
        """The casefolded text, or ``None`` when ``find`` offsets would lie.

        Built once per document. The length check is the whole correctness
        argument: folding is a per-character map, so equal lengths mean equal
        offsets, and that is what lets one ``find`` index serve both strings.
        """
        if self.folded is None:
            folded = self.text.casefold()
            self.folded = folded if len(folded) == len(self.text) else ""
        return self.folded or None


class _TextCache(Protocol):
    """The slice of ``functools.lru_cache`` this module needs.

    Declaring the protocol is what lets :func:`configure_text_cache` rebind the
    cache without a ``# type: ignore``: the decorated function is a
    ``_lru_cache_wrapper``, a private generic type that strict pyright refuses
    to name, but it satisfies this shape structurally.
    """

    def __call__(self, path: str) -> _Document | None: ...

    def cache_info(self) -> Any: ...

    def cache_clear(self) -> None: ...


def _new_cache(maxsize: int) -> _TextCache:
    """Build the document cache; the only place that knows about ``lru_cache``."""
    cached = functools.lru_cache(maxsize=maxsize)(_load)
    return cached


def _load(path: str) -> _Document | None:
    """Read and whitespace-normalise one document; ``None`` if it vanished.

    ``lru_cache`` pays off when ten hits come from the same book (a 5 MB file
    re-read for every result is the slowest part of a query); the bound keeps
    the memory bill finite. Only the normalised text is built here: the
    casefolded twin arrives on the document's second use (see ``_Document``),
    so a one-shot CLI query never pays for it.

    The size is deliberately modest: a CLI answers *one* query per process. A
    server answers thousands of queries over the same corpus, so the web layer
    calls :func:`configure_text_cache` once at startup and raises it to the
    number of documents it actually serves.
    """
    try:
        raw = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    return _Document(text=" ".join(raw.split()))


_load_document: _TextCache = _new_cache(_DEFAULT_CACHE)


def configure_text_cache(maxsize: int) -> int:
    """Replace the document cache with one of ``maxsize`` entries.

    Returns the previous size so a caller can restore it (the tests do). The
    old cache is dropped whole rather than resized - ``lru_cache`` cannot change
    its bound in place - which is exactly what a server wants at startup when
    the cache is empty anyway.

    Why this exists: normalising a 5.3 MB book costs ~196 ms and scanning it for
    matches another ~345 ms (Lab 3), so re-reading the same ten books for every
    request is the single largest cost of ``GET /search``. Raising the bound
    from 8 to the corpus size removes both.
    """
    global _load_document
    previous = int(_load_document.cache_info().maxsize or 0)
    _load_document = _new_cache(maxsize)
    return previous


def _pattern(terms: Iterable[str]) -> re.Pattern[str] | None:
    """One alternation matching any query term on word boundaries.

    ``(?<![^\\\\W_])`` / ``(?![^\\\\W_])`` are "not a letter or digit before/after":
    ``war`` then does not match inside ``warning``, while ``don't`` and
    ``well-known`` still match as whole tokens, because those characters are
    inside the term, not at its edge.
    """
    unique = sorted({term for term in terms if term}, key=len, reverse=True)
    if not unique:
        return None
    alternation = "|".join(re.escape(term) for term in unique)
    return re.compile(rf"(?<![^\W_])(?:{alternation})(?![^\W_])", re.IGNORECASE)


# --------------------------------------------------------------------------- offsets
def _fold_safe(term: str) -> bool:
    """True when ``term.casefold()`` has the same length, so offsets survive."""
    return len(term.casefold()) == len(term)


def _bounded(text: str, start: int, end: int, *, at_start: int, at_end: int) -> bool:
    """Word boundaries of ``text[start:end]``, measured inside a slice.

    The regex ran over ``text[at_start:at_end]``, so a match touching the slice
    edge counts as bounded: the slice simply has no character there. Reproducing
    that detail is what keeps the new windows byte-identical to Lab 3's.
    """
    before_ok = start == at_start or not text[start - 1].isalnum()
    after_ok = end == at_end or not text[end].isalnum()
    return before_ok and after_ok


def _scan(text: str, term: str, limit: int) -> tuple[int, ...]:
    """Character offsets of ``term`` as a whole word, left to right.

    ``str.find`` is the C two-way search: 2.6 ms for a full pass over 5.4 MB
    against 345 ms for ``re.finditer`` with ``IGNORECASE`` on the same text.
    The loop stops at ``limit`` matches, so a common term costs far less than a
    full pass.
    """
    found: list[int] = []
    size = len(text)
    width = len(term)
    position = text.find(term)
    while position != -1:
        end = position + width
        before_ok = position == 0 or not text[position - 1].isalnum()
        after_ok = end == size or not text[end].isalnum()
        if before_ok and after_ok:
            found.append(position)
            if len(found) >= limit:
                break
        position = text.find(term, position + 1)
    return tuple(found)


def _occurrences(folded: str, document: _Document, term: str) -> tuple[int, ...]:
    """The memoised offsets of ``term`` in ``document`` - one scan, then a dict."""
    cached = document.offsets.get(term)
    if cached is None:
        if len(document.offsets) >= _OFFSET_MEMO:
            document.offsets.clear()
        cached = _scan(folded, term, _MAX_CANDIDATES)
        document.offsets[term] = cached
    return cached


def _candidates(
    folded: str, document: _Document, terms: Sequence[str]
) -> list[tuple[int, int]]:
    """Spans of the first ``_MAX_CANDIDATES`` matches, in the order ``re`` found them.

    ``finditer`` walks left to right and, at one position, prefers the longest
    alternative (``_pattern`` sorts the terms by length). Sorting by
    ``(start, -length)`` and dropping anything that overlaps the previous span
    reproduces that exactly - the one place where a plain per-term scan and the
    alternation could disagree.
    """
    spans: list[tuple[int, int]] = []
    for term in terms:
        width = len(term)
        spans.extend(
            (start, start + width) for start in _occurrences(folded, document, term)
        )
    spans.sort(key=lambda span: (span[0], span[0] - span[1]))

    merged: list[tuple[int, int]] = []
    last_end = -1
    for start, end in spans:
        if start < last_end:  # already covered by an earlier, longer match
            continue
        merged.append((start, end))
        last_end = end
        if len(merged) >= _MAX_CANDIDATES:
            break
    return merged


def _distinct_in(text: str, terms: Sequence[str], start: int, end: int) -> int:
    """How many distinct query terms have a whole-word match inside the window.

    This is Lab 3's
    ``len({m.group(0).casefold() for m in pattern.finditer(inside)})``, computed
    with ``str.find`` over ~160 characters instead of a regex: 200 candidate
    windows cost 0.2 ms instead of 7.6 ms.
    """
    matched: set[str] = set()
    for term in terms:
        position = text.find(term, start, end)
        while position != -1:
            stop = position + len(term)
            if stop <= end and _bounded(
                text, position, stop, at_start=start, at_end=end
            ):
                matched.add(term)
                break
            position = text.find(term, position + 1, end)
    return len(matched)


# --------------------------------------------------------------------------- windows
def _window_by_offsets(
    document: _Document, folded: str, terms: Sequence[str], width: int
) -> tuple[int, int] | None:
    """Pick the window covering the most distinct terms, from the offset memo."""
    text = document.text
    best_score = (-1, 0)
    best: tuple[int, int] | None = None
    for match_start, match_end in _candidates(folded, document, terms):
        start = max(0, match_start - width)
        end = min(len(text), match_end + width)
        score = (_distinct_in(folded, terms, start, end), -start)
        if score > best_score:
            best_score, best = score, (start, end)
    return best


def _window_by_regex(
    text: str, pattern: re.Pattern[str], width: int
) -> tuple[int, int] | None:
    """Lab 3's scan, kept as the fallback when case folding moved the offsets."""
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


def _window(
    document: _Document, terms: Sequence[str], width: int, pattern: re.Pattern[str]
) -> tuple[int, int] | None:
    """Best window for ``terms``: the fast path, or the regex when folding moved."""
    folded = [term.casefold() for term in terms if term]
    twin = document.twin() if document.uses > 1 else None
    if twin is not None and folded and all(_fold_safe(term) for term in folded):
        # Deduplicate by folded form: two spellings of one term are one hit.
        return _window_by_offsets(
            document, twin, tuple(dict.fromkeys(folded)), width
        )
    return _window_by_regex(document.text, pattern, width)


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
    # Materialise once: ``terms`` may be a generator, and both the pattern and
    # the offset scan need to read it.
    vocabulary = tuple(terms)
    document = _load_document(index.doc_meta[doc_id].path)
    if document is None or not document.text:
        return ""
    document.uses += 1
    text = document.text

    opening = text[: 2 * width] + ("…" if len(text) > 2 * width else "")
    pattern = _pattern(vocabulary)
    if pattern is None:
        return opening
    found = _window(document, vocabulary, width, pattern)
    if found is None:
        return opening

    start, end = _cut_on_words(text, *found)
    window = text[start:end]
    highlighted = pattern.sub(lambda m: f"{mark[0]}{m.group(0)}{mark[1]}", window)
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    return f"{prefix}{highlighted}{suffix}"

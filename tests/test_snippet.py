"""Lab 3 - snippets: choosing a window, marking terms, degrading gracefully."""

from __future__ import annotations

from pathlib import Path

import pytest

from findex.index import DocMeta, Index, build_index
from findex.snippet import snippet


@pytest.fixture()
def long_corpus(tmp_path: Path) -> Path:
    body = " ".join(f"filler{i}" for i in range(400))
    (tmp_path / "long.txt").write_text(
        f"Title: Long\n\nbody begins. {body} the treasure lies under the old oak "
        f"near the river bank. {body} body ends.\n",
        encoding="utf-8",
    )
    return tmp_path


def test_snippet_marks_the_matched_term(corpus: Path) -> None:
    index = build_index(corpus)
    text = snippet(index, 0, {"fox"})
    assert "[fox]" in text
    assert text.lower().count("fox") >= 1


def test_snippet_is_case_insensitive(corpus: Path) -> None:
    index = build_index(corpus)
    assert "[fox]" in snippet(index, 0, {"FOX"})


def test_snippet_cuts_a_window_and_adds_ellipses(long_corpus: Path) -> None:
    index = build_index(long_corpus)
    text = snippet(index, 0, {"treasure"}, width=30)
    assert "[treasure]" in text
    assert text.startswith("…")
    assert text.endswith("…")
    assert len(text) < 200


def test_snippet_normalises_whitespace(long_corpus: Path) -> None:
    index = build_index(long_corpus)
    assert "\n" not in snippet(index, 0, {"treasure"})


def test_no_match_returns_the_opening(corpus: Path) -> None:
    index = build_index(corpus)
    text = snippet(index, 0, {"zzz"})
    assert text.startswith("Title: Alpha")
    assert "[" not in text


def test_empty_terms_return_the_opening(corpus: Path) -> None:
    index = build_index(corpus)
    assert snippet(index, 0, set()).startswith("Title: Alpha")


def test_window_prefers_the_spot_with_more_distinct_terms(long_corpus: Path) -> None:
    index = build_index(long_corpus)
    text = snippet(index, 0, {"treasure", "oak", "river"}, width=40)
    assert "[treasure]" in text
    assert "[oak]" in text
    assert "[river]" in text


def test_missing_file_returns_empty_string(corpus: Path) -> None:
    index = build_index(corpus)
    index.doc_meta[0] = DocMeta(0, "does/not/exist.txt", "Gone", 10)
    assert snippet(index, 0, {"fox"}) == ""


def test_custom_markers(corpus: Path) -> None:
    index = build_index(corpus)
    assert "**fox**" in snippet(index, 0, {"fox"}, mark=("**", "**"))


def test_short_document_has_no_ellipsis(corpus: Path) -> None:
    index = build_index(corpus)
    text = snippet(index, 0, {"fox"})
    assert not text.startswith("…")


def test_snippet_does_not_match_inside_a_word(tmp_path: Path) -> None:
    (tmp_path / "war.txt").write_text(
        "Title: War\n\nwarfare is not war, but warning is not either.\n",
        encoding="utf-8",
    )
    index = build_index(tmp_path)
    text = snippet(index, 0, {"war"})
    assert "[war]" in text
    assert "[war]fare" not in text
    assert "[war]ning" not in text


def test_snippet_works_on_a_hand_built_index(tmp_path: Path) -> None:
    path = tmp_path / "doc.txt"
    path.write_text("Title: X\n\nnothing to see here", encoding="utf-8")
    index = Index(
        postings={},
        doc_lengths={0: 4},
        doc_meta={0: DocMeta(0, str(path), "X", 4)},
        n_tokens=4,
    )
    assert "[see]" in snippet(index, 0, {"see"})


def test_snippet_accepts_a_generator_of_terms(corpus: Path) -> None:
    """``terms`` is an ``Iterable``: a generator must not be consumed twice.

    The fast path reads the vocabulary once for the pattern and once for the
    offset scan; a generator emptied by the first read would silently drop to
    the regex path (and, before the fix, to an empty vocabulary).
    """
    index = build_index(corpus)
    terms = (term for term in ["fox"])
    assert "[fox]" in snippet(index, 0, terms)


def test_offset_path_matches_the_regex_path_on_a_second_use(corpus: Path) -> None:
    """First use walks the regex, the second walks offsets; both must agree."""
    index = build_index(corpus)
    first = snippet(index, 0, {"fox", "dog"})
    second = snippet(index, 0, {"fox", "dog"})
    assert first == second
    assert "[fox]" in second


def test_case_folding_that_changes_length_falls_back_to_regex(tmp_path: Path) -> None:
    """``ß`` folds to ``ss`` and moves every offset; the module must notice."""
    (tmp_path / "de.txt").write_text(
        "Title: Strasse\n\ndie Straße ist lang und die Strasse auch\n",
        encoding="utf-8",
    )
    index = build_index(tmp_path)
    # Twice: the first call uses the regex, the second would use offsets.
    text = snippet(index, 0, {"strasse"})
    assert snippet(index, 0, {"strasse"}) == text
    assert "Strasse" in text or "Straße" in text

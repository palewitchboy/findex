"""Checks for the tokenizer and the lazy corpus stream (plain asserts)."""

import inspect
from pathlib import Path

from findex.corpus import iter_documents
from findex.tokenize import tokenize


def toks(text: str) -> list[str]:
    return list(tokenize(text))


def test_is_generator():
    assert inspect.isgeneratorfunction(tokenize)
    assert inspect.isgeneratorfunction(iter_documents)


def test_mixed_case():
    assert toks("Hello WORLD Python") == ["hello", "world", "python"]


def test_cyrillic():
    assert toks("Привіт, Світе! Київ") == ["привіт", "світе", "київ"]


def test_combining_mark_matches_precomposed():
    assert toks("café") == toks("café") == ["café"]


def test_casefold_not_lower():
    assert toks("Straße") == toks("STRASSE") == ["strasse"]


def test_punctuation_is_separator():
    assert toks("one, two; three... (four)!") == ["one", "two", "three", "four"]


def test_apostrophes_inside_words_kept():
    assert toks("don't п'ять п’ять") == ["don't", "п'ять", "п'ять"]
    assert toks("'quoted'") == ["quoted"]


def test_hyphens_inside_words_kept():
    assert toks("well-known IT-фахівець - dash -x") == [
        "well-known",
        "it-фахівець",
        "dash",
        "x",
    ]


def test_digits_kept_underscore_splits():
    got = toks("python3 in 2026 snake_case")
    assert got == ["python3", "in", "2026", "snake", "case"]


def test_empty_string():
    assert toks("") == []
    assert toks("   ...  ") == []


def test_iter_documents_txt_and_bad_encoding(tmp_path: Path):
    (tmp_path / "a.txt").write_text("перший документ", encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.txt").write_bytes(b"bad \xff\xfe bytes")  # invalid UTF-8
    (sub / "skip.md").write_text("not a txt", encoding="utf-8")

    it = iter_documents(tmp_path)
    first = next(it)  # lazy: works before the rest is read
    rest = list(it)
    docs = [first, *rest]

    assert len(docs) == 2
    assert sorted(d.doc_id for d in docs) == [0, 1]
    assert any("перший" in d.text for d in docs)


def test_iter_documents_jsonl(tmp_path: Path):
    f = tmp_path / "docs.jsonl"
    f.write_text('{"text": "one"}\nnot json\n{"text": "two"}\n', encoding="utf-8")
    assert [d.text for d in iter_documents(f)] == ["one", "two"]

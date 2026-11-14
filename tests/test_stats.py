"""Lab 4 - the Lab 1 pipeline and its numbers: lazy stats, eager comparison."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from findex.corpus import iter_documents
from findex.eager import eager_stats, markdown_table
from findex.stats import fmt_bytes, lazy_stats, measure


def test_lazy_stats_counts_the_fixture_corpus(corpus: Path) -> None:
    stats = lazy_stats(corpus, top=3)
    assert stats.documents == 3
    assert stats.tokens == 40
    assert stats.vocabulary == 23
    assert len(stats.top) == 3
    assert stats.top[0][0] in {"the", "fox"}


def test_lazy_stats_honours_limit(corpus: Path) -> None:
    assert lazy_stats(corpus, limit=2).documents == 2


def test_lazy_stats_top_is_bounded(corpus: Path) -> None:
    assert len(lazy_stats(corpus, top=1).top) == 1


def test_measure_reports_time_and_memory(corpus: Path) -> None:
    measurement = measure(lambda: lazy_stats(corpus))
    assert measurement.result.documents == 3
    assert measurement.elapsed >= 0
    assert measurement.peak_bytes > 0


def test_eager_and_lazy_agree(corpus: Path) -> None:
    assert eager_stats(corpus) == lazy_stats(corpus)


def test_eager_honours_limit(corpus: Path) -> None:
    assert eager_stats(corpus, limit=2) == lazy_stats(corpus, limit=2)


def test_markdown_table_has_both_versions(corpus: Path) -> None:
    eager = measure(lambda: eager_stats(corpus))
    lazy = measure(lambda: lazy_stats(corpus))
    table = markdown_table(eager, lazy)
    assert "| eager (lists) |" in table
    assert "| lazy (generators) |" in table


@pytest.mark.parametrize(
    ("size", "expected"),
    [(0, "0.0 B"), (512, "512.0 B"), (2048, "2.0 KiB")],
)
def test_fmt_bytes(size: int, expected: str) -> None:
    assert fmt_bytes(size) == expected


def test_jsonl_stream_skips_bad_lines(tmp_path: Path) -> None:
    path = tmp_path / "docs.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"text": "first document"}),
                "",
                "not json at all",
                json.dumps({"body": "no text field"}),
                json.dumps({"text": "second document"}),
            ]
        ),
        encoding="utf-8",
    )
    documents = list(iter_documents(path))
    assert [doc.text for doc in documents] == ["first document", "second document"]
    assert [doc.doc_id for doc in documents] == [0, 1]


def test_gzipped_jsonl_is_read(tmp_path: Path) -> None:
    import gzip

    path = tmp_path / "docs.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps({"text": "compressed document"}) + "\n")
    documents = list(iter_documents(path))
    assert [doc.text for doc in documents] == ["compressed document"]


def test_unknown_root_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "notes.md"
    path.write_text("not a corpus", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        list(iter_documents(path))

"""Lab 4 - the typer CLI: commands, JSON, exit codes, logging."""

from __future__ import annotations

import importlib.metadata
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from findex import cli
from findex.cli import app, index_main, search_main, stats_main
from findex.store import load, storage_of

runner = CliRunner()
SRC = Path(__file__).resolve().parents[1] / "src"


def invoke(*args: str) -> Any:
    return runner.invoke(app, list(args))


@pytest.fixture()
def built(corpus: Path, tmp_path: Path) -> Path:
    """A pickle index over the three-document fixture corpus."""
    out = tmp_path / "idx.pkl"
    result = invoke("index", str(corpus), "--out", str(out), "--json")
    assert result.exit_code == 0, result.stderr
    return out


# --------------------------------------------------------------------------- root
def test_help_lists_the_three_commands() -> None:
    result = invoke("--help")
    assert result.exit_code == 0
    for command in ("index", "search", "stats"):
        assert command in result.stdout


def test_version_option() -> None:
    result = invoke("--version")
    assert result.exit_code == 0
    assert result.stdout.startswith("findex ")


def test_version_falls_back_to_source_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", explode)
    assert cli._version() == "0.8.0"


# --------------------------------------------------------------------------- index
def test_index_writes_a_file_and_reports_json(corpus: Path, tmp_path: Path) -> None:
    out = tmp_path / "idx.pkl"
    result = invoke("index", str(corpus), "--out", str(out), "--json")
    assert result.exit_code == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["documents"] == 3
    assert payload["terms"] > 0
    assert payload["positions"] is False
    assert payload["file_bytes"] == out.stat().st_size
    assert out.exists()


def test_index_positions_and_array_storage(corpus: Path, tmp_path: Path) -> None:
    out = tmp_path / "idx.pkl"
    result = invoke(
        "index",
        str(corpus),
        "--out",
        str(out),
        "--positions",
        "--storage",
        "arrays",
        "--limit",
        "2",
        "--json",
    )
    assert result.exit_code == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["positions"] is True
    assert payload["storage"] == "arrays"
    assert payload["documents"] == 2
    restored = load(out)
    assert restored.positions is not None
    assert storage_of(restored) == "arrays"


def test_index_json_format_loses_the_array_layout(corpus: Path, tmp_path: Path) -> None:
    """JSON is the portable format: postings become lists, arrays are gone."""
    out = tmp_path / "idx.json"
    result = invoke(
        "index",
        str(corpus),
        "--out",
        str(out),
        "--positions",
        "--storage",
        "arrays",
        "--format",
        "json",
        "--json",
    )
    assert result.exit_code == 0, result.stderr
    restored = load(out)
    assert restored.positions is not None
    assert storage_of(restored) == "objects"


def test_index_human_output_is_a_summary(corpus: Path, tmp_path: Path) -> None:
    out = tmp_path / "idx.pkl"
    result = invoke("index", str(corpus), "--out", str(out))
    assert result.exit_code == 0
    assert "documents" in result.stdout
    assert "build time" in result.stdout
    assert "saved" in result.stdout


def test_index_cannot_write_to_a_directory(corpus: Path, tmp_path: Path) -> None:
    result = invoke("index", str(corpus), "--out", str(tmp_path))
    assert result.exit_code == 1
    assert "error:" in result.stderr


def test_index_rejects_a_plain_txt_file(tmp_path: Path) -> None:
    path = tmp_path / "single.txt"
    path.write_text("Title: One\n\nfox", encoding="utf-8")
    result = invoke("index", str(path))
    assert result.exit_code == 1
    assert "error:" in result.stderr


# --------------------------------------------------------------------------- search
def test_search_json_is_machine_readable(built: Path) -> None:
    result = invoke("search", str(built), "fox", "--json")
    assert result.exit_code == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["query"] == "fox"
    assert payload["mode"] == "ranked"
    assert payload["count"] == len(payload["results"]) == 3
    scores = [hit["score"] for hit in payload["results"]]
    assert scores == sorted(scores, reverse=True)
    assert "[fox]" in payload["results"][0]["snippet"]
    assert payload["results"][0]["title"] == "Alpha"


def test_search_human_output_is_a_table(built: Path) -> None:
    result = invoke("search", str(built), "fox")
    assert result.exit_code == 0
    assert "hit(s)" in result.stdout
    assert "Alpha" in result.stdout
    assert "score" in result.stdout


def test_search_scorer_choice_is_honoured(built: Path) -> None:
    result = invoke("search", str(built), "fox", "--scorer", "tfidf", "--json")
    assert result.exit_code == 0, result.stderr
    assert json.loads(result.stdout)["scorer"] == "tfidf"


def test_search_explain_is_inside_the_json(built: Path) -> None:
    result = invoke("search", str(built), "fox", "--explain", "--json")
    payload = json.loads(result.stdout)
    assert payload["explain"].startswith("tree    :")


def test_search_explain_is_printed_in_human_mode(built: Path) -> None:
    result = invoke("search", str(built), "fox", "--explain")
    assert result.exit_code == 0
    assert "tree    :" in result.stdout
    assert "df=" in result.stdout


def test_search_boolean_json(built: Path) -> None:
    result = invoke("search", str(built), "fox NOT quick", "--boolean", "--json")
    assert result.exit_code == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["mode"] == "boolean"
    assert payload["engine"] == "merge"
    assert [hit["doc_id"] for hit in payload["results"]] == [1]
    assert payload["results"][0]["tf"] == {"fox": 1}


def test_search_boolean_human_table(built: Path) -> None:
    result = invoke("search", str(built), "fox OR rabbits", "--boolean")
    assert result.exit_code == 0
    assert "boolean" in result.stdout
    assert "Beta" in result.stdout


def test_search_boolean_explain_json(built: Path) -> None:
    result = invoke(
        "search", str(built), "fox OR rabbits", "--boolean", "--explain", "--json"
    )
    payload = json.loads(result.stdout)
    assert payload["explain"].startswith("group 1:")


def test_search_no_snippets(built: Path) -> None:
    result = invoke("search", str(built), "fox", "--no-snippets", "--json")
    assert json.loads(result.stdout)["results"][0]["snippet"] == ""


def test_search_caps_hits_with_k(built: Path) -> None:
    result = invoke("search", str(built), "fox", "--k", "1", "--json")
    assert json.loads(result.stdout)["count"] == 1


def test_missing_index_exits_1(tmp_path: Path) -> None:
    result = invoke("search", str(tmp_path / "nope.pkl"), "fox")
    assert result.exit_code == 1
    assert "index not found" in result.stderr


def test_corrupt_index_exits_1(tmp_path: Path) -> None:
    path = tmp_path / "garbage.bin"
    path.write_bytes(b"?not an index")
    result = invoke("search", str(path), "fox")
    assert result.exit_code == 1
    assert "cannot read index" in result.stderr


def test_empty_query_exits_2(built: Path) -> None:
    result = invoke("search", str(built), "   ")
    assert result.exit_code == 2
    assert "empty query" in result.stderr


def test_bad_syntax_exits_2(built: Path) -> None:
    result = invoke("search", str(built), "(")
    assert result.exit_code == 2
    assert "error:" in result.stderr


def test_phrase_without_positions_exits_2(built: Path) -> None:
    result = invoke("search", str(built), '"quick brown"')
    assert result.exit_code == 2
    assert "positions" in result.stderr


def test_unknown_scorer_exits_2(built: Path) -> None:
    result = invoke("search", str(built), "fox", "--scorer", "cosine")
    assert result.exit_code == 2


def test_k_must_be_positive(built: Path) -> None:
    result = invoke("search", str(built), "fox", "--k", "0")
    assert result.exit_code == 2


# --------------------------------------------------------------------------- stats
def test_stats_on_a_corpus_json(corpus: Path) -> None:
    result = invoke("stats", str(corpus), "--top", "3", "--json")
    assert result.exit_code == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["mode"] == "corpus"
    assert payload["documents"] == 3
    assert payload["tokens"] > 0
    assert len(payload["top"]) == 3


def test_stats_on_an_index_json(built: Path) -> None:
    result = invoke("stats", str(built), "--json")
    assert result.exit_code == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["mode"] == "index"
    assert payload["documents"] == 3
    assert payload["storage"] == "objects"
    assert payload["top"]


def test_stats_human_tables(corpus: Path) -> None:
    result = invoke("stats", str(corpus))
    assert result.exit_code == 0
    assert "documents" in result.stdout
    assert "top 10 terms" in result.stdout


def test_stats_human_on_an_index(built: Path) -> None:
    result = invoke("stats", str(built), "--top", "2")
    assert result.exit_code == 0
    assert "postings" in result.stdout
    assert "avg doc length" in result.stdout
    assert "top 2 terms" in result.stdout


def test_stats_missing_path_is_a_usage_error() -> None:
    result = invoke("stats", "definitely-not-here")
    assert result.exit_code == 2


# --------------------------------------------------------------------------- shims
def test_index_main_shim(corpus: Path, tmp_path: Path) -> None:
    out = tmp_path / "shim.pkl"
    with pytest.raises(SystemExit) as excinfo:
        index_main([str(corpus), "--out", str(out)])
    assert excinfo.value.code == 0
    assert out.exists()


def test_search_main_shim(built: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        search_main([str(built), "fox", "--json"])
    assert excinfo.value.code == 0
    assert json.loads(capsys.readouterr().out)["count"] == 3


def test_stats_main_shim(corpus: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        stats_main([str(corpus), "--json"])
    assert excinfo.value.code == 0
    assert json.loads(capsys.readouterr().out)["documents"] == 3


def test_python_dash_m_findex_runs_the_same_app() -> None:
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    result = subprocess.run(
        [sys.executable, "-m", "findex", "--version"],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0
    assert result.stdout.startswith("findex ")


def test_verbose_logs_go_to_stderr_and_json_to_stdout(built: Path) -> None:
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    result = subprocess.run(
        [sys.executable, "-m", "findex", "-vv", "search", str(built), "fox", "--json"],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0
    payload = json.loads(result.stdout)  # stdout is pure JSON
    assert payload["count"] == 3
    assert "DEBUG" in result.stderr
    assert "took" in result.stderr


# --------------------------------------------------------------------------- logging
def test_configure_logging_is_idempotent() -> None:
    cli.configure_logging(1)
    first = list(logging.getLogger().handlers)
    cli.configure_logging(2)
    assert logging.getLogger().level == logging.DEBUG
    assert first  # a handler was installed
    assert logging.getLogger().handlers == first  # ...and not installed twice

"""Lab 5 - parallel index construction: identity, spawn safety, worker errors."""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import pytest
import typer.main
from typer.testing import CliRunner

from findex.cli import app
from findex.index import (
    DEFAULT_EXECUTOR,
    EXECUTORS,
    PartialIndex,
    build_index,
    build_partial,
    build_partials,
    default_workers,
    document_paths,
    merge,
    validate_workers,
)
from findex.store import load

runner = CliRunner()
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

EXECUTOR_CASES = [("threads", 2), ("processes", 2)]


def invoke(*args: str) -> Any:
    return runner.invoke(app, list(args))


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    return subprocess.run(
        [sys.executable, "-m", "findex", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )


# --------------------------------------------------------------------------- identity
@pytest.mark.parametrize(("executor", "workers"), EXECUTOR_CASES)
def test_parallel_build_is_byte_identical_to_serial(
    corpus: Path, executor: str, workers: int
) -> None:
    serial = build_index(corpus)
    parallel = build_index(corpus, executor=executor, workers=workers)  # type: ignore[arg-type]
    assert parallel.to_dict() == serial.to_dict()
    assert pickle.dumps(parallel) == pickle.dumps(serial)


@pytest.mark.parametrize(("executor", "workers"), EXECUTOR_CASES)
def test_parallel_positions_are_identical_too(
    corpus: Path, executor: str, workers: int
) -> None:
    serial = build_index(corpus, positions=True)
    parallel = build_index(
        corpus, positions=True, executor=executor, workers=workers  # type: ignore[arg-type]
    )
    assert parallel.positions is not None
    assert parallel.positions == serial.positions
    assert pickle.dumps(parallel) == pickle.dumps(serial)


@pytest.mark.parametrize("storage", ["objects", "arrays"])
def test_parallel_honours_the_storage_layout(corpus: Path, storage: str) -> None:
    serial = build_index(corpus, storage=storage)  # type: ignore[arg-type]
    parallel = build_index(
        corpus, storage=storage, executor="processes", workers=2  # type: ignore[arg-type]
    )
    assert parallel.to_dict() == serial.to_dict()
    assert type(parallel.postings["fox"]) is type(serial.postings["fox"])


def test_merge_of_all_partials_equals_the_serial_build(corpus: Path) -> None:
    paths = document_paths(corpus)
    partials = [build_partial(str(path), doc_id) for doc_id, path in enumerate(paths)]
    assert merge(partials) == build_index(corpus)
    assert pickle.dumps(merge(partials)) == pickle.dumps(build_index(corpus))


def test_merge_concatenates_in_worker_order(corpus: Path) -> None:
    paths = document_paths(corpus)
    partials = [build_partial(str(path), doc_id) for doc_id, path in enumerate(paths)]
    merged = merge(partials)
    assert list(merged.doc_ids("fox")) == [0, 1, 2]
    assert list(merged.doc_ids("dog")) == [0, 1]


# ------------------------------------------------------------------ merge edges
@pytest.mark.parametrize("storage", ["objects", "arrays"])
def test_merge_of_nothing_is_an_empty_index(storage: str) -> None:
    index = merge([], storage=storage)  # type: ignore[arg-type]
    assert len(index) == 0
    assert index.num_docs == 0
    assert index.n_tokens == 0
    assert index.positions is None
    assert index.to_dict()["postings"] == {}


def test_merge_rejects_overlapping_doc_ids(corpus: Path) -> None:
    paths = document_paths(corpus)
    first = build_partial(str(paths[0]), 0)
    second = build_partial(str(paths[1]), 0)  # same id on purpose
    with pytest.raises(ValueError, match="duplicate doc_id 0"):
        merge([first, second])


def test_merge_rejects_an_unknown_storage() -> None:
    with pytest.raises(ValueError, match="unknown storage"):
        merge([], storage="columns")  # type: ignore[arg-type]


def test_merge_keeps_positions_when_partials_have_them(corpus: Path) -> None:
    paths = document_paths(corpus)
    partials = [
        build_partial(str(path), doc_id, positions=True)
        for doc_id, path in enumerate(paths)
    ]
    merged = merge(partials)
    assert merged.positions is not None
    assert merged.positions["fox"][0] == (5, 12)


# --------------------------------------------------------------------------- partials
def test_build_partial_returns_one_document(corpus: Path) -> None:
    partial = build_partial(str(corpus / "alpha.txt"), 7)
    assert isinstance(partial, PartialIndex)
    assert partial.doc_lengths == {7: 15}
    assert partial.doc_meta[7].title == "Alpha"
    assert partial.n_tokens == 15
    assert partial.positions is None
    assert partial.postings["fox"] == [(7, 2)]
    ids = {doc_id for pairs in partial.postings.values() for doc_id, _ in pairs}
    assert ids == {7}


def test_build_partial_is_deterministic(corpus: Path) -> None:
    left = build_partial(str(corpus / "beta.txt"), 1)
    right = build_partial(corpus / "beta.txt", 1)
    assert pickle.dumps(left) == pickle.dumps(right)


def test_build_partial_reports_the_path_it_could_not_read(tmp_path: Path) -> None:
    missing = tmp_path / "nope.txt"
    with pytest.raises(OSError, match="nope.txt"):
        build_partial(str(missing), 0)


def test_build_partials_serial_matches_the_map_reduce_path(corpus: Path) -> None:
    paths = document_paths(corpus)
    tasks = [(str(path), doc_id) for doc_id, path in enumerate(paths)]
    assert merge(build_partials(tasks)) == build_index(corpus)


@pytest.mark.parametrize(
    ("executor", "workers"), [("threads", 2), ("processes", 2)]
)
def test_a_worker_exception_surfaces_with_the_worker_traceback(
    corpus: Path, executor: str, workers: int
) -> None:
    tasks = [(str(corpus / "alpha.txt"), 0), (str(corpus / "gone.txt"), 1)]
    with pytest.raises(OSError, match="gone.txt") as excinfo:
        build_partials(tasks, executor=executor, workers=workers)  # type: ignore[arg-type]
    rendered = "".join(traceback.format_exception(excinfo.value))
    # Threads re-raise in the caller with the worker's frames; processes wrap
    # the remote traceback in __cause__, which format_exception follows.
    assert "build_partial" in rendered
    assert "OSError" in rendered


def test_on_partial_reports_progress(corpus: Path) -> None:
    paths = document_paths(corpus)
    tasks = [(str(path), doc_id) for doc_id, path in enumerate(paths)]
    seen: list[int] = []
    partials = build_partials(
        tasks, executor="processes", workers=2, on_partial=seen.append
    )
    assert len(partials) == 3
    assert seen == [1, 2, 3]


# --------------------------------------------------------------------------- paths
def test_document_paths_is_sorted_and_honours_the_limit(corpus: Path) -> None:
    paths = document_paths(corpus)
    assert [path.name for path in paths] == ["alpha.txt", "beta.txt", "gamma.txt"]
    assert document_paths(corpus, limit=2) == paths[:2]
    assert document_paths(corpus) == document_paths(corpus)  # stable


def test_document_paths_rejects_a_streamed_corpus(tmp_path: Path) -> None:
    jsonl = tmp_path / "corpus.jsonl"
    jsonl.write_text('{"text": "one"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="directory"):
        document_paths(jsonl)
    with pytest.raises(ValueError, match="directory"):
        build_index(jsonl, executor="processes", workers=1)


def test_parallel_build_rejects_an_unknown_executor(corpus: Path) -> None:
    with pytest.raises(ValueError, match="unknown executor"):
        build_index(corpus, executor="fibers")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- workers
def test_default_workers_never_exceeds_the_cap() -> None:
    workers = default_workers()
    assert 1 <= workers <= 8


def test_validate_workers_resolves_defaults() -> None:
    assert validate_workers(None, "serial") == 1
    assert validate_workers(None, "processes") == default_workers()
    assert validate_workers(3, "processes") == 3


@pytest.mark.parametrize(
    ("workers", "executor", "message"),
    [
        (0, "processes", "workers must be >= 1"),
        (-2, "threads", "workers must be >= 1"),
        (4, "serial", "no effect with --executor serial"),
    ],
)
def test_validate_workers_rejects_nonsense(
    workers: int, executor: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_workers(workers, executor)  # type: ignore[arg-type]


def test_validate_workers_rejects_an_unknown_executor() -> None:
    with pytest.raises(ValueError, match="unknown executor"):
        validate_workers(2, "quantum")  # type: ignore[arg-type]


def test_workers_are_clamped_to_the_document_count(corpus: Path) -> None:
    index = build_index(corpus, executor="processes", workers=64)
    assert index.num_docs == 3


# --------------------------------------------------------------------------- cli
def test_cli_executor_flags_reach_the_build(corpus: Path, tmp_path: Path) -> None:
    out = tmp_path / "threads.pkl"
    result = invoke(
        "index",
        str(corpus),
        "--out",
        str(out),
        "--executor",
        "threads",
        "--workers",
        "2",
        "--json",
    )
    assert result.exit_code == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["executor"] == "threads"
    assert payload["workers"] == 2
    assert payload["documents"] == 3
    serial = tmp_path / "serial.pkl"
    assert invoke("index", str(corpus), "--out", str(serial), "--json").exit_code == 0
    assert load(out).to_dict() == load(serial).to_dict()


def test_cli_default_executor_is_the_measured_winner() -> None:
    """The default is a decision from bench/lab5.py, so assert it on the command.

    Checking the rendered --help text was brittle: on an 80-column CI terminal
    rich wraps the option table and the flags are no longer findable as strings.
    """
    assert DEFAULT_EXECUTOR in EXECUTORS
    command = typer.main.get_command(app).commands["index"]
    executor = next(p for p in command.params if "--executor" in p.opts)  # type: ignore[attr-defined]
    assert executor.default == DEFAULT_EXECUTOR  # type: ignore[attr-defined]


def test_cli_rejects_workers_with_serial(corpus: Path, tmp_path: Path) -> None:
    result = invoke(
        "index",
        str(corpus),
        "--out",
        str(tmp_path / "x.pkl"),
        "--executor",
        "serial",
        "--workers",
        "4",
    )
    assert result.exit_code == 2
    assert "no effect" in result.stderr


def test_cli_rejects_zero_workers(corpus: Path, tmp_path: Path) -> None:
    result = invoke(
        "index",
        str(corpus),
        "--out",
        str(tmp_path / "x.pkl"),
        "--workers",
        "0",
    )
    assert result.exit_code == 2


def test_cli_rejects_an_unknown_executor(corpus: Path, tmp_path: Path) -> None:
    result = invoke(
        "index",
        str(corpus),
        "--out",
        str(tmp_path / "x.pkl"),
        "--executor",
        "fibers",
    )
    assert result.exit_code == 2


def test_cli_positions_survive_a_process_build(corpus: Path, tmp_path: Path) -> None:
    out = tmp_path / "pos.pkl"
    result = invoke(
        "index",
        str(corpus),
        "--out",
        str(out),
        "--positions",
        "--executor",
        "processes",
        "--workers",
        "2",
        "--json",
    )
    assert result.exit_code == 0, result.stderr
    loaded = load(out)
    assert loaded.positions is not None
    assert loaded.positions["fox"][0] == (5, 12)


def test_cli_falls_back_to_serial_for_a_streamed_corpus(tmp_path: Path) -> None:
    """The default executor must not break `.jsonl`: there are no file chunks."""
    jsonl = tmp_path / "corpus.jsonl"
    jsonl.write_text(
        '{"text": "the quick brown fox"}\n{"text": "a lazy dog"}\n', encoding="utf-8"
    )
    out = tmp_path / "jsonl.pkl"
    result = invoke("index", str(jsonl), "--out", str(out), "--json")
    assert result.exit_code == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["executor"] == "serial"
    assert payload["documents"] == 2
    assert "using serial" in result.stderr


def test_cli_does_not_trace_memory_unless_asked(corpus: Path, tmp_path: Path) -> None:
    """``tracemalloc`` costs ~5x, so it must be opt-in for the index command."""
    result = invoke(
        "index",
        str(corpus),
        "--out",
        str(tmp_path / "plain.pkl"),
        "--executor",
        "serial",
        "--json",
    )
    assert result.exit_code == 0, result.stderr
    assert json.loads(result.stdout)["peak_bytes"] is None


def test_cli_trace_memory_reports_a_peak(corpus: Path, tmp_path: Path) -> None:
    result = invoke(
        "index",
        str(corpus),
        "--out",
        str(tmp_path / "traced.pkl"),
        "--executor",
        "serial",
        "--trace-memory",
        "--json",
    )
    assert result.exit_code == 0, result.stderr
    payload = json.loads(result.stdout)
    assert isinstance(payload["peak_bytes"], int)
    assert payload["peak_bytes"] > 0


def test_spawn_safe_entry_point_without_a_main_guard(
    corpus: Path, tmp_path: Path
) -> None:
    """``python -m findex`` must be able to spawn workers on its own.

    Nothing in this test defines ``if __name__ == "__main__"`` for the child:
    the console entry point is the guard, and if it were missing the spawn
    would recurse until the timeout instead of printing JSON.
    """
    out = tmp_path / "spawn.pkl"
    result = run_cli(
        "index",
        str(corpus),
        "--out",
        str(out),
        "--executor",
        "processes",
        "--workers",
        "2",
        "--json",
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["executor"] == "processes"
    assert payload["workers"] == 2
    assert payload["documents"] == 3
    assert out.exists()

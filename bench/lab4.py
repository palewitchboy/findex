"""Lab 4 benchmark: CLI start-up, command latency, packaging, test health.

    uv run python bench/lab4.py            # write bench/lab4_results.json

What is measured, and how to read it honestly:

* **cold** is the first invocation in this process tree (the OS file cache is
  whatever the previous step left it as - on Windows it is never truly cold);
* **warm** is the median of ``WARM_RUNS`` later invocations;
* every command is run through the *installed console script* (``.venv``,
  ``[project.scripts]``) when it exists, because that is what a user runs;
* JSON vs human compares the bytes on stdout for the same query - the number
  that decides whether ``--json`` is worth piping.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TMP = ROOT / "bench" / "_tmp" / "lab4"
RESULTS = ROOT / "bench" / "lab4_results.json"
WARM_RUNS = 5
CORPUS_DOCS = 6


def findex_command() -> list[str]:
    """Prefer the installed console script: that is the artefact under test."""
    name = "findex.exe" if os.name == "nt" else "findex"
    script = Path(sys.executable).with_name(name)
    if script.exists():
        return [str(script)]
    return [sys.executable, "-m", "findex"]


def run(args: list[str], *, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [*findex_command(), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=timeout,
        check=False,
    )


def timed(args: list[str]) -> dict[str, float]:
    """One cold run, then ``WARM_RUNS`` warm runs; milliseconds."""
    started = time.perf_counter()
    cold = run(args)
    cold_ms = (time.perf_counter() - started) * 1000
    if cold.returncode != 0:
        raise SystemExit(f"command failed: {args}\n{cold.stderr}")
    warm: list[float] = []
    for _ in range(WARM_RUNS):
        start = time.perf_counter()
        result = run(args)
        warm.append((time.perf_counter() - start) * 1000)
        if result.returncode != 0:
            raise SystemExit(f"command failed: {args}\n{result.stderr}")
    return {
        "cold_ms": round(cold_ms, 1),
        "warm_median_ms": round(statistics.median(warm), 1),
        "warm_min_ms": round(min(warm), 1),
        "warm_max_ms": round(max(warm), 1),
    }


def build_corpus() -> Path:
    """The smallest real books, copied once - full-corpus indexing is 200 s."""
    corpus = TMP / "corpus"
    if corpus.exists():
        shutil.rmtree(corpus)
    corpus.mkdir(parents=True)
    sources = sorted((ROOT / "data").glob("*.txt"), key=lambda p: p.stat().st_size)
    if sources:
        for source in sources[:CORPUS_DOCS]:
            shutil.copy2(source, corpus / source.name)
    else:
        for number in range(CORPUS_DOCS):
            (corpus / f"doc{number}.txt").write_text(
                f"Title: Synthetic {number}\n\n"
                + ("the quick brown fox jumps over the lazy dog. " * 40)
                + f"\nunique{number} marker\n",
                encoding="utf-8",
            )
    return corpus


def packaging() -> dict[str, Any]:
    uv = shutil.which("uv")
    command = [uv, "build"] if uv else [sys.executable, "-m", "uv", "build"]
    subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    wheels = sorted((ROOT / "dist").glob("*.whl"))
    sdists = sorted((ROOT / "dist").glob("*.tar.gz"))
    wheel = wheels[-1]
    sdist = sdists[-1]
    return {
        "wheel": wheel.name,
        "wheel_bytes": wheel.stat().st_size,
        "sdist": sdist.name,
        "sdist_bytes": sdist.stat().st_size,
    }


def test_health() -> dict[str, Any]:
    collected = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    match = re.search(r"(\d+) tests? collected", collected.stdout)
    tests = int(match.group(1)) if match else -1

    coverage_json = TMP / "coverage.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--cov=findex",
            f"--cov-report=json:{coverage_json}",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    percent = 0.0
    by_module: dict[str, float] = {}
    if coverage_json.exists():
        report = json.loads(coverage_json.read_text(encoding="utf-8"))
        percent = round(float(report["totals"]["percent_covered"]), 1)
        by_module = {
            Path(name).name: round(float(data["summary"]["percent_covered"]), 1)
            for name, data in sorted(report["files"].items())
        }

    fast = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-m", "not slow"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    fast_match = re.search(r"(\d+)/(\d+) tests collected", fast.stdout)
    tests_fast = int(fast_match.group(1)) if fast_match else -1

    pyright = subprocess.run(
        [sys.executable, "-m", "pyright", "--outputjson"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    errors = json.loads(pyright.stdout)["summary"]["errorCount"]

    ignores = 0
    for path in (ROOT / "src" / "findex").glob("*.py"):
        ignores += path.read_text(encoding="utf-8").count("# type: ignore")

    return {
        "tests": tests,
        "tests_fast": tests_fast,
        "coverage_percent": percent,
        "coverage_by_module": by_module,
        "pyright_errors": errors,
        "type_ignore_comments": ignores,
        "python": sys.version.split()[0],
    }


def main() -> int:
    TMP.mkdir(parents=True, exist_ok=True)
    corpus = build_corpus()
    index = TMP / "lab4.pkl"

    build = timed(["index", str(corpus), "--out", str(index), "--json"])
    search_json = timed(["search", str(index), "the", "--k", "5", "--json"])
    search_human = timed(["search", str(index), "the", "--k", "5"])
    stats = timed(["stats", str(corpus), "--top", "5", "--json"])
    startup = timed(["--version"])

    json_bytes = len(
        run(["search", str(index), "the", "--k", "5", "--json"]).stdout.encode()
    )
    human_bytes = len(run(["search", str(index), "the", "--k", "5"]).stdout.encode())

    results: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "platform": sys.platform,
        "corpus_docs": len(list(corpus.glob("*.txt"))),
        "corpus_bytes": sum(p.stat().st_size for p in corpus.glob("*.txt")),
        "warm_runs": WARM_RUNS,
        "commands": {
            "startup (--version)": startup,
            "index": build,
            "search --json": search_json,
            "search (human)": search_human,
            "stats": stats,
        },
        "output_bytes": {"json": json_bytes, "human": human_bytes},
        "packaging": packaging(),
        "health": test_health(),
    }
    RESULTS.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

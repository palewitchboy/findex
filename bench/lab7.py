"""Lab 7 benchmark: what the search API actually costs, and who pays for it.

    uv run python bench/lab7.py                     # build the index, run everything
    uv run python bench/lab7.py --skip-build        # reuse bench/_tmp/lab7/index.pkl
    uv run python bench/lab7.py --oha C:/path/to/oha.exe

What is measured
----------------
* **index** - 48 books from ``data/``, no positions, ``arrays`` postings, pickle.
  Built once, timed, and measured again for a **cold load in a fresh process**:
  the number that decides whether a container startup is acceptable.
* **core** - the search engine on its own, no HTTP: parse + match + BM25 scoring
  (``snippets=False``) versus the same query with snippets. The gap between the
  two is the whole story of this lab's latency, and it is not the search.
* **HTTP** - four ``oha`` runs at 8 concurrent connections, 10 s each:

  ==============================  =========================================
  1 worker, ``def``, bm25         the shipped route: FastAPI runs it in a
                                  thread pool, so the loop stays free
  1 worker, ``def``, tfidf        the other scorer, same everything else
  1 worker, ``async def``, bm25   the control: same code on the event loop
  4 workers, ``def``, bm25        4 processes, 4 GILs (lab 5)
  ==============================  =========================================

Every run is preceded by 40 warm-up requests. That is not cheating, it is the
measurement: the snippet reader caches normalised document text, and without a
warm-up the first eight concurrent requests would each read the same ten books
from disk - a thundering herd that measures ``open()``, not the server. The
README reports the cold-cache effect separately.

Numbers land in ``bench/lab7_results.json``; the README tables are copied from
it.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:  # --no-project runs have no installed pkg
    sys.path.insert(0, str(ROOT / "src"))

from findex.index import build_index  # noqa: E402
from findex.ranking import Searcher  # noqa: E402
from findex.store import load, save_pickle  # noqa: E402

WORK = ROOT / "bench" / "_tmp" / "lab7"
INDEX = WORK / "index.pkl"
RESULTS = ROOT / "bench" / "lab7_results.json"

#: 48 books, not 120: the point is an HONEST server benchmark, and each snippet
#: reads and scans a whole book, so the full corpus would be 2.5x the wait for
#: the same lesson.
DOCUMENTS = 48
QUERY = "love and death"
K = 10
CONCURRENCY = 8
DURATION = "10s"
WARMUP_REQUESTS = 40

SCENARIOS: tuple[dict[str, Any], ...] = (
    {
        "label": "1 worker · def · bm25",
        "workers": 1,
        "route": "/search",
        "mode": "def",
        "scorer": "bm25",
    },
    {
        "label": "1 worker · def · tfidf",
        "workers": 1,
        "route": "/search",
        "mode": "def",
        "scorer": "tfidf",
    },
    {
        "label": "1 worker · async def · bm25",
        "workers": 1,
        "route": "/search/async",
        "mode": "async def",
        "scorer": "bm25",
    },
    {
        "label": "4 workers · def · bm25",
        "workers": 4,
        "route": "/search",
        "mode": "def",
        "scorer": "bm25",
    },
)


# --------------------------------------------------------------------------- http
def free_port() -> int:
    """Ask the OS for an unused port; the socket is closed immediately."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def get(url: str, timeout: float = 30.0) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:  # 4xx/5xx are answers, not failures
        return exc.code, exc.read()


def wait_for_health(base: str, timeout: float = 60.0) -> float:
    """Block until /health answers 200; return how long startup took."""
    started = time.perf_counter()
    while time.perf_counter() - started < timeout:
        try:
            status, _ = get(f"{base}/health", timeout=2.0)
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(0.2)
            continue
        if status == 200:
            return time.perf_counter() - started
        time.sleep(0.2)
    raise TimeoutError(f"{base}/health did not become ready in {timeout:.0f}s")


def start_server(port: int, workers: int) -> subprocess.Popen[bytes]:
    env = dict(os.environ)
    env["INDEX_PATH"] = str(INDEX)
    env["LOG_LEVEL"] = "WARNING"
    env["TEXT_CACHE"] = "32"
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "findex.web:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--workers",
        str(workers),
        "--log-level",
        "warning",
    ]
    return subprocess.Popen(  # noqa: S603 - a fixed argv, no shell
        command,
        cwd=ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def stop_server(process: subprocess.Popen[bytes]) -> None:
    """Kill the whole tree: with --workers 4 the children outlive the parent."""
    if process.poll() is not None:
        return
    if platform.system() == "Windows":
        subprocess.run(  # noqa: S603,S607 - taskkill is the supported way here
            ["taskkill", "/F", "/T", "/PID", str(process.pid)],
            capture_output=True,
            check=False,
        )
    else:
        process.terminate()
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:  # pragma: no cover
        process.kill()


def warm_up(base: str, requests: int) -> None:
    """Populate every worker's snippet cache before the clock starts."""
    for _ in range(requests):
        get(f"{base}/search?{urlencode({'q': QUERY, 'k': K})}")


# --------------------------------------------------------------------------- oha
def find_oha(explicit: str | None) -> Path:
    for candidate in (explicit, os.environ.get("OHA_BIN"), shutil.which("oha")):
        if candidate:
            path = Path(candidate)
            if path.exists():
                return path
    raise SystemExit(
        "oha not found: pass --oha PATH, set OHA_BIN, or put it on PATH\n"
        "https://github.com/hatoo/oha/releases"
    )


def run_oha(oha: Path, url: str, duration: str) -> dict[str, Any]:
    """One load run. ``--output-format json`` is oha's machine-readable mode."""
    command = [
        str(oha),
        "-z",
        duration,
        "-c",
        str(CONCURRENCY),
        "--no-tui",
        "--output-format",
        "json",
        url,
    ]
    completed = subprocess.run(  # noqa: S603 - fixed argv
        command, capture_output=True, check=False, timeout=600
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "oha failed ("
            f"{completed.returncode}): "
            f"{completed.stderr.decode('utf-8', 'replace')}"
        )
    return json.loads(completed.stdout.decode("utf-8"))


def percentile(oha_json: dict[str, Any], name: str) -> float:
    """oha reports latency in seconds in ``latencyPercentiles`` and in ms in
    ``metrics.latency_ms``; the ms copy is the one used here, so a README table
    cannot be off by 1000x."""
    metrics = oha_json.get("metrics", {})
    latency = metrics.get("latency_ms", {})
    if name in latency:
        return float(latency[name])
    return float(oha_json["latencyPercentiles"][name]) * 1000.0


# --------------------------------------------------------------------------- index
def build_compact_index() -> dict[str, Any]:
    started = time.perf_counter()
    index = build_index(
        ROOT / "data", limit=DOCUMENTS, positions=False, storage="arrays"
    )
    build_seconds = time.perf_counter() - started
    save_pickle(index, INDEX)
    info = {
        "documents": index.num_docs,
        "terms": len(index),
        "postings": index.n_postings(),
        "tokens": index.n_tokens,
        "avg_doc_length": round(index.avg_doc_length, 1),
        "positions": False,
        "storage": "arrays",
        "format": "pickle",
        "file_bytes": INDEX.stat().st_size,
        "build_seconds": round(build_seconds, 2),
    }
    index.close()
    return info


LOAD_PROBE = """
import sys, time
sys.path.insert(0, {src!r})
from findex.store import load
started = time.perf_counter()
index = load({path!r})
print(f"{{(time.perf_counter() - started) * 1000:.1f}} {{index.num_docs}}")
"""


def cold_load_seconds(runs: int = 3) -> list[float]:
    """Load the pickle in a **fresh interpreter** - that is a container start."""
    code = LOAD_PROBE.format(src=str(ROOT / "src"), path=str(INDEX))
    samples: list[float] = []
    for _ in range(runs):
        completed = subprocess.run(  # noqa: S603 - fixed argv
            [sys.executable, "-c", code], capture_output=True, check=True, timeout=300
        )
        milliseconds = float(completed.stdout.split()[0])
        samples.append(round(milliseconds / 1000.0, 4))
    return samples


def core_numbers() -> dict[str, Any]:
    """The engine without HTTP: scoring alone vs scoring plus snippets."""
    index = load(INDEX)
    try:
        # Ranking only: the query cache is per Searcher, exactly like the route,
        # so every iteration pays parse + match + score for real.
        started = time.perf_counter()
        for _ in range(200):
            Searcher(index, snippets=False).search(QUERY, k=K)
        score_ms = (time.perf_counter() - started) / 200 * 1000

        Searcher(index).search(QUERY, k=K)  # warm the text cache
        started = time.perf_counter()
        for _ in range(20):
            Searcher(index).search(QUERY, k=K)
        full_ms = (time.perf_counter() - started) / 20 * 1000

        biggest = max(index.doc_meta.values(), key=lambda meta: meta.n_tokens)
        from findex import snippet as snippet_module  # noqa: PLC0415

        snippet_module._load_document.cache_clear()
        started = time.perf_counter()
        snippet_module._load_document(biggest.path)
        read_ms = (time.perf_counter() - started) * 1000
        size_mb = Path(biggest.path).stat().st_size / 1e6
        return {
            "score_ms": round(score_ms, 3),
            "full_ms": round(full_ms, 1),
            "snippets_ms": round(max(0.0, full_ms - score_ms), 1),
            "biggest_doc_mb": round(size_mb, 2),
            "biggest_read_ms": round(read_ms, 1),
        }
    finally:
        index.close()


# --------------------------------------------------------------------------- run
def run_scenario(oha: Path, scenario: dict[str, Any]) -> dict[str, Any]:
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    workers = int(scenario["workers"])
    query = urlencode({"q": QUERY, "k": K, "scorer": scenario["scorer"]})
    url = f"{base}{scenario['route']}?{query}"
    process = start_server(port, workers)
    try:
        startup = wait_for_health(base)
        time.sleep(0.3)
        warm_up(base, WARMUP_REQUESTS)
        measured = run_oha(oha, url, DURATION)
    finally:
        stop_server(process)

    counts: dict[str, int] = {
        str(code): int(count)
        for code, count in measured["statusCodeDistribution"].items()
    }
    return {
        **scenario,
        "url": scenario["route"],
        "requests": sum(counts.values()),
        "ok": counts.get("200", 0),
        "rps": round(float(measured["summary"]["requestsPerSec"]), 2),
        "p50_ms": round(percentile(measured, "p50"), 1),
        "p95_ms": round(percentile(measured, "p95"), 1),
        "p99_ms": round(percentile(measured, "p99"), 1),
        "max_ms": round(percentile(measured, "max"), 1),
        "startup_s": round(startup, 2),
    }


def run_interference(oha: Path, workers: int = 1) -> dict[str, Any]:
    """A trivial route under search load: the point of ``def`` in one number.

    Four ``oha`` connections hammer ``/search`` (or ``/search/async``) for 14 s;
    one second in, eight more connections time ``/health``. With ``async def``
    the search blocks the event loop, so even a handler that returns a constant
    has to wait for a whole scoring pass. With ``def`` the search runs in a
    worker thread and the loop answers immediately - same throughput, completely
    different service.
    """
    results: dict[str, Any] = {}
    for mode, route in (("def", "/search"), ("async def", "/search/async")):
        port = free_port()
        base = f"http://127.0.0.1:{port}"
        query = urlencode({"q": QUERY, "k": K})
        process = start_server(port, workers)
        try:
            wait_for_health(base)
            time.sleep(0.3)
            warm_up(base, WARMUP_REQUESTS)
            load = subprocess.Popen(  # noqa: S603 - fixed argv
                [
                    str(oha),
                    "-z",
                    "14s",
                    "-c",
                    "4",
                    "--no-tui",
                    "--output-format",
                    "json",
                    f"{base}{route}?{query}",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            try:
                time.sleep(1.5)
                measured = run_oha(oha, f"{base}/health", "8s")
            finally:
                load.wait(timeout=60)
        finally:
            stop_server(process)
        results[mode] = {
            "route": route,
            "requests": sum(
                int(count) for count in measured["statusCodeDistribution"].values()
            ),
            "rps": round(float(measured["summary"]["requestsPerSec"]), 2),
            "p50_ms": round(percentile(measured, "p50"), 1),
            "p95_ms": round(percentile(measured, "p95"), 1),
            "p99_ms": round(percentile(measured, "p99"), 1),
        }
    return results


def machine_info(oha: Path) -> dict[str, Any]:
    import fastapi  # noqa: PLC0415
    import uvicorn  # noqa: PLC0415

    return {
        "cpu": platform.processor() or platform.machine(),
        "logical_cpus": os.cpu_count(),
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "fastapi": fastapi.__version__,
        "uvicorn": uvicorn.__version__,
        "oha": subprocess.run(  # noqa: S603 - fixed argv
            [str(oha), "--version"], capture_output=True, text=True, check=False
        ).stdout.strip(),
    }


def print_table(results: dict[str, Any]) -> None:
    header = f"{'scenario':<28}{'reqs':>6}{'rps':>8}{'p50':>9}{'p95':>9}{'p99':>9}"
    print(header)
    print("-" * len(header))
    for row in results["rows"]:
        print(
            f"{row['label']:<28}{row['requests']:>6}{row['rps']:>8.2f}"
            f"{row['p50_ms']:>9.1f}{row['p95_ms']:>9.1f}{row['p99_ms']:>9.1f}"
        )
    core = results["core"]
    index = results["index"]
    print()
    print(
        f"index      {index['documents']} docs · {index['terms']:,} terms · "
        f"{index['file_bytes'] / 1e6:.1f} MB · built in {index['build_seconds']} s"
    )
    print(
        f"cold load  {min(results['index']['load_seconds']) * 1000:.0f} ms "
        f"(best of {len(results['index']['load_seconds'])})"
    )
    print(
        f"engine     score {core['score_ms']} ms · with snippets {core['full_ms']} ms "
        f"· biggest book {core['biggest_doc_mb']} MB "
        f"read in {core['biggest_read_ms']} ms"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(__doc__ or "lab 7 benchmark").splitlines()[0]
    )
    parser.add_argument(
        "--skip-build", action="store_true", help="reuse an existing index"
    )
    parser.add_argument("--oha", default=None, help="path to the oha binary")
    parser.add_argument(
        "--only", default=None, help="run one scenario label (substring)"
    )
    args = parser.parse_args(argv)

    oha = find_oha(args.oha)
    WORK.mkdir(parents=True, exist_ok=True)

    if args.skip_build and INDEX.exists():
        print(
            f"reusing {INDEX.relative_to(ROOT)} ({INDEX.stat().st_size / 1e6:.1f} MB)"
        )
        previous: dict[str, Any] = {}
        if RESULTS.exists():
            previous = json.loads(RESULTS.read_text(encoding="utf-8")).get("index", {})
        if previous:
            index_info: dict[str, Any] = previous
        else:
            index = load(INDEX)
            index_info = {
                "documents": index.num_docs,
                "terms": len(index),
                "postings": index.n_postings(),
                "tokens": index.n_tokens,
                "avg_doc_length": round(index.avg_doc_length, 1),
                "positions": False,
                "storage": "arrays",
                "format": "pickle",
                "file_bytes": INDEX.stat().st_size,
                "build_seconds": None,
            }
            index.close()
    else:
        print(f"building a {DOCUMENTS}-document index from data/ ...")
        index_info = build_compact_index()
        print(
            f"  {index_info['documents']} docs, "
            f"{index_info['file_bytes'] / 1e6:.1f} MB, "
            f"{index_info['build_seconds']} s"
        )

    print("cold load in fresh processes ...")
    index_info["load_seconds"] = cold_load_seconds()
    print(f"  {[f'{value * 1000:.0f} ms' for value in index_info['load_seconds']]}")

    print("engine without HTTP ...")
    core = core_numbers()
    print(f"  score {core['score_ms']} ms · full {core['full_ms']} ms")

    rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        if args.only and args.only not in scenario["label"]:
            continue
        print(f"oha: {scenario['label']} ... ", end="", flush=True)
        row = run_scenario(oha, scenario)
        rows.append(row)
        print(
            f"{row['requests']} reqs, {row['rps']:.2f} rps, p50 {row['p50_ms']:.0f} ms"
        )

    interference: dict[str, Any] = {}
    if not args.only:
        print("interference: /health while /search is saturated ...")
        interference = run_interference(oha)
        for mode, measured in interference.items():
            print(f"  {mode:<10} p50 {measured['p50_ms']:.0f} ms")

    results: dict[str, Any] = {
        "machine": machine_info(oha),
        "index": index_info,
        "core": core,
        "workload": {
            "query": QUERY,
            "k": K,
            "concurrency": CONCURRENCY,
            "duration": DURATION,
            "warmup_requests": WARMUP_REQUESTS,
            "text_cache": 32,
        },
        "rows": rows,
        "interference": interference,
    }
    RESULTS.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print()
    print_table(results)
    print(f"\nwrote {RESULTS.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

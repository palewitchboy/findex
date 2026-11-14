"""Lab 6 benchmark: how much of a crawl is waiting, and what politeness costs.

    uv run python bench/lab6.py                 # fixture matrix + failure demo
    uv run python bench/lab6.py --with-real     # ... plus a polite gutenberg crawl
    uv run python bench/lab6.py --only-real     # just the real crawl

What is measured:

* **fixture** - ``http.server`` in a thread, 200 pages, every response delayed
  by 50 ms, every 17th page answering 503 once before it succeeds. The delay is
  what a network is: 200 x 50 ms of *waiting* that one thread can overlap.
* **naive** - the same crawl with a synchronous ``httpx.Client``: fetch, parse,
  fetch the next. No event loop, no overlap. This is the baseline the table
  needs, because "asyncio is fast" is not a thing - *overlap* is the thing, and
  concurrency 1 should land on top of the naive row.
* **asyncio x {1, 5, 20}** - the same work through ``findex.crawl``, same
  parser, same frontier, same per-host limits (pyhost = concurrency, delay 0:
  there is nothing to be polite to on loopback).
* **memory** - peak RSS of this process, sampled every 20 ms by a thread. Not
  ``tracemalloc``: a crawler's memory is sockets, buffers and pending bodies,
  and tracemalloc would slow the run it is supposed to describe.

The real crawl is deliberately tiny - 12 pages, one request per second, an
honest User-Agent - and it proves the part a fixture cannot: that ``robots.txt``
is fetched and obeyed on a real server, that the JSONL is a corpus Lab 1 can
read, and that ``findex index`` eats it without a flag.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:  # --no-project runs have no installed pkg
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "bench") not in sys.path:  # reuse lab5's machine probe
    sys.path.insert(0, str(ROOT / "bench"))

import httpx  # noqa: E402

from findex.crawl import (  # noqa: E402
    USER_AGENT,
    CrawlConfig,
    Crawler,
    CrawlStats,
    canonicalize,
    make_client,
    parse_html,
)
from lab5 import cpu_name, logical_cpus, physical_cpus  # noqa: E402

try:  # psutil is a dev dependency; the sampler degrades to None without it
    import psutil
except ImportError:  # pragma: no cover - only on an install without dev deps
    psutil = None  # type: ignore[assignment]

TMP = ROOT / "bench" / "_tmp" / "lab6"
RESULTS = ROOT / "bench" / "lab6_results.json"
FIXTURE_PAGES = 200
FIXTURE_DELAY = 0.05
FLAKY_EVERY = 17
LEVELS = (1, 5, 20)
REAL_MAX_PAGES = 12
REAL_SEED = "https://www.gutenberg.org/browse/scores/top"


# --------------------------------------------------------------------------- fixture
def page_html(index: int, total: int) -> bytes:
    """One page: a title, some words, a link to itself and to its neighbours."""
    words = " ".join(f"word{index}_{n}" for n in range(40))
    neighbours = "".join(
        f'<a href="/p/{other}">page {other}</a>'
        for other in range(index + 1, min(index + 6, total))
    )
    body = (
        f"<html><head><title>Page {index}</title></head><body>"
        f"<h1>Page {index}</h1><p>{words}</p>{neighbours}"
        f'<a href="/p/{index}">self</a></body></html>'
    )
    return body.encode("utf-8")


class Fixture:
    """200 pages behind a 50 ms sleep, plus two permanently broken endpoints."""

    def __init__(
        self, *, pages: int = FIXTURE_PAGES, delay: float = FIXTURE_DELAY
    ) -> None:
        self.pages = pages
        self.delay = delay
        self.hits: dict[str, int] = {}
        self._lock = threading.Lock()
        self._bodies = {f"/p/{i}": page_html(i, pages) for i in range(pages)}
        self._bodies["/"] = (
            "<html><head><title>Index</title></head><body>"
            + "".join(f'<a href="/p/{i}">page {i}</a> ' for i in range(pages))
            + "</body></html>"
        ).encode("utf-8")
        # canonicalize() strips the trailing slash, so the page must answer
        # on "/failure" as well as on "/failure/".
        self._bodies["/failure"] = (
            b'<html><head><title>Broken</title></head><body>'
            b'<a href="/broken/404">gone</a><a href="/broken/503">down</a>'
            b'<a href="/private/x">secret</a></body></html>'
        )
        self._bodies["/failure/"] = (
            b'<html><head><title>Broken</title></head><body>'
            b'<a href="/broken/404">gone</a><a href="/broken/503">down</a>'
            b'<a href="/private/x">secret</a></body></html>'
        )
        self._bodies["/private/x"] = b"<html><body><p>do not index</p></body></html>"
        fixture = self

        class Server(ThreadingHTTPServer):
            # The default backlog is 5: twenty simultaneous connections would
            # be refused, and the benchmark would measure connect errors
            # instead of overlap. It has to be a class attribute - listen()
            # runs inside __init__, before any instance attribute could be set.
            request_queue_size = 256
            daemon_threads = True

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self) -> None:  # noqa: N802 - http.server's spelling
                path = self.path.split("?", 1)[0]
                status, body, headers = fixture.reply(path)
                if fixture.delay:
                    time.sleep(fixture.delay)
                self.send_response(status)
                for name, value in headers.items():
                    self.send_header(name, value)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                pass

        self.server = Server(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def reset(self) -> None:
        """Forget the flaky pages' hit counts, so every run meets them fresh."""
        with self._lock:
            self.hits.clear()

    def reply(self, path: str) -> tuple[int, bytes, dict[str, str]]:
        with self._lock:
            self.hits[path] = self.hits.get(path, 0) + 1
            first = self.hits[path] == 1
        if path == "/robots.txt":
            return 200, b"User-agent: *\nDisallow: /private\n", {}
        if path == "/broken/404":
            return 404, b"gone", {}
        if path == "/broken/503":
            return 503, b"down", {}
        if path.startswith("/p/"):
            number = path.rsplit("/", 1)[-1]
            if number.isdigit() and int(number) % FLAKY_EVERY == 0 and first:
                # Retry-After: 0 is the server saying "ask again now". Without
                # it every retry would sit out our 0.5-1 s backoff, and four
                # retries at concurrency 1 would add three seconds of *sleeping*
                # to a ten-second crawl: the table would measure backoff, not
                # concurrency.
                return 503, b"try again", {"Retry-After": "0"}
        body = self._bodies.get(path)
        if body is None:
            return 404, b"missing", {}
        return 200, body, {}

    def __enter__(self) -> Fixture:
        self.thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


# --------------------------------------------------------------------------- memory
class RssSampler(threading.Thread):
    """Peak RSS of this process, sampled every 20 ms (see lab5 for the why)."""

    def __init__(self, interval: float = 0.02) -> None:
        super().__init__(daemon=True, name="lab6-rss")
        self.interval = interval
        self._stop = threading.Event()
        self.peak = 0

    def run(self) -> None:
        process = psutil.Process() if psutil is not None else None
        while not self._stop.wait(self.interval):
            if process is None:  # pragma: no cover - dev deps always present
                continue
            try:
                self.peak = max(self.peak, process.memory_info().rss)
            except Exception:  # pragma: no cover - process vanished
                return

    def stop(self) -> None:
        self._stop.set()
        self.join(timeout=1)


# --------------------------------------------------------------------------- runs
def measure_async(
    fixture: Fixture, concurrency: int, *, pages: int
) -> dict[str, Any]:
    """One crawl at one concurrency level; the client is built off the clock."""
    fixture.reset()
    seed = f"{fixture.base_url}/"
    config = CrawlConfig(
        max_pages=pages,
        concurrency=concurrency,
        per_host=concurrency,
        delay=0.0,
        max_attempts=3,
    )
    client = make_client(config)  # ~1 s: the TLS context loads here, not in the loop
    stats = CrawlStats()

    async def drive() -> int:
        delivered = 0
        crawler = Crawler(config, stats=stats, client=client)
        async for _page in crawler.run([seed]):
            delivered += 1
        await client.aclose()
        return delivered

    sampler = RssSampler()
    sampler.start()
    started = time.perf_counter()
    delivered = asyncio.run(drive())
    wall = time.perf_counter() - started
    sampler.stop()
    return {
        "mode": "asyncio",
        "concurrency": concurrency,
        "pages": delivered,
        "wall": wall,
        "pages_per_second": delivered / wall if wall else 0.0,
        "peak_rss": sampler.peak,
        "retries": stats.retries,
        "errors": stats.errors,
        "duplicates": stats.duplicates,
        "requests": stats.started,
    }


def measure_naive(fixture: Fixture, *, pages: int) -> dict[str, Any]:
    """The same crawl with a synchronous client: one request at a time."""
    fixture.reset()
    seed = f"{fixture.base_url}/"
    client = httpx.Client(
        headers={"User-Agent": USER_AGENT}, timeout=httpx.Timeout(10.0),
        follow_redirects=True,
    )
    root = canonicalize(seed)
    host = httpx.URL(root).host
    frontier: deque[str] = deque([root])
    seen = {root}
    delivered = 0
    errors = 0
    retries = 0
    sampler = RssSampler()
    sampler.start()
    try:
        started = time.perf_counter()
        while frontier and delivered < pages:
            url = frontier.popleft()
            response = None
            for _attempt in range(3):
                try:
                    response = client.get(url)
                except httpx.HTTPError:
                    response = None
                if response is not None and response.status_code < 500:
                    break
                retries += 1  # the fixture says Retry-After: 0, so retry now
            if response is None or response.status_code != 200:
                errors += 1
                continue
            delivered += 1
            for link in parse_html(response.text, base_url=url).links:
                if link not in seen and httpx.URL(link).host == host:
                    seen.add(link)
                    frontier.append(link)
        wall = time.perf_counter() - started
    finally:
        client.close()
        sampler.stop()
    return {
        "mode": "naive (sync httpx, one at a time)",
        "concurrency": 1,
        "pages": delivered,
        "wall": wall,
        "pages_per_second": delivered / wall if wall else 0.0,
        "peak_rss": sampler.peak,
        "retries": retries,
        "errors": errors,
        "duplicates": 0,
        "requests": delivered + errors,
    }


# --------------------------------------------------------------------------- logging
@contextmanager
def captured(level: int = logging.INFO) -> Iterator[list[str]]:
    """Collect the crawler's own log lines (the same ones ``crawl.log`` holds)."""
    lines: list[str] = []

    class Sink(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            lines.append(f"{record.levelname:8s} {record.getMessage()}")

    logger = logging.getLogger("findex.crawl")
    handler = Sink()
    handler.setLevel(level)
    previous, previous_propagate = logger.level, logger.propagate
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    try:
        yield lines
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)
        logger.propagate = previous_propagate


def failure_demo(fixture: Fixture) -> dict[str, Any]:
    """One 404 and one permanent 503, handled, logged, and the crawl goes on."""
    seed = f"{fixture.base_url}/failure/"
    config = CrawlConfig(max_pages=5, concurrency=2, delay=0.0, max_attempts=3)
    client = make_client(config)
    stats = CrawlStats()

    async def drive() -> int:
        delivered = 0
        crawler = Crawler(config, stats=stats, client=client)
        async for _page in crawler.run([seed]):
            delivered += 1
        await client.aclose()
        return delivered

    with captured(logging.DEBUG) as lines:
        delivered = asyncio.run(drive())
    TMP.mkdir(parents=True, exist_ok=True)
    log_path = TMP / "failure.log"
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "seed": seed,
        "pages": delivered,
        "errors": stats.errors,
        "retries": stats.retries,
        "robots_skipped": stats.robots_skipped,
        "log": str(log_path.relative_to(ROOT)),
        "log_lines": lines,
    }


# --------------------------------------------------------------------------- real
def real_crawl(*, pages: int = REAL_MAX_PAGES) -> dict[str, Any]:
    """Twelve pages from gutenberg.org, one request per second, honest UA."""
    TMP.mkdir(parents=True, exist_ok=True)
    corpus = TMP / "gutenberg.jsonl"
    log_path = TMP / "gutenberg.log"
    config = CrawlConfig(
        max_pages=pages,
        concurrency=2,
        per_host=1,  # one request in flight ...
        delay=1.0,  # ... and one per second
        timeout=20.0,
        max_attempts=3,
    )
    client = make_client(config)
    stats = CrawlStats()
    records: list[dict[str, Any]] = []

    async def drive() -> None:
        crawler = Crawler(config, stats=stats, client=client)
        async for page in crawler.run([REAL_SEED]):
            records.append(
                {
                    "url": page.url,
                    "title": page.title,
                    "text": page.text,
                    "status": page.status,
                    "bytes": page.n_bytes,
                    "depth": page.depth,
                    "elapsed": round(page.elapsed, 3),
                    "fetched_at": page.fetched_at_iso,
                }
            )
        await client.aclose()

    with captured(logging.INFO) as lines:
        started = time.perf_counter()
        asyncio.run(drive())
        wall = time.perf_counter() - started
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with corpus.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    indexed = index_the_corpus(corpus)
    example = dict(records[1]) if len(records) > 1 else {}
    if example:
        head = example["text"][:180]
        example["text"] = head + ("..." if len(example["text"]) > 180 else "")
    return {
        "seed": REAL_SEED,
        "user_agent": USER_AGENT,
        "max_pages": pages,
        "concurrency": config.concurrency,
        "per_host": config.per_host,
        "delay": config.delay,
        "pages": len(records),
        "bytes": sum(record["bytes"] for record in records),
        "wall": wall,
        "pages_per_second": len(records) / wall if wall else 0.0,
        "errors": stats.errors,
        "retries": stats.retries,
        "robots_skipped": stats.robots_skipped,
        "urls": [record["url"] for record in records],
        "corpus": str(corpus.relative_to(ROOT)),
        "corpus_bytes": corpus.stat().st_size if corpus.exists() else 0,
        "log": str(log_path.relative_to(ROOT)),
        "log_lines": lines[:40],
        "index": indexed,
        "example": example,
    }


def index_the_corpus(corpus: Path) -> dict[str, Any]:
    """Prove the JSONL is a corpus: ``findex index`` it in a child process."""
    index_path = TMP / "gutenberg.pkl"
    command = [
        sys.executable,
        "-m",
        "findex",
        "index",
        str(corpus),
        "--out",
        str(index_path),
        "--json",
    ]
    env = {
        "PYTHONPATH": str(ROOT / "src"),
        "PYTHONIOENCODING": "utf-8",
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
        "PATH": os.environ.get("PATH", ""),
    }
    result = subprocess.run(
        command, capture_output=True, text=True, encoding="utf-8", env=env, timeout=300
    )
    payload: dict[str, Any] = {
        "command": " ".join(command),
        "exit_code": result.returncode,
        "stderr": result.stderr.strip()[-800:],
    }
    if result.returncode == 0 and result.stdout.strip():
        stats = json.loads(result.stdout)
        payload.update(
            {
                "documents": stats.get("documents"),
                "terms": stats.get("terms"),
                "tokens": stats.get("tokens"),
                "postings": stats.get("postings"),
                "build_seconds": stats.get("build_seconds"),
                "index_bytes": stats.get("file_bytes"),
            }
        )
    return payload


# --------------------------------------------------------------------------- main
def machine() -> dict[str, Any]:
    total_ram = int(psutil.virtual_memory().total) if psutil is not None else None
    return {
        "cpu": cpu_name(),
        "physical_cores": physical_cpus(),
        "logical_cores": logical_cpus(),
        "ram_bytes": total_ram,
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
    }


def print_table(rows: list[dict[str, Any]]) -> None:
    print(f"{'mode':34s} {'c':>3s} {'pages':>6s} {'wall s':>8s} {'pages/s':>8s} "
          f"{'RSS MB':>7s} {'retry':>6s} {'errors':>6s}")
    for row in rows:
        print(
            f"{row['mode']:34s} {row['concurrency']:3d} {row['pages']:6d} "
            f"{row['wall']:8.2f} {row['pages_per_second']:8.2f} "
            f"{row['peak_rss'] / 1e6:7.1f} {row['retries']:6d} {row['errors']:6d}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-real", action="store_true", help="also crawl gutenberg.org"
    )
    parser.add_argument("--only-real", action="store_true", help="only the real crawl")
    parser.add_argument("--pages", type=int, default=FIXTURE_PAGES)
    parser.add_argument("--real-pages", type=int, default=REAL_MAX_PAGES)
    parser.add_argument(
        "--attach",
        default=None,
        help="merge the 'real' section of an earlier run instead of crawling again",
    )
    parser.add_argument(
        "--runs", type=int, default=2, help="repeats for the async rows"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    results: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "machine": machine(),
    }

    if not args.only_real:
        print(f"fixture: {args.pages} pages, {FIXTURE_DELAY * 1000:.0f} ms each, "
              f"{1 / FLAKY_EVERY:.0%} flaky\n", flush=True)
        with Fixture(pages=args.pages) as fixture:
            rows: list[dict[str, Any]] = [measure_naive(fixture, pages=args.pages)]
            print_table(rows)
            for level in LEVELS:
                samples = [
                    measure_async(fixture, level, pages=args.pages)
                    for _ in range(args.runs)
                ]
                row = dict(samples[0])
                row["wall"] = statistics.median(s["wall"] for s in samples)
                row["pages_per_second"] = (
                    row["pages"] / row["wall"] if row["wall"] else 0.0
                )
                row["peak_rss"] = max(s["peak_rss"] for s in samples)
                row["runs"] = len(samples)
                rows.append(row)
                print_table([row])
            results["fixture"] = {
                "pages": args.pages,
                "delay_ms": FIXTURE_DELAY * 1000,
                "flaky_every": FLAKY_EVERY,
                "runs": args.runs,
            }
            results["rows"] = rows
            baseline = rows[0]["wall"]
            for row in rows:
                row["speedup_vs_naive"] = baseline / row["wall"] if row["wall"] else 0.0

            print("\nfailure demo ...", flush=True)
            results["failure"] = failure_demo(fixture)

    if args.attach:
        earlier = json.loads(Path(args.attach).read_text(encoding="utf-8"))
        if "real" not in earlier:
            parser.error(f"{args.attach} has no 'real' section")
        results["real"] = earlier["real"]
        print(f"attached the real crawl from {args.attach}")

    if args.with_real or args.only_real:
        print(f"\nreal crawl: {REAL_SEED}")
        print(f"  max {args.real_pages} pages, one request per second ...", flush=True)
        results["real"] = real_crawl(pages=args.real_pages)
        real = results["real"]
        print(f"  {real['pages']} pages, {real['bytes']} bytes, "
              f"{real['wall']:.1f}s, errors={real['errors']}")
        print(f"  index: {real['index'].get('documents')} documents")

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {RESULTS.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

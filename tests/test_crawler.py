"""Lab 6 - the async crawler: fetching, robots, limits, the frontier, the CLI.

No test talks to the internet. HTTP is faked twice:

* ``httpx.MockTransport`` for everything that needs a *specific* failure
  (a 429 with ``Retry-After``, a 500 that clears up, a timeout);
* a real ``http.server`` in a thread on ``127.0.0.1`` for the end-to-end
  paths - the CLI, the JSONL file, ``findex index`` eating that file - because
  a mock transport never proves that a socket, a pool and an event loop agree.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import aclosing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, NamedTuple

import httpx
import pytest
import typer.main
from rich.console import Console
from typer.testing import CliRunner

from findex import cli
from findex.cli import app
from findex.crawl import (
    USER_AGENT,
    CrawlConfig,
    CrawledPage,
    Crawler,
    CrawlStats,
    FetchError,
    HostLimiter,
    RobotsCache,
    backoff_delay,
    canonicalize,
    crawl,
    fetch,
    host_allowed,
    make_client,
    parse_html,
    parse_retry_after,
)

runner = CliRunner()
SITE = "https://site.test"
SRC = Path(__file__).resolve().parents[1] / "src"


# --------------------------------------------------------------------------- helpers
def no_sleep(recorded: list[float]) -> Any:
    """An ``asyncio.sleep`` stand-in that records instead of waiting."""

    async def sleep(delay: float) -> None:
        recorded.append(delay)

    return sleep


def mock_site(
    pages: dict[str, tuple[int, str]],
    *,
    robots: str | None = None,
    hits: list[str] | None = None,
) -> httpx.MockTransport:
    """A transport serving ``pages`` by URL, 404 for anything else."""

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if hits is not None:
            hits.append(url)
        if robots is not None and url.endswith("/robots.txt"):
            return httpx.Response(200, text=robots)
        page = pages.get(url)
        if page is None:
            return httpx.Response(404, text="missing")
        status, body = page
        return httpx.Response(
            status,
            text=body,
            headers={"content-type": "text/html; charset=utf-8"},
        )

    return httpx.MockTransport(handler)


def async_client(transport: httpx.MockTransport) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=transport, headers={"User-Agent": USER_AGENT})


def collect(
    seeds: str | list[str], **kwargs: Any
) -> tuple[list[CrawledPage], CrawlStats]:
    """Run a whole crawl in a fresh loop; return its pages and counters."""

    async def main() -> tuple[list[CrawledPage], CrawlStats]:
        stats = CrawlStats()
        pages = [page async for page in crawl(seeds, stats=stats, **kwargs)]
        return pages, stats

    return asyncio.run(main())


class _Reply(NamedTuple):
    status: int
    body: bytes
    delay: float = 0.0
    content_type: str = "text/html; charset=utf-8"


def html(body: str, *, delay: float = 0.0) -> _Reply:
    return _Reply(200, body.encode("utf-8"), delay=delay)


class _Site:
    """A real HTTP server on loopback; ``pages`` is filled in per test."""

    def __init__(self) -> None:
        self.pages: dict[str, _Reply] = {}
        pages = self.pages

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - http.server's spelling
                path = self.path.split("?", 1)[0]
                reply = pages.get(path, _Reply(404, b"missing"))
                if reply.delay:
                    time.sleep(reply.delay)
                self.send_response(reply.status)
                self.send_header("Content-Type", reply.content_type)
                self.send_header("Content-Length", str(len(reply.body)))
                self.end_headers()
                self.wfile.write(reply.body)

            def log_message(self, format: str, *args: object) -> None:
                pass  # a test server should not print

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


@pytest.fixture()
def site() -> Iterator[_Site]:
    server = _Site()
    try:
        yield server
    finally:
        server.close()


# --------------------------------------------------------------------------- fetch
def test_fetch_returns_the_body_and_the_counters() -> None:
    transport = mock_site({f"{SITE}/": (200, "hello")})

    async def main() -> Any:
        async with async_client(transport) as client:
            return await fetch(client, f"{SITE}/")

    result = asyncio.run(main())
    assert result.status == 200
    assert result.text == "hello"
    assert result.n_bytes == 5
    assert result.attempts == 1
    assert result.retries == 0
    assert result.elapsed >= 0.0


def test_fetch_retries_a_500_then_succeeds() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(500, text="boom")
        return httpx.Response(200, text="finally")

    delays: list[float] = []

    async def main() -> Any:
        async with async_client(httpx.MockTransport(handler)) as client:
            return await fetch(client, f"{SITE}/", sleep=no_sleep(delays))

    result = asyncio.run(main())
    assert result.text == "finally"
    assert result.attempts == 3
    assert result.retries == 2
    assert len(delays) == 2
    assert delays[0] < delays[1]  # exponential, not constant


def test_fetch_honours_retry_after_on_429() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "7"}, text="slow down")
        return httpx.Response(200, text="ok")

    delays: list[float] = []

    async def main() -> Any:
        async with async_client(httpx.MockTransport(handler)) as client:
            return await fetch(client, f"{SITE}/", sleep=no_sleep(delays))

    result = asyncio.run(main())
    assert result.status == 200
    assert delays == [7.0]  # the header, not our own backoff


def test_fetch_gives_up_after_the_attempt_budget() -> None:
    hits: list[str] = []

    def five_hundreds(request: httpx.Request) -> httpx.Response:
        hits.append(str(request.url))
        return httpx.Response(503, text="down")

    async def main() -> None:
        async with async_client(httpx.MockTransport(five_hundreds)) as client:
            await fetch(client, f"{SITE}/x", max_attempts=3, sleep=no_sleep([]))

    with pytest.raises(FetchError) as caught:
        asyncio.run(main())
    assert caught.value.status == 503
    assert caught.value.attempts == 3
    assert len(hits) == 3


def test_fetch_never_retries_a_404() -> None:
    hits: list[str] = []
    transport = mock_site({}, hits=hits)

    async def main() -> None:
        async with async_client(transport) as client:
            await fetch(client, f"{SITE}/gone", max_attempts=3, sleep=no_sleep([]))

    with pytest.raises(FetchError) as caught:
        asyncio.run(main())
    assert caught.value.status == 404
    assert caught.value.attempts == 1
    assert len(hits) == 1  # a 404 is an answer, not a failure to retry


def test_fetch_times_out() -> None:
    calls = {"n": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        await asyncio.sleep(0.3)
        return httpx.Response(200, text="too late")

    async def main() -> None:
        async with async_client(httpx.MockTransport(handler)) as client:
            await fetch(
                client, f"{SITE}/", timeout=0.05, max_attempts=2, sleep=no_sleep([])
            )

    with pytest.raises(FetchError) as caught:
        asyncio.run(main())
    assert "timed out" in str(caught.value)
    assert caught.value.attempts == 2
    assert calls["n"] == 2


def test_fetch_sends_the_honest_user_agent() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("user-agent", ""))
        return httpx.Response(200, text="ok")

    async def main() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await fetch(client, f"{SITE}/")

    asyncio.run(main())
    assert seen and seen[0].startswith("findex-crawler/")
    assert "github.com" in seen[0]  # a contact URL, not an anonymous client


def test_fetch_rejects_a_silly_budget() -> None:
    async def main() -> None:
        async with async_client(mock_site({})) as client:
            await fetch(client, f"{SITE}/", max_attempts=0)

    with pytest.raises(ValueError):
        asyncio.run(main())


# --------------------------------------------------------------------------- backoff
def test_backoff_is_exponential_then_capped() -> None:
    assert backoff_delay(1, rng=lambda: 0.0) == pytest.approx(0.5)
    assert backoff_delay(2, rng=lambda: 0.0) == pytest.approx(1.0)
    assert backoff_delay(3, rng=lambda: 0.0) == pytest.approx(2.0)
    assert backoff_delay(10, rng=lambda: 0.0) == pytest.approx(8.0)  # capped


def test_backoff_adds_jitter() -> None:
    assert backoff_delay(1, rng=lambda: 1.0) == pytest.approx(1.0)  # 0.5 + 0.5
    assert backoff_delay(1, jitter=0.0) == pytest.approx(0.5)


def test_parse_retry_after_reads_both_forms() -> None:
    assert parse_retry_after("12") == 12.0
    assert parse_retry_after("0") == 0.0
    assert parse_retry_after(None) is None
    assert parse_retry_after("not a number") is None
    now = time.time()
    date = time.strftime("%a, %d %b %Y %H:%M:%S GMT", time.gmtime(now + 30))
    parsed = parse_retry_after(date, now=now)
    assert parsed is not None
    assert 25.0 <= parsed <= 31.0


# --------------------------------------------------------------------------- robots
def test_robots_blocks_a_disallowed_path() -> None:
    transport = mock_site(
        {f"{SITE}/": (200, "<p>ok</p>")},
        robots="User-agent: *\nDisallow: /private\n",
    )

    async def main() -> tuple[bool, bool]:
        async with async_client(transport) as client:
            robots = RobotsCache(client)
            return (
                await robots.allowed(f"{SITE}/"),
                await robots.allowed(f"{SITE}/private/x"),
            )

    allowed, denied = asyncio.run(main())
    assert allowed is True
    assert denied is False


def test_robots_is_fetched_once_per_host() -> None:
    hits: list[str] = []
    transport = mock_site({}, robots="User-agent: *\nDisallow: /x\n", hits=hits)

    async def main() -> None:
        async with async_client(transport) as client:
            robots = RobotsCache(client)
            for path in ("/a", "/b", "/c"):
                await robots.allowed(f"{SITE}{path}")
            assert robots.fetched == 1

    asyncio.run(main())
    assert [url for url in hits if url.endswith("robots.txt")] == [f"{SITE}/robots.txt"]


def test_robots_404_means_everything_is_allowed() -> None:
    transport = mock_site({f"{SITE}/": (200, "<p>ok</p>")})

    async def main() -> bool:
        async with async_client(transport) as client:
            return await RobotsCache(client).allowed(f"{SITE}/anything")

    assert asyncio.run(main()) is True


def test_robots_server_error_fails_closed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(503, text="down")
        return httpx.Response(200, text="<p>ok</p>")

    async def main() -> bool:
        async with async_client(httpx.MockTransport(handler)) as client:
            robots = RobotsCache(client, sleep=no_sleep([]))
            return await robots.allowed(f"{SITE}/")

    assert asyncio.run(main()) is False  # "when in doubt, don't"


def test_robots_crawl_delay_is_actually_read() -> None:
    """``parse()`` leaves ``mtime`` at 0 and ``crawl_delay()`` then returns None.

    This is the trap in ``urllib.robotparser``: only ``read()`` or
    ``modified()`` make the directives visible, and we never call ``read()``.
    """
    transport = mock_site({}, robots="User-agent: *\nCrawl-delay: 3\n")

    async def main() -> float:
        async with async_client(transport) as client:
            return await RobotsCache(client).min_delay(f"{SITE}/")

    assert asyncio.run(main()) == 3.0


def test_robots_request_rate_becomes_a_delay() -> None:
    transport = mock_site({}, robots="User-agent: *\nRequest-rate: 1/5\n")

    async def main() -> float:
        async with async_client(transport) as client:
            return await RobotsCache(client).min_delay(f"{SITE}/")

    assert asyncio.run(main()) == 5.0


# --------------------------------------------------------------------------- urls
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("HTTPS://Example.ORG/Path", "https://example.org/Path"),
        ("http://example.org:80/a", "http://example.org/a"),
        ("https://example.org:443/a", "https://example.org/a"),
        ("https://example.org/a#fragment", "https://example.org/a"),
        ("https://example.org//a///b//", "https://example.org/a/b"),
        ("https://example.org/?b=2&a=1", "https://example.org/?a=1&b=2"),
        ("https://example.org/Path/", "https://example.org/Path"),
        ("https://example.org", "https://example.org/"),
        ("https://example.org:8443/a", "https://example.org:8443/a"),
    ],
)
def test_canonicalize(raw: str, expected: str) -> None:
    assert canonicalize(raw) == expected


def test_canonicalize_resolves_relative_links() -> None:
    assert canonicalize("../b?z=1#x", f"{SITE}/dir/page") == f"{SITE}/b?z=1"


def test_canonicalize_refuses_non_pages() -> None:
    for raw in ("mailto:someone@example.org", "javascript:void(0)", "tel:+123"):
        with pytest.raises(ValueError):
            canonicalize(raw)


def test_host_allowed_treats_www_as_the_same_site() -> None:
    assert host_allowed("www.example.org", ["example.org"]) is True
    assert host_allowed("example.org", ["www.example.org"]) is True
    assert host_allowed("docs.example.org", ["example.org"]) is True
    assert host_allowed("notexample.org", ["example.org"]) is False
    assert host_allowed("anything.test", ()) is True  # empty = no restriction


# --------------------------------------------------------------------------- html
def test_parse_html_takes_text_and_links() -> None:
    markup = """
    <html><head><title>  A &amp; B  </title>
    <style>body { colour: red }</style></head>
    <body><h1>Heading</h1><p>first para</p><p>second <a href="/two">two</a>
    <a href="mailto:x@y.z">mail</a><script>var leak = "no";</script></p></body></html>
    """
    parsed = parse_html(markup, base_url=f"{SITE}/one")
    assert parsed.title == "A & B"
    assert "first para" in parsed.text
    assert "second" in parsed.text
    assert "colour" not in parsed.text  # <style> dropped
    assert "leak" not in parsed.text  # <script> dropped
    assert parsed.links == (f"{SITE}/two",)


def test_parse_html_dedupes_links_in_document_order() -> None:
    markup = '<a href="/a">1</a><a href="/a#x">2</a><a href="/b">3</a>'
    assert parse_html(markup, base_url=f"{SITE}/").links == (f"{SITE}/a", f"{SITE}/b")


def test_parse_html_separates_blocks_with_newlines() -> None:
    parsed = parse_html("<p>one</p><p>two</p>", base_url=f"{SITE}/")
    assert parsed.text == "one\ntwo"


# --------------------------------------------------------------------------- limits
def test_limiter_caps_concurrency_per_host() -> None:
    limiter = HostLimiter(per_host=2, delay=0.0)
    state = {"now": 0, "peak": 0}

    async def job() -> None:
        async with limiter.slot("example.org"):
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
            await asyncio.sleep(0.01)
            state["now"] -= 1

    async def main() -> None:
        async with asyncio.TaskGroup() as group:
            for _ in range(6):
                group.create_task(job())

    asyncio.run(main())
    assert state["peak"] == 2


def test_limiter_spaces_start_times() -> None:
    limiter = HostLimiter(per_host=4, delay=0.05)
    starts: list[float] = []

    async def job() -> None:
        async with limiter.slot("example.org"):
            starts.append(time.perf_counter())

    async def main() -> None:
        await asyncio.gather(*(job() for _ in range(3)))

    asyncio.run(main())
    starts.sort()
    assert starts[1] - starts[0] >= 0.04
    assert starts[2] - starts[1] >= 0.04


# --------------------------------------------------------------------------- crawl
def test_crawl_walks_links_breadth_first() -> None:
    pages = {
        f"{SITE}/": (200, '<a href="/a">a</a><a href="/b">b</a>'),
        f"{SITE}/a": (200, '<a href="/c">c</a>'),
        f"{SITE}/b": (200, "<p>b</p>"),
        f"{SITE}/c": (200, "<p>c</p>"),
    }
    got, stats = collect(SITE, transport=mock_site(pages), concurrency=1, delay=0.0)
    assert [page.url for page in got] == [
        f"{SITE}/",
        f"{SITE}/a",
        f"{SITE}/b",
        f"{SITE}/c",
    ]
    assert stats.pages == 4
    assert stats.errors == 0


def test_crawl_reserves_the_page_budget_before_fetching() -> None:
    links = "".join(f'<a href="/p{i}">x</a>' for i in range(10))
    pages = {f"{SITE}/": (200, links)}
    pages.update({f"{SITE}/p{i}": (200, "<p>x</p>") for i in range(10)})
    hits: list[str] = []
    got, stats = collect(
        SITE,
        transport=mock_site(pages, hits=hits),
        max_pages=3,
        concurrency=8,
        delay=0.0,
    )
    fetched = [url for url in hits if not url.endswith("robots.txt")]
    assert len(fetched) == 3  # eight workers, one budget, no overshoot
    assert stats.pages == 3
    assert len(got) == 3


def test_crawl_never_fetches_the_same_url_twice() -> None:
    pages = {
        f"{SITE}/": (200, '<a href="/a">a</a><a href="/a#x">a again</a>'),
        f"{SITE}/a": (200, '<a href="/">home</a><a href="/a">self</a>'),
    }
    hits: list[str] = []
    got, stats = collect(SITE, transport=mock_site(pages, hits=hits), delay=0.0)
    fetched = [url for url in hits if not url.endswith("robots.txt")]
    assert sorted(fetched) == [f"{SITE}/", f"{SITE}/a"]
    assert stats.duplicates >= 1
    assert len(got) == 2


def test_crawl_stays_inside_the_allowlist() -> None:
    pages = {
        f"{SITE}/": (200, '<a href="/a">a</a><a href="https://elsewhere.test/x">out</a>'),
        f"{SITE}/a": (200, "<p>a</p>"),
    }
    hits: list[str] = []
    got, stats = collect(
        SITE,
        transport=mock_site(pages, hits=hits),
        delay=0.0,
        allowed_domains=["site.test"],
    )
    assert all("elsewhere" not in url for url in hits)
    assert stats.domain_skipped == 1
    assert len(got) == 2


def test_crawl_refuses_a_seed_outside_the_allowlist() -> None:
    async def main() -> None:
        async for _ in crawl(
            SITE, allowed_domains=["other.test"], transport=mock_site({})
        ):
            pass

    with pytest.raises(ValueError):
        asyncio.run(main())


def test_crawl_skips_what_robots_disallows() -> None:
    pages = {
        f"{SITE}/": (200, '<a href="/private/x">secret</a><a href="/a">a</a>'),
        f"{SITE}/private/x": (200, "<p>secret</p>"),
        f"{SITE}/a": (200, "<p>a</p>"),
    }
    hits: list[str] = []
    got, stats = collect(
        SITE,
        transport=mock_site(
            pages, robots="User-agent: *\nDisallow: /private\n", hits=hits
        ),
        delay=0.0,
    )
    assert f"{SITE}/private/x" not in hits
    assert stats.robots_skipped == 1
    assert [page.url for page in got] == [f"{SITE}/", f"{SITE}/a"]


def test_robots_skips_do_not_eat_the_page_budget() -> None:
    links = '<a href="/private/x">no</a><a href="/a">a</a><a href="/b">b</a>'
    pages = {
        f"{SITE}/": (200, links),
        f"{SITE}/private/x": (200, "<p>secret</p>"),
        f"{SITE}/a": (200, "<p>a</p>"),
        f"{SITE}/b": (200, "<p>b</p>"),
    }
    got, stats = collect(
        SITE,
        transport=mock_site(pages, robots="User-agent: *\nDisallow: /private\n"),
        delay=0.0,
        max_pages=3,
        concurrency=1,
    )
    assert stats.robots_skipped == 1
    assert [page.url for page in got] == [f"{SITE}/", f"{SITE}/a", f"{SITE}/b"]


def test_crawl_survives_a_broken_page() -> None:
    pages = {
        f"{SITE}/": (200, '<a href="/gone">gone</a><a href="/a">a</a>'),
        f"{SITE}/a": (200, "<p>a</p>"),
    }
    got, stats = collect(SITE, transport=mock_site(pages), delay=0.0, max_attempts=1)
    assert stats.errors == 1
    assert [page.url for page in got] == [f"{SITE}/", f"{SITE}/a"]


def test_crawl_counts_retries() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404, text="")
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, text="try later")
        return httpx.Response(200, text="<p>finally</p>")

    async def main() -> CrawlStats:
        stats = CrawlStats()
        async with async_client(httpx.MockTransport(handler)) as client:
            crawler = Crawler(
                CrawlConfig(max_pages=1, concurrency=1, delay=0.0),
                stats=stats,
                sleep=no_sleep([]),
                client=client,
            )
            async for _page in crawler.run([SITE]):
                pass
        return stats

    stats = asyncio.run(main())
    assert stats.pages == 1
    assert stats.retries == 1
    assert stats.errors == 0


def test_crawl_respects_max_depth() -> None:
    pages = {
        f"{SITE}/": (200, '<a href="/a">a</a>'),
        f"{SITE}/a": (200, '<a href="/b">b</a>'),
        f"{SITE}/b": (200, "<p>b</p>"),
    }
    got, _ = collect(SITE, transport=mock_site(pages), delay=0.0, max_depth=1)
    assert [page.url for page in got] == [f"{SITE}/", f"{SITE}/a"]
    assert got[0].depth == 0
    assert got[1].depth == 1


def test_crawl_stops_promptly_when_the_consumer_breaks() -> None:
    """Closing the generator must cancel the workers, not wait for them."""
    bodies = {
        f"{SITE}/": "".join(f'<a href="/p{i}">x</a>' for i in range(20)),
        **{f"{SITE}/p{i}": "<p>x</p>" for i in range(20)},
    }

    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.2)
        if request.url.path == "/robots.txt":
            return httpx.Response(404, text="")
        return httpx.Response(200, text=bodies[str(request.url)])

    async def main() -> float:
        started = time.perf_counter()
        async with async_client(httpx.MockTransport(handler)) as client:
            stream = crawl(SITE, concurrency=8, delay=0.0, max_pages=50, client=client)
            async with aclosing(stream):
                async for _page in stream:
                    break
        return time.perf_counter() - started

    assert asyncio.run(main()) < 1.0


def test_crawl_is_quiet_under_debug_mode(caplog: pytest.LogCaptureFixture) -> None:
    """``debug=True`` logs a warning whenever a callback blocks the loop."""
    pages = {f"{SITE}/": (200, '<a href="/a">a</a>'), f"{SITE}/a": (200, "<p>a</p>")}

    async def main() -> None:
        async with async_client(mock_site(pages)) as client:
            crawler = Crawler(
                CrawlConfig(max_pages=5, concurrency=3, delay=0.0),
                stats=CrawlStats(),
                client=client,
            )
            async for _page in crawler.run([SITE]):
                pass

    with caplog.at_level(logging.WARNING, logger="asyncio"):
        asyncio.run(main(), debug=True)
    noisy = [
        record.getMessage()
        for record in caplog.records
        if record.name.startswith("asyncio")
    ]
    assert noisy == []


def test_crawl_rejects_a_bad_seed() -> None:
    async def main() -> None:
        async for _ in crawl("mailto:someone@example.org"):
            pass

    with pytest.raises(ValueError):
        asyncio.run(main())


def test_crawl_config_validates_its_numbers() -> None:
    for kwargs in (
        {"max_pages": 0},
        {"concurrency": 0},
        {"per_host": 0},
        {"delay": -1.0},
        {"max_attempts": 0},
    ):
        with pytest.raises(ValueError):
            CrawlConfig(**kwargs)


def test_make_client_sends_the_configured_agent() -> None:
    config = CrawlConfig(transport=mock_site({}))
    client = make_client(config)
    assert client.headers["user-agent"] == USER_AGENT
    asyncio.run(client.aclose())


# --------------------------------------------------------------------------- CLI
def climb(site: _Site, tmp_path: Path, *args: str) -> Any:
    return runner.invoke(
        app,
        [
            "crawl",
            f"{site.base_url}/",
            "--delay",
            "0",
            "--out",
            str(tmp_path / "corpus.jsonl"),
            "--log-file",
            str(tmp_path / "crawl.log"),
            *args,
        ],
    )


def test_cli_crawl_streams_jsonl_and_reports(site: _Site, tmp_path: Path) -> None:
    site.pages["/"] = html('<html><title>Root</title><a href="/a">a</a></html>')
    site.pages["/a"] = html("<html><title>A</title><p>alpha beta</p></html>")
    site.pages["/robots.txt"] = html("User-agent: *\nDisallow: /nope\n")
    result = climb(site, tmp_path, "--json", "--max-pages", "5")
    assert result.exit_code == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["pages"] == 2
    assert payload["seed"].startswith("http://127.0.0.1:")
    assert payload["respect_robots"] is True
    lines = (tmp_path / "corpus.jsonl").read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    assert [record["url"].rsplit("/", 1)[-1] or "root" for record in records] == [
        "root",
        "a",
    ]
    assert records[1]["text"] == "alpha beta"
    assert all(record["status"] == 200 for record in records)


def test_cli_crawl_writes_a_per_url_log(site: _Site, tmp_path: Path) -> None:
    site.pages["/"] = html('<a href="/a">a</a>')
    site.pages["/a"] = html("<p>a</p>")
    result = climb(site, tmp_path, "--json", "--retries", "1")
    assert result.exit_code == 0, result.stderr
    log_text = (tmp_path / "crawl.log").read_text(encoding="utf-8")
    assert "robots.txt" in log_text
    assert "GET http://127.0.0.1:" in log_text
    assert "-> 200" in log_text


def test_cli_crawl_output_is_indexable(site: _Site, tmp_path: Path) -> None:
    site.pages["/"] = html('<html><title>Root</title><a href="/a">a</a></html>')
    site.pages["/a"] = html("<html><title>A</title><p>alpha beta gamma</p></html>")
    assert climb(site, tmp_path, "--json").exit_code == 0
    built = runner.invoke(
        app,
        [
            "index",
            str(tmp_path / "corpus.jsonl"),
            "--out",
            str(tmp_path / "idx.pkl"),
            "--json",
        ],
    )
    assert built.exit_code == 0, built.stderr
    assert json.loads(built.stdout)["documents"] == 2


def test_cli_crawl_records_a_handled_failure(site: _Site, tmp_path: Path) -> None:
    site.pages["/"] = html('<a href="/broken">broken</a><a href="/ok">ok</a>')
    site.pages["/broken"] = _Reply(404, b"no")
    site.pages["/ok"] = html("<p>ok</p>")
    result = climb(site, tmp_path, "--json", "--retries", "1")
    assert result.exit_code == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["pages"] == 2
    assert payload["errors"] == 1
    log_text = (tmp_path / "crawl.log").read_text(encoding="utf-8")
    assert "404" in log_text
    assert "giving up on" in log_text


def test_cli_crawl_renders_the_live_line(
    site: _Site, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The panel is only drawn for a terminal; draw it into a buffer to test it."""
    site.pages["/"] = html('<a href="/slow">slow</a>')
    site.pages["/slow"] = html("<p>slow</p>", delay=0.35)
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=True, width=120)
    monkeypatch.setattr(cli, "err_console", console)
    result = climb(site, tmp_path)
    assert result.exit_code == 0, result.stderr
    drawn = buffer.getvalue()
    assert "crawl" in drawn
    assert "pages" in drawn


def test_cli_crawl_exits_2_on_a_bad_seed(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["crawl", "ftp://example.org/", "--out", str(tmp_path / "x.jsonl")],
    )
    assert result.exit_code == 2
    assert "http(s)" in result.stderr


def test_cli_crawl_exits_1_when_nothing_can_be_fetched(
    site: _Site, tmp_path: Path
) -> None:
    site.pages["/"] = _Reply(500, b"down")
    result = climb(site, tmp_path, "--retries", "1")
    assert result.exit_code == 1
    assert "no page could be fetched" in result.stderr


def test_cli_crawl_needs_a_start_url() -> None:
    assert runner.invoke(app, ["crawl"]).exit_code != 0


def test_cli_crawl_help_lists_the_politeness_options() -> None:
    """The flags exist on the command, not just somewhere in the help text.

    The help table is rendered by rich to the terminal width; on CI it is 80
    columns and the option column is wrapped, so the old string check failed
    there while passing on a wide local console.
    """
    assert runner.invoke(app, ["crawl", "--help"]).exit_code == 0
    command = typer.main.get_command(app).commands["crawl"]
    declared = {opt for p in command.params for opt in getattr(p, "opts", ())}
    assert {"--max-pages", "--concurrency", "--per-host", "--allow-domain"} <= declared


def test_crawl_module_runs_as_a_script() -> None:
    env = {
        "PYTHONPATH": str(SRC),
        "PYTHONIOENCODING": "utf-8",
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
        "PATH": os.environ.get("PATH", ""),
    }
    result = subprocess.run(
        [sys.executable, "-m", "findex.crawl", "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "START_URL" in result.stdout

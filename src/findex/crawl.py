"""M7 - the async crawler: one thread, many in-flight requests, good manners.

The whole point of the lab is that a network request is mostly *waiting*, and
waiting is the one thing a single thread can do a thousand times at once. So
this module is built out of four pieces that can each be tested alone:

* :func:`fetch` - one URL, a timeout, bounded retries with exponential backoff
  and jitter, ``Retry-After`` honoured, every status logged;
* :class:`RobotsCache` - ``robots.txt`` fetched once per host, parsed with the
  stdlib :class:`urllib.robotparser.RobotFileParser`, cached, and *asked* before
  every fetch;
* :class:`HostLimiter` - a per-host :class:`asyncio.Semaphore` (how many
  requests may be in flight to one server) plus a per-host minimum delay
  (how often we may *start* one), and a global semaphore for the whole crawl;
* :class:`Crawler` - an :class:`asyncio.Queue` frontier, ``concurrency`` worker
  tasks inside a :class:`asyncio.TaskGroup`, URL canonicalization with a
  ``seen`` set so nothing is fetched twice, and an async generator over the
  pages as they arrive.

Two rules the code keeps to, and the README explains at length:

* **no blocking call inside a coroutine.** HTTP is ``httpx.AsyncClient``;
  HTML parsing is CPU work and goes to a thread with
  :func:`asyncio.to_thread`; the JSONL write is the caller's business and the
  CLI does it with ``to_thread`` as well. ``asyncio.run(..., debug=True)``
  stays silent - a test asserts it.
* **politeness is not optional.** An honest ``User-Agent`` with a contact URL,
  ``robots.txt`` obeyed, one host at a time within the per-host budget, a
  minimum delay between requests to the same host, a domain allowlist, and a
  hard ``max_pages`` that is reserved *before* the request so the budget can
  never be overshot.

The crawler never indexes anything: it writes raw documents and exits, which is
exactly the boundary Lab 1 drew (``iter_documents``) and Lab 5 paid for
(GIL - CPU work does not belong on the I/O path).
"""

from __future__ import annotations

import asyncio
import email.utils
import logging
import random
import re
import time
from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    Mapping,
    Sequence,
)
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from html.parser import HTMLParser
from typing import Final
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import httpx

log = logging.getLogger(__name__)

__all__ = [
    "BACKOFF_BASE",
    "BACKOFF_CAP",
    "CrawledPage",
    "CrawlConfig",
    "CrawlStats",
    "Crawler",
    "FetchError",
    "FetchResult",
    "HostLimiter",
    "ParsedHtml",
    "RobotsCache",
    "USER_AGENT",
    "canonicalize",
    "crawl",
    "fetch",
    "host_allowed",
    "make_client",
    "parse_html",
    "parse_retry_after",
]

#: Honest identification: what we are, where the code lives, why we crawl.
#: A ``mailto:`` or a project URL is what robots.txt etiquette asks for; an
#: anonymous ``python-httpx/0.28`` string is what gets a crawler blocked.
USER_AGENT: Final = (
    "findex-crawler/0.5 (+https://github.com/rmalkevy/Programming-Practice-Projects; "
    "lab-06 educational crawler)"
)

#: The product token alone - ``robotparser`` matches against this, not the
#: whole header, and ``crawl-delay`` directives are written for it.
ROBOTS_AGENT: Final = "findex-crawler"

DEFAULT_TIMEOUT: Final = 10.0
DEFAULT_ATTEMPTS: Final = 3
BACKOFF_BASE: Final = 0.5
BACKOFF_CAP: Final = 8.0
BACKOFF_JITTER: Final = 0.5
#: A ``Retry-After`` larger than this is treated as "come back much later":
#: we cap it rather than let one header stall the whole crawl.
MAX_RETRY_AFTER: Final = 60.0
#: Statuses we never retry: the server understood and said no.
#: 429 is the exception (it is a "later", not a "no") and 5xx is a "maybe".
RETRY_STATUSES: Final = frozenset({429, 500, 502, 503, 504})
DEFAULT_PORTS: Final = {"http": 80, "https": 443}

Sleep = Callable[[float], Awaitable[None]]
Rng = Callable[[], float]


# --------------------------------------------------------------------------- fetching
@dataclass(frozen=True, slots=True)
class FetchResult:
    """One successful fetch: what came back, how big, how long, how many tries."""

    url: str
    status: int
    text: str
    n_bytes: int
    elapsed: float
    attempts: int
    content_type: str = ""

    @property
    def retries(self) -> int:
        """How many times we had to ask again before this body arrived."""
        return max(0, self.attempts - 1)


class FetchError(Exception):
    """A URL that could not be fetched inside the retry budget.

    Carries ``status`` (``None`` for a timeout or a connection error) and
    ``attempts`` so the caller can count retries without parsing the message.
    """

    def __init__(
        self,
        url: str,
        message: str,
        *,
        status: int | None = None,
        attempts: int = 0,
    ) -> None:
        super().__init__(f"{url}: {message}")
        self.url = url
        self.message = message
        self.status = status
        self.attempts = attempts


def parse_retry_after(value: str | None, *, now: float | None = None) -> float | None:
    """``Retry-After`` as seconds - either delta-seconds or an HTTP date.

    RFC 9110 allows both forms; a server that means "in 30 s" often sends
    ``Retry-After: 30``, a proxy in front of it sends a date. Garbage, a
    negative delta and a date in the past all collapse to ``0.0`` (retry now,
    with our backoff still applied as a floor by the caller).
    """
    if not value:
        return None
    value = value.strip()
    try:
        seconds = float(value)
    except ValueError:
        pass
    else:
        return max(0.0, seconds)
    try:
        moment = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    reference = time.time() if now is None else now
    return max(0.0, moment.timestamp() - reference)


def backoff_delay(
    attempt: int,
    *,
    base: float = BACKOFF_BASE,
    cap: float = BACKOFF_CAP,
    jitter: float = BACKOFF_JITTER,
    rng: Rng = random.random,
) -> float:
    """``min(cap, base * 2**(attempt-1))`` plus a random fraction of a second.

    The exponential part backs off; the jitter is what stops twenty workers
    that hit the same 503 from retrying in lockstep forever (a thundering herd
    is rude to the server and useless to us).
    """
    delay = min(cap, base * 2 ** (attempt - 1))
    return delay + (rng() * jitter if jitter else 0.0)


async def fetch(
    client: httpx.AsyncClient,
    url: str,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    max_attempts: int = DEFAULT_ATTEMPTS,
    sleep: Sleep = asyncio.sleep,
    rng: Rng = random.random,
    headers: Mapping[str, str] | None = None,
) -> FetchResult:
    """GET ``url`` with a timeout, retries and ``Retry-After`` support.

    Retried: timeouts, connection errors, 429 and 5xx. Not retried: every other
    4xx - a 404 will still be a 404 in two seconds, and hammering it is abuse.

    ``sleep`` is injected so the tests can retry instantly; in production it is
    :func:`asyncio.sleep`, the only sleep that does not freeze the loop.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    if timeout <= 0:
        raise ValueError("timeout must be > 0")
    request_headers = (
        dict(headers) if headers is not None else {"User-Agent": USER_AGENT}
    )

    last_status: int | None = None
    last_message = "no attempt was made"
    for attempt in range(1, max_attempts + 1):
        started = time.perf_counter()
        retry_after: float | None = None
        try:
            async with asyncio.timeout(timeout):
                response = await client.get(
                    url, headers=request_headers, follow_redirects=True
                )
        except TimeoutError:
            last_message = f"timed out after {timeout:.1f}s"
            log.warning(
                "GET %s: %s (attempt %d/%d)", url, last_message, attempt, max_attempts
            )
        except httpx.HTTPError as exc:
            last_message = f"{type(exc).__name__}: {exc}"
            log.warning(
                "GET %s: %s (attempt %d/%d)", url, last_message, attempt, max_attempts
            )
        else:
            elapsed = time.perf_counter() - started
            last_status = response.status_code
            if 200 <= last_status < 300:
                body = response.text
                log.info(
                    "GET %s -> %d, %d bytes, %.2fs (attempt %d)",
                    url,
                    last_status,
                    len(response.content),
                    elapsed,
                    attempt,
                )
                return FetchResult(
                    url=url,
                    status=last_status,
                    text=body,
                    n_bytes=len(response.content),
                    elapsed=elapsed,
                    attempts=attempt,
                    content_type=response.headers.get("content-type", ""),
                )
            last_message = f"HTTP {last_status}"
            retry_after = parse_retry_after(response.headers.get("Retry-After"))
            if last_status == 429 or last_status >= 500:
                log.warning(
                    "GET %s -> %d, retrying (attempt %d/%d, retry-after=%s)",
                    url,
                    last_status,
                    attempt,
                    max_attempts,
                    "none" if retry_after is None else f"{retry_after:.1f}s",
                )
            else:
                log.info("GET %s -> %d, not retrying", url, last_status)
                raise FetchError(
                    url, last_message, status=last_status, attempts=attempt
                )

        if attempt < max_attempts:
            delay = (
                retry_after
                if retry_after is not None
                else backoff_delay(attempt, rng=rng)
            )
            delay = min(delay, MAX_RETRY_AFTER)
            log.debug("sleeping %.2fs before retry %d of %s", delay, attempt + 1, url)
            await sleep(delay)

    raise FetchError(
        url,
        f"{last_message} after {max_attempts} attempt(s)",
        status=last_status,
        attempts=max_attempts,
    )


# --------------------------------------------------------------------------- robots
class RobotsCache:
    """``robots.txt`` per host: fetched once, cached, consulted before every GET.

    Failure policy, and why it is the conservative one:

    * ``401``/``403``/``404``/``410`` - there is no robots file, or it is not
      for us: everything is allowed (the RFC's "unavailable" case);
    * a 5xx or a connection error - we *cannot know* what the site allows, so
      we do not crawl it at all. "When in doubt, don't" is the only defensible
      reading of a 503 on ``robots.txt``, and it fails closed in a visible way
      (a warning line, not a traceback).

    ``Crawl-delay`` and ``Request-rate`` are read as a *minimum* delay between
    requests to that host; :class:`HostLimiter` applies it on top of whatever
    ``--delay`` the user asked for.
    """

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        user_agent: str = USER_AGENT,
        timeout: float = DEFAULT_TIMEOUT,
        before_request: Callable[[str], Awaitable[None]] | None = None,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self._client = client
        self._agent = user_agent
        self._robots_agent = ROBOTS_AGENT
        self._timeout = timeout
        self._before = before_request
        self._sleep = sleep
        self._lock = asyncio.Lock()
        self._cache: dict[str, RobotFileParser | None] = {}
        self.fetched = 0

    async def parser_for(self, url: str) -> RobotFileParser | None:
        """The parser for ``url``'s origin, or ``None`` if it could not be read."""
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin in self._cache:
            return self._cache[origin]
        async with self._lock:
            # Double-checked under the lock: twenty workers asking about the
            # same host must produce one request, not twenty.
            if origin in self._cache:
                return self._cache[origin]
            parser = await self._load(origin, parts.hostname or "")
            self._cache[origin] = parser
            return parser

    async def _load(self, origin: str, host: str) -> RobotFileParser | None:
        if self._before is not None:
            await self._before(host)
        robots_url = f"{origin}/robots.txt"
        try:
            result = await fetch(
                self._client,
                robots_url,
                timeout=self._timeout,
                max_attempts=2,
                sleep=self._sleep,
                headers={"User-Agent": self._agent},
            )
        except FetchError as exc:
            if exc.status in (401, 403, 404, 410):
                log.info(
                    "robots.txt for %s: HTTP %s - no rules, everything allowed",
                    origin,
                    exc.status,
                )
            else:
                log.warning(
                    "robots.txt for %s is unavailable (%s) - refusing to crawl it",
                    origin,
                    exc.message,
                )
                return None
            empty = RobotFileParser()
            empty.parse([])
            return empty
        self.fetched += 1
        parser = RobotFileParser()
        parser.parse(result.text.splitlines())
        # `crawl_delay()` and `request_rate()` return None until `mtime()` is
        # non-zero, and `parse()` never sets it - only `read()` (the network
        # version we are replacing) and `modified()` do. Without this line a
        # `Crawl-delay: 5` in robots.txt would be parsed, stored and ignored.
        parser.modified()
        log.info(
            "robots.txt for %s: %d bytes, %d line(s)",
            origin,
            result.n_bytes,
            len(result.text.splitlines()),
        )
        return parser

    async def allowed(self, url: str) -> bool:
        """May we fetch ``url``? ``False`` also covers "robots.txt was unreadable"."""
        parser = await self.parser_for(url)
        if parser is None:
            return False
        return parser.can_fetch(self._agent, url)

    async def min_delay(self, url: str) -> float:
        """``Crawl-delay`` (or ``Request-rate``) for this host, in seconds."""
        parser = await self.parser_for(url)
        if parser is None:
            return 0.0
        delay = parser.crawl_delay(self._robots_agent)
        if delay:
            return float(delay)
        rate = parser.request_rate(self._robots_agent)
        if rate is not None and rate.requests > 0:
            return float(rate.seconds) / float(rate.requests)
        return 0.0


# --------------------------------------------------------------------------- limits
class HostLimiter:
    """Per-host concurrency *and* per-host pacing, plus a global ceiling.

    The semaphore answers "how many requests may be in flight to this host at
    once"; the delay answers "how often may we start one". They are different
    questions: with ``per_host=2, delay=1`` two requests overlap but their
    start times are a second apart, which is what a site owner actually cares
    about. The slot is *reserved* before the sleep, not after, so two coroutines
    holding the same semaphore cannot pick the same start time.
    """

    def __init__(self, *, per_host: int = 2, delay: float = 1.0) -> None:
        if per_host < 1:
            raise ValueError("per_host must be >= 1")
        if delay < 0:
            raise ValueError("delay must be >= 0")
        self.per_host = per_host
        self.delay = delay
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._next_slot: dict[str, float] = {}

    def semaphore(self, host: str) -> asyncio.Semaphore:
        """The semaphore for ``host``, created on first use (lazily, per host)."""
        semaphore = self._semaphores.get(host)
        if semaphore is None:
            semaphore = self._semaphores[host] = asyncio.Semaphore(self.per_host)
        return semaphore

    async def wait(self, host: str, delay: float | None = None) -> None:
        """Reserve the next start slot for ``host`` and sleep until it."""
        effective = self.delay if delay is None else max(self.delay, delay)
        now = asyncio.get_running_loop().time()
        slot_at = max(now, self._next_slot.get(host, now))
        self._next_slot[host] = slot_at + effective
        if slot_at > now:
            await asyncio.sleep(slot_at - now)

    @asynccontextmanager
    async def slot(
        self, host: str, delay: float | None = None
    ) -> AsyncGenerator[None, None]:
        """Hold one of the host's concurrency slots for the duration of a GET."""
        async with self.semaphore(host):
            await self.wait(host, delay)
            yield


# --------------------------------------------------------------------------- urls
_SLASHES = re.compile(r"/{2,}")
_WHITESPACE = re.compile(r"[ \t\r\f\v\u00a0]+")


def canonicalize(url: str, base: str | None = None) -> str:
    """One URL, one spelling - so ``seen`` is a real dedupe and not a guess.

    Lowercase scheme and host, drop the default port, drop the fragment (it is
    the client's business, never the server's), collapse repeated slashes,
    strip a trailing slash, and sort the query parameters. Anything that is
    not ``http``/``https`` raises :class:`ValueError` - ``mailto:`` and
    ``javascript:`` are not pages.
    """
    joined = urljoin(base, url) if base else url
    parts = urlsplit(joined.strip())
    scheme = parts.scheme.lower()
    if scheme not in DEFAULT_PORTS:
        raise ValueError(f"unsupported scheme {scheme!r}")
    host = (parts.hostname or "").lower()
    if not host:
        raise ValueError("no host")
    if host.endswith("."):
        host = host[:-1]
    try:
        port = parts.port
    except ValueError as exc:  # a port that is not a number
        raise ValueError(str(exc)) from exc
    netloc = host if port in (None, DEFAULT_PORTS[scheme]) else f"{host}:{port}"
    path = _SLASHES.sub("/", parts.path) or "/"
    if len(path) > 1:
        path = path.rstrip("/") or "/"
    query = urlencode(sorted(parse_qsl(parts.query, keep_blank_values=True)))
    return urlunsplit((scheme, netloc, path, query, ""))


def _bare_host(host: str) -> str:
    """``www.example.com`` and ``example.com`` are one site to a human."""
    host = host.lower().rstrip(".")
    return host[4:] if host.startswith("www.") else host


def host_allowed(host: str, allowed: Iterable[str]) -> bool:
    """Is ``host`` on the allowlist (or a subdomain of an entry)?

    Empty allowlist means "no restriction" - the caller derives one from the
    seeds; the crawler never widens it on its own.
    """
    entries = [_bare_host(entry) for entry in allowed]
    if not entries:
        return True
    candidate = _bare_host(host)
    return any(
        candidate == entry or candidate.endswith("." + entry) for entry in entries
    )


# --------------------------------------------------------------------------- html
_SKIP_TAGS: Final = frozenset(
    {"script", "style", "noscript", "template", "svg", "canvas"}
)
_BLOCK_TAGS: Final = frozenset(
    {
        "address",
        "article",
        "aside",
        "blockquote",
        "br",
        "dd",
        "div",
        "dl",
        "dt",
        "figcaption",
        "figure",
        "footer",
        "form",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hr",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "td",
        "th",
        "tr",
        "ul",
    }
)


class _Extractor(HTMLParser):
    """Text and links out of HTML with nothing but the standard library.

    ``convert_charrefs=True`` turns ``&amp;`` into ``&`` before we see it; the
    ``<script>``/``<style>`` bodies are dropped by depth so a nested
    ``<noscript><style>`` cannot leak CSS into the corpus; block-level tags
    become newlines so words from two paragraphs do not fuse into one token.
    """

    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self._base = base_url
        self._skip_depth = 0
        self._in_title = False
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        name = tag.lower()
        if name in _SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if name == "title":
            self._in_title = True
            return
        if name in _BLOCK_TAGS:
            self.text_parts.append("\n")
        if name == "a":
            href = dict(attrs).get("href")
            if href:
                try:
                    self.links.append(canonicalize(href, self._base))
                except ValueError:
                    pass  # mailto:, javascript:, tel: - not pages

    def handle_endtag(self, tag: str) -> None:
        name = tag.lower()
        if name in _SKIP_TAGS:
            if self._skip_depth:
                self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        if name == "title":
            self._in_title = False
            return
        if name in _BLOCK_TAGS:
            self.text_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title:
            self.title_parts.append(data)
            return
        self.text_parts.append(data)


def _normalize(text: str) -> str:
    """Collapse runs of spaces, trim every line, drop empty ones."""
    text = _WHITESPACE.sub(" ", text)
    lines = (line.strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line).strip()


@dataclass(frozen=True, slots=True)
class ParsedHtml:
    """What the crawler needs from a page: a title, readable text, links."""

    title: str
    text: str
    links: tuple[str, ...]


def parse_html(markup: str, *, base_url: str) -> ParsedHtml:
    """Extract ``(title, text, links)`` from ``markup``; never raises on bad HTML.

    ``html.parser`` is forgiving by design, but a page can still confuse a
    hand-written handler; a parse error must degrade that page, not abort a
    crawl of two hundred. Links are canonicalized against ``base_url`` and
    de-duplicated in document order.
    """
    extractor = _Extractor(base_url)
    try:
        extractor.feed(markup)
        extractor.close()
    except Exception as exc:  # pragma: no cover - defensive, html.parser is lenient
        log.warning("cannot parse %s: %s", base_url, exc)
    title = _normalize(" ".join(part for part in extractor.title_parts))
    text = _normalize("".join(extractor.text_parts))
    return ParsedHtml(
        title=title, text=text, links=tuple(dict.fromkeys(extractor.links))
    )


# --------------------------------------------------------------------------- crawl
@dataclass(frozen=True, slots=True)
class CrawledPage:
    """One fetched, parsed page - exactly what the JSONL writer needs."""

    url: str
    title: str
    text: str
    status: int
    n_bytes: int
    depth: int
    links: tuple[str, ...]
    elapsed: float
    fetched_at: float

    @property
    def fetched_at_iso(self) -> str:
        return datetime.fromtimestamp(self.fetched_at, UTC).isoformat(
            timespec="seconds"
        )


@dataclass
class CrawlStats:
    """Live counters. The CLI reads them while the crawl is still running."""

    pages: int = 0
    bytes: int = 0
    errors: int = 0
    retries: int = 0
    robots_skipped: int = 0
    domain_skipped: int = 0
    duplicates: int = 0
    started: int = 0
    queue_peak: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "pages": self.pages,
            "bytes": self.bytes,
            "errors": self.errors,
            "retries": self.retries,
            "robots_skipped": self.robots_skipped,
            "domain_skipped": self.domain_skipped,
            "duplicates": self.duplicates,
            "queue_peak": self.queue_peak,
        }


@dataclass(frozen=True, slots=True)
class CrawlConfig:
    """Everything a crawl needs to know, in one immutable object."""

    max_pages: int = 100
    concurrency: int = 10
    per_host: int = 2
    delay: float = 1.0
    max_depth: int | None = None
    allowed_domains: tuple[str, ...] = ()
    user_agent: str = USER_AGENT
    timeout: float = DEFAULT_TIMEOUT
    max_attempts: int = DEFAULT_ATTEMPTS
    respect_robots: bool = True
    #: Testing seam (and a local-mirror seam): an ``httpx`` transport to use
    #: instead of the network. ``httpx.MockTransport`` in the tests.
    transport: httpx.AsyncBaseTransport | None = None

    def __post_init__(self) -> None:
        if self.max_pages < 1:
            raise ValueError("max_pages must be >= 1")
        if self.concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        if self.per_host < 1:
            raise ValueError("per_host must be >= 1")
        if self.delay < 0:
            raise ValueError("delay must be >= 0")
        if self.max_depth is not None and self.max_depth < 0:
            raise ValueError("max_depth must be >= 0")
        if self.timeout <= 0:
            raise ValueError("timeout must be > 0")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")


def make_client(config: CrawlConfig) -> httpx.AsyncClient:
    """One client for the whole crawl - connection reuse is half the speedup."""
    return httpx.AsyncClient(
        transport=config.transport,
        headers={"User-Agent": config.user_agent},
        timeout=httpx.Timeout(config.timeout),
        limits=httpx.Limits(
            max_connections=max(4, config.concurrency * 2),
            max_keepalive_connections=max(2, config.concurrency),
        ),
        follow_redirects=True,
    )


class Crawler:
    """The frontier, the workers and the bookkeeping behind :func:`crawl`.

    The shape is producer/consumer on two queues:

    * ``frontier`` - URLs to visit, fed by the seeds and then by every page's
      links (breadth-first, with a ``depth`` per item);
    * ``results`` - pages that have been fetched and parsed, bounded so a slow
      consumer (a disk, a terminal) pushes back on the network instead of
      letting the crawl buffer two hundred pages in memory.

    Liveness is a counter, not a timeout: ``_pending`` is "URLs pushed but not
    yet finished". When it hits zero the crawl is *over* and the last worker
    drops one sentinel per worker plus one for the consumer. Because every
    mutation of ``_pending`` happens between two ``await``s, it needs no lock -
    that is the one asyncio invariant worth trusting, and the price of breaking
    it (an ``await`` inside the read-modify-write) is a hang, not a wrong count.
    """

    def __init__(
        self,
        config: CrawlConfig,
        *,
        stats: CrawlStats | None = None,
        sleep: Sleep = asyncio.sleep,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.config = config
        self.stats = stats if stats is not None else CrawlStats()
        self._sleep = sleep
        # An injected client is not closed by us - the caller owns it. That is
        # how the CLI builds the pool (and its ~1 s TLS context) *before* the
        # loop starts, so the loop never blocks on certificate loading.
        self._client = client
        self._limiter = HostLimiter(per_host=config.per_host, delay=config.delay)
        self._seen: set[str] = set()
        self._allowed: tuple[str, ...] = tuple(config.allowed_domains)
        self._host_delays: dict[str, float] = {}
        self._pending = 0
        self._frontier: asyncio.Queue[tuple[str, int] | None] = asyncio.Queue()
        self._results: asyncio.Queue[CrawledPage | None] = asyncio.Queue()
        self._robots: RobotsCache | None = None

    # ------------------------------------------------------------------ public
    async def run(self, seeds: Sequence[str]) -> AsyncIterator[CrawledPage]:
        """Crawl from ``seeds``, yielding pages as they arrive."""
        roots: list[str] = []
        for seed in seeds:
            try:
                roots.append(canonicalize(seed))
            except ValueError as exc:
                raise ValueError(f"bad seed URL {seed!r}: {exc}") from exc
        if not roots:
            return
        if not self._allowed:
            hosts = {urlsplit(root).hostname or "" for root in roots}
            self._allowed = tuple(sorted(host for host in hosts if host))
        for root in roots:
            host = urlsplit(root).hostname or ""
            if not host_allowed(host, self._allowed):
                raise ValueError(f"seed {root} is outside {self._allowed!r}")
        self._results = asyncio.Queue(maxsize=max(1, self.config.concurrency * 2))
        for root in roots:
            self._push(root, 0)
        if self._pending == 0:  # every seed was a duplicate of another
            await self._results.put(None)
            return

        client = self._client
        owned = client is None
        if client is None:
            client = make_client(self.config)
        try:
            async for page in self._drive(client):
                yield page
        finally:
            if owned:
                await client.aclose()

    async def _drive(self, client: httpx.AsyncClient) -> AsyncIterator[CrawledPage]:
        self._robots = RobotsCache(
            client,
            user_agent=self.config.user_agent,
            timeout=self.config.timeout,
            before_request=self._wait_for_host,
            sleep=self._sleep,
        )
        async with asyncio.TaskGroup() as group:
            workers = [
                group.create_task(self._worker(client), name=f"crawl-worker-{i}")
                for i in range(self.config.concurrency)
            ]
            try:
                while True:
                    page = await self._results.get()
                    if page is None:
                        # The sentinel is queued by the last _finish(), i.e.
                        # *after* every page that was already fetched - so by
                        # here the budget-exhausted frontier is fully drained.
                        return
                    yield page
            finally:
                # Clean stop: cancel the workers *inside* the group, so the
                # group's own exit has nothing left to cancel and no
                # "Task was destroyed but it is pending" reaches stderr.
                for worker in workers:
                    worker.cancel()

    # ------------------------------------------------------------------ worker
    async def _worker(self, client: httpx.AsyncClient) -> None:
        while True:
            item = await self._frontier.get()
            self._frontier.task_done()
            if item is None:
                return
            url, depth = item
            try:
                await self._visit(client, url, depth)
            except asyncio.CancelledError:
                raise
            except Exception:
                self.stats.errors += 1
                log.exception("crawl worker: unexpected failure on %s", url)
            finally:
                await self._finish()

    async def _finish(self) -> None:
        """One URL is done; when the last one lands, wake everybody up."""
        self._pending -= 1
        if self._pending > 0:
            return
        for _ in range(self.config.concurrency):
            self._frontier.put_nowait(None)
        await self._results.put(None)

    async def _visit(self, client: httpx.AsyncClient, url: str, depth: int) -> None:
        host = urlsplit(url).hostname or ""
        robots = self._robots
        if self.config.respect_robots and robots is not None:
            # Checked *before* the budget is reserved: a page that robots.txt
            # forbids is not "fetched", so it must not eat a slot of --max-pages.
            if not await robots.allowed(url):
                self.stats.robots_skipped += 1
                log.info("robots.txt disallows %s - skipped", url)
                return
            delay = await robots.min_delay(url)
            if delay:
                self._host_delays[host] = max(self._host_delays.get(host, 0.0), delay)

        # The budget is reserved *before* the request and with no `await`
        # between the check and the increment, so twenty workers cannot all
        # squeeze past `max_pages` at once.
        if self.stats.started >= self.config.max_pages:
            return
        self.stats.started += 1

        try:
            async with self._limiter.slot(host, self._host_delays.get(host)):
                result = await fetch(
                    client,
                    url,
                    timeout=self.config.timeout,
                    max_attempts=self.config.max_attempts,
                    sleep=self._sleep,
                )
        except FetchError as exc:
            self.stats.errors += 1
            self.stats.retries += max(0, exc.attempts - 1)
            log.warning("giving up on %s: %s", url, exc)
            return

        self.stats.retries += result.retries
        # CPU work leaves the loop: html.parser on a 100 KB page is ~10 ms of
        # pure Python, and ten of those in a row is a visibly stalled crawl.
        parsed = await asyncio.to_thread(
            partial(parse_html, result.text, base_url=url)
        )
        self.stats.pages += 1
        self.stats.bytes += result.n_bytes
        page = CrawledPage(
            url=url,
            title=parsed.title,
            text=parsed.text,
            status=result.status,
            n_bytes=result.n_bytes,
            depth=depth,
            links=parsed.links,
            elapsed=result.elapsed,
            fetched_at=time.time(),
        )
        self._push_links(parsed.links, depth + 1)
        await self._results.put(page)

    # ------------------------------------------------------------------ queue
    async def _wait_for_host(self, host: str) -> None:
        """Robots.txt requests queue behind the same per-host pacing."""
        await self._limiter.wait(host, self._host_delays.get(host))

    def _push_links(self, links: Iterable[str], depth: int) -> None:
        if self.config.max_depth is not None and depth > self.config.max_depth:
            return
        for link in links:
            if not host_allowed(urlsplit(link).hostname or "", self._allowed):
                self.stats.domain_skipped += 1
                continue
            self._push(link, depth)

    def _push(self, url: str, depth: int) -> bool:
        """Add ``url`` to the frontier once; ``False`` if it was already seen."""
        if url in self._seen:
            self.stats.duplicates += 1
            return False
        self._seen.add(url)
        self._frontier.put_nowait((url, depth))
        self._pending += 1
        self.stats.queue_peak = max(self.stats.queue_peak, self._frontier.qsize())
        return True


async def crawl(
    seeds: str | Sequence[str],
    *,
    max_pages: int = 100,
    concurrency: int = 10,
    per_host: int = 2,
    delay: float = 1.0,
    max_depth: int | None = None,
    allowed_domains: Sequence[str] = (),
    user_agent: str = USER_AGENT,
    timeout: float = DEFAULT_TIMEOUT,
    max_attempts: int = DEFAULT_ATTEMPTS,
    respect_robots: bool = True,
    transport: httpx.AsyncBaseTransport | None = None,
    stats: CrawlStats | None = None,
    sleep: Sleep = asyncio.sleep,
    client: httpx.AsyncClient | None = None,
) -> AsyncIterator[CrawledPage]:
    """Crawl ``seeds`` politely, yielding each page as soon as it is parsed.

    This is the interface the lab asks for: an **async generator**, so the
    consumer starts writing page one to disk while page twenty is still in
    flight - the Lab 1 streaming idea, one layer up.

        async for page in crawl("https://example.org/", max_pages=12):
            ...

    ``allowed_domains`` defaults to the hosts of the seeds and is never widened:
    a link to another site is counted (``stats.domain_skipped``) and dropped.
    """
    config = CrawlConfig(
        max_pages=max_pages,
        concurrency=concurrency,
        per_host=per_host,
        delay=delay,
        max_depth=max_depth,
        allowed_domains=tuple(allowed_domains),
        user_agent=user_agent,
        timeout=timeout,
        max_attempts=max_attempts,
        respect_robots=respect_robots,
        transport=transport,
    )
    crawler = Crawler(config, stats=stats, sleep=sleep, client=client)
    seed_list = [seeds] if isinstance(seeds, str) else list(seeds)
    async for page in crawler.run(seed_list):
        yield page


# ------------------------------------------------------------------ entry point
if __name__ == "__main__":  # pragma: no cover - see findex.cli
    from findex.cli import crawl_main

    crawl_main()

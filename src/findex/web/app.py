"""Lab 7 - the ASGI application: routes, lifespan, dependencies, templates.

What this module is trying to prove
-----------------------------------
Every piece of FastAPI was already built by hand in labs 1-6, so the code here
is mostly *wiring*, and the wiring is deliberately explicit:

``lifespan``
    An ``@asynccontextmanager`` - lab 3's ``open_index`` in async form - that
    loads the index **once**, records how long it took, and closes it on
    shutdown. Not per request: a 48-document index is a few megabytes, the full
    120-book corpus is not, and re-reading either one on every ``GET /search``
    would put disk I/O on the hot path.

``Depends(get_index)`` / ``Depends(get_settings)``
    Functions, not globals. ``app.dependency_overrides[get_index] = lambda: tiny``
    replaces the real index in the tests without the route knowing that it
    happened - that is the whole reason the dependency is a function.

``def`` vs ``async def``
    ``GET /search`` is a plain ``def``: BM25 scoring is CPU-bound, and FastAPI
    runs sync handlers in a thread pool, so the event loop keeps accepting while
    one thread scores. ``GET /search/async`` is the same logic written as
    ``async def`` and is the **control** for the experiment - it blocks the loop
    for the whole scoring pass. Both are shipped so ``bench/lab7.py`` can
    measure them against the same index in the same process; the README has the
    numbers and the verdict.

Why Swagger lives at ``/api/docs``
----------------------------------
``GET /docs/{doc_id}`` is the document view this lab asks for, and FastAPI's
default Swagger UI route is ``/docs``. They collide, and the document route is
the product while the schema browser is a convenience, so the built-in docs
move: ``docs_url="/api/docs"``, ``openapi_url="/api/openapi.json"``. The JSON
schema is at ``/api/openapi.json`` for the same reason.

Status codes
------------
``422`` - a parameter violated its declared type or range (FastAPI), or ``k``
exceeded ``MAX_K`` (this module, because OpenAPI cannot read settings).
``400`` - the query string does not parse; it is content, not a parameter type.
``404`` - unknown ``doc_id``.
``503`` - the index is not loaded; ``/health`` says so without raising.
``500`` - anything unexpected, logged with the request id that is also returned
in the body and in the ``X-Request-ID`` header.
"""

from __future__ import annotations

import logging
import math
import pickle
import time
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Annotated, cast
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markupsafe import Markup, escape

from findex import __version__
from findex.index import Index
from findex.query import MissingPositions, QuerySyntaxError, parse
from findex.ranking import BM25, Scorer, Searcher, TfIdf
from findex.snippet import configure_text_cache, mark_terms
from findex.store import load
from findex.web.models import (
    DocModel,
    HealthModel,
    HitModel,
    ScorerName,
    SearchResponse,
    StatsModel,
)
from findex.web.settings import Settings

log = logging.getLogger(__name__)

#: The absolute ceiling on ``?k=``, baked into the OpenAPI schema. The
#: *configurable* limit is ``Settings.max_k`` (100 by default) and is checked in
#: the handler; this constant only keeps the generated schema from advertising
#: an unbounded integer.
MAX_K_CEILING = 1_000

#: Snippet markers. JSON keeps lab 3's human-readable ``[term]``; HTML escapes
#: the text first and then swaps two control characters for ``<mark>`` - text
#: containing a literal ``[war]`` must not be able to forge a highlight.
_JSON_MARK = ("[", "]")
_HTML_MARK = ("\x01", "\x02")

_HERE = Path(__file__).resolve().parent

templates = Jinja2Templates(directory=str(_HERE / "templates"))


# --------------------------------------------------------------------------- filters
def highlight(value: str) -> Markup:
    """Escape ``value``, then turn the HTML snippet markers into ``<mark>``."""
    escaped = str(escape(value))
    opened = escaped.replace(_HTML_MARK[0], "<mark>")
    return Markup(opened.replace(_HTML_MARK[1], "</mark>"))


templates.env.filters["highlight"] = highlight


def basename(value: str) -> str:
    """Last path component: the page shows ``pg100.txt``, not the server's disk.

    The absolute path stays in the ``DocModel`` (useful for the API and for
    debugging), but a public page has no business printing
    ``C:\\Users\\<someone>\\...`` - it leaks the deployment layout and means
    nothing to a reader.
    """
    return Path(value).name


templates.env.filters["basename"] = basename


# --------------------------------------------------------------------------- deps
@lru_cache(maxsize=1)
def default_settings() -> Settings:
    """The environment, read and validated exactly once per process."""
    return Settings()


def get_settings(request: Request) -> Settings:
    """Per-app settings: ``create_app(settings)`` wins, the environment is next.

    A function with a ``Request`` parameter is a FastAPI dependency like any
    other, and ``dependency_overrides[get_settings]`` still replaces it - the
    override is consulted before the callable itself is inspected.
    """
    settings: Settings | None = getattr(request.app.state, "settings", None)
    return settings if settings is not None else default_settings()


def get_index(request: Request) -> Index:
    """The index loaded by the lifespan, or a 503 explaining why there is none."""
    index: Index | None = getattr(request.app.state, "index", None)
    if index is None:
        reason: str = (
            getattr(request.app.state, "index_error", "") or "index not loaded"
        )
        raise HTTPException(status_code=503, detail=reason)
    return index


def get_uptime(request: Request) -> float:
    """Seconds since the lifespan started - ``0.0`` for an app never started."""
    started: float | None = getattr(request.app.state, "started_at", None)
    return 0.0 if started is None else time.monotonic() - started


# --------------------------------------------------------------------------- lifespan
def _configure_logging(level: str) -> None:
    """Set the root level from ``LOG_LEVEL`` without ever raising on a typo."""
    resolved = logging.getLevelNamesMapping().get(level.upper())
    if resolved is None:
        log.warning("unknown LOG_LEVEL %r; using INFO", level)
        resolved = logging.INFO
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=resolved,
            format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
        )
    else:
        root.setLevel(resolved)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup: load the index once. Shutdown: release it.

    A missing or unreadable index does **not** abort startup. The process comes
    up with ``app.state.index = None``, logs why, and answers 503 everywhere -
    which is a debuggable state: ``/health`` is reachable and the log says what
    to fix. A container that refuses to boot gives a platform nothing to probe
    and no request to trace.
    """
    settings: Settings = app.state.settings
    _configure_logging(settings.log_level)
    app.state.started_at = time.monotonic()
    app.state.index = None
    app.state.index_error = ""
    app.state.index_load_s = 0.0

    # Lab 3 caches 8 documents because a CLI answers one query per process. A
    # server answers thousands over the same corpus, so the cache is raised to
    # the configured size here - at startup, where the memory decision belongs.
    previous_cache = configure_text_cache(settings.text_cache)
    log.info(
        "snippet text cache: %d documents (was %d)",
        settings.text_cache,
        previous_cache,
    )

    started = time.perf_counter()
    try:
        app.state.index = load(settings.index_path)
    except (OSError, TypeError, ValueError, EOFError, pickle.UnpicklingError) as exc:
        app.state.index_error = f"{type(exc).__name__}: {exc}"
        log.error("cannot load index from %s: %s", settings.index_path, exc)
    else:
        app.state.index_load_s = time.perf_counter() - started
        log.info(
            "loaded %s in %.0f ms: %r",
            settings.index_path,
            app.state.index_load_s * 1000,
            app.state.index,
        )

    try:
        yield
    finally:
        index = app.state.index
        if index is not None:
            index.close()
        app.state.index = None
        configure_text_cache(previous_cache)
        log.info("index released, shutdown complete")


# --------------------------------------------------------------------------- search
def _scorer(name: ScorerName) -> Scorer:
    return BM25() if name == "bm25" else TfIdf()


def _run_search(
    index: Index,
    settings: Settings,
    *,
    q: str,
    k: int,
    scorer: ScorerName,
    page: int,
    mark: tuple[str, str] = _JSON_MARK,
) -> SearchResponse:
    """The one implementation both ``/search`` routes call.

    Pagination is "rank ``page * k`` documents, keep the last ``k``". That is
    the honest simple version and it is O(page) expensive; every real engine
    instead ranks once and keeps an offset (Lucene) or a search-after cursor.
    ``min(page * k, num_docs)`` caps the request: ranking can never return more
    documents than the index has, so ``?page=1000000`` costs the same as the
    last real page instead of a million-element heap.
    """
    if k > settings.max_k:
        raise HTTPException(
            status_code=422,
            detail=f"k must be <= {settings.max_k} (MAX_K)",
        )

    started = time.perf_counter()
    searcher = Searcher(index, _scorer(scorer), snippets=True, mark=mark)
    try:
        total = searcher.count(q)
        wanted = min(page * k, index.num_docs)
        ranked = searcher.search(q, k=wanted)
    except (QuerySyntaxError, MissingPositions) as exc:
        raise HTTPException(status_code=400, detail=f"bad query: {exc}") from exc

    window = ranked[(page - 1) * k :]
    took_ms = (time.perf_counter() - started) * 1000
    return SearchResponse(
        query=q,
        total=total,
        page=page,
        pages=max(1, math.ceil(total / k)),
        k=k,
        scorer=scorer,
        took_ms=round(took_ms, 3),
        results=[
            HitModel(
                doc_id=hit.doc_id,
                title=hit.title,
                score=round(hit.score, 6),
                snippet=hit.snippet,
            )
            for hit in window
        ],
    )


def _document(index: Index, settings: Settings, doc_id: int) -> DocModel:
    meta = index.doc_meta.get(doc_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"document {doc_id} is not indexed")
    try:
        text = Path(meta.path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        text = ""
    truncated = len(text) > settings.doc_max_chars
    return DocModel(
        doc_id=meta.doc_id,
        title=meta.title,
        path=meta.path,
        n_tokens=meta.n_tokens,
        text=text[: settings.doc_max_chars],
        truncated=truncated,
    )


def _page_url(q: str, scorer: str, k: int, page: int) -> str:
    return "/?" + urlencode({"q": q, "scorer": scorer, "k": k, "page": page})


# --------------------------------------------------------------------------- routes
router = APIRouter()


@router.get(
    "/search", response_model=SearchResponse, summary="Rank documents (threadpool)"
)
def search(
    q: Annotated[
        str,
        Query(min_length=1, max_length=200, pattern=r"\S", description="Query string."),
    ],
    k: Annotated[int, Query(ge=1, le=MAX_K_CEILING)] = 10,
    scorer: Annotated[ScorerName, Query()] = "bm25",
    page: Annotated[int, Query(ge=1)] = 1,
    index: Index = Depends(get_index),
    settings: Settings = Depends(get_settings),
) -> SearchResponse:
    """Ranked search. A plain ``def``: the work is CPU-bound, so FastAPI
    offloads it to a worker thread and the event loop stays free."""
    return _run_search(index, settings, q=q, k=k, scorer=scorer, page=page)


@router.get(
    "/search/async",
    response_model=SearchResponse,
    summary="Rank documents (inline, blocks the loop - benchmark control)",
)
async def search_inline(
    q: Annotated[
        str,
        Query(min_length=1, max_length=200, pattern=r"\S", description="Query string."),
    ],
    k: Annotated[int, Query(ge=1, le=MAX_K_CEILING)] = 10,
    scorer: Annotated[ScorerName, Query()] = "bm25",
    page: Annotated[int, Query(ge=1)] = 1,
    index: Index = Depends(get_index),
    settings: Settings = Depends(get_settings),
) -> SearchResponse:
    """Identical logic written as ``async def`` on purpose.

    This is lab 6's bug with an HTTP face: there is no ``await`` inside, so the
    whole scoring pass runs *on the event loop*, and every other request on this
    worker waits for it. Kept as a route so the benchmark measures the mistake
    instead of asserting it.
    """
    return _run_search(index, settings, q=q, k=k, scorer=scorer, page=page)


@router.get("/stats", response_model=StatsModel, summary="Index statistics")
def stats(request: Request, index: Index = Depends(get_index)) -> StatsModel:
    return StatsModel(
        documents=index.num_docs,
        terms=len(index),
        postings=index.n_postings(),
        tokens=index.n_tokens,
        avg_doc_length=round(index.avg_doc_length, 3),
        uptime_s=round(get_uptime(request), 3),
    )


@router.get("/health", response_model=HealthModel, summary="Liveness + index readiness")
def health(request: Request) -> Response:
    """200 when the index is loaded, **503** when it is not.

    Deliberately does not depend on ``get_index``: a health check that raises is
    a health check that reports 503 with a stack trace instead of a body the
    platform's dashboard can show.
    """
    index = cast("Index | None", getattr(request.app.state, "index", None))
    reason: str = getattr(request.app.state, "index_error", "")
    model = HealthModel(
        status="ok" if index is not None else "index-not-loaded",
        index_loaded=index is not None,
        documents=0 if index is None else index.num_docs,
        uptime_s=round(get_uptime(request), 3),
        detail="" if index is not None else reason,
    )
    return JSONResponse(
        status_code=200 if index is not None else 503,
        content=model.model_dump(),
    )


@router.get("/api/doc/{doc_id}", response_model=DocModel, summary="One document (JSON)")
def document(
    doc_id: int,
    index: Index = Depends(get_index),
    settings: Settings = Depends(get_settings),
) -> DocModel:
    return _document(index, settings, doc_id)


# --------------------------------------------------------------------------- html
@router.get("/", response_class=HTMLResponse, include_in_schema=False)
def search_page(
    request: Request,
    q: Annotated[str, Query(max_length=200)] = "",
    scorer: Annotated[ScorerName, Query()] = "bm25",
    page: Annotated[int, Query(ge=1)] = 1,
    k: Annotated[int | None, Query(ge=1, le=MAX_K_CEILING)] = None,
    index: Index = Depends(get_index),
    settings: Settings = Depends(get_settings),
) -> Response:
    """The search page: a form, highlighted snippets, pagination, doc links."""
    per_page = settings.page_size if k is None else k
    context: dict[str, object] = {
        "version": __version__,
        "q": q,
        "scorer": scorer,
        "k": per_page,
        "page": page,
        "error": "",
        "response": None,
        "prev_url": None,
        "next_url": None,
    }
    if q.strip():
        try:
            result = _run_search(
                index,
                settings,
                q=q,
                k=per_page,
                scorer=scorer,
                page=page,
                mark=_HTML_MARK,
            )
        except HTTPException as exc:
            # A malformed query is content: keep the form, show the message.
            context["error"] = str(exc.detail)
            context["q"] = q
            return templates.TemplateResponse(
                request, "search.html", context, status_code=200
            )
        context["response"] = result
        context["prev_url"] = (
            _page_url(q, scorer, per_page, page - 1) if page > 1 else None
        )
        context["next_url"] = (
            _page_url(q, scorer, per_page, page + 1) if page < result.pages else None
        )
    return templates.TemplateResponse(request, "search.html", context)


@router.get("/docs/{doc_id}", response_class=HTMLResponse, include_in_schema=False)
def doc_page(
    request: Request,
    doc_id: int,
    q: Annotated[str, Query(max_length=200)] = "",
    index: Index = Depends(get_index),
    settings: Settings = Depends(get_settings),
) -> Response:
    """The full document, with the query terms marked.

    ``/docs`` is why Swagger moved to ``/api/docs``: this route owns the path.
    """
    model = _document(index, settings, doc_id)
    terms = sorted(_terms_of(q))
    body = mark_terms(model.text, terms, mark=_HTML_MARK) if terms else model.text
    return templates.TemplateResponse(
        request,
        "doc.html",
        {
            "version": __version__,
            "doc": model,
            "q": q,
            "terms": terms,
            "body": body,
            "back_url": "/?" + urlencode({"q": q}) if q else "/",
        },
    )


def _terms_of(q: str) -> frozenset[str]:
    """Positive terms of a query, or nothing when it does not parse at all."""
    if not q.strip():
        return frozenset()
    try:
        return parse(q).terms()
    except ValueError:
        return frozenset()


# --------------------------------------------------------------------------- errors
async def request_id_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Give every request an id, echo it in the header, log any 500 under it."""
    request_id = uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


async def http_error_handler(request: Request, exc: Exception) -> Response:
    """JSON ``{"detail": ...}`` for clients; a rendered page for browsers.

    Typed as ``Exception`` because that is what Starlette's ``ExceptionHandler``
    protocol promises; the ``isinstance`` guard is what makes the annotation
    true rather than wishful.
    """
    if not isinstance(exc, HTTPException):
        raise exc
    if "text/html" in request.headers.get("accept", ""):
        return templates.TemplateResponse(
            request,
            "error.html",
            {
                "version": __version__,
                "status": exc.status_code,
                "detail": exc.detail,
                "q": "",
            },
            status_code=exc.status_code,
        )
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
        headers=getattr(exc, "headers", None),
    )


async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """One clean 500 with a request id; the traceback goes to the log, not the wire."""
    request_id: str = getattr(request.state, "request_id", "-")
    log.exception(
        "unhandled %s on %s %s (request %s)",
        type(exc).__name__,
        request.method,
        request.url.path,
        request_id,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "internal server error", "request_id": request_id},
        headers={"X-Request-ID": request_id},
    )


# --------------------------------------------------------------------------- app
def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the ASGI app. ``settings=None`` means "read the environment"."""
    resolved = default_settings() if settings is None else settings
    app = FastAPI(
        title="findex",
        version=__version__,
        summary="A small search engine, on the web.",
        lifespan=lifespan,
        # /docs belongs to the document view; the schema browser moves aside.
        docs_url="/api/docs",
        redoc_url=None,
        openapi_url="/api/openapi.json",
    )
    app.state.settings = resolved
    app.state.index = None
    app.state.index_error = ""
    app.state.started_at = None

    app.include_router(router)
    app.mount("/static", StaticFiles(directory=str(_HERE / "static")), name="static")
    app.middleware("http")(request_id_middleware)
    app.add_exception_handler(HTTPException, http_error_handler)
    app.add_exception_handler(Exception, unhandled_error_handler)
    return app


__all__ = [
    "MAX_K_CEILING",
    "create_app",
    "default_settings",
    "get_index",
    "get_settings",
    "get_uptime",
    "lifespan",
    "router",
    "templates",
]

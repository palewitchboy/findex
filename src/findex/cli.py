"""The command-line interface: ``findex index|search|stats``.

This module is the *only* place in findex that touches the terminal, the exit
code and the logging configuration. Everything below it is a library that
returns values; the CLI turns those values into a rich table or into one JSON
document on stdout, and every diagnostic goes to stderr.

Three rules, all of them testable:

* **stdout carries data, stderr carries diagnostics.** ``findex search q --json``
  must pipe into ``jq`` without a single log line in the way.
* **Expected errors exit nonzero with one clear line** - a missing index is
  exit 1, a bad query is exit 2 (the same code click uses for usage errors),
  and no traceback ever reaches the user.
* **Logging is configured here and only here.** Library modules call
  ``logging.getLogger(__name__)``; ``-v`` turns the root logger to INFO,
  ``-vv`` to DEBUG, and a ``RichHandler`` writes to stderr.

``python -m findex.index`` and ``python -m findex.search`` keep working: they
delegate to :func:`index_main` / :func:`search_main`, thin shims that prepend
the subcommand name and call the same typer app.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.metadata
import json
import logging
import os
import pickle
import sys
import time
import tracemalloc
from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from pathlib import Path
from typing import Annotated, Literal, NoReturn, cast

import httpx
import typer
from rich.console import Console
from rich.live import Live
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

from findex import __version__
from findex.crawl import (
    USER_AGENT,
    CrawlConfig,
    CrawledPage,
    Crawler,
    CrawlStats,
    make_client,
)
from findex.index import (
    DEFAULT_EXECUTOR,
    ExecutorName,
    Index,
    build_index,
    document_paths,
    validate_workers,
)
from findex.query import parse
from findex.ranking import BM25, Scorer, Searcher, TfIdf
from findex.ranking import explain as explain_ranking
from findex.search import Engine
from findex.search import explain as explain_boolean
from findex.search import parse as parse_boolean
from findex.search import search as boolean_search
from findex.stats import Measurement, Stats, fmt_bytes, lazy_stats, measure
from findex.store import load, save, storage_of

#: Exit codes. 2 is what click itself uses for a malformed command line, so a
#: script can tell "you typed it wrong" from "the thing you asked for failed".
EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_USAGE = 2

Storage = Literal["objects", "arrays"]
Format = Literal["pickle", "json"]
ScorerName = Literal["bm25", "tfidf"]

app = typer.Typer(
    name="findex",
    help="A small search engine: build an inverted index, rank documents, "
    "print snippets.",
    no_args_is_help=True,
    add_completion=True,
)

console = Console()
err_console = Console(stderr=True)
log = logging.getLogger(__name__)

_logging_configured = False


# --------------------------------------------------------------------------- plumbing
def configure_logging(verbose: int) -> None:
    """Configure the root logger once, at the CLI boundary.

    ``-v`` selects INFO, ``-vv`` (and anything above) selects DEBUG, no flag
    keeps WARNING - so a normal run is silent unless something is wrong. The
    handler is a :class:`RichHandler` on stderr: colour when a terminal is
    attached, plain text when the output is redirected, and never stdout.

    Called again in the same process (tests, ``CliRunner``) it only adjusts the
    level; adding a second handler would duplicate every line.
    """
    level = (
        logging.DEBUG
        if verbose >= 2
        else logging.INFO
        if verbose == 1
        else logging.WARNING
    )
    global _logging_configured
    root = logging.getLogger()
    if _logging_configured:
        root.setLevel(level)
        return
    handler = RichHandler(
        console=err_console,
        show_path=False,
        show_time=verbose >= 2,
        markup=False,
        rich_tracebacks=False,
    )
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[handler],
        force=True,  # this is the entry point; take the root logger over
    )
    _logging_configured = True


def _version() -> str:
    """Installed distribution version, falling back to the source tree."""
    try:
        return importlib.metadata.version("findex")
    except importlib.metadata.PackageNotFoundError:
        return __version__


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"findex {_version()}")
        raise typer.Exit(EXIT_OK)


def _fail(message: str, code: int = EXIT_FAILURE) -> NoReturn:
    """Print one line to stderr and exit nonzero - never a traceback."""
    err_console.print(Text(f"error: {message}", style="bold red"))
    raise typer.Exit(code)


def _print_json(payload: dict[str, object]) -> None:
    """The machine-readable contract: one JSON document, nothing else.

    ``ensure_ascii`` keeps the output writable on any console encoding (a
    Windows cp1251 stdout would otherwise choke on a Cyrillic title); ``jq``
    and Python's ``json`` both decode the escapes back to the same string.
    """
    sys.stdout.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _load_index(path: Path) -> Index:
    """Load an index or exit 1 with a readable message."""
    try:
        return load(path)
    except FileNotFoundError:
        _fail(f"index not found: {path} (build one with 'findex index')")
    except (OSError, ValueError, pickle.UnpicklingError, EOFError) as exc:
        _fail(f"cannot read index {path}: {exc}")


def _scorer(name: ScorerName) -> Scorer:
    return BM25() if name == "bm25" else TfIdf()


def _document_count(root: Path, limit: int | None) -> int | None:
    """Cheap upper bound for the progress bar; ``None`` when unknown (JSONL)."""
    if not root.is_dir():
        return None
    return len(document_paths(root, limit))


def _summary_table(title: str, rows: list[tuple[str, str]]) -> Table:
    table = Table(title=title, title_justify="left", show_header=False, box=None)
    table.add_column(style="dim", no_wrap=True)
    table.add_column(justify="right", style="bold")
    for label, value in rows:
        table.add_row(label, value)
    return table


# --------------------------------------------------------------------------- app
@app.callback()
def main(
    verbose: Annotated[
        int,
        typer.Option(
            "-v",
            "--verbose",
            count=True,
            help="Log to stderr: -v for INFO, -vv for DEBUG.",
        ),
    ] = 0,
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the version and exit.",
        ),
    ] = False,
) -> None:
    """findex - a search engine in one installable command."""
    configure_logging(verbose)
    del version  # consumed by the eager callback above


@app.command()
def index(
    corpus: Annotated[
        Path,
        typer.Argument(
            exists=True,
            readable=True,
            help="Directory of .txt files, or a .jsonl/.jsonl.gz file.",
        ),
    ],
    out: Annotated[
        Path, typer.Option("--out", "-o", help="Index file to write.")
    ] = Path("index.pkl"),
    limit: Annotated[
        int | None,
        typer.Option("--limit", min=1, help="Index only the first N documents."),
    ] = None,
    positions: Annotated[
        bool,
        typer.Option("--positions", help="Store token positions (needed for phrases)."),
    ] = False,
    storage: Annotated[
        Storage, typer.Option("--storage", help="Postings layout.")
    ] = "objects",
    executor: Annotated[
        ExecutorName,
        typer.Option(
            "--executor",
            "-e",
            help="Run the build in one process, a thread pool or a process pool.",
            case_sensitive=False,
        ),
    ] = DEFAULT_EXECUTOR,
    workers: Annotated[
        int | None,
        typer.Option(
            "--workers",
            "-w",
            min=1,
            help="Pool size for --executor threads|processes (default: min(8, CPUs)).",
        ),
    ] = None,
    fmt: Annotated[
        Format, typer.Option("--format", help="Serialization format.")
    ] = "pickle",
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit one JSON document on stdout.")
    ] = False,
    trace_memory: Annotated[
        bool,
        typer.Option(
            "--trace-memory",
            help="Measure peak memory with tracemalloc (accurate, ~5x slower).",
        ),
    ] = False,
) -> None:
    """Build the inverted index and save it to a file."""
    try:
        resolved_workers = validate_workers(workers, executor)
    except ValueError as exc:
        _fail(str(exc), EXIT_USAGE)
    if executor != "serial" and not corpus.is_dir():
        # A .jsonl corpus is one stream: there are no per-file chunks to map.
        # Falling back keeps `findex index corpus.jsonl` working out of the box;
        # the library call still raises for an *explicit* parallel request.
        log.warning("parallel indexing needs a directory of .txt files; using serial")
        executor = "serial"
        resolved_workers = 1
    total = _document_count(corpus, limit)
    peak: int | None = None
    if trace_memory:
        tracemalloc.start()
    started = time.perf_counter()
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold]indexing[/bold] {task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=err_console,
            transient=True,
            disable=as_json,
        ) as progress:
            task = progress.add_task(corpus.name or "corpus", total=total)

            def advance(_: int) -> None:
                progress.update(task, advance=1)

            try:
                built = build_index(
                    corpus,
                    limit=limit,
                    positions=positions,
                    storage=storage,
                    on_document=advance,
                    executor=executor,
                    workers=resolved_workers,
                )
            except (OSError, ValueError) as exc:
                # -vv logs the traceback (worker exceptions carry the remote
                # one as __cause__); the user still gets one line and exit 1.
                log.debug("index build failed", exc_info=True)
                _fail(str(exc))
            progress.update(task, completed=total or progress.tasks[0].total)
    finally:
        if trace_memory:
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
    elapsed = time.perf_counter() - started

    save_started = time.perf_counter()
    try:
        save(built, out, fmt=fmt)
    except OSError as exc:
        _fail(f"cannot write {out}: {exc}")
    save_elapsed = time.perf_counter() - save_started
    file_bytes = out.stat().st_size

    log.info(
        "indexed %d documents into %s (%s, %s)",
        built.num_docs,
        out,
        storage,
        fmt,
    )
    if as_json:
        _print_json(
            {
                "index": str(out),
                "format": fmt,
                "storage": storage,
                "executor": executor,
                "workers": resolved_workers,
                "documents": built.num_docs,
                "terms": len(built),
                "tokens": built.n_tokens,
                "postings": built.n_postings(),
                "positions": built.positions is not None,
                "build_seconds": round(elapsed, 3),
                "save_seconds": round(save_elapsed, 3),
                "peak_bytes": peak,
                "file_bytes": file_bytes,
            }
        )
        return

    rows: list[tuple[str, str]] = [
        ("documents", f"{built.num_docs:,}"),
        ("tokens", f"{built.n_tokens:,}"),
        ("executor", f"{executor} x {resolved_workers}"),
        ("build time", f"{elapsed:.2f} s"),
    ]
    if peak is not None:
        rows.append(("peak memory", fmt_bytes(peak)))
    rows.extend(
        [
            ("saved", f"{out} ({fmt}, {fmt_bytes(file_bytes)})"),
            ("save time", f"{save_elapsed:.2f} s"),
        ]
    )
    console.print(_summary_table(f"built {built!r}", rows))


def _ranked_payload(
    index: Index,
    query: str,
    scorer_name: ScorerName,
    k: int,
    explain: bool,
    snippets: bool,
) -> tuple[dict[str, object], str]:
    """Rank ``query`` and return the JSON payload plus the human explanation."""
    started = time.perf_counter()
    hits = Searcher(index, _scorer(scorer_name), snippets=snippets).search(query, k=k)
    elapsed_ms = (time.perf_counter() - started) * 1000
    explanation = explain_ranking(index, parse(query)) if explain else ""
    payload: dict[str, object] = {
        "query": query,
        "mode": "ranked",
        "scorer": scorer_name,
        "k": k,
        "count": len(hits),
        "elapsed_ms": round(elapsed_ms, 3),
        "results": [
            {
                "doc_id": hit.doc_id,
                "score": round(hit.score, 6),
                "title": hit.title,
                "path": index.doc_meta[hit.doc_id].path,
                "snippet": hit.snippet,
            }
            for hit in hits
        ],
    }
    if explain:
        payload["explain"] = explanation
    return payload, explanation


def _term_frequencies(index: Index, query: str, doc_id: int) -> dict[str, int]:
    """Positive query terms and their tf in ``doc_id`` (Boolean mode)."""
    parts: dict[str, int] = {}
    for group in parse_boolean(query):
        for clause in group:
            if clause.negated:
                continue
            for doc, tf in index.tf_items(clause.term):
                if doc == doc_id:
                    parts[clause.term] = tf
                    break
    return parts


def _boolean_payload(
    index: Index, query: str, engine: Engine, k: int, explain: bool
) -> tuple[dict[str, object], str]:
    started = time.perf_counter()
    hits = boolean_search(index, query, engine=engine, k=k)
    elapsed_ms = (time.perf_counter() - started) * 1000
    explanation = explain_boolean(index, query) if explain else ""
    payload: dict[str, object] = {
        "query": query,
        "mode": "boolean",
        "engine": engine,
        "k": k,
        "count": len(hits),
        "elapsed_ms": round(elapsed_ms, 3),
        "results": [
            {
                "doc_id": doc_id,
                "title": index.doc_meta[doc_id].title,
                "path": index.doc_meta[doc_id].path,
                "n_tokens": index.doc_meta[doc_id].n_tokens,
                "tf": _term_frequencies(index, query, doc_id),
            }
            for doc_id in hits
        ],
    }
    if explain:
        payload["explain"] = explanation
    return payload, explanation


@app.command()
def search(
    index_path: Annotated[
        Path,
        typer.Argument(metavar="INDEX", help="Index file written by 'findex index'."),
    ],
    query: Annotated[
        list[str],
        typer.Argument(
            help='Query: "war AND peace", "war OR (peace NOT treaty)", "a phrase".'
        ),
    ],
    scorer_name: Annotated[
        ScorerName, typer.Option("--scorer", "-s", help="Ranking function.")
    ] = "bm25",
    k: Annotated[int, typer.Option("--k", "-k", min=1, help="How many hits.")] = 10,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit one JSON document on stdout.")
    ] = False,
    explain: Annotated[
        bool, typer.Option("--explain", help="Print the parse tree and df per term.")
    ] = False,
    boolean: Annotated[
        bool,
        typer.Option("--boolean", help="Lab 2 Boolean matching (doc ids, no scores)."),
    ] = False,
    engine: Annotated[
        Engine, typer.Option("--engine", help="Boolean engine (with --boolean).")
    ] = "merge",
    no_snippets: Annotated[
        bool, typer.Option("--no-snippets", help="Skip the +/-80 char context window.")
    ] = False,
) -> None:
    """Search an index and print ranked (or Boolean) results."""
    query_text = " ".join(query).strip()
    if not query_text:
        _fail("empty query", EXIT_USAGE)

    index = _load_index(index_path)
    try:
        try:
            if boolean:
                payload, explanation = _boolean_payload(
                    index, query_text, engine, k, explain
                )
            else:
                payload, explanation = _ranked_payload(
                    index, query_text, scorer_name, k, explain, not no_snippets
                )
        except ValueError as exc:
            _fail(str(exc), EXIT_USAGE)

        if as_json:
            _print_json(payload)
            return

        if explain:
            console.print(Text(explanation, style="dim"))
        if boolean:
            _print_boolean_table(payload)
        else:
            _print_ranked_table(payload)
    finally:
        index.close()


def _print_ranked_table(payload: dict[str, object]) -> None:
    query = str(payload["query"])
    scorer_name = str(payload["scorer"])
    count = int(str(payload["count"]))
    table = Table(
        title=f'{count} hit(s) for "{query}" - {scorer_name}',
        title_justify="left",
        header_style="bold cyan",
    )
    table.add_column("#", justify="right", width=3)
    table.add_column("score", justify="right", width=8)
    table.add_column("doc", justify="right", width=4, style="dim")
    table.add_column("title", overflow="fold")
    table.add_column("snippet", overflow="fold", style="dim")
    results = cast("list[dict[str, object]]", payload["results"])
    for rank, item in enumerate(results, start=1):
        table.add_row(
            str(rank),
            f"{float(str(item['score'])):.3f}",
            str(item["doc_id"]),
            Text(str(item["title"])),
            Text(str(item["snippet"])),
        )
    console.print(table)


def _print_boolean_table(payload: dict[str, object]) -> None:
    query = str(payload["query"])
    engine = str(payload["engine"])
    count = int(str(payload["count"]))
    table = Table(
        title=f'{count} hit(s) for "{query}" - boolean, {engine}',
        title_justify="left",
        header_style="bold cyan",
        expand=True,
    )
    table.add_column("#", justify="right", width=3)
    table.add_column("doc", justify="right", width=4, style="dim")
    table.add_column("title", overflow="fold")
    table.add_column("tokens", justify="right")
    table.add_column("tf", style="dim")
    results = cast("list[dict[str, object]]", payload["results"])
    for rank, item in enumerate(results, start=1):
        tf = cast("dict[str, int]", item["tf"])
        table.add_row(
            str(rank),
            str(item["doc_id"]),
            Text(str(item["title"])),
            f"{int(str(item['n_tokens'])):,}",
            ", ".join(f"{term}={freq}" for term, freq in tf.items()),
        )
    console.print(table)


def _looks_like_corpus(path: Path) -> bool:
    return path.is_dir() or path.suffix.lower() in {".jsonl", ".ndjson", ".gz"}


def _index_stats_payload(index: Index, source: Path, top: int) -> dict[str, object]:
    ranked = sorted(index.postings.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:top]
    return {
        "source": str(source),
        "mode": "index",
        "storage": storage_of(index),
        "documents": index.num_docs,
        "terms": len(index),
        "tokens": index.n_tokens,
        "postings": index.n_postings(),
        "avg_doc_length": round(index.avg_doc_length, 3),
        "file_bytes": source.stat().st_size,
        "top": [{"term": term, "df": len(postings)} for term, postings in ranked],
    }


@app.command()
def stats(
    source: Annotated[
        Path,
        typer.Argument(
            exists=True,
            readable=True,
            help="A corpus (directory / .jsonl) or an index file.",
        ),
    ],
    top: Annotated[int, typer.Option("--top", min=1, help="How many top terms.")] = 10,
    limit: Annotated[
        int | None,
        typer.Option(
            "--limit", min=1, help="Only the first N documents (corpus mode)."
        ),
    ] = None,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit one JSON document on stdout.")
    ] = False,
) -> None:
    """Corpus statistics (Lab 1) or a summary of a saved index."""
    if _looks_like_corpus(source):
        measurement: Measurement = measure(lambda: lazy_stats(source, limit, top))
        corpus_stats: Stats = measurement.result
        payload: dict[str, object] = {
            "source": str(source),
            "mode": "corpus",
            "documents": corpus_stats.documents,
            "tokens": corpus_stats.tokens,
            "vocabulary": corpus_stats.vocabulary,
            "elapsed_seconds": round(measurement.elapsed, 3),
            "peak_bytes": measurement.peak_bytes,
            "top": [{"term": term, "count": count} for term, count in corpus_stats.top],
        }
    else:
        index = _load_index(source)
        try:
            payload = _index_stats_payload(index, source, top)
        finally:
            index.close()

    if as_json:
        _print_json(payload)
        return

    rows: list[tuple[str, str]] = [
        ("source", str(payload["source"])),
        ("documents", f"{int(str(payload['documents'])):,}"),
        ("tokens", f"{int(str(payload['tokens'])):,}"),
    ]
    if payload["mode"] == "corpus":
        rows.append(("vocabulary", f"{int(str(payload['vocabulary'])):,}"))
        rows.append(("elapsed", f"{float(str(payload['elapsed_seconds'])):.2f} s"))
        rows.append(("peak memory", fmt_bytes(int(str(payload["peak_bytes"])))))
    else:
        rows.append(("terms", f"{int(str(payload['terms'])):,}"))
        rows.append(("postings", f"{int(str(payload['postings'])):,}"))
        rows.append(("avg doc length", f"{float(str(payload['avg_doc_length'])):.1f}"))
        rows.append(("storage", str(payload["storage"])))
    console.print(_summary_table(f"stats: {payload['mode']}", rows))

    table = Table(
        title=f"top {top} terms", title_justify="left", header_style="bold cyan"
    )
    table.add_column("#", justify="right", width=3)
    table.add_column("term")
    column = "df" if payload["mode"] == "index" else "count"
    table.add_column(column, justify="right")
    top_rows = cast("list[dict[str, object]]", payload["top"])
    for rank, item in enumerate(top_rows, start=1):
        table.add_row(str(rank), str(item["term"]), f"{int(str(item[column])):,}")
    console.print(table)


# --------------------------------------------------------------------------- crawl
def _page_record(page: CrawledPage) -> dict[str, object]:
    """One JSON line. ``text`` is the field Lab 1's ``iter_documents`` reads."""
    return {
        "url": page.url,
        "text": page.text,
        "title": page.title,
        "status": page.status,
        "bytes": page.n_bytes,
        "depth": page.depth,
        "links": len(page.links),
        "elapsed": round(page.elapsed, 3),
        "fetched_at": page.fetched_at_iso,
    }


def _crawl_line(stats: CrawlStats, started: float, config: CrawlConfig) -> Text:
    """The one-line live display: pages, bytes, errors, retries, rate."""
    elapsed = max(time.perf_counter() - started, 1e-9)
    in_flight = max(0, stats.started - stats.pages - stats.errors)
    line = Text()
    line.append("crawl ", style="bold")
    line.append(f"{stats.pages}/{config.max_pages} pages", style="bold cyan")
    line.append(f" · {fmt_bytes(stats.bytes)}")
    line.append(f" · {stats.pages / elapsed:.2f} pages/s")
    line.append(f" · {in_flight} in flight")
    if stats.errors:
        line.append(f" · {stats.errors} errors", style="yellow")
    if stats.retries:
        line.append(f" · {stats.retries} retries", style="yellow")
    line.append(f" · {elapsed:.1f}s", style="dim")
    return line


async def _refresh_live(
    live: Live,
    stats: CrawlStats,
    started: float,
    config: CrawlConfig,
    interval: float = 0.25,
) -> None:
    while True:
        live.update(_crawl_line(stats, started, config))
        await asyncio.sleep(interval)


@contextmanager
def _crawl_log(path: Path) -> Generator[logging.Logger, None, None]:
    """Attach a per-URL log file to ``findex.crawl`` for the duration of a run.

    The library only calls ``logging.getLogger(__name__)``; the file handler is
    wired here, at the CLI boundary, exactly like the stderr handler in
    :func:`configure_logging`. ``propagate`` is switched off unless the user
    asked for ``-v``: without that, an INFO line would reach the ``rich``
    handler on stderr (handler levels, unlike logger levels, are not consulted
    during propagation) and a normal run would stop being quiet.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("findex.crawl")
    previous_level, previous_propagate = logger.level, logger.propagate
    handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = logging.getLogger().level <= logging.INFO
    try:
        yield logger
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
        logger.propagate = previous_propagate
        handler.close()


async def _crawl_async(
    config: CrawlConfig,
    seed: str,
    out: Path,
    client: httpx.AsyncClient,
    *,
    live: Live | None,
    stats: CrawlStats,
) -> tuple[int, float]:
    """Stream pages to ``out`` as they arrive; return (pages written, seconds).

    The file is opened and written with :func:`asyncio.to_thread`: a 200 KB
    write is fast, but it is still blocking I/O, and "fast" is not "not
    blocking". The event loop stays free to keep the next requests in flight.
    """
    started = time.perf_counter()
    written = 0
    refresher: asyncio.Task[None] | None = None
    handle = await asyncio.to_thread(out.open, "w", encoding="utf-8", newline="\n")
    try:
        if live is not None:
            refresher = asyncio.create_task(_refresh_live(live, stats, started, config))
        crawler = Crawler(config, stats=stats, client=client)
        async for page in crawler.run([seed]):
            line = json.dumps(_page_record(page), ensure_ascii=False) + "\n"
            await asyncio.to_thread(handle.write, line)
            written += 1
    finally:
        if refresher is not None:
            refresher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await refresher
        await asyncio.to_thread(handle.close)
    return written, time.perf_counter() - started


@app.command()
def crawl(
    start_url: Annotated[
        str, typer.Argument(metavar="START_URL", help="Seed URL (http or https).")
    ],
    max_pages: Annotated[
        int, typer.Option("--max-pages", min=1, help="Hard cap on fetched pages.")
    ] = 50,
    out: Annotated[
        Path,
        typer.Option("--out", "-o", help="JSON Lines corpus to write as pages arrive."),
    ] = Path("crawl.jsonl"),
    concurrency: Annotated[
        int, typer.Option("--concurrency", "-c", min=1, help="Worker coroutines.")
    ] = 10,
    per_host: Annotated[
        int, typer.Option("--per-host", min=1, help="In-flight requests per host.")
    ] = 2,
    delay: Annotated[
        float,
        typer.Option(
            "--delay", min=0.0, help="Minimum seconds between requests to a host."
        ),
    ] = 1.0,
    allow_domain: Annotated[
        list[str] | None,
        typer.Option(
            "--allow-domain",
            help="Only crawl these hosts (repeatable; default: the seed's host).",
        ),
    ] = None,
    max_depth: Annotated[
        int | None,
        typer.Option(
            "--max-depth", min=0, help="Stop following links past this depth."
        ),
    ] = None,
    timeout: Annotated[
        float, typer.Option("--timeout", min=0.1, help="Per-request timeout, seconds.")
    ] = 10.0,
    retries: Annotated[
        int, typer.Option("--retries", min=1, help="Attempts per URL (1 = no retry).")
    ] = 3,
    log_file: Annotated[
        Path, typer.Option("--log-file", help="Per-URL status log.")
    ] = Path("crawl.log"),
    user_agent: Annotated[
        str, typer.Option("--user-agent", help="Honest UA with a contact URL.")
    ] = USER_AGENT,
    no_robots: Annotated[
        bool,
        typer.Option(
            "--no-robots",
            help="Ignore robots.txt. Only for a local mirror of a site you own.",
        ),
    ] = False,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit one JSON document on stdout.")
    ] = False,
) -> None:
    """Crawl a site politely and stream the pages into a JSON Lines corpus."""
    try:
        config = CrawlConfig(
            max_pages=max_pages,
            concurrency=concurrency,
            per_host=per_host,
            delay=delay,
            max_depth=max_depth,
            allowed_domains=tuple(allow_domain or ()),
            user_agent=user_agent,
            timeout=timeout,
            max_attempts=retries,
            respect_robots=not no_robots,
        )
    except ValueError as exc:
        _fail(str(exc), EXIT_USAGE)
    if not start_url.lower().startswith(("http://", "https://")):
        _fail(f"seed must be an http(s) URL: {start_url!r}", EXIT_USAGE)

    stats = CrawlStats()
    # A live display is for a human at a terminal; under --json or a redirect
    # it would be noise on stderr (and in the tests it would be a race).
    show_live = not as_json and err_console.is_terminal
    written = 0
    elapsed = 0.0
    try:
        # Built *outside* the loop: creating an httpx client loads the CA
        # bundle, which is ~1 s of pure blocking on a cold cache. Doing it here
        # keeps the event loop free from the first instruction.
        client = make_client(config)
        started = time.perf_counter()
        live = Live(
            _crawl_line(stats, started, config),
            console=err_console,
            refresh_per_second=8,
            auto_refresh=False,
            transient=True,
        )
        screen: AbstractContextManager[object] = (
            live if show_live else nullcontext()
        )
        with _crawl_log(log_file), screen:
            written, elapsed = asyncio.run(
                _crawl_async(
                    config,
                    start_url,
                    out,
                    client,
                    live=live if show_live else None,
                    stats=stats,
                )
            )
    except ValueError as exc:  # a bad seed, a bad allowlist, a bad config
        _fail(str(exc), EXIT_USAGE)
    except OSError as exc:
        _fail(f"cannot write {out}: {exc}")

    rate = written / elapsed if elapsed else 0.0
    log.info(
        "crawled %d page(s), %s, %d error(s) from %s",
        written,
        fmt_bytes(stats.bytes),
        stats.errors,
        start_url,
    )
    if written == 0:
        _fail(f"no page could be fetched from {start_url} (see {log_file})")

    if as_json:
        _print_json(
            {
                "seed": start_url,
                "out": str(out),
                "pages": written,
                "bytes": stats.bytes,
                "errors": stats.errors,
                "retries": stats.retries,
                "robots_skipped": stats.robots_skipped,
                "domain_skipped": stats.domain_skipped,
                "duplicates": stats.duplicates,
                "queue_peak": stats.queue_peak,
                "max_pages": config.max_pages,
                "concurrency": config.concurrency,
                "per_host": config.per_host,
                "delay": config.delay,
                "allowed_domains": list(config.allowed_domains),
                "respect_robots": config.respect_robots,
                "user_agent": config.user_agent,
                "elapsed_seconds": round(elapsed, 3),
                "pages_per_second": round(rate, 3),
                "log": str(log_file),
            }
        )
        return

    rows: list[tuple[str, str]] = [
        ("pages", f"{written:,} of {config.max_pages:,} allowed"),
        ("bytes", fmt_bytes(stats.bytes)),
        ("errors", f"{stats.errors:,}"),
        ("retries", f"{stats.retries:,}"),
        ("elapsed", f"{elapsed:.2f} s"),
        ("speed", f"{rate:.2f} pages/s"),
        ("written", str(out)),
        ("log", str(log_file)),
    ]
    console.print(_summary_table(f"crawled {start_url}", rows))


# --------------------------------------------------------------------------- serve
@app.command()
def serve(
    index_path: Annotated[
        Path,
        typer.Option("--index", "-i", help="Index file written by 'findex index'."),
    ] = Path("index.pkl"),
    host: Annotated[str, typer.Option("--host", help="Bind address.")] = "127.0.0.1",
    port: Annotated[
        int, typer.Option("--port", min=1, max=65535, help="TCP port to listen on.")
    ] = 8000,
    workers: Annotated[
        int,
        typer.Option("--workers", "-w", min=1, help="Server processes (1 per core)."),
    ] = 1,
) -> None:
    """Serve the search index over HTTP (Lab 7's FastAPI app, via uvicorn)."""
    if not index_path.exists():
        _fail(f"index not found: {index_path} (build one with 'findex index')")
    # Configuration goes through the environment, not through the import: the
    # app reads INDEX_PATH with pydantic-settings when it is created, and with
    # --workers > 1 uvicorn re-imports it in every child process.
    os.environ["INDEX_PATH"] = str(index_path)
    try:
        import uvicorn  # noqa: PLC0415 - fastapi/uvicorn stay out of every other command
    except ImportError:  # pragma: no cover - the dependency is declared
        _fail("uvicorn is not installed; run 'uv sync'")
    log.info(
        "serving %s on http://%s:%d with %d worker(s)", index_path, host, port, workers
    )
    uvicorn.run(
        "findex.web:app",
        host=host,
        port=port,
        workers=workers,
        log_level="info",
    )


# --------------------------------------------------------------------------- shims
def _run(command: str, argv: list[str] | None) -> None:
    """Run ``app`` as if the user had typed ``findex <command> ...``."""
    args = list(sys.argv[1:] if argv is None else argv)
    app(args=[command, *args])


def index_main(argv: list[str] | None = None) -> None:
    """``python -m findex.index`` - kept as a thin shim over ``findex index``."""
    _run("index", argv)


def search_main(argv: list[str] | None = None) -> None:
    """``python -m findex.search`` - kept as a thin shim over ``findex search``."""
    _run("search", argv)


def stats_main(argv: list[str] | None = None) -> None:
    """``python -m findex.stats`` - kept as a thin shim over ``findex stats``."""
    _run("stats", argv)


def crawl_main(argv: list[str] | None = None) -> None:
    """``python -m findex.crawl`` - kept as a thin shim over ``findex crawl``."""
    _run("crawl", argv)


__all__ = [
    "app",
    "configure_logging",
    "crawl_main",
    "index_main",
    "main",
    "search_main",
    "stats_main",
]

"""Lab 8 benchmark: profile first, optimise second, prove the output did not move.

    uv run python bench/lab8.py                # everything (~4 minutes)
    uv run python bench/lab8.py --skip-identity  # skip the 500-snippet A/B

What is measured, and why
-------------------------
1. **Profiles** - ``cProfile`` on ``findex search`` with snippets, without
   snippets, and on ``findex index``. The ranking was already known to be fast
   (0.46 ms in Lab 7); this lab starts by measuring *where the rest goes*.
2. **Flame graphs** - a real sampling profile of the two implementations, drawn
   as SVG by :func:`render_flame_svg`. ``py-spy 0.4.3`` cannot attach on this
   Windows host (``Failed to find python version from target process``), so the
   sampler here reads ``sys._current_frames()`` from a thread once per
   millisecond and the SVG is rendered from those samples. Same idea, no
   ptrace, no admin rights; the run is in-process, so the numbers include the
   same work the profiler would see.
3. **The snippet A/B** - Lab 3's implementation (kept verbatim in
   ``bench/lab8_legacy.py``) against the shipped one, in one process, on the
   same corpus: cold, warm-up and steady state, plus a **byte-for-byte** check
   of 50 queries × k=10.
4. **The scorer A/B** - the Python per-posting loop against the NumPy one, on
   the same matched sets, for common / medium / rare / multi-term queries, with
   an equality check on every ranking.
5. **Scaling** - the same two scorers on synthetic posting lists of increasing
   ``df``, because 48 documents are too few to show what vectorisation is for.
6. **The price of positions** - the index rebuilt with ``positions=True``:
   bytes and seconds, the measurement the brief asks for.

Every number lands in ``bench/lab8_results.json``.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import math
import os
import platform
import pstats
import re
import subprocess
import sys
import threading
import time
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:  # --no-project runs have no installed pkg
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "bench") not in sys.path:
    sys.path.insert(0, str(ROOT / "bench"))

import lab8_legacy as legacy  # noqa: E402
from findex import snippet as shipped  # noqa: E402
from findex.index import Index, build_index  # noqa: E402
from findex.query import parse  # noqa: E402
from findex.ranking import BM25, Searcher  # noqa: E402
from findex.store import load, save_pickle  # noqa: E402

WORK = ROOT / "bench" / "_tmp" / "lab8"
INDEX = WORK / "index.pkl"
RESULTS = ROOT / "bench" / "lab8_results.json"
FIGURES = ROOT / "docs" / "figures"

#: 48 books, the same subset Labs 5-7 used: the full 120 would multiply every
#: snippet measurement by 2.5 for the same lesson.
DOCUMENTS = 48
QUERY = "love and death"
K = 10

#: The A/B query set: single terms, multi-term queries, operators, words that
#: match inside nothing (``warfare``), a missing term, and the two most common
#: words in the corpus. 50 of them, as the brief asks.
QUERIES: tuple[str, ...] = (
    "love", "death", "war", "peace", "the", "king", "queen", "sea", "ship",
    "love and death", "war and peace", "the king and the queen",
    "man woman child", "night day light", "heart soul mind", "god heaven hell",
    "money power law", "mother father son", "time world life", "book letter word",
    "sword blood battle", "ship sea storm", "house door window", "horse road town",
    "doctor patient medicine", "school teacher child", "river bank oak",
    "treasure gold silver", "fox wolf bear", "sun moon star", "fire water earth",
    "friend enemy stranger", "city street crowd", "music song dance",
    "science nature reason", "history people nation", "war OR peace",
    "love OR hate", "king NOT queen", "death AND life", "war AND NOT peace",
    "don't", "well-known", "warfare", "warning", "nothing", "zzzznotaterm",
    "great", "little", "young",
)  # fmt: skip


# --------------------------------------------------------------------------- utils
def ms(value: float) -> float:
    """Seconds to milliseconds, rounded to something a table can hold."""
    return round(value * 1000.0, 3)


def timed(fn: Callable[[], object], repeat: int = 3) -> float:
    """Best-of-``repeat`` wall time in seconds - the minimum, not the mean.

    The minimum is the least noisy statistic available on a shared machine: the
    fastest run is the one with the fewest interruptions, which is what the
    other machine would also see.
    """
    fn()
    best = math.inf
    for _ in range(repeat):
        started = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - started)
    return best


def machine_info() -> dict[str, Any]:
    import numpy as np

    return {
        "cpu": platform.processor() or platform.machine(),
        "logical_cpus": os.cpu_count(),
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "numpy": np.__version__,
    }


# --------------------------------------------------------------------------- sampler
class Sampler:
    """A self-profiler: read the calling thread's stack, count it, draw it.

    ``py-spy`` is a ptrace-based sampler and cannot attach on this host, but the
    idea - read the stack, count it, draw it - needs no privileges at all when
    the process profiles itself: ``sys._current_frames()`` hands back the frame
    of every thread, and walking ``f_back`` gives the call chain.

    ``threading.Event.wait(0.001)`` is *not* a 1 ms timer on Windows - it is a
    ~31 ms timer (measured), which would yield thirty samples for a whole
    query. So the loop samples as fast as the interpreter allows (~125 000
    samples/s on this host) and counts identical stacks instead of storing
    every sample: the profile is uniform in time either way, and the memory
    stays bounded by the number of distinct stacks.
    """

    def __init__(self) -> None:
        self.counts: Counter[tuple[str, ...]] = Counter()
        self._target = threading.get_ident()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(0):
            frame = sys._current_frames().get(self._target)
            if frame is None:
                continue
            stack: list[str] = []
            while frame is not None:
                code = frame.f_code
                stack.append(f"{Path(code.co_filename).name}:{code.co_name}")
                frame = frame.f_back
            stack.reverse()
            self.counts[tuple(stack)] += 1

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def __enter__(self) -> Sampler:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


def _flame_tree(samples: Mapping[tuple[str, ...], int]) -> dict[str, Any]:
    root: dict[str, Any] = {"name": "all", "value": 0, "children": {}}
    for stack, count in samples.items():
        node = root
        node["value"] += count
        for frame in stack:
            node = node["children"].setdefault(
                frame, {"name": frame, "value": 0, "children": {}}
            )
            node["value"] += count
    return root


def render_flame_svg(
    samples: Mapping[tuple[str, ...], int], path: Path, *, title: str
) -> int:
    """Draw a flame graph from stack samples; returns the deepest row drawn.

    Layout is the classic one: depth on y, time on x, every frame's width equal
    to the number of samples that passed through it. Colours are a stable hash
    of the frame name, so the same function is the same colour in both graphs.
    """
    tree = _flame_tree(samples)
    total = max(1, tree["value"])
    row_height = 17
    width = 1200
    rows = 0

    rects: list[str] = []
    labels: list[str] = []

    def colour(name: str) -> str:
        hue = sum(ord(char) * 37 for char in name) % 360
        return f"hsl({hue}, 62%, 58%)"

    def walk(node: dict[str, Any], x: int, depth: int) -> int:
        nonlocal rows
        rows = max(rows, depth)
        span = int(width * node["value"] / total)
        if node["name"] != "all" and span > 0:
            y = depth * row_height
            rects.append(
                f'<rect x="{x}" y="{y}" width="{span}" height="{row_height - 1}" '
                f'fill="{colour(str(node["name"]))}" stroke="#1b1b1f" '
                f'stroke-width="0.5"/>'
            )
            if span > 60:
                text = str(node["name"]).replace("&", "&amp;").replace("<", "&lt;")
                labels.append(
                    f'<text x="{x + 4}" y="{y + 12}" font-size="11" '
                    f'fill="#101014">{text}</text>'
                )
        cursor = x
        for child in sorted(
            node["children"].values(), key=lambda item: item["value"], reverse=True
        ):
            cursor = walk(child, cursor, depth + 1)
        return x + span

    walk(tree, 0, 0)
    height = (rows + 1) * row_height + 46
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#14141a"/>',
        f'<text x="10" y="20" font-size="14" fill="#e8e8f0" '
        f'font-family="monospace">{title}</text>',
        f'<text x="10" y="36" font-size="11" fill="#9a9aa8" '
        f'font-family="monospace">{total} samples · 1 ms interval · '
        f"width = share of samples</text>",
        '<g transform="translate(0, 42)">',
        *rects,
        *labels,
        "</g>",
        "</svg>",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(svg), encoding="utf-8")
    return rows


# --------------------------------------------------------------------------- profiles
def top_functions(profile: cProfile.Profile, limit: int = 12) -> list[dict[str, Any]]:
    """The top ``limit`` functions by cumulative time, as plain dicts."""
    stream = io.StringIO()
    stats = pstats.Stats(profile, stream=stream)
    stats.sort_stats("cumulative")
    rows: list[dict[str, Any]] = []
    for (filename, line, name), values in stats.stats.items():
        calls, _ncalls, tottime, cumtime = values[:4]
        rows.append(
            {
                "function": f"{Path(filename).name}:{line}({name})",
                "file": str(Path(filename).name),
                "calls": calls,
                "tottime_ms": ms(tottime),
                "cumtime_ms": ms(cumtime),
            }
        )
    rows.sort(key=lambda row: float(row["cumtime_ms"]), reverse=True)
    return rows[:limit]


def profile_callable(fn: Callable[[], object], limit: int = 12) -> dict[str, Any]:
    """Run ``fn`` under cProfile and return the wall time plus the top functions.

    ``cProfile`` instruments every call, so ``wall_ms`` is inflated - that is
    expected and is why the ranking of functions, not the absolute number, is
    what gets read.
    """
    profile = cProfile.Profile()
    started = time.perf_counter()
    profile.enable()
    fn()
    profile.disable()
    elapsed = time.perf_counter() - started
    return {"wall_ms": ms(elapsed), "top": top_functions(profile, limit)}


def _snippet_terms() -> frozenset[str]:
    return parse(QUERY).terms()


def _all_snippets(
    index: Index, module: Any, hits: Sequence[int], terms: frozenset[str]
) -> None:
    """One query worth of snippets with ``module``'s implementation.

    ``hits`` and ``terms`` are passed in so the loop measures **snippets**, not
    a fresh search per iteration: a new ``Searcher`` per call would rebuild the
    match cache and the NumPy posting views inside the flame graph.
    """
    for doc_id in hits:
        module.snippet(index, doc_id, terms)


# --------------------------------------------------------------------------- index
def build_compact_index() -> dict[str, Any]:
    """Build the 48-document index the way Lab 7's server embeds it."""
    started = time.perf_counter()
    index = build_index(
        ROOT / "data", limit=DOCUMENTS, positions=False, storage="arrays"
    )
    seconds = time.perf_counter() - started
    save_pickle(index, INDEX)
    info = {
        "documents": index.num_docs,
        "terms": len(index),
        "postings": index.n_postings(),
        "tokens": index.n_tokens,
        "avg_doc_length": round(index.avg_doc_length, 1),
        "file_bytes": INDEX.stat().st_size,
        "build_seconds": round(seconds, 2),
    }
    index.close()
    return info


def positions_cost() -> dict[str, Any]:
    """How much the index grows when token positions are stored (brief option a)."""
    plain_seconds = timed(
        lambda: build_index(
            ROOT / "data", limit=DOCUMENTS, positions=False, storage="arrays"
        ),
        repeat=1,
    )
    placed_seconds = timed(
        lambda: build_index(
            ROOT / "data", limit=DOCUMENTS, positions=True, storage="arrays"
        ),
        repeat=1,
    )
    sizes: dict[int, int] = {}
    for positions in (False, True):
        index = build_index(
            ROOT / "data", limit=DOCUMENTS, positions=positions, storage="arrays"
        )
        path = WORK / f"positions_{int(positions)}.pkl"
        save_pickle(index, path)
        sizes[int(positions)] = path.stat().st_size
        index.close()
    return {
        "without_bytes": sizes[0],
        "with_bytes": sizes[1],
        "ratio": round(sizes[1] / sizes[0], 3),
        "build_seconds_without": round(plain_seconds, 2),
        "build_seconds_with": round(placed_seconds, 2),
    }


# --------------------------------------------------------------------------- snippets
def snippet_ab(index: Index, *, check_identity: bool = True) -> dict[str, Any]:
    """Lab 3's snippet against the shipped one: latency and byte-identical output."""
    hits = [hit.doc_id for hit in Searcher(index, snippets=False).search(QUERY, k=K)]
    terms = parse(QUERY).terms()

    def run(module: Any) -> float:
        started = time.perf_counter()
        for doc_id in hits:
            module.snippet(index, doc_id, terms)
        return time.perf_counter() - started

    shipped.configure_text_cache(len(index.doc_meta))
    legacy.configure_text_cache(len(index.doc_meta))

    legacy._read_document.cache_clear()
    cold_legacy = run(legacy)
    shipped._load_document.cache_clear()
    cold_shipped = run(shipped)
    second_shipped = run(shipped)
    steady_shipped = min(run(shipped) for _ in range(3))
    second_legacy = run(legacy)
    steady_legacy = min(run(legacy) for _ in range(3))

    identity = {"queries": 0, "snippets": 0, "mismatches": 0}
    if check_identity:
        for query in QUERIES:
            query_terms = parse(query).terms()
            identity["queries"] += 1
            for hit in Searcher(index, snippets=False).search(query, k=K):
                identity["snippets"] += 1
                if legacy.snippet(index, hit.doc_id, query_terms) != shipped.snippet(
                    index, hit.doc_id, query_terms
                ):
                    identity["mismatches"] += 1

    def legacy_query() -> None:
        """A whole search: score, then Lab 3's snippets."""
        for hit in Searcher(index, BM25(), snippets=False).search(QUERY, k=K):
            legacy.snippet(index, hit.doc_id, terms)

    def shipped_query() -> None:
        """A whole search through the shipped path, a fresh Searcher per query."""
        Searcher(index, BM25(), snippets=True).search(QUERY, k=K)

    full_legacy = timed(legacy_query, repeat=5)
    full_shipped = timed(shipped_query, repeat=5)

    fast_snippet_stats = {
        "documents": len(hits),
        "steady_ms": ms(steady_shipped),
        "cold_ms": ms(cold_shipped),
        "warmup_ms": ms(second_shipped),
        "legacy_steady_ms": ms(steady_legacy),
        "legacy_cold_ms": ms(cold_legacy),
        "legacy_warmup_ms": ms(second_legacy),
        "steady_speedup": round(steady_legacy / steady_shipped, 1),
        "cold_ratio": round(cold_shipped / cold_legacy, 2),
        "full_ms": ms(full_shipped),
        "legacy_full_ms": ms(full_legacy),
        "full_speedup": round(full_legacy / full_shipped, 1),
        "identity": identity,
        "cache_maxsize": shipped._load_document.cache_info().maxsize,
    }
    return fast_snippet_stats


# --------------------------------------------------------------------------- scoring
def scoring_ab(index: Index) -> dict[str, Any]:
    """Python loop against the NumPy scorer, on identical match sets."""
    labels = {
        "love and death": "3 terms, all common",
        "the": "1 term, df = 48",
        "love": "1 term, df = 42",
        "treasure": "1 term, df = 28",
        "buffalo": "1 term, df = 9",
        "zzzznotaterm": "1 term, df = 0",
        "war and peace": "3 terms, df = 38",
    }
    rows: list[dict[str, Any]] = []
    identical = True
    for query, label in labels.items():
        reference = Searcher(index, BM25(), snippets=False, engine="python")
        vectorised = Searcher(index, BM25(), snippets=False, engine="numpy")

        def run_python(ref: Searcher = reference, text: str = query) -> object:
            return ref.search(text, k=K)

        def run_numpy(vec: Searcher = vectorised, text: str = query) -> object:
            return vec.search(text, k=K)

        old = [(hit.doc_id, hit.score) for hit in reference.search(query, k=K)]
        new = [(hit.doc_id, hit.score) for hit in vectorised.search(query, k=K)]
        same = [doc for doc, _ in old] == [doc for doc, _ in new] and all(
            abs(left[1] - right[1]) <= 1e-9
            for left, right in zip(old, new, strict=True)
        )
        identical = identical and same
        python_ms = ms(timed(run_python, repeat=20))
        numpy_ms = ms(timed(run_numpy, repeat=20))
        rows.append(
            {
                "query": query,
                "label": label,
                "docs": len(reference._match(query).docs),
                "python_ms": python_ms,
                "numpy_ms": numpy_ms,
                "speedup": round(python_ms / numpy_ms, 2) if numpy_ms else None,
                "identical": same,
            }
        )
    return {"rows": rows, "identical": identical}


def _python_scoring(
    ids_list: list[int], tfs_list: list[int], idf: float, norm: Any, k1: float
) -> dict[int, float]:
    """The per-posting loop as a standalone function, so it can be timed."""
    scores: dict[int, float] = {}
    for position, doc_id in enumerate(ids_list):
        tf = tfs_list[position]
        scores[doc_id] = (
            scores.get(doc_id, 0.0) + idf * tf * (k1 + 1.0) / (tf + float(norm[doc_id]))
        )
    return scores


def _numpy_scoring(
    doc_ids: Any, tfs: Any, idf: float, norm: Any, k1: float, n_docs: int
) -> Any:
    """The same arithmetic as one array expression."""
    import numpy as np

    scores = np.zeros(n_docs, dtype=np.float64)
    weights = tfs.astype(np.float64)
    scores[doc_ids] += idf * weights * (k1 + 1.0) / (weights + norm[doc_ids])
    return scores


def vector_scaling() -> dict[str, Any]:
    """The same arithmetic at four sizes: where vectorisation actually pays.

    A synthetic posting list is enough here - the question is "how does the cost
    per posting scale", not "what does this corpus contain". ``df`` postings
    scored against a 100 000-document length array, Python loop versus NumPy
    expression, best of five.
    """
    from functools import partial

    import numpy as np

    rows: list[dict[str, Any]] = []
    n_docs = 100_000
    doc_len = np.full(n_docs, 120.0)
    avgdl = 120.0
    k1, b = 1.5, 0.75
    norm = k1 * (1.0 - b + b * doc_len / avgdl)
    rng = np.random.default_rng(20261112)

    for df in (48, 480, 4_800, 48_000):
        doc_ids = np.sort(rng.choice(n_docs, size=df, replace=False)).astype(np.int32)
        tfs = rng.integers(1, 12, size=df).astype(np.int32)
        idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
        python_call = partial(
            _python_scoring, doc_ids.tolist(), tfs.tolist(), idf, norm, k1
        )
        numpy_call = partial(_numpy_scoring, doc_ids, tfs, idf, norm, k1, n_docs)
        python_s = timed(python_call, repeat=5)
        numpy_s = timed(numpy_call, repeat=5)
        rows.append(
            {
                "df": df,
                "python_ms": ms(python_s),
                "numpy_ms": ms(numpy_s),
                "speedup": round(python_s / numpy_s, 2) if numpy_s else None,
                "python_ns_per_posting": round(python_s / df * 1e9, 1),
                "numpy_ns_per_posting": round(numpy_s / df * 1e9, 1),
            }
        )
    return {"rows": rows, "documents": n_docs}


# --------------------------------------------------------------------------- tokenizer
_LEGACY_APOSTROPHES = str.maketrans(
    {"\u2019": "'", "\u02bc": "'", "\u2018": "'"}
)
_LEGACY_WORD = re.compile(r"[^\W_]+(?:['\-][^\W_]+)*")


def _legacy_tokenize(text: str) -> Iterator[str]:
    """Lab 2's tokenizer, kept verbatim for the A/B: ``str.translate`` per active char.

    ``str.translate`` with a mapping walks every character through a
    Python-level table; that was 39% of a serial index build and the profile says
    so. The shipped version replaces three characters with three guarded
    ``str.replace`` calls instead. Same tokens, different machine code path -
    and ``build_ab`` proves the resulting indexes are ``==``.
    """
    text = unicodedata.normalize("NFC", text).casefold().translate(_LEGACY_APOSTROPHES)
    for match in _LEGACY_WORD.finditer(text):
        yield match.group()


def _build_with(tokenizer: Callable[[str], Iterator[str]]) -> Index:
    """Run ``build_index`` with ``findex.index.tokenize`` swapped out.

    ``build_index`` looks the function up on the module at call time, so a
    temporary patch is enough to build "the old way" in the same process -
    without a second copy of ``index.py``.
    """
    import findex.index as index_module  # noqa: PLC0415

    original = index_module.tokenize
    index_module.tokenize = tokenizer  # type: ignore[assignment]
    try:
        return build_index(
            ROOT / "data", limit=DOCUMENTS, positions=False, storage="arrays"
        )
    finally:
        index_module.tokenize = original


def build_ab() -> dict[str, Any]:
    """``findex index`` before and after the tokenizer fix, on the same corpus.

    The index itself must not change: ``legacy == shipped`` is the proof that
    the 39% saving is a speedup and not a quieter bug.
    """
    started = time.perf_counter()
    legacy_index = _build_with(_legacy_tokenize)
    legacy_seconds = time.perf_counter() - started
    shipped_seconds = timed(
        lambda: build_index(
            ROOT / "data", limit=DOCUMENTS, positions=False, storage="arrays"
        ),
        repeat=1,
    )
    shipped_index = build_index(
        ROOT / "data", limit=DOCUMENTS, positions=False, storage="arrays"
    )
    identical = legacy_index == shipped_index
    tokens = shipped_index.n_tokens
    legacy_index.close()
    shipped_index.close()
    return {
        "legacy_seconds": round(legacy_seconds, 2),
        "shipped_seconds": round(shipped_seconds, 2),
        "speedup": round(legacy_seconds / shipped_seconds, 2),
        "identical": identical,
        "tokens": tokens,
        "seconds_per_million_tokens_legacy": round(legacy_seconds / (tokens / 1e6), 2),
        "seconds_per_million_tokens_shipped": round(
            shipped_seconds / (tokens / 1e6), 2
        ),
    }


# --------------------------------------------------------------------------- pytest
def benchmark_tests() -> dict[str, Any]:
    """Run the ``pytest-benchmark`` gate as CI does and read back its JSON.

    The README should quote the same numbers CI enforces, so the benchmark step
    is executed here rather than retyped: ``--benchmark-json`` is the machine
    readable form of the table pytest prints.
    """
    out = WORK / "benchmark.json"
    if out.exists():
        out.unlink()
    started = time.perf_counter()
    completed = subprocess.run(  # noqa: S603 - a fixed argv, no shell
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_bench_perf.py",
            "--benchmark-only",
            "--benchmark-json",
            str(out),
            "-q",
            "--no-header",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    seconds = time.perf_counter() - started
    if completed.returncode != 0 or not out.exists():
        return {"error": (completed.stdout + completed.stderr)[-800:]}
    data = json.loads(out.read_text(encoding="utf-8"))
    rows = [
        {
            "name": str(item["name"]).removeprefix("test_bench_"),
            "min_ms": round(float(item["stats"]["min"]) * 1000, 3),
            "median_ms": round(float(item["stats"]["median"]) * 1000, 3),
            "rounds": int(item["stats"]["rounds"]),
        }
        for item in data["benchmarks"]
    ]
    collected = subprocess.run(  # noqa: S603 - a fixed argv, no shell
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "--no-header"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    total = 0
    for line in collected.stdout.splitlines():
        words = line.split()
        if len(words) >= 2 and words[1] == "tests" and words[0].isdigit():
            total = int(words[0])
    return {"rows": rows, "seconds": round(seconds, 1), "tests": total}


# --------------------------------------------------------------------------- run
def run_sampled(title: str, fn: Callable[[], object], path: Path) -> dict[str, Any]:
    """Sample ``fn`` and draw the flame graph it produced."""
    with Sampler() as sampler:
        fn()
    rows = render_flame_svg(sampler.counts, path, title=title)
    return {
        "file": str(path.relative_to(ROOT)).replace("\\", "/"),
        "samples": sampler.total,
        "stacks": len(sampler.counts),
        "depth": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="reuse bench/_tmp/lab8/index.pkl when it already exists",
    )
    parser.add_argument(
        "--skip-identity", action="store_true", help="skip the 50-query byte check"
    )
    args = parser.parse_args(argv)

    WORK.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    if args.skip_build and INDEX.exists():
        print(f"reusing {INDEX.relative_to(ROOT)}")
        existing = load(INDEX)
        info = {
            "documents": existing.num_docs,
            "terms": len(existing),
            "postings": existing.n_postings(),
            "tokens": existing.n_tokens,
            "avg_doc_length": round(existing.avg_doc_length, 1),
            "file_bytes": INDEX.stat().st_size,
            "build_seconds": None,
        }
        existing.close()
    else:
        print(f"building a {DOCUMENTS}-document index from data/ ...")
        info = build_compact_index()
        print(
            f"  {info['documents']} docs, {info['terms']:,} terms, "
            f"{info['file_bytes'] / 1e6:.1f} MB, {info['build_seconds']} s"
        )

    index = load(INDEX)
    try:
        print("profiling search with snippets (cold text cache) ...")
        shipped.configure_text_cache(len(index.doc_meta))

        def search_with_snippets() -> None:
            shipped._load_document.cache_clear()
            Searcher(index, BM25(), snippets=True).search(QUERY, k=K)

        profile_snippets = profile_callable(search_with_snippets)

        def search_without() -> None:
            Searcher(index, BM25(), snippets=False).search(QUERY, k=K)

        print("profiling search without snippets ...")
        profile_plain = profile_callable(search_without)
        print("profiling build_index (serial, 48 documents) ...")
        profile_build = profile_callable(
            lambda: build_index(
                ROOT / "data", limit=DOCUMENTS, positions=False, storage="arrays"
            )
        )

        print("sampling: search with snippets, Lab 3 implementation ...")
        legacy.configure_text_cache(len(index.doc_meta))
        sampled_hits = [
            hit.doc_id for hit in Searcher(index, snippets=False).search(QUERY, k=K)
        ]
        sampled_terms = _snippet_terms()

        def legacy_snippets() -> None:
            # One second of work, cold text cache, so the flame graph is the
            # cost of a query rather than the cost of one query's warm-up.
            legacy._read_document.cache_clear()
            deadline = time.perf_counter() + 1.0
            while time.perf_counter() < deadline:
                _all_snippets(index, legacy, sampled_hits, sampled_terms)

        flame_before = run_sampled(
            "search with snippets, cold text cache - Lab 3 implementation (before)",
            legacy_snippets,
            FIGURES / "lab8_flame.svg",
        )
        print("sampling: search with snippets, shipped implementation ...")
        # Two passes first: the first builds the casefolded twins, the second
        # is the steady state a server lives in.
        _all_snippets(index, shipped, sampled_hits, sampled_terms)
        _all_snippets(index, shipped, sampled_hits, sampled_terms)

        def shipped_snippets() -> None:
            deadline = time.perf_counter() + 1.0
            while time.perf_counter() < deadline:
                _all_snippets(index, shipped, sampled_hits, sampled_terms)

        flame_after = run_sampled(
            "search with snippets, steady state - Lab 8 implementation (after)",
            shipped_snippets,
            FIGURES / "lab8_flame_after.svg",
        )

        print("snippet A/B (cold, warm-up, steady, identity) ...")
        snippets = snippet_ab(index, check_identity=not args.skip_identity)
        print(
            f"  steady {snippets['legacy_steady_ms']} ms -> "
            f"{snippets['steady_ms']} ms, "
            f"{snippets['identity']['mismatches']} mismatches"
        )

        print("scorer A/B (python vs numpy) ...")
        scoring = scoring_ab(index)
        print(f"  identical rankings: {scoring['identical']}")

        print("scaling: the same expression at four posting sizes ...")
        scaling = vector_scaling()

        print("build A/B: tokenizer (translate vs guarded replace) ...")
        build = build_ab()
        print(
            f"  {build['legacy_seconds']} s -> {build['shipped_seconds']} s, "
            f"identical: {build['identical']}"
        )

        print("price of positions ...")
        positions = positions_cost()

        print("pytest-benchmark gate ...")
        bench = benchmark_tests()
        print(f"  {len(bench.get('rows', []))} tests")

        try:
            import pytest_benchmark  # noqa: PLC0415

            benchmark_version = getattr(pytest_benchmark, "__version__", "?")
        except ImportError:  # pragma: no cover
            benchmark_version = "absent"

        results: dict[str, Any] = {
            "machine": machine_info(),
            "workload": {"query": QUERY, "k": K, "documents": DOCUMENTS},
            "index": info,
            "profiles": {
                "search_snippets": profile_snippets,
                "search_no_snippets": profile_plain,
                "build_index": profile_build,
            },
            "flame": {
                "before": flame_before,
                "after": flame_after,
                "sampler": (
                    "in-process sampling thread over sys._current_frames(), "
                    "unthrottled (~15 000 samples/s; a 1 ms timer is a 31 ms "
                    "timer on Windows)"
                ),
                "pyspy": PYSPY_NOTE,
            },
            "snippets": snippets,
            "scoring": scoring,
            "scaling": scaling,
            "build": build,
            "benchmark": bench,
            "positions": positions,
            "pytest_benchmark": benchmark_version,
        }
    finally:
        index.close()

    RESULTS.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\nwrote {RESULTS.relative_to(ROOT)}")
    return 0


#: What ``py-spy 0.4.3`` does on this host, recorded verbatim so the README can
#: say *why* the flame graph comes from an in-process sampler instead.
PYSPY_NOTE = (
    "py-spy 0.4.3 fails on this Windows host: 'Failed to find python version "
    "from target process' when attaching with --pid, and 'Failed to get process "
    "executable name ... (os error 31)' when spawning. The flame graphs here "
    "come from an in-process sampling thread instead (sys._current_frames, "
    "1 ms interval) rendered to SVG by render_flame_svg()."
)


if __name__ == "__main__":
    raise SystemExit(main())

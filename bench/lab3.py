"""Lab 3 measurements: sanity checks, precision@5, the cache, and top-k cost.

Run:  uv run python bench/lab3.py [corpus_dir] [--limit N]

Writes ``bench/lab3_results.json``. Every number in the README's Lab 3 section
comes from that file.

Sections
--------
1. build   - the index *with* positions (phrases need them), timed and traced.
2. sanity  - the three checks the lab asks for:
             a rare term lifts its own document,
             a phrase query is more precise than its AND,
             BM25 prefers the short document, TF-IDF cannot.
3. p@5     - the ten hand-labelled queries, both scorers.
4. cache   - cold vs warm vs no cache, with the log line as evidence.
5. top-k   - ``heapq.nlargest`` against ``sorted(...)[:k]`` on 1k/100k/1M docs.
"""

from __future__ import annotations

import argparse
import heapq
import json
import logging
import random
import re
import statistics
import sys
import time
import tracemalloc
from operator import itemgetter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from findex.index import Index, build_index  # noqa: E402
from findex.query import parse  # noqa: E402
from findex.ranking import BM25, Searcher, TfIdf, rank  # noqa: E402
from findex.store import load, save_pickle  # noqa: E402
from lab3_queries import PHRASE_CANDIDATES, QUERIES  # noqa: E402

TMP = HERE / "_tmp"
K = 5


# --------------------------------------------------------------------------- helpers
def chrono(fn):
    """Time a call without tracemalloc (tracing slows the I/O path 10x)."""
    start = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - start


def timed_peak(fn):
    """Return ``(result, seconds, peak_bytes)`` with tracemalloc around the call."""
    tracemalloc.start()
    start = time.perf_counter()
    try:
        result = fn()
        elapsed = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, elapsed, peak


def median_ms(fn, reps: int = 5) -> float:
    samples = []
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return round(statistics.median(samples) * 1000, 4)


def mb(n_bytes: int) -> float:
    return round(n_bytes / (1024 * 1024), 1)


def round3(x: float) -> float:
    return round(x, 3)


class _ListHandler(logging.Handler):
    """Collect log messages so the JSON can quote the real 'cache hit' line."""

    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


# --------------------------------------------------------------------------- 1. build
def build_study(root: Path, limit: int | None) -> tuple[Index, dict[str, object]]:
    index, build_s, peak = timed_peak(
        lambda: build_index(root, limit=limit, positions=True)
    )
    assert isinstance(index, Index)

    TMP.mkdir(exist_ok=True)
    path = TMP / "l3.pkl"
    save_s = chrono(lambda: save_pickle(index, path))[1]
    size_mb = mb(path.stat().st_size)
    _, load_s = chrono(lambda: load(path))

    row = {
        "docs": index.num_docs,
        "tokens": index.n_tokens,
        "terms": len(index),
        "postings": index.n_postings(),
        "avg_doc_length": round(index.avg_doc_length, 1),
        "positions": True,
        "build_s": round(build_s, 1),
        "peak_mb": mb(peak),
        "file_mb": size_mb,
        "save_s": round(save_s, 2),
        "load_s": round(load_s, 2),
    }
    return index, row


def positions_cost(root: Path, limit: int | None) -> dict[str, object]:
    """The price of the positions Lab 3 needs, on a small slice of the corpus."""
    small = min(limit or 20, 20)
    without, _, peak_off = timed_peak(lambda: build_index(root, limit=small))
    with_pos, _, peak_on = timed_peak(
        lambda: build_index(root, limit=small, positions=True)
    )
    TMP.mkdir(exist_ok=True)
    off, on = TMP / "pos_off3.pkl", TMP / "pos_on3.pkl"
    save_pickle(without, off)
    save_pickle(with_pos, on)
    return {
        "limit": small,
        "peak_off_mb": mb(peak_off),
        "peak_on_mb": mb(peak_on),
        "file_off_mb": mb(off.stat().st_size),
        "file_on_mb": mb(on.stat().st_size),
    }


def lab2_reference() -> dict[str, object] | None:
    """Lab 2's full-corpus build (no positions), for an honest comparison."""
    path = HERE / "lab2_results.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    slots = next(r for r in data["memory_study"] if r["variant"] == "slots")
    return {
        "docs": slots["docs"],
        "build_s": slots["build_s"],
        "peak_mb": slots["peak_mb"],
        "file_mb": slots["file_mb"],
        "positions": False,
    }


# --------------------------------------------------------------------------- 2. sanity
def pick_rare_and_common(index: Index) -> tuple[str, str]:
    """A genuinely rare word and a mid-frequency one, deterministically.

    ``df == 1`` alone would hand back OCR debris like ``aaarh``, so the rare
    term must look like a word (alphabetic, 6-12 letters, a vowel, no ``aaa``)
    and come from the document whose length is closest to the corpus average -
    "the rarest word in the most typical book", not in the 1.3M-token
    encyclopedia or the 3k-token poem.
    """
    values = set("aeiouy")
    candidates = [
        term
        for term in index
        if term.isalpha()
        and 6 <= len(term) <= 12
        and index.df(term) == 1
        and not re.search(r"(.)\1\1", term)
        and any(ch in values for ch in term)
    ]
    average = index.avg_doc_length

    def distance(term: str) -> tuple[float, str]:
        doc_id = next(iter(index.doc_ids(term)))
        return (abs(index.doc_length(doc_id) - average), term)

    rare = min(candidates, key=distance)
    target = index.num_docs // 2
    common = min(
        (t for t in index if t.isalpha() and len(t) >= 4),
        key=lambda t: (abs(index.df(t) - target), t),
    )
    return rare, common


def rare_term_study(index: Index) -> dict[str, object]:
    """A rare term must lift its only document above everything else."""
    rare, common = pick_rare_and_common(index)
    rare_doc = next(iter(index.doc_ids(rare)))
    query = f"{common} OR {rare}"

    row: dict[str, object] = {
        "query": query,
        "rare_term": rare,
        "rare_df": index.df(rare),
        "rare_doc": rare_doc,
        "rare_doc_title": index.doc_meta[rare_doc].title,
        "common_term": common,
        "common_df": index.df(common),
        "docs": index.num_docs,
    }
    for name, scorer in (("bm25", BM25()), ("tfidf", TfIdf())):
        hits = rank(index, parse(query), scorer, k=index.num_docs)
        order = [hit.doc_id for hit in hits]
        rare_score = next(h.score for h in hits if h.doc_id == rare_doc)
        other = max((h.score for h in hits if h.doc_id != rare_doc), default=0.0)
        row[f"rank_{name}"] = order.index(rare_doc) + 1
        row[f"matched_{name}"] = len(hits)
        row[f"rare_score_{name}"] = round3(rare_score)
        row[f"best_other_{name}"] = round3(other)
        row[f"ratio_{name}"] = round3(rare_score / other) if other else None
    return row


def phrase_study(index: Index) -> dict[str, object]:
    """A phrase query keeps only documents where the words are adjacent."""
    assert index.positions is not None, "the phrase check needs positions"
    chosen: dict[str, object] | None = None
    best_gap = -1
    for phrase in PHRASE_CANDIDATES:
        words = phrase.split()
        and_node = parse(" ".join(words))
        phrase_node = parse(f'"{phrase}"')
        and_docs = and_node.evaluate(index)
        phrase_docs = phrase_node.evaluate(index)
        gap = len(and_docs) - len(phrase_docs)
        if gap > best_gap:
            best_gap = gap
            and_hits = rank(index, and_node, BM25(), k=K)
            phrase_hits = rank(index, phrase_node, BM25(), k=K)
            chosen = {
                "phrase": phrase,
                "and_count": len(and_docs),
                "phrase_count": len(phrase_docs),
                "and_top5_with_phrase": sum(
                    1 for h in and_hits if h.doc_id in phrase_docs
                ),
                "phrase_top5_with_phrase": sum(
                    1 for h in phrase_hits if h.doc_id in phrase_docs
                ),
                "and_top5": [h.title for h in and_hits],
                "phrase_top5": [h.title for h in phrase_hits],
            }
    assert chosen is not None
    and_count = int(chosen["and_count"])  # type: ignore[call-overload]
    phrase_count = int(chosen["phrase_count"])  # type: ignore[call-overload]
    chosen["reduction"] = round(and_count / phrase_count, 1) if phrase_count else None
    return chosen


def length_study(index: Index) -> dict[str, object]:
    """BM25 normalises by document length; TF-IDF does not."""
    term = ""
    short = long = 0
    short_len = long_len = 0
    best_spread = 0.0
    for candidate, postings in index.postings.items():
        if not candidate.isalpha() or len(candidate) < 5:
            continue  # a word makes a readable example; "1994" does not
        if not 2 <= len(postings) <= 30:
            continue  # with df == N, TF-IDF's idf collapses to zero
        shortest: tuple[int, int] | None = None
        longest: tuple[int, int] | None = None
        for doc_id, tf in postings.items():
            if tf != 1:
                continue
            length = index.doc_length(doc_id)
            if shortest is None or length < shortest[1]:
                shortest = (doc_id, length)
            if longest is None or length > longest[1]:
                longest = (doc_id, length)
        if shortest is None or longest is None or shortest[0] == longest[0]:
            continue
        spread = longest[1] / shortest[1]
        if spread > best_spread:
            best_spread = spread
            term = candidate
            short, short_len = shortest
            long, long_len = longest
    assert term, "no term occurs exactly once in two documents"

    bm25, tfidf = BM25(), TfIdf()
    bm25_short = bm25(term, short, 1, index)
    bm25_long = bm25(term, long, 1, index)
    tfidf_short = tfidf(term, short, 1, index)
    tfidf_long = tfidf(term, long, 1, index)

    # Saturation, on the same term and document: what the 20th occurrence adds.
    one = bm25(term, short, 1, index)
    nineteen = bm25(term, short, 19, index)
    twenty = bm25(term, short, 20, index)

    return {
        "term": term,
        "short_doc": short,
        "short_title": index.doc_meta[short].title,
        "short_len": short_len,
        "long_doc": long,
        "long_title": index.doc_meta[long].title,
        "long_len": long_len,
        "spread": round(best_spread, 1),
        "avg_doc_length": round(index.avg_doc_length, 1),
        "bm25_short": round3(bm25_short),
        "bm25_long": round3(bm25_long),
        "bm25_ratio": round3(bm25_short / bm25_long),
        "tfidf_short": round3(tfidf_short),
        "tfidf_long": round3(tfidf_long),
        "tfidf_ratio": round3(tfidf_short / tfidf_long) if tfidf_long else None,
        "saturation": {
            "tf1": round3(one),
            "tf19": round3(nineteen),
            "tf20": round3(twenty),
            "gain_20th_pct_of_first": round(100 * (twenty - nineteen) / one, 2),
        },
    }


# --------------------------------------------------------------------------- 3. p@5
_WARNED: set[str] = set()


def resolve(index: Index, stems: tuple[str, ...]) -> set[int]:
    """Map path stems to doc ids; warn (do not lie) when a label is missing."""
    by_stem = {Path(meta.path).stem: doc_id for doc_id, meta in index.doc_meta.items()}
    missing = [stem for stem in stems if stem not in by_stem and stem not in _WARNED]
    if missing:
        _WARNED.update(missing)
        print(f"  ! labels not in this index (partial corpus?): {missing}")
    return {by_stem[stem] for stem in stems if stem in by_stem}


def precision_study(index: Index, scorer, name: str) -> dict[str, object]:
    """precision@5 plus MRR, with the denominator written down.

    With 120 books and 1-2 known-relevant documents per query, strict
    precision@5 (divide by 5) can never exceed 0.40 and says nothing about
    order. So each row reports ``found`` (how many relevant documents made the
    top 5), ``p_at_5`` = found / min(5, |relevant|), and the reciprocal rank of
    the first relevant hit. The mean MRR is what actually separates the two
    scorers here.
    """
    rows = []
    for case in QUERIES:
        relevant = resolve(index, case.relevant)
        hits = rank(index, parse(case.query), scorer, k=K)
        ids = [hit.doc_id for hit in hits]
        found = len(set(ids) & relevant)
        first = next(
            (pos + 1 for pos, doc_id in enumerate(ids) if doc_id in relevant), None
        )
        rows.append(
            {
                "query": case.query,
                "relevant": len(relevant),
                "found": found,
                "p_at_5": round(found / min(K, len(relevant)), 2) if relevant else 0.0,
                "strict_p_at_5": round(found / K, 2),
                "first_rank": first,
                "rr": round(1 / first, 3) if first else 0.0,
                "top5": [hit.title for hit in hits],
            }
        )
    return {
        "scorer": name,
        "k": K,
        "mean_precision": round(sum(r["p_at_5"] for r in rows) / len(rows), 2),
        "mean_strict_precision": round(
            sum(r["strict_p_at_5"] for r in rows) / len(rows), 2
        ),
        "mrr": round(sum(r["rr"] for r in rows) / len(rows), 3),
        "first_at_1": sum(1 for r in rows if r["first_rank"] == 1),
        "rows": rows,
    }


def cache_study(index: Index, rounds: int = 20) -> dict[str, object]:
    query = "war AND peace"
    cached = Searcher(index, BM25(), snippets=False)
    _, cold_s = chrono(lambda: cached.search(query, k=10))
    warm_ms = median_ms(lambda: cached.search(query, k=10), reps=50)
    info = cached.cache_info()

    logger = logging.getLogger("findex.ranking")
    handler = _ListHandler()
    previous_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        cached.search(query, k=10)  # this one must log a cache hit
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)

    def run_all(searcher: Searcher) -> float:
        start = time.perf_counter()
        for _ in range(rounds):
            for case in QUERIES:
                searcher.search(case.query, k=K)
        return (time.perf_counter() - start) * 1000

    no_cache = Searcher(index, BM25(), snippets=False, cache_size=0)
    with_cache = Searcher(index, BM25(), snippets=False)
    no_cache_ms = run_all(no_cache)
    with_cache_ms = run_all(with_cache)
    calls = rounds * len(QUERIES)

    return {
        "query": query,
        "rounds": rounds,
        "cold_ms": round(cold_s * 1000, 3),
        "warm_ms": warm_ms,
        "hits": info.hits,
        "misses": info.misses,
        "maxsize": info.maxsize,
        "no_cache_total_ms": round(no_cache_ms, 2),
        "cached_total_ms": round(with_cache_ms, 2),
        "calls": calls,
        "no_cache_us_per_call": round(no_cache_ms * 1000 / calls, 1),
        "cached_us_per_call": round(with_cache_ms * 1000 / calls, 1),
        "speedup": round(no_cache_ms / with_cache_ms, 2) if with_cache_ms else None,
        "cache_hits": with_cache.cache_info().hits,
        "cache_misses": with_cache.cache_info().misses,
        "log_line": handler.messages[-1] if handler.messages else "",
    }


# --------------------------------------------------------------------------- 5. top-k
def topk_study() -> list[dict[str, object]]:
    rows = []
    rng = random.Random(7)
    for n in (1_000, 100_000, 1_000_000):
        items = [(i, rng.random()) for i in range(n)]
        nlargest_ms = median_ms(
            lambda items=items: heapq.nlargest(10, items, key=itemgetter(1)), reps=15
        )
        sorted_ms = median_ms(
            lambda items=items: sorted(items, key=itemgetter(1), reverse=True)[:10],
            reps=15,
        )
        rows.append(
            {
                "n": n,
                "nlargest_ms": nlargest_ms,
                "sorted_ms": sorted_ms,
                "ratio": round(sorted_ms / nlargest_ms, 2) if nlargest_ms else None,
            }
        )
    return rows


# --------------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path("data"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--skip-positions-cost",
        action="store_true",
        help="skip the two extra 20-document builds",
    )
    args = parser.parse_args()

    print(f"corpus: {args.root} (limit={args.limit})")
    index, build_row = build_study(args.root, args.limit)
    print(
        f"built {index!r} in {build_row['build_s']} s, "
        f"peak {build_row['peak_mb']} MB, file {build_row['file_mb']} MB"
    )

    pos = None if args.skip_positions_cost else positions_cost(args.root, args.limit)
    if pos:
        print(
            f"positions ({pos['limit']} docs): peak {pos['peak_off_mb']} -> "
            f"{pos['peak_on_mb']} MB, file {pos['file_off_mb']} -> "
            f"{pos['file_on_mb']} MB"
        )

    rare = rare_term_study(index)
    print(
        f"rare term: {rare['rare_term']!r} (df={rare['rare_df']}) is rank "
        f"{rare['rank_bm25']} for {rare['query']!r}"
    )

    phrase = phrase_study(index)
    print(
        f"phrase: {phrase['phrase']!r} keeps {phrase['phrase_count']} of "
        f"{phrase['and_count']} AND matches"
    )

    length = length_study(index)
    print(
        f"length: {length['term']!r} BM25 {length['bm25_short']} vs "
        f"{length['bm25_long']} (ratio {length['bm25_ratio']}), "
        f"TF-IDF {length['tfidf_short']} vs {length['tfidf_long']}"
    )

    tfidf = precision_study(index, TfIdf(), "tfidf")
    bm25 = precision_study(index, BM25(), "bm25")
    print(
        f"p@5: tfidf {tfidf['mean_precision']}, bm25 {bm25['mean_precision']} | "
        f"MRR: tfidf {tfidf['mrr']}, bm25 {bm25['mrr']}"
    )

    cache = cache_study(index)
    print(
        f"cache: cold {cache['cold_ms']} ms, warm {cache['warm_ms']} ms, "
        f"hits={cache['cache_hits']}/{cache['calls']}"
    )

    topk = topk_study()
    print(
        "top-k: "
        + ", ".join(f"n={r['n']}: {r['nlargest_ms']}/{r['sorted_ms']} ms" for r in topk)
    )

    results = {
        "corpus": str(args.root),
        "limit": args.limit,
        "index": build_row,
        "positions_cost": pos,
        "lab2_reference": lab2_reference(),
        "sanity": {
            "rare_term": rare,
            "phrase_vs_and": phrase,
            "bm25_length": length,
        },
        "precision": {"tfidf": tfidf, "bm25": bm25},
        "cache": cache,
        "topk": topk,
    }
    out = HERE / "lab3_results.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()

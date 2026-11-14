"""Lab 5 benchmark: serial vs threads vs processes, and why the numbers look so.

    uv run python bench/lab5.py                     # matrix + full corpus + plot
    uv run python bench/lab5.py --executor threads --workers 8 \\
        --out bench/lab5_freethreaded.json          # one row (free-threaded run)

What is measured, and how:

* the work is exactly what ``findex index`` does - ``build_partials`` then
  ``merge`` - so the *merge* column is the reduce step on its own, not a guess;
* every configuration runs once as a warm-up (spawn, imports, file cache) and
  then ``RUNS`` times; the table reports the **median** and the spread;
* **wall** is ``perf_counter`` around the whole build, **CPU** is the parent's
  ``psutil`` CPU time plus a 20 ms sampler over the process tree (children die
  before the run ends, so a final reading would miss them);
* **RSS** is the peak summed resident set of the tree, sampled the same way;
* **speedup** is ``serial_median / median``; **efficiency** divides it by the
  worker count.

The free-threaded build has no ``psutil`` wheel (and no ``matplotlib``), so this
file imports both lazily: ``psutil`` is optional, and the plot is a separate
step. Under ``python3.13t`` the CPU column falls back to ``time.process_time()``,
which is exact for threads (one process) and *parent-only* for processes - the
JSON says which one it is.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import pickle
import platform
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:  # --no-project runs have no installed pkg
    sys.path.insert(0, str(ROOT / "src"))

from findex.index import (  # noqa: E402
    DEFAULT_EXECUTOR,
    ExecutorName,
    build_partials,
    document_paths,
    merge,
)

try:  # psutil is a dev dependency and has no free-threaded wheel
    import psutil
except ImportError:  # pragma: no cover - exercised by the 3.13t run
    psutil = None  # type: ignore[assignment]

TMP = ROOT / "bench" / "_tmp" / "lab5"
RESULTS = ROOT / "bench" / "lab5_results.json"
FREETHREADED = ROOT / "bench" / "lab5_freethreaded.json"
FIGURE = ROOT / "docs" / "figures" / "lab5_speedup.png"
RUNS = 3
SUBSET_DOCS = 48
SAMPLE_SECONDS = 0.02
WORKER_LEVELS = (1, 2, 4, 8)


# --------------------------------------------------------------------------- machine
class _MemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def peak_working_set() -> int | None:
    """Peak RSS of this process - the fallback when ``psutil`` is missing."""
    if os.name != "nt":  # pragma: no cover - the project is Windows-first
        return None
    counters = _MemoryCounters()
    counters.cb = ctypes.sizeof(_MemoryCounters)
    try:
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        psapi = ctypes.windll.psapi  # type: ignore[attr-defined]
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_MemoryCounters),
            ctypes.c_ulong,
        ]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        handle = kernel32.GetCurrentProcess()
        ok = psapi.GetProcessMemoryInfo(
            handle, ctypes.byref(counters), counters.cb
        )
    except (AttributeError, OSError):  # pragma: no cover - defensive
        return None
    return int(counters.PeakWorkingSetSize) if ok else None


def logical_cpus() -> int:
    if psutil is not None:
        return int(psutil.cpu_count(logical=True) or os.cpu_count() or 1)
    return os.cpu_count() or 1


def physical_cpus() -> int | None:
    if psutil is None:
        return None
    return psutil.cpu_count(logical=False)


def cpu_name() -> str:
    """Friendly CPU name from the registry when available."""
    if os.name == "nt":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            with key:
                value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            return str(value).strip()
        except (OSError, ImportError):  # pragma: no cover - non-Windows
            pass
    return platform.processor() or platform.machine()


def machine() -> dict[str, Any]:
    total_ram = None
    if psutil is not None:
        total_ram = int(psutil.virtual_memory().total)
    return {
        "cpu": cpu_name(),
        "physical_cores": physical_cpus(),
        "logical_cores": logical_cpus(),
        "ram_bytes": total_ram,
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "gil_enabled": _gil_enabled(),
    }


def _gil_enabled() -> bool | None:
    probe = getattr(sys, "_is_gil_enabled", None)
    return bool(probe()) if probe is not None else None


# --------------------------------------------------------------------------- sampling
class TreeSampler(threading.Thread):
    """Sample CPU time and RSS of this process *and its children* every 20 ms.

    ``ProcessPoolExecutor`` shuts its workers down before ``merge`` returns, and
    a dead process cannot be asked how much CPU it used. Sampling while the pool
    is alive is the honest approximation: per-pid times are monotone, so taking
    the max per pid keeps a child's last reading as a constant contribution.
    """

    def __init__(self) -> None:
        super().__init__(daemon=True, name="lab5-sampler")
        self._stop = threading.Event()
        self.peak_rss = 0
        self.cpu: dict[int, float] = {}
        self._parent_base = 0.0

    def run(self) -> None:
        assert psutil is not None
        me = psutil.Process()
        times = me.cpu_times()
        # cpu_times() is cumulative since the process was created; the parent
        # already ran warm-ups, so subtract its reading at sampler start.
        self._parent_base = times.user + times.system
        while not self._stop.wait(SAMPLE_SECONDS):
            try:
                processes = [me, *me.children(recursive=True)]
            except psutil.Error:  # pragma: no cover - process vanished
                processes = [me]
            rss = 0
            for process in processes:
                try:
                    rss += process.memory_info().rss
                    times = process.cpu_times()
                except psutil.Error:
                    continue
                used = times.user + times.system
                if process.pid == me.pid:
                    used -= self._parent_base
                if used > self.cpu.get(process.pid, 0.0):
                    self.cpu[process.pid] = used
            self.peak_rss = max(self.peak_rss, rss)

    def stop(self) -> None:
        self._stop.set()


# --------------------------------------------------------------------------- measuring
def measure_once(
    tasks: list[tuple[str, int]],
    executor: ExecutorName,
    workers: int | None,
    *,
    positions: bool = False,
) -> dict[str, float]:
    """One timed build: partials, then merge, with the tree sampled throughout."""
    sampler = TreeSampler() if psutil is not None else None
    cpu_start = time.process_time()
    if sampler is not None:
        sampler.start()
    started = time.perf_counter()
    partials = build_partials(
        tasks, executor=executor, workers=workers, positions=positions
    )
    built = time.perf_counter()
    index = merge(partials)
    finished = time.perf_counter()
    cpu_parent = time.process_time() - cpu_start
    if sampler is not None:
        sampler.stop()
        sampler.join()

    cpu = cpu_parent if sampler is None else sum(sampler.cpu.values())
    rss = sampler.peak_rss if sampler is not None else peak_working_set()
    return {
        "wall": finished - started,
        "cpu": cpu,
        "cpu_parent": cpu_parent,
        "rss": float(rss or 0),
        "merge": finished - built,
        "terms": float(len(index)),
        "postings": float(index.n_postings()),
        "tokens": float(index.n_tokens),
    }


def summarize(samples: list[dict[str, float]]) -> dict[str, float]:
    """Median of every column, plus min/max wall so the spread is visible."""
    return {
        "wall": statistics.median(s["wall"] for s in samples),
        "wall_min": min(s["wall"] for s in samples),
        "wall_max": max(s["wall"] for s in samples),
        "cpu": statistics.median(s["cpu"] for s in samples),
        "cpu_parent": statistics.median(s["cpu_parent"] for s in samples),
        "rss": statistics.median(s["rss"] for s in samples),
        "merge": statistics.median(s["merge"] for s in samples),
        "terms": samples[0]["terms"],
        "postings": samples[0]["postings"],
        "tokens": samples[0]["tokens"],
    }


def run_config(
    tasks: list[tuple[str, int]],
    executor: ExecutorName,
    workers: int | None,
    runs: int = RUNS,
) -> dict[str, Any]:
    """Warm-up, then ``runs`` timed builds; returns the aggregated row."""
    measure_once(tasks, executor, workers)  # warm-up: imports, spawn, file cache
    samples = [measure_once(tasks, executor, workers) for _ in range(runs)]
    row: dict[str, Any] = {"executor": executor, "workers": workers or 1}
    row.update(summarize(samples))
    row["runs"] = runs
    return row


def subset_corpus(count: int) -> tuple[Path, list[Path]]:
    """The ``count`` smallest books, copied once into a stable directory."""
    corpus = TMP / f"corpus{count}"
    sources = sorted(
        (ROOT / "data").glob("*.txt"), key=lambda path: (path.stat().st_size, path.name)
    )
    if not sources:
        raise SystemExit("data/ is empty - run 'uv run python download_books.py'")
    chosen = sources[:count]
    if not corpus.exists() or len(list(corpus.glob("*.txt"))) != len(chosen):
        corpus.mkdir(parents=True, exist_ok=True)
        for source in chosen:
            (corpus / source.name).write_bytes(source.read_bytes())
    return corpus, sorted(corpus.glob("*.txt"))


# --------------------------------------------------------------------------- probes
def pickling_probe(tasks: list[tuple[str, int]], workers: int) -> dict[str, Any]:
    """How many bytes the workers send back, and what it costs to pickle them.

    This is a *parent-side* round trip: ``dumps`` measures the serialize cost
    that the pool pays before writing to the pipe, ``loads`` the deserialize
    cost the parent pays on arrival. The pipe itself is not measured here (the
    pool does not expose it), so the numbers are a floor, not the full tax.
    """
    partials = build_partials(tasks, executor="processes", workers=workers)
    total_bytes = 0
    send_bytes = 0
    dump_seconds = 0.0
    load_seconds = 0.0
    for task in tasks:
        send_bytes += len(
            pickle.dumps(task, protocol=pickle.HIGHEST_PROTOCOL)
        )
    for partial in partials:
        start = time.perf_counter()
        blob = pickle.dumps(partial, protocol=pickle.HIGHEST_PROTOCOL)
        dump_seconds += time.perf_counter() - start
        total_bytes += len(blob)
        start = time.perf_counter()
        pickle.loads(blob)
        load_seconds += time.perf_counter() - start
    return {
        "partials": len(partials),
        "sent_bytes": send_bytes,
        "bytes": total_bytes,
        "bytes_per_doc": total_bytes / max(1, len(partials)),
        "dump_seconds": dump_seconds,
        "load_seconds": load_seconds,
        "roundtrip_seconds": dump_seconds + load_seconds,
    }


def io_probe(root: Path, workers: int, runs: int = 3) -> dict[str, Any]:
    """Read every file twice: one thread, then a thread pool.

    The GIL is released during ``read()``, so threads overlap the waits. This is
    the counter-example the lab asks for: the *same* pool that does nothing for
    tokenization is a win for I/O.
    """
    paths = sorted(root.glob("*.txt"))
    total = 0

    def read_all() -> int:
        return sum(len(path.read_bytes()) for path in paths)

    serial_samples = []
    for _ in range(runs):
        start = time.perf_counter()
        total = read_all()
        serial_samples.append(time.perf_counter() - start)

    threaded_samples = []
    for _ in range(runs):
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            sizes = list(pool.map(lambda p: len(p.read_bytes()), paths))
        total = sum(sizes)
        threaded_samples.append(time.perf_counter() - start)

    serial_median = statistics.median(serial_samples)
    threaded_median = statistics.median(threaded_samples)
    return {
        "files": len(paths),
        "bytes": total,
        "workers": workers,
        "serial_seconds": serial_median,
        "threads_seconds": threaded_median,
        "speedup": serial_median / threaded_median if threaded_median else 0.0,
        "serial_mb_s": total / serial_median / 1e6 if serial_median else 0.0,
        "threads_mb_s": total / threaded_median / 1e6 if threaded_median else 0.0,
    }


# --------------------------------------------------------------------------- plot
def plot(results: dict[str, Any], path: Path) -> bool:
    """Speedup and wall time vs workers; returns False when matplotlib is absent."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - the 3.13t run
        return False

    matrix = results["matrix"]
    lanes = {"threads": ("#c0392b", "o"), "processes": ("#1f6feb", "s")}
    serial_wall = next(
        row["wall"] for row in matrix if row["executor"] == "serial"
    )

    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=140)
    ceiling = results.get("amdahl", {}).get("max_speedup")

    for executor, (color, marker) in lanes.items():
        rows = sorted(
            (row for row in matrix if row["executor"] == executor),
            key=lambda row: row["workers"],
        )
        xs = [row["workers"] for row in rows]
        left.plot(xs, [row["wall"] for row in rows], marker=marker, color=color,
                  label=executor)
        right.plot(xs, [row["speedup"] for row in rows], marker=marker, color=color,
                   label=executor)

    xs = sorted({row["workers"] for row in matrix})
    left.plot(xs, [serial_wall for _ in xs], linestyle="--", color="#555555",
              label="serial")
    right.plot(xs, xs, linestyle=":", color="#555555", label="ideal (linear)")
    if ceiling:
        right.axhline(ceiling, linestyle="-.", color="#888888", linewidth=1,
                      label=f"Amdahl ceiling {ceiling:.2f}x")

    free = results.get("free_threaded", {}).get("row")
    if free:
        version = results["free_threaded"]["machine"].get("python", "3.13t")
        right.scatter([free["workers"]], [free["speedup"]], marker="*", s=180,
                      color="#e67e22", zorder=5,
                      label=f"threads, free-threaded {version}")

    for axes, title in ((left, "Wall time"), (right, "Speedup")):
        axes.set_xlabel("workers")
        axes.grid(alpha=0.25)
        axes.set_title(title, loc="left")
        axes.legend(fontsize=8, frameon=False)
    left.set_ylabel("seconds")
    right.set_ylabel("x serial")
    right.set_xscale("log", base=2)
    right.set_xticks(xs)
    right.set_xticklabels([str(x) for x in xs])
    figure.suptitle(
        f"findex index build, {results['subset']['docs']} documents "
        f"({results['subset']['megabytes']:.1f} MB), median of {RUNS}",
        fontsize=10,
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)
    return True


# --------------------------------------------------------------------------- main
def full_corpus_rows(root: Path, workers: int | None) -> dict[str, Any]:
    """Serial vs processes on the whole corpus - one run each, labelled as such."""
    paths = document_paths(root)
    tasks = [(str(path), doc_id) for doc_id, path in enumerate(paths)]
    serial = measure_once(tasks, "serial", 1)
    parallel = measure_once(tasks, "processes", workers)
    return {
        "docs": len(paths),
        "megabytes": sum(path.stat().st_size for path in paths) / 1e6,
        "runs": 1,
        "rows": [
            {
                "executor": "serial",
                "workers": 1,
                "wall": serial["wall"],
                "cpu": serial["cpu"],
                "rss": serial["rss"],
                "merge": serial["merge"],
                "speedup": 1.0,
            },
            {
                "executor": "processes",
                "workers": workers or 1,
                "wall": parallel["wall"],
                "cpu": parallel["cpu"],
                "rss": parallel["rss"],
                "merge": parallel["merge"],
                "speedup": serial["wall"] / parallel["wall"],
            },
        ],
        "terms": serial["terms"],
        "postings": serial["postings"],
        "tokens": serial["tokens"],
    }


def single_mode(args: argparse.Namespace) -> int:
    """One configuration, printed and (optionally) written - the 3.13t path."""
    _, paths = subset_corpus(args.subset)
    tasks = [(str(path), doc_id) for doc_id, path in enumerate(paths)]
    row = run_config(tasks, args.executor, args.workers, runs=args.runs)
    # The baseline has to come from the *same* interpreter: a free-threaded
    # build is 5-10% slower single-threaded, and hiding that would flatter the
    # threaded row.
    serial = run_config(tasks, "serial", 1, runs=args.runs)
    row["serial_wall"] = serial["wall"]
    row["speedup"] = serial["wall"] / row["wall"] if row["wall"] else 0.0
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "machine": machine(),
        "subset": {
            "docs": len(paths),
            "megabytes": sum(path.stat().st_size for path in paths) / 1e6,
        },
        "runs": args.runs,
        "row": row,
        "serial": serial,
        "cpu_source": "psutil tree sampler"
        if psutil is not None
        else "time.process_time (parent only)",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


def attach(path: str) -> int:
    """Merge a single-run JSON (the free-threaded row) into the main results."""
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    extra = json.loads(Path(path).read_text(encoding="utf-8"))
    results["free_threaded"] = extra
    RESULTS.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"attached {path} as free_threaded")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executor", choices=("serial", "threads", "processes"))
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--subset", type=int, default=SUBSET_DOCS)
    parser.add_argument("--runs", type=int, default=RUNS)
    parser.add_argument("--out", default=str(RESULTS))
    parser.add_argument("--attach", default=None)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--skip-full", action="store_true")
    args = parser.parse_args(argv)

    if args.attach:
        return attach(args.attach)
    if args.executor:
        return single_mode(args)
    if args.plot_only:
        plot(json.loads(RESULTS.read_text(encoding="utf-8")), FIGURE)
        return 0

    TMP.mkdir(parents=True, exist_ok=True)
    _, paths = subset_corpus(args.subset)
    tasks = [(str(path), doc_id) for doc_id, path in enumerate(paths)]
    all_workers = logical_cpus()

    matrix: list[dict[str, Any]] = [run_config(tasks, "serial", 1, args.runs)]
    for executor in ("threads", "processes"):
        for workers in (*WORKER_LEVELS, all_workers):
            print(f"benchmarking {executor} x {workers} ...", flush=True)
            matrix.append(run_config(tasks, executor, workers, args.runs))

    serial_wall = matrix[0]["wall"]
    for row in matrix:
        row["speedup"] = serial_wall / row["wall"] if row["wall"] else 0.0
        row["efficiency"] = row["speedup"] / row["workers"]
    merge_fraction = matrix[0]["merge"] / serial_wall if serial_wall else 0.0

    print("measuring pickling tax ...", flush=True)
    pickling = pickling_probe(tasks, min(8, all_workers))
    print("measuring I/O ...", flush=True)
    io = io_probe(ROOT / "data", min(8, all_workers))

    results: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "machine": machine(),
        "method": {
            "runs": args.runs,
            "warmup": 1,
            "sample_seconds": SAMPLE_SECONDS,
            "merge_measured": "build_partials vs merge, separately",
            "subset": f"{args.subset} smallest books",
            "all_workers": all_workers,
            "default_executor": DEFAULT_EXECUTOR,
        },
        "subset": {
            "docs": len(paths),
            "megabytes": sum(path.stat().st_size for path in paths) / 1e6,
        },
        "matrix": matrix,
        "amdahl": {
            "merge_fraction": merge_fraction,
            "max_speedup": (1.0 / merge_fraction) if merge_fraction else None,
        },
        "pickling": pickling,
        "io": io,
    }
    if not args.skip_full:
        print("full corpus: serial vs processes ...", flush=True)
        results["full_corpus"] = full_corpus_rows(
            ROOT / "data", min(8, all_workers)
        )
    if FREETHREADED.exists():
        results["free_threaded"] = json.loads(
            FREETHREADED.read_text(encoding="utf-8")
        )

    RESULTS.write_text(json.dumps(results, indent=2), encoding="utf-8")
    if plot(results, FIGURE):
        print(f"plot -> {FIGURE.relative_to(ROOT)}")
    else:
        print("matplotlib is not installed; plot skipped")
    print(json.dumps({k: v for k, v in results.items() if k != "matrix"}, indent=2))
    for row in matrix:
        print(
            f"{row['executor']:9s} {row['workers']:2d} "
            f"wall={row['wall']:6.2f}s cpu={row['cpu']:6.2f}s "
            f"rss={row['rss'] / 1e6:6.1f}MB merge={row['merge']:5.2f}s "
            f"speedup={row['speedup']:4.2f}x"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

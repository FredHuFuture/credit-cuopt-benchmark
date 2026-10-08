#!/usr/bin/env python
"""M3 scaling sweep: solve time vs problem size on the CREDIT model.

The question M2 left open. At 10M nonzeros the GPU lost once the desk limits
were required to hold, because crossover has to clean up against the whole
constraint set and a credit mandate has six families of limits (nine named
constraint blocks once expanded) where the equity problem had three. That
attribution is a hypothesis, not a measurement — scripts/run_ablation.py is
the controlled test of it. First-order methods are supposed to win as the scenario
block grows. Does the ranking flip, and where? (The M2 timings themselves live
in results/ and in the README — not quoted here, where they go stale.)

Design notes:

- **x-axis is nonzeros, not N or S.** Two universe sizes are swept across
  scenario counts so the cells overlap in nnz. If the curves collapse onto a
  single line in nnz, the scenario block is what governs cost; if they
  separate, something else does. Either answer is worth having.
- **Every solver sees identical problem data.** The universe, the scenario
  matrix and the limits are generated once per cell and passed to each worker
  as files (the GPU and CPU scenario generators do not share a random
  stream — regenerating per worker would race solvers on different problems).
- **One clean subprocess per solver configuration**, as always: highspy and
  OR-Tools bundle clashing HiGHS builds and only one can live in a process.
- **DNFs are data.** A configuration that exceeds the time limit is recorded
  as a DNF at that size, never silently dropped — the size at which the CPU
  stops finishing is the most interesting number in the study.

Three things this sweep did not do until now, each of which made a published
column mean less than it looked like:

- **The error bar excluded the dominant noise source.** `run_spec` runs all
  `--runs` timed solves inside ONE worker process, so median-of-3 measured
  within-process spread while this repo's documented failure mode is across
  process and session (22.0s vs 38.8s on consecutive runs of one crossover
  configuration; a separate 3.3x cross-session disagreement on an identical
  configuration). `--replicates N` invokes the worker N times per
  configuration — genuinely fresh processes, distinct PIDs recorded — and
  reports the two spreads in SEPARATE columns. See `aggregate_replicates`
  for why they are not collapsed and why the summary is a quartile spread.
- **The ground-truth reference was wired but never used.** `run_bench.py`
  solves a HiGHS feasibility-1e-9 reference and hands it to every worker;
  this script did not, so every row of `results/m3_sweep.csv` had an empty
  `gap_vs_ref_bp` and the visible consequence went uncaught: at 1000x5,000
  the cuOpt row reported an objective 0.61bp BETTER than the HiGHS vertex
  while holding sum(w) = 1.0004. One reference solve per CELL is now run
  before the configurations, in its own subprocess, and is never part of any
  reported timing. A reference that fails or times out is recorded as
  missing in `ref_status`, never skipped silently.
- **Nothing scored the mandate.** `check_ok` covers the budget, the box and
  an objective recomputation — not the rating bands, sector caps, issuer cap,
  duration match, DTS band or turnover budget the LP was actually built with.
  Every row now carries both gate verdicts from `src/credit.py`
  (`bench_worker.mandate_columns` documents which limit sheet, and why the
  sweep has one at all).

Run the small validation grid first:
    python scripts/run_sweep.py --grid small --out results/m3_sweep_small.csv
Then the full sweep (hours; fire and forget):
    nohup python scripts/run_sweep.py --grid full --runs 3 --replicates 5 \
        --replicate-only cuopt_pdlp+xover,highs --out results/m3_sweep.csv &
"""

from __future__ import annotations

import argparse
import json
import pickle
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.bench import (  # noqa: E402
    DEFAULT_THREADS, assert_gpu_idle, capture_environment, write_csv,
)
from src.credit import (  # noqa: E402
    DESK_WEIGHT_GATE_BP, NOTIONAL_USD, PURITY_REL_GATE, default_limits,
)
from src.scenarios import credit_scenarios, warmup_gpu  # noqa: E402
from src.universe import make_universe  # noqa: E402

WORKER = REPO / "scripts" / "bench_worker.py"
MARKER = "RESULT_JSON:"

#: (n_bonds, n_scenarios). Cells overlap in nnz across the two universe sizes
#: on purpose — see the docstring.
GRIDS = {
    "small": [(300, 2_000), (300, 8_000), (800, 5_000)],
    "full": [
        (1_000, 5_000),     # ~5.0e6 nnz
        (1_000, 20_000),    # ~2.0e7
        (2_500, 8_000),     # ~2.0e7  (same nnz, different shape)
        (1_000, 50_000),    # ~5.0e7
        (2_500, 20_000),    # ~5.0e7  (same nnz, different shape)
    ],
    # Cells past the host's memory ceiling. Kept as a named grid rather than
    # deleted: "this is where a 16 GB consumer box stops being able to build
    # the problem at all" is a result, not an embarrassment. Running these
    # needs either more RAM or a matrix-level model builder that skips CVXPY
    # canonicalization (see src/interactive.py for that path).
    "beyond": [(1_000, 100_000), (2_500, 40_000)],
}

#: The benchmarked configurations, in run order, keyed by the ORCHESTRATOR's
#: own job name. That name — not the solver label — is what --solvers,
#: --replicate-only and --resume all match on: the label in the CSV is built
#: by the spec factory ("cuopt-pdlp+xover@0.0001"), carries the tolerance, and
#: never equalled the factory name, which is why the old per-configuration
#: resume test (`solver.startswith("cuopt_pdlp+xover")`) could not fire.
JOB_ORDER = ("cuopt_pdlp", "cuopt_pdlp+xover", "ortools_pdlp", "highs")
GPU_JOBS = frozenset({"cuopt_pdlp", "cuopt_pdlp+xover"})

#: Recovers the job from a `solver` label written before the `job` column
#: existed, so --resume still works against a committed CSV. Longest prefix
#: first: the crossover label extends the plain one.
_LEGACY_LABEL_TO_JOB = (
    ("cuopt-pdlp+xover", "cuopt_pdlp+xover"),
    ("cuopt-pdlp", "cuopt_pdlp"),
    ("ortools-pdlp", "ortools_pdlp"),
    ("highs-ipm+xover", "highs_ipm+xover"),
    ("highs-ipm", "highs_ipm"),
    ("highs-simplex", "highs"),
)

#: Peak host memory is dominated by CVXPY canonicalization, not by the dense
#: scenario matrix: roughly the CSR copy (12 B/nnz) several times over, plus
#: the dense R itself (8 B/nnz). Calibrated against the failure that taught us
#: this: a 1.0e8-nnz cell reached 8.5 GB RSS before the OOM killer took it,
#: which is ~90 B/nnz. Estimate from the measurement, not from the arithmetic
#: that predicted 48 and was wrong by half.
BYTES_PER_NNZ_PEAK = 90


def estimate_peak_gib(nnz: int) -> float:
    return nnz * BYTES_PER_NNZ_PEAK / 1024 ** 3


def available_gib() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable"):
                    return int(line.split()[1]) / 1024 ** 2
    except OSError:
        pass
    return float("inf")


def run_worker(args: list[str], timeout_s: float) -> tuple[dict | None, str]:
    cmd = [sys.executable, str(WORKER)] + args
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return None, f"worker wall timeout after {timeout_s:.0f}s"
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(MARKER):
            return json.loads(line[len(MARKER):]), ""
    tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-6:]
    return None, f"worker exit {proc.returncode}: " + " | ".join(tail)


# --------------------------------------------------------------------------
# Resume bookkeeping
# --------------------------------------------------------------------------

def job_of(row: dict) -> str:
    """Which configuration produced this CSV row."""
    j = str(row.get("job", "") or "").strip()
    if j:
        return j
    label = str(row.get("solver", "") or "").strip()
    if label in JOB_ORDER:
        return label      # memory-refusal rows already record the job name
    for prefix, job in _LEGACY_LABEL_TO_JOB:
        if label.startswith(prefix):
            return job
    return label


def _as_bool(v: Any) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes")


def _as_int(v: Any, default: int = 0) -> int:
    try:
        return int(float(str(v).strip()))
    except (TypeError, ValueError):
        return default


def row_is_complete(row: dict, want_replicates: int) -> bool:
    """Is this row a finished answer for `want_replicates` processes?

    A DNF (or a memory refusal) is finished at any replicate count: it cost
    up to two time limits to establish and re-establishing it changes
    nothing. A converged row is finished only if it already carries at least
    as many independent processes as this invocation asks for.

    A row with FEWER is re-run in full rather than topped up. Topping up
    would need the per-process check, status and mandate records of the
    replicates already done, and the CSV keeps only their timings — a row
    rebuilt from what survives would be an aggregate whose parts could not be
    reconstructed, which is worse than paying for the configuration again.
    Rows written before the `n_replicates` column existed count as one
    replicate, which is exactly what they were.
    """
    if _as_bool(row.get("dnf")):
        return True
    return _as_int(row.get("n_replicates"), 1) >= want_replicates


# --------------------------------------------------------------------------
# Replicate aggregation
# --------------------------------------------------------------------------

def _quartiles(xs: list[float]) -> tuple[float, float] | None:
    """(p25, p75) by linear interpolation; None with fewer than 2 points."""
    if len(xs) < 2:
        return None
    q = statistics.quantiles(sorted(xs), n=4, method="inclusive")
    return float(q[0]), float(q[2])


def _r4(x: float) -> float:
    return round(float(x), 4)


def aggregate_replicates(recs: list[dict], runs_per_replicate: int) -> dict:
    """Collapse one configuration's replicate records into one CSV row.

    Two spreads, reported in separate columns because they are different
    quantities and collapsing them is the defect this function exists to fix:

    - `wall_s_iqr_within` — the median across processes of each process's own
      interquartile range. This is all that median-of-`--runs` ever measured,
      and on its own it understates the bar: every timing inside it shares
      one CUDA context, one allocator state and one canonicalized problem.
    - `wall_s_iqr_across` — the interquartile range of the per-process
      MEDIANS. This is where the 22.0s/38.8s and the 3.3x cross-session
      disagreements in the README live, and no number of runs inside a single
      process can estimate it. Empty below two replicates, which is honest:
      one process cannot report a spread across processes.

    IQR rather than min/max (the systems reviewer's request): min and max are
    the extreme order statistics, so their expected spread grows with the
    sample size without bound — a min/max bar taken over 5 replicates is not
    comparable with one taken over 3, while a quartile spread estimates a
    fixed feature of the distribution and is. `wall_s_min`/`wall_s_max` stay
    in the CSV because they cost nothing and are what a reader checks a
    suspicious row against, but the bar to quote is the IQR.

    The headline `wall_s_median` is the median OF THE PER-PROCESS MEDIANS,
    which at one replicate is exactly the number this sweep reported before,
    so the default changes no published value.

    Every individual timing survives twice: flat in `wall_s_all` (the
    existing convention) and grouped by process in `wall_s_by_replicate`, so
    any statistic in this docstring can be recomputed from the CSV alone.

    A DNF ends the configuration — there is no point paying for four more
    time limits — but the replicates that DID finish are kept: "converged in
    two processes and hit the limit in the third" is a stronger statement
    than either half, and it is the across-process story in its rawest form.
    """
    per_rep = [[float(t) for t in (r.get("wall_s_all") or [])] for r in recs]
    ok_idx = [i for i, r in enumerate(recs) if not r.get("dnf")]
    ok_reps = [per_rep[i] for i in ok_idx if per_rep[i]]
    medians = [statistics.median(t) for t in ok_reps]
    pooled = [t for run in per_rep for t in run]

    dnf_at = next((i for i, r in enumerate(recs) if r.get("dnf")), None)
    base = dict(recs[dnf_at if dnf_at is not None else 0])

    if dnf_at is None and medians:
        internals = [float(r["solver_internal_s"]) for r in recs
                     if str(r.get("solver_internal_s", "")).strip() != ""]
        base["wall_s_median"] = _r4(statistics.median(medians))
        base["wall_s_min"] = _r4(min(pooled))
        base["wall_s_max"] = _r4(max(pooled))
        base["n_runs"] = sum(len(t) for t in ok_reps)
        if internals:
            internal = statistics.median(internals)
            base["solver_internal_s"] = _r4(internal)
            base["overhead_s"] = _r4(base["wall_s_median"] - internal)

    within = [q[1] - q[0]
              for q in (_quartiles(t) for t in ok_reps) if q is not None]
    across = _quartiles(medians)
    iqr_within = statistics.median(within) if within else None
    iqr_across = (across[1] - across[0]) if across else None

    base.update({
        "n_replicates": len(recs),
        "n_replicates_ok": len(ok_reps),
        "runs_per_replicate": runs_per_replicate,
        "replicate_pids": json.dumps([r.get("pid", "") for r in recs]),
        "replicate_medians": json.dumps([_r4(m) for m in medians]),
        "wall_s_by_replicate": json.dumps(
            [[_r4(t) for t in run] for run in per_rep]),
        "wall_s_all": [_r4(t) for t in pooled],
        "wall_s_iqr_within": _r4(iqr_within) if iqr_within is not None else "",
        "wall_s_p25_across": _r4(across[0]) if across else "",
        "wall_s_p75_across": _r4(across[1]) if across else "",
        "wall_s_iqr_across": _r4(iqr_across) if iqr_across is not None else "",
        "iqr_across_over_within": (
            round(iqr_across / iqr_within, 3)
            if (iqr_across is not None and iqr_within) else ""),
        "replicate_check_ok": json.dumps([r.get("check_ok", "")
                                          for r in recs]),
        "replicate_gate_verdict": json.dumps([r.get("gate_verdict", "")
                                              for r in recs]),
    })

    notes = [str(base.get("notes", "") or "")]
    if dnf_at is not None and len(ok_reps):
        notes.append(f"DNF on replicate {dnf_at + 1}; "
                     f"{len(ok_reps)} earlier replicate(s) converged "
                     f"(medians {[_r4(m) for m in medians]})")
    # A verdict or a check that flips between two runs of the SAME
    # configuration on the SAME data is a result, not noise to average over.
    if len({r.get("check_ok", "") for r in recs}) > 1:
        notes.append("check_ok DISAGREES across replicates")
    verdicts = {r.get("gate_verdict", "") for r in recs
                if r.get("gate_verdict", "")}
    if len(verdicts) > 1:
        notes.append(f"mandate verdict DISAGREES across replicates: "
                     f"{sorted(verdicts)}")
    base["notes"] = "; ".join(p for p in notes if p)
    return base


def build_jobs(args: argparse.Namespace) -> list[tuple[str, str, dict, str]]:
    """(job, factory, factory kwargs, hardware) for each selected job."""
    catalogue = {
        "cuopt_pdlp": ("cuopt_pdlp",
                       {"tol": args.tol, "time_limit_s": args.time_limit},
                       "GPU"),
        "cuopt_pdlp+xover": ("cuopt_pdlp",
                             {"tol": args.tol,
                              "time_limit_s": args.time_limit,
                              "crossover": True}, "GPU"),
        "ortools_pdlp": ("ortools_pdlp",
                         {"tol": args.tol, "time_limit_s": args.time_limit,
                          "threads": args.threads}, "CPU"),
        "highs": ("highs", {"time_limit_s": args.time_limit,
                            "threads": args.threads}, "CPU"),
    }
    return [(job, *catalogue[job]) for job in JOB_ORDER
            if job in args.solver_set]


def _csv_list(text: str) -> list[str]:
    return [p.strip() for p in text.split(",") if p.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=sorted(GRIDS), default="small")
    ap.add_argument("--runs", type=int, default=3,
                    help="timed solves INSIDE each worker process")
    ap.add_argument("--replicates", type=int, default=1,
                    help="independent worker PROCESSES per configuration. "
                         "The default of 1 reproduces this sweep's previous "
                         "behaviour exactly. Above 1, the across-process "
                         "spread — this project's dominant noise source — "
                         "gets its own columns instead of being invisible")
    ap.add_argument("--replicate-only", default="",
                    help="comma-separated job names that get --replicates "
                         "processes; every other configuration gets 1. The "
                         "sweep already runs for hours and replicating all "
                         "four multiplies that, but only the vertex-producing "
                         "rows carry the paper's central claim. What I would "
                         "actually run: --runs 3 --replicates 5 "
                         "--replicate-only cuopt_pdlp+xover,highs")
    ap.add_argument("--solvers", default=",".join(JOB_ORDER),
                    help="comma-separated subset of "
                         f"{','.join(JOB_ORDER)} to run")
    ap.add_argument("--time-limit", type=float, default=300.0)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=0.95)
    ap.add_argument("--risk-aversion", type=float, default=3.0)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--scen-seed", type=int, default=11)
    ap.add_argument("--no-reference", action="store_true",
                    help="skip the per-cell ground-truth solve. Leaves "
                         "gap_vs_ref_bp empty on every row and records why — "
                         "for harness checks only, never for a published run")
    ap.add_argument("--ref-time-limit", type=float, default=0.0,
                    help="time limit for the reference solve (default: 2x "
                         "--time-limit, matching reference_spec)")
    ap.add_argument("--no-gpu", action="store_true",
                    help="host has no usable GPU (or must not touch it): "
                         "generate scenarios with NumPy and skip the CUDA "
                         "warmup and the idle check. CPU-only validation of "
                         "the harness; the timings are not comparable with a "
                         "published run and the CSV header says so")
    ap.add_argument("--allow-busy-gpu", action="store_true",
                    help="proceed even if another workload is using the GPU "
                         "(results will not be publishable)")
    ap.add_argument("--only", default="",
                    help="run a single cell, e.g. 2500x20000. One cell per "
                         "process keeps a crash in one from costing the rest.")
    ap.add_argument("--resume", action="store_true",
                    help="skip configurations already present in --out (this "
                         "sweep takes hours; losing finished cells to a crash "
                         "in a later one is not acceptable). A configuration "
                         "counts as done only if it carries at least as many "
                         "replicates as this invocation asks for")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    args.solver_set = _csv_list(args.solvers)
    unknown = [s for s in args.solver_set if s not in JOB_ORDER]
    if unknown:
        print(f"unknown solver job(s) {unknown}; known: {list(JOB_ORDER)}")
        return 1
    if not args.solver_set:
        print("--solvers selected nothing")
        return 1
    replicate_only = _csv_list(args.replicate_only)
    unknown = [s for s in replicate_only if s not in args.solver_set]
    if unknown:
        print(f"--replicate-only names {unknown}, which --solvers does not "
              f"run")
        return 1
    gpu_selected = sorted(set(args.solver_set) & GPU_JOBS)
    if args.no_gpu and gpu_selected:
        print(f"--no-gpu cannot run {gpu_selected}; restrict --solvers "
              f"to the CPU jobs")
        return 1

    def want_replicates(job: str) -> int:
        if replicate_only and job not in replicate_only:
            return 1
        return max(1, args.replicates)

    # A sweep that takes hours is worth nothing if something else is using the
    # card. Check before spending the time, not after reading the numbers.
    if not args.no_gpu:
        assert_gpu_idle(raise_on_busy=not args.allow_busy_gpu)

    ref_time_limit = args.ref_time_limit or 2.0 * args.time_limit
    jobs = build_jobs(args)

    cells = GRIDS[args.grid]
    if args.only:
        cells = [c for c in cells if f"{c[0]}x{c[1]}" == args.only]
        if not cells:
            print(f"no cell named {args.only} in grid '{args.grid}'")
            return 1
    print(f"Sweep '{args.grid}': {len(cells)} cells, "
          f"time limit {args.time_limit:.0f}s, {args.runs} timed runs each, "
          f"replicates {{" +
          ", ".join(f"{j}:{want_replicates(j)}" for j, *_ in jobs) + "}",
          flush=True)
    print(f"  reference: "
          + ("DISABLED (--no-reference): gap_vs_ref_bp stays empty"
             if args.no_reference else
             f"HiGHS simplex @1e-9, one solve per cell, "
             f"{ref_time_limit:.0f}s limit, excluded from every timing"),
          flush=True)

    rows: list[dict] = []
    prior: dict[tuple[str, str], dict] = {}
    if args.resume and args.out and Path(args.out).exists():
        import csv as _csv
        with open(args.out) as f:
            body = [ln for ln in f if not ln.startswith("#")]
        for r in _csv.DictReader(body):
            r = dict(r)
            rows.append(r)
            prior[(r.get("cell", ""), job_of(r))] = r
        cells_seen = sorted({c for c, _ in prior})
        print(f"resuming: {len(prior)} configuration row(s) already on disk "
              f"across cells {', '.join(cells_seen)}", flush=True)

    t_start = time.perf_counter()

    for ci, (n_bonds, n_scen) in enumerate(cells, 1):
        nnz = n_scen * (n_bonds + 2)
        cell = f"{n_bonds}x{n_scen}"
        todo = [j for j in jobs
                if not (prior.get((cell, j[0])) is not None
                        and row_is_complete(prior[(cell, j[0])],
                                            want_replicates(j[0])))]
        if not todo:
            print(f"\n[cell {ci}/{len(cells)}] {cell} — already measured, "
                  f"skipping", flush=True)
            continue
        print(f"\n[cell {ci}/{len(cells)}] {n_bonds} bonds x {n_scen:,} "
              f"scenarios  (~{nnz:,} nnz); {len(todo)}/{len(jobs)} "
              f"configuration(s) to run", flush=True)

        need, have = estimate_peak_gib(nnz), available_gib()
        print(f"  memory: needs ~{need:.1f} GiB peak, {have:.1f} GiB "
              f"available", flush=True)
        # 0.5, not 0.85. The guest cannot see the real constraint: WSL2 grows
        # its VM on demand, so a cell that fits in *guest* free memory can
        # still be killed when the Windows host cannot satisfy the growth —
        # which takes the orchestrator down with no in-guest OOM record.
        # Measured on this host: a 5.0e7-nnz cell estimated at 4.2 GiB against
        # 5.9 GiB free died repeatedly, non-deterministically, depending on
        # what Windows was doing at the time.
        if need > have * 0.50:
            # Record the refusal instead of letting the OOM killer end the
            # whole sweep — a cell the host cannot build is a finding.
            print("  SKIPPED: would exceed host memory", flush=True)
            for job, _factory, _kwargs, hardware in todo:
                # A cell that used to fit and no longer does would otherwise
                # end up with two rows for one configuration: the partial
                # measurement and the refusal.
                stale = prior.pop((cell, job), None)
                if stale is not None:
                    rows = [r for r in rows if r is not stale]
                rows.append({
                    "job": job, "solver": job, "hardware": hardware,
                    "status": "skipped_memory", "dnf": True,
                    "wall_s_median": "", "n_bonds": n_bonds,
                    "n_scenarios": n_scen, "nnz": nnz, "cell": cell,
                    "notes": f"needs ~{need:.1f} GiB peak, "
                             f"{have:.1f} GiB available",
                })
            if args.out:
                _write(args, rows, ref_time_limit)
            continue

        uni = make_universe(n_bonds=n_bonds, seed=args.seed)
        limits = default_limits(uni)
        if not args.no_gpu:
            warmup_gpu(uni)
        R, mu, sinfo = credit_scenarios(uni, n_scen=n_scen,
                                        seed=args.scen_seed,
                                        use_gpu=not args.no_gpu)
        print(f"  scenarios: {sinfo['seconds']:.2f}s on {sinfo['device']}; "
              f"matrix {R.nbytes / 1e9:.2f} GB", flush=True)

        tmp = Path(tempfile.mkdtemp(prefix=f"sweep_{n_bonds}x{n_scen}_"))
        data_npz, uni_pkl = str(tmp / "data.npz"), str(tmp / "uni.pkl")
        ref_npz = str(tmp / "ref.npz")
        np.savez(data_npz, R=R, mu=mu, alpha=args.alpha,
                 risk_aversion=args.risk_aversion, w_max=limits.w_max)
        with open(uni_pkl, "wb") as f:
            pickle.dump((uni, limits), f)
        del R, mu   # the workers own the memory from here

        # ---- ground truth, once per cell -------------------------------
        # Once per CELL, not per configuration: it is a full HiGHS solve at
        # feasibility 1e-9 on the same constrained problem (--universe is
        # passed, so the reference respects the mandate too — a reference
        # solved without the limits would be a different LP and its objective
        # would not bound anything). It runs in its own subprocess before any
        # timing starts and appears in no wall_s_* column.
        ref_cols = {"ref_status": "disabled (--no-reference)",
                    "ref_obj": "", "ref_solve_s": ""}
        ref_arg: list[str] = []
        if not args.no_reference:
            print(f"  {'reference @1e-9':<22}", end="", flush=True)
            t_ref = time.perf_counter()
            ref, diag = run_worker(
                ["--data", data_npz, "--universe", uni_pkl,
                 "--factory", "reference", "--mode", "reference",
                 "--out-ref", ref_npz,
                 "--kwargs", json.dumps({"time_limit_s": ref_time_limit})],
                timeout_s=ref_time_limit + 300)
            wall_ref = time.perf_counter() - t_ref
            if ref is None:
                # Missing, and recorded as missing on every row of the cell.
                # A reference that timed out is a fact about the cell (HiGHS
                # at 1e-9 cannot finish it in the budget), not a licence to
                # report unchecked objectives as if nothing were expected.
                # The LAST segment of the diagnostic is kept rather than the
                # first: run_worker joins the tail of the traceback, and the
                # sentence naming the failure is at the end of it.
                reason = diag.rsplit(" | ", 1)[-1].strip() or diag
                ref_cols = {"ref_status": f"MISSING: {reason[:200]}",
                            "ref_obj": "", "ref_solve_s": round(wall_ref, 3)}
                print(f"MISSING ({reason[:60]})", flush=True)
            else:
                ref_cols = {"ref_status": "ok",
                            "ref_obj": ref["ref_obj"],
                            "ref_solve_s": ref.get("ref_solve_s",
                                                   round(wall_ref, 3))}
                ref_arg = ["--ref", ref_npz]
                print(f"{ref_cols['ref_solve_s']}s  obj="
                      f"{ref['ref_obj']:.10f}", flush=True)

        for job, factory, kwargs, hardware in todo:
            stale = prior.pop((cell, job), None)
            if stale is not None:
                # Fewer replicates than asked for: the configuration is re-run
                # in full and the partial row must not survive alongside it.
                rows = [r for r in rows if r is not stale]
                print(f"  {job:<22}re-running "
                      f"({_as_int(stale.get('n_replicates'), 1)} replicate(s) "
                      f"on disk, {want_replicates(job)} wanted)", flush=True)
            want = want_replicates(job)
            print(f"  {job:<22}{want} replicate(s)", flush=True)
            # A DNF costs at most two time limits, not runs+1: the harness
            # abandons a configuration whose warmup solve fails, but retries
            # that warmup once first (src/bench.py run_spec), so the wall
            # ceiling has to cover warmup + retry + runs.
            budget = args.time_limit * (args.runs + 2) + 300
            recs: list[dict] = []
            for k in range(1, want + 1):
                rec, diag = run_worker(
                    ["--data", data_npz, "--universe", uni_pkl,
                     "--factory", factory, "--kwargs", json.dumps(kwargs),
                     "--runs", str(args.runs), "--replicate", str(k)]
                    + ref_arg,
                    timeout_s=budget)
                if rec is None:
                    rec = {"solver": job, "hardware": hardware,
                           "status": "worker_failure", "dnf": True,
                           "wall_s_median": "", "notes": diag}
                    print(f"    rep {k}/{want}  DNF ({diag[:56]})",
                          flush=True)
                else:
                    mark = ("DNF" if rec["dnf"]
                            else f"{rec['wall_s_median']:.3f}s")
                    print(f"    rep {k}/{want}  pid {rec.get('pid', '?')}  "
                          f"{mark}  check_ok={rec.get('check_ok', '')}  "
                          f"{rec.get('gate_verdict', '')}", flush=True)
                recs.append(rec)
                if rec.get("dnf"):
                    break   # four more time limits buy nothing

            row = aggregate_replicates(recs, args.runs)
            row.update({"job": job, "n_bonds": n_bonds,
                        "n_scenarios": n_scen, "nnz": nnz, "cell": cell,
                        "scenario_gen_s": round(sinfo["seconds"], 3),
                        # ASCII-folded: describe() carries a plus-minus and an
                        # element-of sign, and write_csv uses the platform
                        # default encoding.
                        "mandate_limits": (limits.describe()
                                           .replace("±", "+/-")
                                           .replace("∈", " in ")),
                        **ref_cols})
            med = ("DNF" if _as_bool(row.get("dnf"))
                   else f"{row['wall_s_median']}s")
            print(f"    -> median {med}  "
                  f"within-IQR {row['wall_s_iqr_within'] or '-'}  "
                  f"across-IQR {row['wall_s_iqr_across'] or '-'}  "
                  f"vs_ref {row.get('gap_vs_ref_bp', '') or '-'}bp  "
                  f"mandate {row.get('gate_verdict', '') or '-'}", flush=True)
            rows.append(row)
            # Checkpoint after every CONFIGURATION, not every cell. A cell can
            # die part-way (the host reclaiming the VM under memory pressure
            # takes the orchestrator with it, with no chance to flush), and
            # losing measurements that already succeeded is pure waste. A
            # configuration is written only once all its replicates are in, so
            # a partial replicate set never reaches the CSV at all.
            if args.out:
                _write(args, rows, ref_time_limit)

        for p in (data_npz, uni_pkl, ref_npz):
            Path(p).unlink(missing_ok=True)
        tmp.rmdir()

        if args.out:   # checkpoint after every cell — this run is long
            _write(args, rows, ref_time_limit)

    print(f"\nSweep finished in {(time.perf_counter() - t_start) / 60:.1f} min")
    if args.out:
        _write(args, rows, ref_time_limit)
        print(f"Wrote {args.out}")
    return 0


def _write(args, rows: list[dict], ref_time_limit: float) -> None:
    env = capture_environment()
    env["sweep_grid"] = args.grid
    env["time_limit_s"] = str(args.time_limit)
    env["runs_per_config"] = str(args.runs)
    env["replicates"] = str(args.replicates)
    env["replicate_only"] = args.replicate_only or "(all selected solvers)"
    env["solvers"] = args.solvers
    env["threads_cpu_solvers"] = str(args.threads)
    env["pdlp_tolerance"] = str(args.tol)
    env["problem"] = "credit Mean-CVaR with the full desk limit set"
    env["alpha"] = str(args.alpha)
    env["risk_aversion"] = str(args.risk_aversion)
    env["isolation"] = "one clean subprocess per solver configuration"
    env["replication"] = (
        "one subprocess per REPLICATE; wall_s_median is the median of the "
        "per-process medians, wall_s_iqr_within the median within-process "
        "IQR, wall_s_iqr_across the IQR of the per-process medians (empty "
        "below 2 replicates). Every timing is kept: wall_s_all flat, "
        "wall_s_by_replicate grouped by process, replicate_pids alongside")
    env["reference"] = (
        "DISABLED (--no-reference): gap_vs_ref_bp empty on every row"
        if args.no_reference else
        f"highs-simplex feasibility 1e-9 on the SAME constrained problem, "
        f"one solve per cell, {ref_time_limit:.0f}s limit; ref_solve_s is "
        f"orchestrator overhead and is in no wall_s_* column")
    env["mandate"] = (
        "default_limits(uni) derived per cell from that cell's own "
        "benchmark; families 1-6 only (te_bp=inf, so the LP never carried "
        "the tracking-error family and it is not scored). Verdicts compare "
        "solvers WITHIN a cell; across cells the limit sheet differs")
    env["purity_gate_rel"] = f"{PURITY_REL_GATE:g}"
    env["desk_gate_bp_weight"] = f"{DESK_WEIGHT_GATE_BP:g}"
    env["desk_gate_notional_usd"] = f"{NOTIONAL_USD:.0f}"
    if args.no_gpu:
        env["gpu"] = "not used (--no-gpu)"
        env["scenario_device"] = "CPU (NumPy)"
        env["publishable"] = ("NO: --no-gpu is a harness check, not a "
                              "measurement")
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    norm = [{k: r.get(k, "") for k in keys} for r in rows]
    for r in norm:
        if isinstance(r.get("wall_s_all"), list):
            r["wall_s_all"] = json.dumps(r["wall_s_all"])
    write_csv(args.out, norm, env)


if __name__ == "__main__":
    sys.exit(main())

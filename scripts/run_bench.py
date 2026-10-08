#!/usr/bin/env python
"""Run the equity Mean-CVaR GPU-vs-CPU benchmark (M1).

Orchestrates one clean subprocess per solver configuration (see
bench_worker.py for why isolation is mandatory) and merges the results into
a table and CSV.

Example:
    python scripts/run_bench.py --n-assets 400 --n-scenarios 10000 \
        --tols 1e-4 1e-6 --runs 3 --out results/m1_equity.csv
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.bench import (  # noqa: E402
    DEFAULT_THREADS, assert_gpu_idle, capture_environment, write_csv,
)
from src.scenarios import equity_scenarios  # noqa: E402

WORKER = REPO / "scripts" / "bench_worker.py"
MARKER = "RESULT_JSON:"


def run_worker(args: list[str], timeout_s: float) -> tuple[dict | None, str]:
    """Run bench_worker.py; return (parsed record or None, diagnostics)."""
    cmd = [sys.executable, str(WORKER)] + args
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return None, f"worker timeout after {timeout_s:.0f}s"
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(MARKER):
            return json.loads(line[len(MARKER):]), ""
    tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-8:]
    return None, f"worker exit {proc.returncode}: " + " | ".join(tail)


def dnf_row(label: str, hardware: str, n_assets: int, n_scen: int,
            reason: str) -> dict:
    return {
        "solver": label, "hardware": hardware, "tolerances": "",
        "threads": "", "n_assets": n_assets, "n_scenarios": n_scen,
        "nnz_est": n_scen * (n_assets + 2), "status": "worker_failure",
        "dnf": True, "n_runs": 0, "wall_s_median": "", "wall_s_min": "",
        "wall_s_max": "", "solver_internal_s": "", "overhead_s": "",
        "compile_s": "",
        "objective": "", "check_ok": "", "check_obj_gap": "",
        "check_budget_resid": "", "check_budget_signed": "",
        "gap_vs_ref_bp": "", "weight_l1_vs_ref": "",
        "dust_positions": "", "positions": "", "config": "", "notes": reason,
        "wall_s_all": [],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-assets", type=int, default=400)
    ap.add_argument("--n-scenarios", type=int, default=10_000)
    ap.add_argument("--tols", type=float, nargs="+", default=[1e-4])
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--time-limit", type=float, default=300.0)
    ap.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    ap.add_argument("--alpha", type=float, default=0.95)
    ap.add_argument("--risk-aversion", type=float, default=1.0)
    ap.add_argument("--w-max", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--crossover", action="store_true")
    ap.add_argument("--thread-sweep", action="store_true",
                    help="sweep CPU thread counts first and use each CPU "
                         "solver's best (hybrid P/E cores make 'all threads' "
                         "an assumption, not a fact)")
    ap.add_argument("--allow-busy-gpu", action="store_true",
                    help="proceed even if another workload is using the GPU "
                         "(results will not be publishable)")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    # Check the card is free BEFORE spending the time, not after reading the
    # numbers: a competing workload makes every configuration slower by an
    # amount stable enough to look like a measurement. Same guard as
    # scripts/run_sweep.py.
    assert_gpu_idle(raise_on_busy=not args.allow_busy_gpu)

    print(f"Generating scenarios: {args.n_assets} assets x "
          f"{args.n_scenarios} scenarios (seed {args.seed})", flush=True)
    R, mu = equity_scenarios(args.n_assets, args.n_scenarios, seed=args.seed)

    tmpdir = tempfile.mkdtemp(prefix="bench_")
    data_npz = str(Path(tmpdir) / "data.npz")
    ref_npz = str(Path(tmpdir) / "ref.npz")
    np.savez(data_npz, R=R, mu=mu, alpha=args.alpha,
             risk_aversion=args.risk_aversion, w_max=args.w_max)

    # Reference solve (own subprocess, HiGHS at 1e-9)
    print("Reference solve (HiGHS @ 1e-9)...", flush=True)
    ref, diag = run_worker(
        ["--data", data_npz, "--factory", "reference", "--mode", "reference",
         "--out-ref", ref_npz,
         "--kwargs", json.dumps({"time_limit_s": 2 * args.time_limit})],
        timeout_s=2 * args.time_limit + 120,
    )
    if ref is None:
        print(f"FATAL: reference solve failed: {diag}", file=sys.stderr)
        return 1
    print(f"  reference objective: {ref['ref_obj']:.8f}", flush=True)

    # Optional thread sweep: on a hybrid 8P+8E CPU, "all 24 threads" can be
    # SLOWER than fewer P-cores for sync-heavy first-order iterations. Give
    # each CPU solver its measured-best thread count, disclose the sweep.
    sweep_notes: dict[str, str] = {}
    best_threads = {"ortools_pdlp": args.threads, "highs": args.threads}
    if args.thread_sweep:
        candidates = sorted({4, 8, 16, args.threads})
        for factory, kw in [
            ("ortools_pdlp", {"tol": args.tols[0],
                              "time_limit_s": args.time_limit}),
            ("highs", {"time_limit_s": args.time_limit}),
        ]:
            timings: dict[int, float] = {}
            for th in candidates:
                rec, diag = run_worker(
                    ["--data", data_npz, "--factory", factory,
                     "--kwargs", json.dumps({**kw, "threads": th}),
                     "--runs", "1", "--ref", ref_npz],
                    timeout_s=args.time_limit * 2 + 180,
                )
                if rec and not rec["dnf"]:
                    timings[th] = float(rec["wall_s_median"])
            if timings:
                best_threads[factory] = min(timings, key=timings.get)
            sweep_notes[factory] = ", ".join(
                f"{th}T={timings.get(th, float('nan')):.3f}s"
                for th in candidates)
            print(f"thread sweep {factory}: {sweep_notes[factory]} "
                  f"-> using {best_threads[factory]}T", flush=True)

    # Benchmark specs: (factory, kwargs, label-ish, hardware)
    jobs: list[tuple[str, dict, str]] = []
    for tol in args.tols:
        jobs.append(("cuopt_pdlp",
                     {"tol": tol, "time_limit_s": args.time_limit}, "GPU"))
        if args.crossover:
            jobs.append(("cuopt_pdlp",
                         {"tol": tol, "time_limit_s": args.time_limit,
                          "crossover": True}, "GPU"))
        jobs.append(("ortools_pdlp",
                     {"tol": tol, "time_limit_s": args.time_limit,
                      "threads": best_threads["ortools_pdlp"]}, "CPU"))
    jobs.append(("highs", {"time_limit_s": args.time_limit,
                           "threads": best_threads["highs"]}, "CPU"))

    rows: list[dict] = []
    for factory, kwargs, hardware in jobs:
        desc = f"{factory}({', '.join(f'{k}={v}' for k, v in kwargs.items())})"
        print(f"Running {desc} ...", flush=True)
        # generous ceiling: warmup + one warmup retry + runs, each up to
        # time_limit, plus startup (see run_spec in src/bench.py)
        timeout = args.time_limit * (args.runs + 2) + 180
        rec, diag = run_worker(
            ["--data", data_npz, "--factory", factory,
             "--kwargs", json.dumps(kwargs), "--runs", str(args.runs),
             "--ref", ref_npz],
            timeout_s=timeout,
        )
        if rec is None:
            rec = dnf_row(desc, hardware, args.n_assets, args.n_scenarios, diag)
            print(f"  DNF: {diag}", flush=True)
        else:
            print(f"  {rec['status']}  wall_med={rec['wall_s_median']}s  "
                  f"internal={rec['solver_internal_s'] or '-'}s  "
                  f"vs_ref={rec['gap_vs_ref_bp'] or '-'}bp  "
                  f"check_ok={rec['check_ok']}", flush=True)
        rows.append(rec)

    # Table
    print("\n" + "=" * 104)
    print(f"{'solver':<26} {'hw':<4} {'wall_med(s)':>11} {'internal(s)':>11} "
          f"{'vs_ref(bp)':>10} {'dust':>5} {'ok':>4}  notes")
    print("-" * 104)
    for r in rows:
        ok = "DNF" if r["dnf"] else ("Y" if r["check_ok"] else "N")
        print(f"{r['solver']:<26} {r['hardware']:<4} "
              f"{str(r['wall_s_median']):>11} {str(r['solver_internal_s']):>11} "
              f"{str(r['gap_vs_ref_bp']) or '-':>10} "
              f"{str(r['dust_positions']):>5} {ok:>4}  {r['notes'][:30]}")
    print("=" * 104)

    if args.out:
        env = capture_environment()
        env["threads_cpu_solvers"] = (
            f"ortools_pdlp={best_threads['ortools_pdlp']}, "
            f"highs={best_threads['highs']}"
        )
        for factory, note in sweep_notes.items():
            env[f"thread_sweep_{factory}"] = note
        env["time_limit_s"] = str(args.time_limit)
        env["runs_per_solver"] = str(args.runs)
        env["reference"] = f"highs-simplex feasibility 1e-9, obj={ref['ref_obj']:.10f}"
        env["isolation"] = "one clean subprocess per solver configuration"
        for r in rows:
            r["wall_s_all"] = json.dumps(r.get("wall_s_all", []))
        # Union of keys, not `rows[0].keys()`: a worker record carries columns
        # `dnf_row` does not (algorithm evidence, the mandate verdicts the
        # worker now scores), and csv.DictWriter raises on a row with a key
        # the header lacks. Whether that fired depended on whether the FIRST
        # configuration happened to fail — a crash at the write step after the
        # solves had already been paid for.
        keys: list[str] = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        write_csv(args.out, [{k: r.get(k, "") for k in keys} for r in rows],
                  env)
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

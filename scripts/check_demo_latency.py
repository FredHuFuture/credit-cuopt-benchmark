#!/usr/bin/env python
"""Which path should the interactive demo use, and at what size?

"Re-optimized while you move the slider" is a latency claim, so it needs a
measurement. Two candidates:

  * the hand-rolled CSR + cuOpt DataModel path (src/interactive.py), built to
    enable PDLP warm start;
  * the plain CVXPY path the benchmarks use.

The hand-rolled path only ever existed to make warm start possible, and warm
start was measured to be a net loss on this problem (it forces presolve off,
and presolve is worth more than it saves). If CVXPY is also faster and more
reliable, the hand-rolled path has no remaining justification.

Median of repeats at distinct knob values, limit compliance checked each time.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.bench import (  # noqa: E402
    assert_gpu_idle, capture_environment, write_csv,
)
from src.credit import (  # noqa: E402
    credit_constraint_builder, default_limits, exposure_report,
    worst_violation,
)
from src.interactive import Knobs, build_interactive, solve  # noqa: E402
from src.model import build_mean_cvar  # noqa: E402
from src.scenarios import credit_scenarios, warmup_gpu  # noqa: E402
from src.universe import make_universe  # noqa: E402

SIZES = (4_000, 10_000, 20_000)
REPS = 3
TOL = 1e-4


def cvxpy_solve(R, mu, uni, limits, lam, tol=TOL):
    """One rebuild-and-solve, the way a Streamlit rerun would do it."""
    import cvxpy as cp
    t0 = time.perf_counter()
    m = build_mean_cvar(R, mu, alpha=0.95, risk_aversion=lam,
                        w_max=limits.w_max,
                        constraint_builder=credit_constraint_builder(
                            uni, limits))
    m.problem.solve(
        solver=cp.CUOPT, verbose=False, warm_start=False,
        solver_method="PDLP", pdlp_solver_mode="Stable2", crossover=0,
        absolute_primal_tolerance=tol, relative_primal_tolerance=tol,
        absolute_dual_tolerance=tol, relative_dual_tolerance=tol,
        absolute_gap_tolerance=tol, relative_gap_tolerance=tol,
        time_limit=120.0)
    return time.perf_counter() - t0, m.weights()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-busy-gpu", action="store_true",
                    help="proceed even if another workload is using the GPU "
                         "(results will not be publishable)")
    ap.add_argument("--out", type=str, default="",
                    help="write the measured rows to CSV, with the same "
                         "environment header the benchmark artifacts carry")
    args = ap.parse_args()

    # Both candidate paths are timed on the GPU, so a contended card decides
    # this comparison instead of the code does. Same guard as the sweep.
    assert_gpu_idle(raise_on_busy=not args.allow_busy_gpu)

    rows: list[dict] = []
    uni = make_universe(n_bonds=500, seed=7)
    warmup_gpu(uni)
    limits = default_limits(uni)
    bench_dts = float(uni.w_benchmark @ uni.dts)
    base = Knobs(alpha=0.95,
                 sector_cap=float(max(limits.sector_caps.values())),
                 issuer_cap=limits.issuer_cap,
                 dts_min_mult=limits.dts_min / bench_dts,
                 dts_max_mult=limits.dts_max / bench_dts,
                 turnover=limits.turnover)

    print(f"500 bonds; median of {REPS} re-solves per size, tol={TOL:g}\n")
    print(f"  {'scenarios':>9} {'nnz':>11}  "
          f"{'hand-rolled':>12} {'breach':>8}   {'CVXPY':>9} {'breach':>8}")
    for n_scen in SIZES:
        R, mu, _ = credit_scenarios(uni, n_scen=n_scen, seed=11, use_gpu=True)

        m = build_interactive(uni, R, mu, limits, base)
        solve(m, base)                                    # discarded warmup
        hand_t, hand_v = [], []
        for i in range(1, REPS + 1):
            kk = Knobs(**{**base.__dict__,
                          "risk_aversion": base.risk_aversion * (1 + 0.1 * i)})
            r = solve(m, kk, tol=TOL)
            hand_t.append(r["seconds"])
            v, _ = worst_violation(exposure_report(uni, r["w"], limits))
            hand_v.append(v)
        nnz = m.A.nnz
        del m

        cvxpy_solve(R, mu, uni, limits, base.risk_aversion)   # warmup
        cvx_t, cvx_v = [], []
        for i in range(1, REPS + 1):
            secs, w = cvxpy_solve(R, mu, uni, limits,
                                  base.risk_aversion * (1 + 0.1 * i))
            cvx_t.append(secs)
            v, _ = worst_violation(exposure_report(uni, w, limits))
            cvx_v.append(v)

        print(f"  {n_scen:>9,} {nnz:>11,}  "
              f"{statistics.median(hand_t):>11.2f}s {statistics.median(hand_v):>8.1e}"
              f"   {statistics.median(cvx_t):>8.2f}s {statistics.median(cvx_v):>8.1e}")
        rows.append({
            "n_bonds": 500, "n_scenarios": n_scen, "nnz": nnz,
            "handrolled_s_median": round(statistics.median(hand_t), 4),
            "handrolled_s_min": round(min(hand_t), 4),
            "handrolled_s_max": round(max(hand_t), 4),
            "handrolled_worst_rel_violation": f"{statistics.median(hand_v):.3e}",
            "cvxpy_s_median": round(statistics.median(cvx_t), 4),
            "cvxpy_s_min": round(min(cvx_t), 4),
            "cvxpy_s_max": round(max(cvx_t), 4),
            "cvxpy_worst_rel_violation": f"{statistics.median(cvx_v):.3e}",
            "reps": REPS, "tol": TOL,
        })
        del R, mu

    if args.out and rows:
        env = capture_environment()
        env["script"] = "scripts/check_demo_latency.py"
        env["question"] = ("which interactive path is faster: the hand-rolled "
                           "CSR + cuOpt DataModel path or plain CVXPY")
        env["timing_boundary"] = ("both paths time model build + solve; the "
                                  "hand-rolled timer starts above the "
                                  "DataModel rebuild (see src/interactive.py)")
        env["reps_per_size"] = str(REPS)
        env["aggregation"] = "median of reps at distinct knob values"
        env["tolerance"] = f"{TOL:g}"
        env["universe_seed"] = "7"
        env["scenario_seed"] = "11"
        write_csv(args.out, rows, env)
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

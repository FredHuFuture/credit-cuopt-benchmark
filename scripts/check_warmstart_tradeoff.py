#!/usr/bin/env python
"""Measure the warm-start / presolve trade-off on the interactive credit LP.

cuOpt's PDLP warm start requires presolve to be OFF (warm-start state is sized
to the problem the solver actually iterated on). That makes "does warm start
pay?" the wrong question. The right one is:

    warm start WITHOUT presolve   vs   cold solve WITH presolve

which is the choice an interactive application actually faces. Median of
repeats, alternating knob values so no run reuses the previous answer.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.bench import (  # noqa: E402
    assert_gpu_idle, capture_environment, write_csv,
)
from src.credit import default_limits, exposure_report, worst_violation  # noqa: E402
from src.interactive import Knobs, build_interactive, solve  # noqa: E402
from src.scenarios import credit_scenarios, warmup_gpu  # noqa: E402
from src.universe import make_universe  # noqa: E402

REPS = 3


def main() -> int:
    # argparse with optional positionals, so the original invocation
    # (`check_warmstart_tradeoff.py 500 20000`) keeps working unchanged.
    ap = argparse.ArgumentParser()
    ap.add_argument("n_bonds", type=int, nargs="?", default=500)
    ap.add_argument("n_scen", type=int, nargs="?", default=20_000)
    ap.add_argument("--allow-busy-gpu", action="store_true",
                    help="proceed even if another workload is using the GPU "
                         "(results will not be publishable)")
    ap.add_argument("--out", type=str, default="",
                    help="write the measured rows to CSV, with the same "
                         "environment header the benchmark artifacts carry")
    args = ap.parse_args()
    n_bonds, n_scen = args.n_bonds, args.n_scen

    # Every mode below is timed on the GPU, so a contended card decides the
    # A/B instead of the trade-off does. Same guard as the sweep.
    assert_gpu_idle(raise_on_busy=not args.allow_busy_gpu)

    uni = make_universe(n_bonds=n_bonds, seed=7)
    warmup_gpu(uni)
    R, mu, _ = credit_scenarios(uni, n_scen=n_scen, seed=11, use_gpu=True)
    limits = default_limits(uni)
    bench_dts = float(uni.w_benchmark @ uni.dts)
    base = Knobs(alpha=0.95, sector_cap=float(max(limits.sector_caps.values())),
                 issuer_cap=limits.issuer_cap,
                 dts_min_mult=limits.dts_min / bench_dts,
                 dts_max_mult=limits.dts_max / bench_dts,
                 turnover=limits.turnover)
    m = build_interactive(uni, R, mu, limits, base)
    print(f"{n_bonds} bonds x {n_scen:,} scenarios — LP {m.A.shape[0]:,} rows "
          f"x {m.A.shape[1]:,} cols, {m.A.nnz:,} nnz\n")

    def knobs_for(i: int) -> Knobs:
        return Knobs(**{**base.__dict__,
                        "risk_aversion": base.risk_aversion * (1 + 0.10 * i)})

    modes = (
        ("cold + presolve", dict(warm=False, presolve=True)),
        ("cold, no presolve", dict(warm=False, presolve=False)),
        ("warm, no presolve", dict(warm=True)),
    )
    out = {}
    for name, kw in modes:
        # One discarded warmup, then REPS timed solves at distinct knob values.
        solve(m, knobs_for(0), **kw)
        times, viols, warm_used = [], [], []
        for i in range(1, REPS + 1):
            r = solve(m, knobs_for(i), **kw)
            times.append(r["seconds"])
            warm_used.append(r["warm"])
            v, _ = worst_violation(exposure_report(uni, r["w"], limits))
            viols.append(v)
        out[name] = (statistics.median(times), min(times), max(times),
                     float(np.median(viols)), any(warm_used))
        med, lo, hi, viol, used = out[name]
        print(f"  {name:<20} {med:7.2f}s  (min {lo:.2f}, max {hi:.2f})   "
              f"worst limit breach {viol:.2e}"
              + ("   [warm start active]" if used else ""))

    cold_p = out["cold + presolve"][0]
    warm_np = out["warm, no presolve"][0]
    cold_np = out["cold, no presolve"][0]
    print(f"\n  warm start saves {100 * (1 - warm_np / cold_np):.0f}% "
          f"against a cold solve on equal terms (both without presolve)")
    print(f"  but presolve alone is worth {cold_np / cold_p:.1f}x")
    verdict = ("COLD + PRESOLVE wins" if cold_p < warm_np
               else "WARM START wins")
    print(f"  -> {verdict}: {cold_p:.2f}s vs {warm_np:.2f}s "
          f"({max(cold_p, warm_np) / min(cold_p, warm_np):.1f}x)")

    if args.out:
        rows = [{
            "mode": name, "n_bonds": n_bonds, "n_scenarios": n_scen,
            "lp_rows": m.A.shape[0], "lp_cols": m.A.shape[1], "nnz": m.A.nnz,
            "wall_s_median": round(med, 4), "wall_s_min": round(lo, 4),
            "wall_s_max": round(hi, 4),
            "worst_rel_violation": f"{viol:.3e}",
            "warm_start_active": used, "reps": REPS,
        } for name, (med, lo, hi, viol, used) in out.items()]
        env = capture_environment()
        env["script"] = "scripts/check_warmstart_tradeoff.py"
        env["question"] = ("warm start WITHOUT presolve vs cold solve WITH "
                           "presolve: the choice an interactive app faces")
        env["reps_per_mode"] = str(REPS)
        env["aggregation"] = "median of reps at distinct knob values"
        env["universe_seed"] = "7"
        env["scenario_seed"] = "11"
        env["verdict"] = verdict
        write_csv(args.out, rows, env)
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

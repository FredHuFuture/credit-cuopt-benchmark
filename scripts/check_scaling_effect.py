#!/usr/bin/env python
"""Does pre-scaling the rows help cuOpt, or fight its own equilibration?

Same model, same settings, only the row scaling differs. Median of repeats at
distinct knob values so nothing reuses a previous answer.
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.credit import default_limits  # noqa: E402
from src.interactive import Knobs, build_interactive, solve  # noqa: E402
from src.scenarios import credit_scenarios, warmup_gpu  # noqa: E402
from src.universe import make_universe  # noqa: E402


def main() -> int:
    n_bonds = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    n_scen = int(sys.argv[2]) if len(sys.argv) > 2 else 20_000
    reps = 3

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
    print(f"{n_bonds} bonds x {n_scen:,} scenarios\n")

    for scaled in (True, False):
        m = build_interactive(uni, R, mu, limits, base, scale_rows=scaled)
        solve(m, base, warm=False, presolve=True)          # discarded warmup
        times = []
        for i in range(1, reps + 1):
            kk = Knobs(**{**base.__dict__,
                          "risk_aversion": base.risk_aversion * (1 + 0.1 * i)})
            times.append(solve(m, kk, warm=False, presolve=True)["seconds"])
        label = "row-scaled" if scaled else "raw units"
        print(f"  {label:<12} median {statistics.median(times):6.2f}s   "
              f"(min {min(times):.2f}, max {max(times):.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

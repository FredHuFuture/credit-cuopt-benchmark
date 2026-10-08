#!/usr/bin/env python
"""Verify the GPU and CPU scenario engines agree — and time how they scale.

CuPy and NumPy do not share a random stream, so the same seed produces
statistically identical but not bit-identical scenario sets. Bit-equality is
therefore the wrong test; distributional agreement is the right one. This
script checks the moments and the tail statistic that actually matters (CVaR
of the benchmark portfolio) agree within sampling error, and reports how
scenario-generation time scales on each device.

Run:  python scripts/check_rng_agreement.py
      python scripts/check_rng_agreement.py --out results/rng_agreement.csv
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.bench import (  # noqa: E402
    assert_gpu_idle, capture_environment, write_csv,
)
from src.check import empirical_cvar  # noqa: E402
from src.scenarios import credit_scenarios, warmup_gpu  # noqa: E402
from src.universe import make_universe  # noqa: E402

#: Kept within the WSL memory allocation: a 100k x 2000 float64 matrix is
#: 1.6 GB on the host and both devices' copies must coexist briefly.
SIZES = ((2_000, 500), (20_000, 500), (100_000, 500), (50_000, 2_000))
REPS = 3   # median over repeats; single-shot timings are too noisy to publish


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-busy-gpu", action="store_true",
                    help="proceed even if another workload is using the GPU "
                         "(results will not be publishable)")
    ap.add_argument("--out", type=str, default="",
                    help="write the measured rows to CSV, with the same "
                         "environment header the benchmark artifacts carry")
    args = ap.parse_args()

    # This script publishes GPU-vs-CPU generation times, so the same guard the
    # sweep uses applies: a contended card makes the comparison meaningless.
    assert_gpu_idle(raise_on_busy=not args.allow_busy_gpu)

    rows = []
    for n_scen, n_bonds in SIZES:
        uni = make_universe(n_bonds=n_bonds, seed=7)
        wb = uni.w_benchmark
        # Warm up on THIS universe: n_bonds changes across sizes, and a
        # generic matmul warms the CUDA context but not this module's
        # kernels — the very anti-pattern the README documents.
        warmup_gpu(uni)

        # Median of REPS: single-shot generation timings vary by tens of
        # percent, which is larger than some of the differences compared.
        gpu_t, cpu_t = [], []
        for i in range(REPS):
            R_g, mu_g, ig = credit_scenarios(uni, n_scen=n_scen, seed=11,
                                             use_gpu=True)
            gpu_t.append(ig["seconds"])
            if i == 0:
                # Reduce to summary statistics and release the matrix before
                # the CPU run allocates its own — both at once does not fit.
                port_g, mean_g = -(R_g @ wb), R_g.mean(axis=0)
            del R_g

            R_c, mu_c, ic = credit_scenarios(uni, n_scen=n_scen, seed=11,
                                             use_gpu=False)
            cpu_t.append(ic["seconds"])
            if i == 0:
                port_c, mean_c = -(R_c @ wb), R_c.mean(axis=0)
            del R_c
        ig = {"seconds": statistics.median(gpu_t)}
        ic = {"seconds": statistics.median(cpu_t)}

        cvar_g, _ = empirical_cvar(port_g, 0.95)
        cvar_c, _ = empirical_cvar(port_c, 0.95)
        # Sampling error of a CVaR95 estimate is roughly sd/sqrt(0.05*S).
        se = float(np.std(port_c)) / np.sqrt(0.05 * n_scen)
        rows.append({
            "n_scen": n_scen, "n_bonds": n_bonds,
            "gpu_s": ig["seconds"], "cpu_s": ic["seconds"],
            "speedup": ic["seconds"] / max(ig["seconds"], 1e-9),
            "cvar_gpu": cvar_g, "cvar_cpu": cvar_c,
            "cvar_diff_in_se": abs(cvar_g - cvar_c) / se,
            "mean_ret_diff_bp": float(np.abs(mean_g - mean_c).max() * 1e4),
            "mu_diff_bp": float(np.abs(mu_g - mu_c).max() * 1e4),
        })
        print(f"  done {n_scen:,} x {n_bonds}", flush=True)

    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False,
                              float_format=lambda v: f"{v:,.4f}"))
    ok = bool((df["cvar_diff_in_se"] < 3.0).all())
    print(f"\n{'PASS' if ok else 'FAIL'}: GPU and CPU scenario sets agree "
          f"within 3 standard errors on benchmark CVaR95 at every size.")

    if args.out and rows:
        env = capture_environment()
        env["script"] = "scripts/check_rng_agreement.py"
        env["reps_per_size"] = str(REPS)
        env["aggregation"] = "median of reps"
        env["universe_seed"] = "7"
        env["scenario_seed"] = "11"
        env["alpha"] = "0.95"
        env["gate"] = "|cvar_gpu - cvar_cpu| < 3 standard errors at every size"
        env["verdict"] = "PASS" if ok else "FAIL"
        write_csv(args.out, rows, env)
        print(f"Wrote {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

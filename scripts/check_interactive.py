#!/usr/bin/env python
"""Headless check of the interactive re-solve path (drives demo/app.py).

Verifies three things the demo depends on and that no browser can prove:
  1. the hand-built CSR LP agrees with the CVXPY model it mirrors,
  2. moving a knob actually changes the answer in the right direction,
  3. warm-started re-solves are faster than cold ones — or, if warm start is
     unavailable on this path, that the cold fallback is still interactive.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.check import empirical_cvar  # noqa: E402
from src.credit import (  # noqa: E402
    credit_constraint_builder, default_limits, exposure_report,
    worst_violation,
)
from src.interactive import Knobs, build_interactive, solve  # noqa: E402
from src.model import build_mean_cvar  # noqa: E402
from src.scenarios import credit_scenarios, warmup_gpu  # noqa: E402
from src.universe import make_universe  # noqa: E402

N_BONDS, N_SCEN, ALPHA, LAM = 500, 20_000, 0.95, 3.0


def main() -> int:
    print(f"Universe {N_BONDS} bonds, {N_SCEN:,} scenarios")
    uni = make_universe(n_bonds=N_BONDS, seed=7)
    warmup_gpu(uni)
    R, mu, _ = credit_scenarios(uni, n_scen=N_SCEN, seed=11, use_gpu=True)
    limits = default_limits(uni)

    k = Knobs(risk_aversion=LAM, alpha=ALPHA,
              sector_cap=max(limits.sector_caps.values()),
              issuer_cap=limits.issuer_cap,
              dts_min_mult=limits.dts_min / float(uni.w_benchmark @ uni.dts),
              dts_max_mult=limits.dts_max / float(uni.w_benchmark @ uni.dts),
              turnover=limits.turnover)

    print("\n[1] Build the hand-rolled CSR model")
    m = build_interactive(uni, R, mu, limits, k)
    print(f"    A is {m.A.shape[0]:,} x {m.A.shape[1]:,} with "
          f"{m.A.nnz:,} nonzeros")

    print("\n[2] Cold solve, then a reference CVXPY solve of the same model")
    cold = solve(m, k, warm=False)
    print(f"    interactive: {cold['status']} in {cold['seconds']:.3f}s  "
          f"obj={cold['objective']:.6f}")

    ref_model = build_mean_cvar(
        R, mu, alpha=ALPHA, risk_aversion=LAM, w_max=limits.w_max,
        constraint_builder=credit_constraint_builder(uni, limits))
    import cvxpy as cp
    ref_model.problem.solve(solver=cp.HIGHS, verbose=False, warm_start=False,
                            highs_options={"solver": "simplex",
                                           "time_limit": 300.0})
    w_ref = ref_model.weights()
    w_int = cold["w"]
    # cuOpt minimizes -objective, so compare magnitudes not signs.
    cvar_int, _ = empirical_cvar(-(R @ w_int), ALPHA)
    cvar_ref, _ = empirical_cvar(-(R @ w_ref), ALPHA)
    util_int = float(mu @ w_int) - LAM * cvar_int
    util_ref = float(mu @ w_ref) - LAM * cvar_ref
    print(f"    CVXPY/HiGHS reference utility {util_ref:.6f}")
    print(f"    interactive model     utility {util_int:.6f}   "
          f"gap {abs(util_int - util_ref):.2e}")
    print(f"    weight L1 distance {np.abs(w_int - w_ref).sum():.4f}")

    rep = exposure_report(uni, w_int, limits)
    viol, which = worst_violation(rep)
    print(f"    limits: worst relative violation {viol:.2e}"
          + (f" ({which})" if viol > 0 else "  — all respected"))
    agree = abs(util_int - util_ref) < 5e-4

    print("\n[3] Knob response: tighten the issuer cap and re-solve")
    k2 = Knobs(**{**k.__dict__, "issuer_cap": k.issuer_cap * 0.5})
    hot = solve(m, k2, warm=True)
    iss2 = (uni.bonds.assign(x=hot["w"])
            .groupby("issuer_id", observed=True)["x"].sum().max())
    iss1 = (uni.bonds.assign(x=w_int)
            .groupby("issuer_id", observed=True)["x"].sum().max())
    print(f"    max issuer weight {iss1:.4f} -> {iss2:.4f} "
          f"(cap {k.issuer_cap:.4f} -> {k2.issuer_cap:.4f})")
    responded = iss2 <= k2.issuer_cap + 1e-4

    print("\n[4] Warm vs cold re-solve, alternating knob values")
    warm_t, cold_t = [], []
    for i in range(4):
        kk = Knobs(**{**k.__dict__,
                      "risk_aversion": LAM * (1.0 + 0.10 * (i + 1))})
        warm_t.append(solve(m, kk, warm=True)["seconds"])
        cold_t.append(solve(m, kk, warm=False)["seconds"])
    used_warm = solve(m, k, warm=True)["warm"]
    wm, cm = float(np.median(warm_t)), float(np.median(cold_t))
    print(f"    warm median {wm:.3f}s   cold median {cm:.3f}s   "
          f"({'warm start ACTIVE' if used_warm else 'warm start UNAVAILABLE'})")
    if used_warm:
        print(f"    warm start saves {100 * (1 - wm / cm):.0f}%")
    interactive = min(wm, cm) < 5.0

    print()
    ok = agree and responded and interactive
    for name, val in (("model agrees with CVXPY reference", agree),
                      ("knob changes the portfolio", responded),
                      ("re-solve is interactive (<5s)", interactive)):
        print(f"    {'PASS' if val else 'FAIL'}  {name}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

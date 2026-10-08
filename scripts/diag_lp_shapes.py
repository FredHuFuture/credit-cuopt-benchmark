#!/usr/bin/env python
"""Why is the hand-built LP 25x slower than the CVXPY one on the same model?

Both describe the same credit Mean-CVaR problem and reach the same solver.
One solves in seconds, the other in a minute. Rather than guess, extract what
each actually hands to cuOpt and compare: shape, nonzeros, coefficient ranges,
bound ranges.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.credit import credit_constraint_builder, default_limits  # noqa: E402
from src.interactive import Knobs, build_interactive  # noqa: E402
from src.model import build_mean_cvar  # noqa: E402
from src.scenarios import credit_scenarios, warmup_gpu  # noqa: E402
from src.universe import make_universe  # noqa: E402


def describe(name, A, c, lo, hi, vlo, vhi):
    d = np.abs(A.data)
    d = d[d > 0]
    print(f"\n--- {name} ---")
    print(f"  A            : {A.shape[0]:,} x {A.shape[1]:,}, "
          f"{A.nnz:,} nnz  ({A.nnz / (A.shape[0] * A.shape[1]):.2%} dense)")
    print(f"  |A| range    : {d.min():.3e} .. {d.max():.3e}  "
          f"(ratio {d.max() / d.min():.2e})")
    fc = np.abs(c[c != 0])
    print(f"  |obj| range  : {fc.min():.3e} .. {fc.max():.3e}")
    for label, v in (("row lo", lo), ("row hi", hi),
                     ("var lo", vlo), ("var hi", vhi)):
        f = v[np.isfinite(v)]
        f = f[f != 0]
        rng = (f"{np.abs(f).min():.3e} .. {np.abs(f).max():.3e}"
               if f.size else "all zero/infinite")
        n_inf = int((~np.isfinite(v)).sum())
        print(f"  {label:<12} : {rng}   ({n_inf:,} infinite)")


def main() -> int:
    n_bonds, n_scen = 500, 20_000
    uni = make_universe(n_bonds=n_bonds, seed=7)
    warmup_gpu(uni)
    R, mu, _ = credit_scenarios(uni, n_scen=n_scen, seed=11, use_gpu=True)
    limits = default_limits(uni)
    bench_dts = float(uni.w_benchmark @ uni.dts)

    # ---- what CVXPY hands to cuOpt ------------------------------------
    import cvxpy as cp
    m_cvx = build_mean_cvar(R, mu, alpha=0.95, risk_aversion=3.0,
                            w_max=limits.w_max,
                            constraint_builder=credit_constraint_builder(
                                uni, limits))
    data, _, _ = m_cvx.problem.get_problem_data(cp.CUOPT)
    A = sp.csr_matrix(data["A"])
    b = np.asarray(data["b"]).ravel()
    c = np.asarray(data["c"]).ravel()
    dims = data["dims"]
    n_eq = dims.zero
    row_lo = np.where(np.arange(A.shape[0]) < n_eq, b, -np.inf)
    row_hi = np.where(np.arange(A.shape[0]) < n_eq, b, b)
    vlo = data.get("lower_bounds")
    vhi = data.get("upper_bounds")
    vlo = np.full(A.shape[1], -np.inf) if vlo is None else np.asarray(vlo)
    vhi = np.full(A.shape[1], np.inf) if vhi is None else np.asarray(vhi)
    describe("CVXPY canonicalization (what the fast path solves)",
             A, c, row_lo, row_hi, vlo, vhi)
    print(f"  cone dims    : zero={dims.zero:,}  nonneg={dims.nonneg:,}")

    # ---- what the hand-built path hands to cuOpt ----------------------
    k = Knobs(alpha=0.95, sector_cap=float(max(limits.sector_caps.values())),
              issuer_cap=limits.issuer_cap,
              dts_min_mult=limits.dts_min / bench_dts,
              dts_max_mult=limits.dts_max / bench_dts,
              turnover=limits.turnover)
    m = build_interactive(uni, R, mu, limits, k)
    lo, hi = m.scaled_bounds()
    describe("hand-built DataModel (the slow path)",
             m.A, m.objective(k), lo, hi, m.var_lb, m.var_ub)

    print("\nRow-scale factors applied by the hand-built path:")
    for name, sl in m.rows.items():
        s = m.row_scale[sl] if isinstance(sl, slice) else m.row_scale[sl:sl + 1]
        s = np.atleast_1d(s)
        print(f"  {name:<16} n={s.size:>6,}  scale {s.min():.3e} .. "
              f"{s.max():.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

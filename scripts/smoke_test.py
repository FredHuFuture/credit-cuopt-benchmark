#!/usr/bin/env python
"""M0 smoke test for bond-cvar-cuopt.

Verifies, in order:
  1. GPU visible from inside WSL2
  2. cuDF imports and runs on the GPU
  3. A trivial 2-variable LP solves to Optimal on cuOpt
  4. A toy Mean-CVaR LP (Rockafellar-Uryasev, 50 assets x 500 scenarios)
     solves on cuOpt with the GPU PDLP method explicitly pinned

Exit code 0 means the GPU path for this project works end to end.

Doc references:
  - cuOpt Python modeling API:
    https://docs.nvidia.com/cuopt/user-guide/latest/cuopt-python/lp-qp-milp/lp-qp-milp-examples.html
  - RAPIDS WSL2 verification snippet: https://docs.rapids.ai/install/
"""

import subprocess
import sys
import time

import numpy as np


def step(msg: str) -> None:
    print(f"\n=== {msg} ===", flush=True)


def main() -> int:
    step("1/4 GPU visible inside WSL2")
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
         "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    print(out.stdout.strip() or out.stderr.strip())
    assert out.returncode == 0, "nvidia-smi failed inside WSL2"

    step("2/4 cuDF")
    import cudf  # noqa: E402
    s = cudf.Series([1, 2, 3])
    total = int(s.sum())
    print(f"cudf {cudf.__version__}, Series sum = {total}")
    assert total == 6

    step("3/4 trivial LP on cuOpt")
    from cuopt.linear_programming.problem import (  # noqa: E402
        Problem, CONTINUOUS, MAXIMIZE,
    )
    from cuopt.linear_programming.solver_settings import (  # noqa: E402
        PDLPSolverMode, SolverMethod, SolverSettings,
    )

    p = Problem()
    x = p.addVariable(lb=0, ub=4, vtype=CONTINUOUS, name="x")
    y = p.addVariable(lb=0, ub=8, vtype=CONTINUOUS, name="y")
    p.addConstraint(x + y <= 10, name="cap")
    p.setObjective(x + y, sense=MAXIMIZE)
    t0 = time.perf_counter()
    # Left on defaults deliberately: this step only asks "does cuOpt solve an
    # LP at all", and makes no claim about which method or device did it.
    p.solve(SolverSettings())
    dt = time.perf_counter() - t0
    status = getattr(p.Status, "name", str(p.Status))
    print(f"status={status} obj={p.ObjValue:.6f} solve_time={dt:.3f}s")
    assert status == "Optimal", f"expected Optimal, got {status}"
    assert abs(p.ObjValue - 10.0) < 1e-4

    step("4/4 toy Mean-CVaR LP (Rockafellar-Uryasev) on cuOpt PDLP (GPU)")
    # max  mu'w - lam * (t + 1/((1-alpha)S) * sum(u))
    # s.t. u_s + t + (R w)_s >= 0,  u >= 0,  sum(w) = 1,  0 <= w <= w_max
    rng = np.random.default_rng(42)
    n_assets, n_scen = 50, 500
    alpha, lam, w_max = 0.95, 1.0, 0.10
    # Fat-tailed synthetic daily returns: Student-t(5), ~1% daily vol
    R = 0.01 * rng.standard_t(df=5, size=(n_scen, n_assets)) + 0.0005
    mu = R.mean(axis=0)

    p2 = Problem()
    w = [p2.addVariable(lb=0.0, ub=w_max, vtype=CONTINUOUS, name=f"w{i}")
         for i in range(n_assets)]
    t_var = p2.addVariable(lb=-1e6, ub=1e6, vtype=CONTINUOUS, name="t")
    u = [p2.addVariable(lb=0.0, ub=1e6, vtype=CONTINUOUS, name=f"u{s_}")
         for s_ in range(n_scen)]

    p2.addConstraint(sum(w) == 1.0, name="budget")
    for s_ in range(n_scen):
        p2.addConstraint(
            u[s_] + t_var + sum(float(R[s_, i]) * w[i]
                                for i in range(n_assets)) >= 0.0,
            name=f"cvar{s_}",
        )
    cvar_term = t_var + (1.0 / ((1.0 - alpha) * n_scen)) * sum(u)
    p2.setObjective(
        sum(float(mu[i]) * w[i] for i in range(n_assets)) - lam * cvar_term,
        sense=MAXIMIZE,
    )

    # Pin PDLP explicitly. A bare SolverSettings() leaves cuOpt in its
    # concurrent mode, which races GPU PDLP against CPU dual simplex and
    # returns whichever finishes first — at this size that is usually the CPU.
    # The line this script prints at the end is a claim about the GPU path, so
    # the GPU path is what has to run. Same pinning as src/interactive.py.
    ss2 = SolverSettings()
    ss2.set_parameter("method", SolverMethod.PDLP)
    ss2.set_parameter("pdlp_solver_mode", PDLPSolverMode.Stable2)
    # Tolerances tightened from PDLP's defaults because the assertions below
    # are on the solution, not on the objective: a first-order method at 1e-4
    # can land sum(w) further from 1 than a simplex vertex ever would.
    for _p in ("absolute_primal_tolerance", "relative_primal_tolerance",
               "absolute_dual_tolerance", "relative_dual_tolerance",
               "absolute_gap_tolerance", "relative_gap_tolerance"):
        ss2.set_parameter(_p, 1e-6)
    t0 = time.perf_counter()
    p2.solve(ss2)
    dt = time.perf_counter() - t0
    status = getattr(p2.Status, "name", str(p2.Status))
    weights = np.array([v.getValue() for v in w])
    print(f"method=PDLP (pinned, GPU)  status={status} "
          f"obj={p2.ObjValue:.6f} solve_time={dt:.3f}s")
    print(f"sum(w)={weights.sum():.6f} max(w)={weights.max():.4f} "
          f"n_nonzero={(weights > 1e-6).sum()}")
    assert status == "Optimal", f"expected Optimal, got {status}"
    assert abs(weights.sum() - 1.0) < 1e-3
    assert weights.max() <= w_max + 1e-4

    print("\nSMOKE_TEST_OK: Mean-CVaR LP solves on the GPU.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

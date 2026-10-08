#!/usr/bin/env python
"""Does a GPU first-order solve survive a real tracking-error mandate? (M6)

M2 found that cuOpt at 1e-4 returned a portfolio that breached the turnover
budget by 9.8% while reporting "Optimal". That was our own tolerance choice
against our own limit, so it invited the obvious rebuttal: tighten the
tolerance and the problem goes away.

This script removes the choice. Bloomberg's optimized fixed income index
methodology 34100 constrains ex-ante tracking error at **10bp**, relaxed in
10bp increments only when the problem proves infeasible. That is a published
production mandate, not a parameter we picked, and it is roughly three times
the per-period return scale a solver run at 1e-4 can resolve. The question is
empirical:

    At what tracking-error budget does a first-order GPU solve stop returning
    a portfolio that respects the mandate — and what does it cost to fix?

Two axes, because the mandate's *form* matters as much as its tightness:

1. **Budget sweep**, `te_mode="mad"`. The Konno-Yamazaki active-MAD form is
   LP-representable, so PDLP applies and the sweep isolates tolerance against
   tightness with the problem class held fixed.
2. **One `te_mode="sigma"` row.** Ex-ante TE as the industry defines it is a
   standard deviation, which is a second-order cone constraint. PDLP cannot
   solve it at all. That row is expected to fail, and the failure is the
   result: the faithful reading of 34100 takes the problem out of the class
   where the GPU advantage lives.

Every TE number in the output is recomputed from the scenario matrix by
`src.credit.tracking_error_bp` — numpy on the returned weights, no LP algebra,
no solver self-report.

Examples:
    # full run, needs a GPU
    python scripts/run_te_mandate.py --out results/m6_te_mandate.csv

    # CPU-only formulation check, no GPU required
    python scripts/run_te_mandate.py --n-bonds 120 --n-scenarios 1500 \
        --solvers clarabel --cpu-only
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.bench import (  # noqa: E402
    assert_gpu_idle, capture_environment, gpu_contention, write_csv,
)
from src.check import check_solution, empirical_cvar  # noqa: E402
from src.credit import (  # noqa: E402
    BLOOMBERG_34100_TE_BP, DESK_TOL_BY_UNIT, UNIT_RETURN_BP,
    credit_constraint_builder, default_limits, exposure_report,
    gate_verdicts, tracking_error_bp, worst_violation,
)
from src.model import build_mean_cvar  # noqa: E402
from src.scenarios import credit_scenarios  # noqa: E402
from src.universe import make_universe  # noqa: E402

#: Solver configurations. `conic` marks the ones that can accept a
#: second-order cone constraint; the rest are LP-only and receive
#: UNSUPPORTED_BY_METHOD on the sigma row, because "PDLP/simplex cannot
#: express this" is a fact about the method class, not a failed solve.
SOLVERS = {
    "cuopt-pdlp-1e-4": dict(
        kind="cuopt", tol=1e-4, crossover=False, conic=False,
        method_class="LP", endpoint="PDLP",
    ),
    "cuopt-pdlp-1e-6": dict(
        kind="cuopt", tol=1e-6, crossover=False, conic=False,
        method_class="LP", endpoint="PDLP",
    ),
    "cuopt-pdlp-xover": dict(
        kind="cuopt", tol=1e-4, crossover=True, conic=False,
        method_class="LP", endpoint="PDLP+crossover",
    ),
    "cuopt-barrier-socp": dict(
        kind="cuopt_barrier", conic=True, method_class="SOCP",
        endpoint="barrier",
    ),
    "clarabel-socp": dict(
        kind="clarabel", conic=True, method_class="SOCP",
        endpoint="interior-point",
    ),
    "highs-simplex": dict(
        kind="highs", conic=False, method_class="LP", endpoint="simplex",
    ),
}

SOLVER_ALIASES = {
    "cuopt@1e-4": "cuopt-pdlp-1e-4",
    "cuopt@1e-6": "cuopt-pdlp-1e-6",
    "cuopt@1e-4+xover": "cuopt-pdlp-xover",
    "clarabel": "clarabel-socp",
}

LEGACY_SOLVER_IDS = {v: k for k, v in SOLVER_ALIASES.items()}

DEFAULT_SOLVERS = (
    "cuopt-pdlp-1e-4", "cuopt-pdlp-1e-6", "cuopt-pdlp-xover",
    "cuopt-barrier-socp", "clarabel-socp", "highs-simplex",
)

#: Sweep points in annualized bp. 10 is the Bloomberg 34100 level; the others
#: bracket it so the failure threshold is located rather than asserted.
DEFAULT_TE_GRID = (5.0, 10.0, 20.0, 50.0, 100.0)

DEFAULT_P0_MAD_GRID_BP = (1.0, 2.0, 3.0, 5.0, 10.0)
DEFAULT_VALIDATION_SCEN_SEEDS = (101, 211, 307)

#: Relative tolerance on the mandate check. A solver run at 1e-4 lands ~1e-4
#: outside every binding constraint by construction; scoring that as a mandate
#: breach measures the tolerance, not the portfolio. The first run of this
#: script had no band at all and reported "NO configuration respected the
#: mandate" on rows whose worst relative violation was 2.5e-7. The band is
#: deliberately looser than any solver tolerance used here and far tighter
#: than an economically meaningful breach — the 2.7%-4.6% turnover breaches
#: this run did find sit three orders of magnitude above it.
MANDATE_REL_TOL = 1e-3

#: Statuses whose weight vector is a converged answer. Anything else (a
#: `user_limit` timeout, a solver error) returns an arbitrary iterate, and
#: recomputing tracking error from it produces a number with no meaning —
#: the first run reported a 261,727bp "breach" from a timed-out simplex.
CONVERGED = ("optimal", "optimal_inaccurate")

#: Desk-materiality tolerance on the TE budget itself, in annualized bp: the
#: precision ex-ante TE is quoted to. `MANDATE_REL_TOL` above is a
#: SOLVER-tolerance band — "is this breach bigger than the arithmetic?" —
#: and this is an economic one — "would the desk see it?". They are different
#: questions with different answers, so both are reported per row and neither
#: is allowed to stand in for the other.
TE_DESK_TOL_BP = DESK_TOL_BY_UNIT[UNIT_RETURN_BP]


@dataclass(frozen=True)
class SolverCallConfig:
    canonical_id: str
    solver: str
    opts: dict[str, Any]


@dataclass(frozen=True)
class SolveResult:
    first_solve_s: float
    steady_state_median_s: float
    steady_state_runs_s: tuple[float, ...]
    status: str
    objective: float
    total_endpoint_s: float


def canonical_solver_id(name: str) -> str:
    canonical = SOLVER_ALIASES.get(name, name)
    if canonical not in SOLVERS:
        raise ValueError(f"unknown solver {name!r}")
    return canonical


def solver_call_config(name: str, time_limit: float) -> SolverCallConfig:
    """CVXPY solver and options for one public solver ID."""
    import cvxpy as cp

    canonical = canonical_solver_id(name)
    spec = SOLVERS[canonical]
    kind = spec["kind"]
    if kind == "cuopt":
        tol = spec["tol"]
        opts = dict(
            solver_method="PDLP", pdlp_solver_mode="Stable2", presolve=2,
            absolute_primal_tolerance=tol, relative_primal_tolerance=tol,
            absolute_dual_tolerance=tol, relative_dual_tolerance=tol,
            absolute_gap_tolerance=tol, relative_gap_tolerance=tol,
            time_limit=time_limit,
            crossover=1 if spec["crossover"] else 0,
        )
        solver = getattr(cp, "CUOPT", "CUOPT")
    elif kind == "cuopt_barrier":
        opts = dict(solver_method="Barrier", time_limit=time_limit)
        solver = getattr(cp, "CUOPT", "CUOPT")
    elif kind == "highs":
        opts = {"highs_options": {
            "solver": "simplex", "time_limit": time_limit, "threads": 16,
            "parallel": "on", "random_seed": 0,
        }}
        solver = cp.HIGHS
    elif kind == "clarabel":
        opts = {}
        solver = cp.CLARABEL
    else:
        raise ValueError(kind)
    return SolverCallConfig(canonical_id=canonical, solver=solver, opts=opts)


def _resource_snapshot() -> dict[str, float]:
    snap = gpu_contention(samples=1, gap_s=0.0)
    return {
        "gpu_util_pct": float(snap.get("gpu_util_pct", math.nan)),
        "gpu_mem_used_mib": float(snap.get("gpu_mem_used_mib", math.nan)),
    }


def _resource_columns(
    before: dict[str, float] | None = None,
    after: dict[str, float] | None = None,
) -> dict[str, Any]:
    before = before or {}
    after = after or {}
    cols = {
        "resource_before_gpu_util_pct": before.get("gpu_util_pct", np.nan),
        "resource_before_gpu_mem_used_mib": before.get(
            "gpu_mem_used_mib", np.nan),
        "resource_after_gpu_util_pct": after.get("gpu_util_pct", np.nan),
        "resource_after_gpu_mem_used_mib": after.get(
            "gpu_mem_used_mib", np.nan),
    }
    unavailable = [
        k for k, v in cols.items()
        if isinstance(v, float) and math.isnan(v)
    ]
    cols["resource_note"] = (
        "GPU telemetry unavailable from nvidia-smi; emitted NaN"
        if unavailable else ""
    )
    return cols


def _timing_columns(
    build_s: float = np.nan,
    first_solve_s: float = np.nan,
    steady_state_median_s: float = np.nan,
    total_endpoint_s: float = np.nan,
) -> dict[str, float]:
    return {
        "build_s": build_s,
        "first_solve_s": first_solve_s,
        "steady_state_median_s": steady_state_median_s,
        "total_endpoint_s": total_endpoint_s,
        # Backward-compatible alias for old plotting/table code.
        "wall_s": steady_state_median_s,
    }


def independent_portfolio_metrics(
    w: np.ndarray,
    R: np.ndarray,
    mu: np.ndarray,
    alpha: float,
    risk_aversion: float,
    solver_objective: float,
) -> dict[str, float]:
    """Recompute row economics from weights and scenarios only."""
    w = np.asarray(w, dtype=np.float64).ravel()
    mu = np.asarray(mu, dtype=np.float64).ravel()
    cvar, _ = empirical_cvar(-(R @ w), alpha)
    expected = float(mu @ w)
    objective = expected - risk_aversion * cvar
    return {
        "solver_objective": float(solver_objective),
        "objective_recomputed": objective,
        "objective_discrepancy": abs(objective - float(solver_objective)),
        "expected_excess_bp_monthly": expected * 1e4,
        "cvar_recomputed": cvar,
    }


def _gate_columns_for_no_portfolio(verdict: str) -> dict:
    """Gate columns for a row where no scoreable portfolio exists.

    NOT-IN-CLASS and ERROR rows must carry the same keys as scored rows —
    `write_csv` takes its fieldnames from the first row — but they must not
    carry a verdict they did not earn. Everything numeric is NaN.
    """
    return {
        "purity_pass": False, "purity_worst_rel": np.nan,
        "purity_worst_limit": "-", "desk_pass": False,
        "desk_worst_multiple": np.nan, "desk_worst_limit": "-",
        "desk_worst_bp_weight": np.nan, "desk_worst_usd": np.nan,
        "n_purity_breaches": np.nan, "n_desk_breaches": np.nan,
        "gate_verdict": verdict,
        "te_breach_desk_multiple": np.nan,
        "respects_mandate_desk": False,
    }


def _base_result_row(
    case: str,
    te_mode: str,
    te_budget_bp: float,
    solver_id: str,
) -> dict[str, Any]:
    canonical = canonical_solver_id(solver_id)
    spec = SOLVERS[canonical]
    return {
        "case": case,
        "te_mode": te_mode,
        "te_budget_bp": te_budget_bp,
        "solver": canonical,
        "solver_requested": solver_id,
        "solver_legacy_id": LEGACY_SOLVER_IDS.get(canonical, ""),
        "method_class": spec["method_class"],
        "endpoint": spec["endpoint"],
        "status": "",
        "status_source": "",
        **_timing_columns(),
        "solver_objective": np.nan,
        "objective_recomputed": np.nan,
        "objective_discrepancy": np.nan,
        "expected_excess_bp_monthly": np.nan,
        "cvar_recomputed": np.nan,
        "te_sigma_bp": np.nan,
        "te_mad_implied_bp": np.nan,
        "te_breach_bp": np.nan,
        "validation_seed": "",
        "validation_summary_stat": "",
        "verdict": "",
        "respects_mandate": False,
        "worst_rel_violation": np.nan,
        "worst_limit": "-",
        "check_ok": False,
        "note": "",
        **_gate_columns_for_no_portfolio(""),
        **_resource_columns(),
    }


def make_unsupported_row(
    case: str,
    te_mode: str,
    te_budget_bp: float,
    solver_id: str,
) -> dict[str, Any]:
    row = _base_result_row(case, te_mode, te_budget_bp, solver_id)
    row.update({
        "status": "UNSUPPORTED_BY_METHOD",
        "status_source": "author-classification",
        "verdict": "UNSUPPORTED_BY_METHOD",
        "note": "LP-only method cannot express the sigma TE SOC constraint",
        **_gate_columns_for_no_portfolio("UNSUPPORTED_BY_METHOD"),
    })
    return row


def make_error_row(
    case: str,
    te_mode: str,
    te_budget_bp: float,
    solver_id: str,
    error: Exception,
    error_stage: str,
    build_s: float = np.nan,
    before: dict[str, float] | None = None,
    after: dict[str, float] | None = None,
) -> dict[str, Any]:
    if error_stage not in {"build", "solver", "check", "runner"}:
        raise ValueError(f"unknown error_stage {error_stage!r}")
    row = _base_result_row(case, te_mode, te_budget_bp, solver_id)
    row.update({
        "status": f"ERROR:{type(error).__name__}",
        "status_source": (
            "solver-returned"
            if error_stage == "solver" else f"runner-{error_stage}-error"
        ),
        "build_s": build_s,
        "verdict": "ERROR",
        "note": str(error)[:200],
        **_gate_columns_for_no_portfolio("ERROR"),
        **_resource_columns(before, after),
    })
    return row


def _is_eligible_recalibration_row(
    row: dict[str, Any],
    sigma_target_bp: float,
    sigma_tol_bp: float,
) -> bool:
    return (
        row.get("te_mode") == "mad" and
        row.get("status") in CONVERGED
        and bool(row.get("check_ok"))
        and bool(row.get("purity_pass"))
        and bool(row.get("desk_pass"))
        and float(row.get("te_sigma_bp", np.inf))
        <= sigma_target_bp + sigma_tol_bp
        and np.isfinite(float(row.get("objective_recomputed", np.nan)))
    )


def select_recalibration_candidate(
    rows: list[dict[str, Any]],
    sigma_target_bp: float,
    sigma_tol_bp: float = 0.0,
) -> dict[str, Any] | None:
    """Pick the highest independently recomputed objective among eligibles."""
    eligible = [
        r for r in rows
        if _is_eligible_recalibration_row(r, sigma_target_bp, sigma_tol_bp)
    ]
    if not eligible:
        return None
    return max(eligible, key=lambda r: float(r["objective_recomputed"]))


def _representative_sigma_row(
    rows: list[dict[str, Any]],
    budget_bp: float,
) -> dict[str, Any] | None:
    eligible = [
        r for r in rows
        if r.get("te_mode") == "mad"
        and float(r.get("te_budget_bp", np.nan)) == float(budget_bp)
        and r.get("status") in CONVERGED
        and bool(r.get("check_ok"))
        and np.isfinite(float(r.get("te_sigma_bp", np.nan)))
        and np.isfinite(float(r.get("objective_recomputed", np.nan)))
    ]
    if not eligible:
        return None
    return max(eligible, key=lambda r: float(r["objective_recomputed"]))


def build_bisection_budgets(
    initial_budgets: tuple[float, ...] | list[float],
    sigma_at_budget: Callable[[float], float],
    sigma_target_bp: float,
    max_steps: int = 8,
    stop_width_bp: float = 0.1,
) -> list[float]:
    """Return initial budgets plus midpoint budgets until the bracket is tight.

    The selector is deliberately separate and non-monotone-safe. Bisection is
    only a budget-generation convenience once adjacent points bracket the
    requested true sigma target.
    """
    budgets = [float(v) for v in initial_budgets]
    scored = [(b, float(sigma_at_budget(b))) for b in budgets]
    bracket: tuple[float, float, float, float] | None = None
    for (lo_b, lo_s), (hi_b, hi_s) in zip(scored, scored[1:]):
        lo_f = lo_s - sigma_target_bp
        hi_f = hi_s - sigma_target_bp
        if not (math.isfinite(lo_f) and math.isfinite(hi_f)):
            continue
        if lo_f * hi_f <= 0.0:
            bracket = (lo_b, hi_b, lo_f, hi_f)
            break
    if bracket is None:
        return budgets

    lo, hi, lo_f, hi_f = bracket
    for _ in range(max_steps):
        if abs(hi - lo) <= stop_width_bp:
            break
        mid = (lo + hi) / 2.0
        if mid not in budgets:
            budgets.append(mid)
        mid_f = float(sigma_at_budget(mid)) - sigma_target_bp
        if not math.isfinite(mid_f):
            break
        if lo_f * mid_f <= 0.0:
            hi, hi_f = mid, mid_f
        else:
            lo, lo_f = mid, mid_f
    return budgets


def solve_once(model, name: str, runs: int = 3, time_limit: float = 300.0):
    """First solve plus `runs` steady-state solves.

    Median of repeats for the same reason as everywhere else in this repo:
    single-shot spread on these solves has been measured above 40%, which is
    larger than several of the differences being compared. DNF/error rows stop
    after this first capped attempt.
    """
    import cvxpy as cp

    cfg = solver_call_config(name, time_limit)
    # warm_start=False: CVXPY defaults it True and HiGHS caches solutions,
    # which would time warm re-solves for one solver only.
    t0 = time.perf_counter()
    try:
        model.problem.solve(solver=cfg.solver, verbose=False,
                            warm_start=False, **cfg.opts)
        status = model.problem.status
    except cp.error.SolverError as e:
        status = f"solver_error: {e}"
    first_solve_s = time.perf_counter() - t0
    first_obj = float(model.problem.value or "nan")
    if status not in CONVERGED:
        return SolveResult(
            first_solve_s=first_solve_s,
            steady_state_median_s=math.nan,
            steady_state_runs_s=(),
            status=status,
            objective=first_obj,
            total_endpoint_s=first_solve_s,
        )

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        try:
            model.problem.solve(solver=cfg.solver, verbose=False,
                                warm_start=False, **cfg.opts)
            status = model.problem.status
        except cp.error.SolverError as e:
            status = f"solver_error: {e}"
        wall = time.perf_counter() - t0
        if status not in CONVERGED:
            return SolveResult(
                first_solve_s=first_solve_s,
                steady_state_median_s=math.nan,
                steady_state_runs_s=tuple(times),
                status=status,
                objective=float(model.problem.value or "nan"),
                total_endpoint_s=first_solve_s + sum(times) + wall,
            )
        times.append(wall)
    return SolveResult(
        first_solve_s=first_solve_s,
        steady_state_median_s=statistics.median(times) if times else math.nan,
        steady_state_runs_s=tuple(times),
        status=status,
        objective=float(model.problem.value),
        total_endpoint_s=first_solve_s + sum(times),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-bonds", type=int, default=500)
    ap.add_argument("--n-scenarios", type=int, default=20_000)
    ap.add_argument("--alpha", type=float, default=0.95)
    ap.add_argument("--risk-aversion", type=float, default=3.0)
    ap.add_argument("--turnover", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--scen-seed", type=int, default=11)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--te-grid", type=float, nargs="*",
                    default=list(DEFAULT_TE_GRID),
                    help="annualized TE budgets in bp for the MAD sweep")
    ap.add_argument("--p0-candidate-grid", action="store_true",
                    help="use the P0 MAD recalibration grid: 1,2,3,5,10bp")
    ap.add_argument("--p0-bisect", action="store_true",
                    help="append MAD bisection candidates around sigma target")
    ap.add_argument("--sigma-target-bp", type=float,
                    default=BLOOMBERG_34100_TE_BP,
                    help="true sigma TE target used for P0 recalibration")
    ap.add_argument("--validation-scen-seeds", type=int, nargs="*",
                    default=None,
                    help="post-selection validation scenario seeds; P0 uses "
                         "three defaults when omitted")
    ap.add_argument("--skip-sigma", action="store_true",
                    help="omit the second-order-cone row")
    ap.add_argument("--mandate-rel-tol", type=float, default=MANDATE_REL_TOL,
                    help="relative band on the mandate check; must exceed the "
                         "loosest solver tolerance in the run")
    ap.add_argument("--time-limit", type=float, default=1800.0,
                    help="per-solve cap in seconds. The MAD form adds S "
                         "auxiliary variables and ~2S rows on top of the CVaR "
                         "block, so simplex and crossover need far more than "
                         "the untracked-TE problem: both timed out at 300s on "
                         "a 500x10000 instance")
    ap.add_argument("--solvers", nargs="*", default=list(DEFAULT_SOLVERS),
                    choices=sorted(set(SOLVERS) | set(SOLVER_ALIASES)),
                    help="solver configs to run")
    ap.add_argument("--cpu-only", action="store_true",
                    help="CPU scenario generation and no GPU-idle check; for "
                         "formulation checks on a machine with no card")
    ap.add_argument("--allow-busy-gpu", action="store_true")
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow replacing an existing --out CSV")
    args = ap.parse_args()
    if args.p0_candidate_grid:
        args.te_grid = list(DEFAULT_P0_MAD_GRID_BP)
    if args.validation_scen_seeds is None:
        args.validation_scen_seeds = (
            list(DEFAULT_VALIDATION_SCEN_SEEDS)
            if args.p0_candidate_grid else []
        )
    if args.out and Path(args.out).exists() and not args.overwrite:
        raise FileExistsError(
            f"{args.out} already exists; pass --overwrite to replace it")

    if not args.cpu_only:
        assert_gpu_idle(raise_on_busy=not args.allow_busy_gpu)

    pd.set_option("display.width", 150)
    pd.set_option("display.max_columns", 24)

    print(f"[1] Universe: {args.n_bonds} bonds (seed {args.seed})")
    uni = make_universe(n_bonds=args.n_bonds, seed=args.seed)

    print(f"[2] Scenarios: {args.n_scenarios:,} monthly excess-return paths")
    R, mu, info = credit_scenarios(uni, n_scen=args.n_scenarios,
                                   seed=args.scen_seed,
                                   use_gpu=not args.cpu_only,
                                   return_mode="excess")
    print(f"    {info['device']}: {info['seconds']:.3f}s")

    # The benchmark's own TE is zero by construction; the incumbent book's is
    # not, and it is the honest scale reference for whether a 10bp budget is
    # even reachable from where the portfolio actually starts.
    te_w0 = tracking_error_bp(uni, uni.w0, R)
    print(f"    incumbent w0 tracking error: "
          f"{te_w0['te_sigma_bp']:.1f}bp sigma / "
          f"{te_w0['te_mad_implied_bp']:.1f}bp MAD-implied (annualized); "
          f"ratio {te_w0['sigma_over_mad_implied']:.3f}")
    if te_w0["sigma_over_mad_implied"] > 1.05:
        print("    NOTE: sigma exceeds the MAD-implied value, i.e. active "
              "returns are fatter-tailed than Gaussian, so a MAD budget is "
              "LOOSER than the sigma budget it was calibrated against.")

    # Unconstrained-TE reference: what does the optimizer do when nobody is
    # watching tracking error? Everything below is relative to this.
    print("\n[3] Reference solve, no TE family")
    limits0 = default_limits(uni, turnover=args.turnover)
    model0 = build_mean_cvar(
        R, mu, alpha=args.alpha, risk_aversion=args.risk_aversion,
        w_max=limits0.w_max,
        constraint_builder=credit_constraint_builder(uni, limits0, R=R))
    ref_solve = solve_once(model0, args.solvers[-1], runs=1,
                           time_limit=args.time_limit)
    st0 = ref_solve.status
    if st0 in CONVERGED:
        te_ref = tracking_error_bp(uni, model0.weights(), R)
        print(f"    {args.solvers[-1]}: {st0}, "
              f"unconstrained TE = {te_ref['te_sigma_bp']:.1f}bp sigma "
              f"({te_ref['te_mad_implied_bp']:.1f}bp MAD-implied)")
    else:
        te_ref = {"te_sigma_bp": np.nan, "te_mad_implied_bp": np.nan}
        print(f"    {args.solvers[-1]}: {st0}, "
              "unconstrained TE unavailable")
    # ASCII only in prints: this script is run from a Windows console under
    # cp1252 as well as from WSL, and an em-dash renders as mojibake there.
    print(f"    Bloomberg 34100 budget is {BLOOMBERG_34100_TE_BP:.0f}bp; "
          f"the unconstrained optimum is "
          f"{te_ref['te_sigma_bp'] / BLOOMBERG_34100_TE_BP:.1f}x that.")

    rows = []

    def run_case(case_name: str, te_kw: dict[str, float | str]) -> None:
        limits = default_limits(uni, turnover=args.turnover, **te_kw)
        is_conic = te_kw.get("te_mode") == "sigma"
        for sname in args.solvers:
            spec = SOLVERS[canonical_solver_id(sname)]
            if is_conic and not spec["conic"]:
                rows.append(make_unsupported_row(
                    case_name, str(te_kw["te_mode"]), float(te_kw["te_bp"]),
                    sname,
                ))
                print(f"    {case_name:<16} {sname:<20} "
                      f"UNSUPPORTED_BY_METHOD")
                continue
            row = _base_result_row(
                case_name, str(te_kw["te_mode"]), float(te_kw["te_bp"]),
                sname,
            )
            before = _resource_snapshot()
            build_s = math.nan
            error_stage = "build"
            try:
                t_build = time.perf_counter()
                model = build_mean_cvar(
                    R, mu, alpha=args.alpha,
                    risk_aversion=args.risk_aversion, w_max=limits.w_max,
                    constraint_builder=credit_constraint_builder(
                        uni, limits, R=R))
                build_s = time.perf_counter() - t_build
                error_stage = "solver"
                solved = solve_once(model, sname, runs=args.runs,
                                    time_limit=args.time_limit)
                after = _resource_snapshot()
                status = solved.status
                converged = status in CONVERGED
                row.update({
                    "status": status,
                    "status_source": "solver-returned",
                    **_timing_columns(
                        build_s=build_s,
                        first_solve_s=solved.first_solve_s,
                        steady_state_median_s=solved.steady_state_median_s,
                        total_endpoint_s=build_s + solved.total_endpoint_s,
                    ),
                    **_resource_columns(before, after),
                })
                if not converged:
                    row.update({
                        "verdict": "UNCONVERGED",
                        **_gate_columns_for_no_portfolio("UNCONVERGED"),
                    })
                    rows.append(row)
                    print(f"    {case_name:<16} {sname:<18} {status:<12} "
                          f"{solved.first_solve_s:7.2f}s  UNCONVERGED")
                    continue

                error_stage = "check"
                w = model.weights()
                te = tracking_error_bp(uni, w, R, limits.periods_per_year)
                chk = check_solution(
                    w=w, solver_objective=solved.objective, R=R, mu=mu,
                    alpha=args.alpha, risk_aversion=args.risk_aversion,
                    w_max=limits.w_max)
                rep = exposure_report(uni, w, limits, R=R)
                viol, which = worst_violation(rep)
                scored = (te["te_mad_implied_bp"] if te_kw["te_mode"] == "mad"
                          else te["te_sigma_bp"])
                breach = scored - float(te_kw["te_bp"])
                band = args.mandate_rel_tol * float(te_kw["te_bp"])
                if breach <= band and viol <= args.mandate_rel_tol:
                    verdict, respects = "OK", True
                else:
                    verdict, respects = "BREACH", False
                gcols = gate_verdicts(rep).as_row()
                te_mult = max(0.0, breach) / TE_DESK_TOL_BP
                gcols["te_breach_desk_multiple"] = te_mult
                gcols["respects_mandate_desk"] = bool(
                    te_mult <= 1.0 and gcols["desk_pass"])
                row.update({
                    **independent_portfolio_metrics(
                        w=w, R=R, mu=mu, alpha=args.alpha,
                        risk_aversion=args.risk_aversion,
                        solver_objective=solved.objective,
                    ),
                    "te_sigma_bp": te["te_sigma_bp"],
                    "te_mad_implied_bp": te["te_mad_implied_bp"],
                    "te_breach_bp": breach,
                    "verdict": verdict,
                    "respects_mandate": respects,
                    "worst_rel_violation": viol,
                    "worst_limit": which if viol > 0 else "-",
                    "check_ok": bool(chk.ok),
                    "_weights": w,
                    **gcols,
                })
                rows.append(row)
                print(f"    {case_name:<16} {sname:<20} {status:<12} "
                      f"{solved.steady_state_median_s:7.2f}s  "
                      f"TE={scored:6.1f}bp ({breach:+.2f}bp)  "
                      f"viol={viol:.1e}  {verdict}")
            except Exception as e:  # noqa: BLE001
                after = _resource_snapshot()
                rows.append(make_error_row(
                    case=case_name, te_mode=str(te_kw["te_mode"]),
                    te_budget_bp=float(te_kw["te_bp"]), solver_id=sname,
                    error=e, error_stage=error_stage, build_s=build_s,
                    before=before, after=after,
                ))
                print(f"    {case_name:<16} {sname:<18} "
                      f"FAILED {type(e).__name__}: {str(e)[:90]}")

    initial_cases = [(f"mad@{bp:g}bp", dict(te_bp=bp, te_mode="mad"))
                     for bp in args.te_grid]
    if args.p0_bisect:
        print(f"\n[4] P0 initial grid: {len(initial_cases)} MAD budgets x "
              f"{len(args.solvers)} solvers")
    else:
        n_cases = len(initial_cases) + (0 if args.skip_sigma else 1)
        print(f"\n[4] {n_cases} TE configurations x "
              f"{len(args.solvers)} solvers")
    for case_name, te_kw in initial_cases:
        run_case(case_name, te_kw)

    if args.p0_bisect:
        completed = {float(bp) for bp in args.te_grid}

        def sigma_for_budget(bp: float) -> float:
            if float(bp) not in completed:
                run_case(f"mad@{bp:g}bp", dict(te_bp=bp, te_mode="mad"))
                completed.add(float(bp))
            rep = _representative_sigma_row(rows, float(bp))
            return math.nan if rep is None else float(rep["te_sigma_bp"])

        build_bisection_budgets(
            initial_budgets=list(args.te_grid),
            sigma_at_budget=sigma_for_budget,
            sigma_target_bp=args.sigma_target_bp,
            max_steps=8,
            stop_width_bp=0.1,
        )

    if not args.skip_sigma:
        run_case(f"sigma@{BLOOMBERG_34100_TE_BP:g}bp",
                 dict(te_bp=BLOOMBERG_34100_TE_BP, te_mode="sigma"))

    selected = None
    if args.p0_candidate_grid:
        selected = select_recalibration_candidate(
            rows, sigma_target_bp=args.sigma_target_bp)
        if selected is not None:
            print(f"\n    P0 selected {selected['case']} / "
                  f"{selected['solver']} by recomputed objective")
        else:
            print("\n    P0 selected no feasible MAD candidate")

    if selected is not None and args.validation_scen_seeds:
        val_sigmas: list[float] = []
        w_sel = np.asarray(selected["_weights"], dtype=np.float64)
        for seed in args.validation_scen_seeds:
            Rv, _, _ = credit_scenarios(
                uni, n_scen=args.n_scenarios, seed=seed,
                use_gpu=not args.cpu_only, return_mode="excess")
            te_v = tracking_error_bp(uni, w_sel, Rv)
            val_sigmas.append(te_v["te_sigma_bp"])
            vrow = _base_result_row(
                f"validation@seed={seed}", "sigma-validation",
                args.sigma_target_bp, selected["solver"])
            vrow.update({
                "status": "VALIDATION_ONLY",
                "status_source": "author-classification",
                "validation_seed": seed,
                "te_sigma_bp": te_v["te_sigma_bp"],
                "te_mad_implied_bp": te_v["te_mad_implied_bp"],
                "te_breach_bp": te_v["te_sigma_bp"] - args.sigma_target_bp,
                "note": f"post-selection scoring for {selected['case']}",
            })
            rows.append(vrow)
        srow = _base_result_row(
            "validation-summary", "sigma-validation",
            args.sigma_target_bp, selected["solver"])
        srow.update({
            "status": "VALIDATION_SUMMARY",
            "status_source": "author-classification",
            "validation_summary_stat": "mean/median/max",
            "te_sigma_bp": float(np.mean(val_sigmas)),
            "te_mad_implied_bp": float(np.median(val_sigmas)),
            "te_breach_bp": float(max(val_sigmas) - args.sigma_target_bp),
            "note": (
                f"sigma mean={np.mean(val_sigmas):.3f}bp; "
                f"median={np.median(val_sigmas):.3f}bp; "
                f"max={max(val_sigmas):.3f}bp"),
        })
        rows.append(srow)

    public_rows = [
        {k: v for k, v in row.items() if not k.startswith("_")}
        for row in rows
    ]
    df = pd.DataFrame(public_rows)
    print("\n[5] Result")
    print(df[["case", "solver", "status", "wall_s", "te_sigma_bp",
              "te_mad_implied_bp", "te_breach_bp", "worst_rel_violation",
              "verdict", "te_breach_desk_multiple", "gate_verdict",
              "respects_mandate_desk"]]
          .to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    # Where the two readings part company. On the committed M6 draw the
    # cuopt@1e-6 / mad@5bp row breached by 2.3e-6 of scale — 1.1e-5 bp of
    # annualized tracking error, four decimal places below the 0.1bp the
    # number is even quoted at.
    split = df[df["respects_mandate"] != df["respects_mandate_desk"]]
    if len(split):
        print(f"\n    {len(split)} row(s) where the solver-tolerance verdict "
              f"and the desk verdict disagree:")
        print(split[["case", "solver", "te_breach_bp", "worst_rel_violation",
                     "verdict", "respects_mandate_desk"]]
              .to_string(index=False, float_format=lambda v: f"{v:,.3g}"))

    # The headline: cheapest configuration that actually respects the mandate
    # at the Bloomberg level, if any.
    at_bbg = df[(df["te_budget_bp"] == BLOOMBERG_34100_TE_BP)
                & df["respects_mandate"]]
    print(f"\n    At the Bloomberg 34100 budget "
          f"({BLOOMBERG_34100_TE_BP:.0f}bp):")
    if len(at_bbg):
        best = at_bbg.loc[at_bbg["wall_s"].idxmin()]
        print(f"      fastest config that respects it: {best['solver']} "
              f"({best['case']}) at {best['wall_s']:.2f}s")
    else:
        print("      NO configuration respected it.")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        env = capture_environment()
        env["script"] = "scripts/run_te_mandate.py (M6 TE-mandate experiment)"
        env["n_bonds"] = str(args.n_bonds)
        env["n_scenarios"] = str(args.n_scenarios)
        env["alpha"] = str(args.alpha)
        env["risk_aversion"] = str(args.risk_aversion)
        env["turnover"] = str(args.turnover)
        env["universe_seed"] = str(args.seed)
        env["scenario_seed"] = str(args.scen_seed)
        env["runs_per_config"] = str(args.runs)
        env["te_grid_bp"] = ",".join(f"{v:g}" for v in args.te_grid)
        env["te_budget_source"] = ("Bloomberg optimized fixed income index "
                                   "methodology 34100: ex-ante TE <= 10bp")
        env["periods_per_year"] = "12"
        env["mandate_rel_tol"] = str(args.mandate_rel_tol)
        env["te_desk_tol_bp"] = f"{TE_DESK_TOL_BP:g}"
        env["time_limit_s"] = str(args.time_limit)
        env["incumbent_te_sigma_bp"] = f"{te_w0['te_sigma_bp']:.3f}"
        env["unconstrained_te_sigma_bp"] = f"{te_ref['te_sigma_bp']:.3f}"
        env["scenario_device"] = info["device"]
        write_csv(args.out, df.to_dict("records"), env)
        print(f"\nWrote {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

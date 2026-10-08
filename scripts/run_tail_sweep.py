#!/usr/bin/env python
"""Is the M6 under-delivery a fat-tail assumption, or a mechanism? (M9)

M6's headline is a point estimate taken under one tail assumption. The
scenario engine draws its systematic factors from a Student-t with
`t_dof = 5.0` (see `CreditScenarioConfig`), and a referee gets to say, in one
line: *you assumed fat tails and then discovered fat tails.* The finding as
committed cannot answer that, because a point cannot.

What is actually being claimed is arithmetic, not empirics. The LP surrogate
constrains `sqrt(pi/2) * mean|A_c|`; the mandate constrains `sd(A_c)`. Those
two functionals coincide exactly on a Gaussian and diverge as the distribution
of active return `A` fattens, so the surrogate's under-delivery must vanish in
the Gaussian limit and grow with the tail. This script measures the whole
curve instead of one point on it.

Three design decisions, each of which could reasonably have gone otherwise.

**1. The curve is indexed by a measurable, not by the model parameter.**
`t_dof` is a knob inside this repo's generator. Nobody can measure it on a
real book, so a curve drawn against it can never be joined to real data.
Excess kurtosis of the realized active-return series can be measured on any
fund's tracking-difference history, and it is what the empirical literature
reports. Every grid point therefore records BOTH the input parameters AND the
realized sample moments of the active-return series, and the x axis of the
figure is the measured kurtosis. A consequence worth stating plainly: because
the axis is measured rather than assumed, a grid point whose *nominal*
parameter is mis-calibrated is not a corrupted observation - it simply sits
somewhere else along the axis.

**2. Two knobs, because in this model they are not interchangeable.** The
active return `A = R (w - b)` is close to systematic-factor-neutral by
construction: `w` and `b` load on the rate and credit factors similarly, so
those loadings largely cancel and what survives in `A` is idiosyncratic. That
makes the Student-t knob, which fattens the SYSTEMATIC factors, weak on the
active axis - and it makes issuer-level DEFAULT JUMPS, which are idiosyncratic
and do not cancel, the dominant active-return tail. So the sweep runs:

  - `t-sweep-nojump` : `pd = 0`, `t_dof` swept. The only tail source is the
    Student-t, so this arm is where the Gaussian limit can be reached and the
    harness validated.
  - `t-sweep-repo`   : the generator exactly as the paper uses it (defaults
    on), `t_dof` swept. This is the arm that answers the referee directly.
  - `jump-sweep`     : `t_dof` at the repo value, default probabilities scaled
    by a multiplier. This is the knob that actually moves measured active
    kurtosis in this model.

If overshoot is a function of measured kurtosis rather than of *how* the
kurtosis was produced, all three arms fall on one curve. That is a checkable
prediction, and the figure is drawn so a reader can check it.

**3. A Gaussian control that must return 1.0, run at several scenario
counts.** `t_dof -> inf` with defaults off makes every return a linear-plus-
quadratic function of Gaussian shocks, so the sigma / MAD-implied ratio of a
FIXED portfolio is 1.0 up to sampling error. The OPTIMIZED portfolio's ratio
is not: the optimizer chooses `w` in an N-dimensional space against one
scenario draw, so it can pick up sample-specific ratio above 1 for free. That
residual is in-sample selection, not a mechanism, and the only honest way to
show which is which is to vary S and watch it shrink. Both numbers are
reported per row - `sigma_over_mad_ref` (incumbent book, no selection) and
the optimized overshoot - so the selection component is visible rather than
argued.

CPU only, by construction: the MAD form makes HiGHS simplex pathological (the
committed M6 run timed it out at 300s on 500x10000, and a 300x3000 instance
ran 13 minutes without finishing), so the sweep uses Clarabel. HiGHS interior
point agrees with it to five significant figures on a 150x4000 instance and is
available via --cross-check-solver as a per-run check rather than an
assertion.

Examples:
    # the committed configuration (about an hour on 24 CPU threads)
    python scripts/run_tail_sweep.py --out results/m9_tail_sweep.csv

    # fast smoke test
    python scripts/run_tail_sweep.py --n-bonds 80 --n-scenarios 800 \
        --dof-grid 5 1e6 --pd-mult-grid --te-grid 10 --replicates 1 \
        --control-scen-grid 800 --out /tmp/pilot.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.bench import capture_environment, write_csv  # noqa: E402
from src.check import check_solution  # noqa: E402
from src.credit import (  # noqa: E402
    credit_constraint_builder, default_limits, exposure_report, gate_verdicts,
    tracking_error_bp, worst_violation,
)
from src.model import build_mean_cvar  # noqa: E402
from src.scenarios import CreditScenarioConfig, credit_scenarios  # noqa: E402
from src.universe import make_universe  # noqa: E402

#: Solver configs. Clarabel is the default for the reason in the docstring;
#: HiGHS interior point is the cross-check. HiGHS SIMPLEX is deliberately
#: absent - on the MAD form it does not finish at any size this sweep would
#: use, and a time-limited simplex iterate is not a portfolio.
SOLVERS = {
    "clarabel": dict(kind="clarabel"),
    "highs-ipm": dict(kind="highs-ipm"),
}

#: Degrees of freedom for the Student-t factors. The generator rescales by
#: sqrt((dof - 2) / dof) for unit variance, so dof must exceed 2; 2.5 is the
#: fattest point taken and is already far outside anything a credit desk would
#: fit. 1e6 is the effectively-Gaussian end: the t's excess kurtosis is
#: 6 / (dof - 4), i.e. 6e-6 there, which no sample of this size can resolve.
#: Note that dof <= 4 has INFINITE population kurtosis - the sample kurtosis
#: recorded at those points is a finite random number, which is precisely what
#: a real data series would also give you, and is flagged in the CSV.
DEFAULT_DOF_GRID = (2.5, 3.0, 4.0, 5.0, 7.0, 12.0, 30.0, 1e6)

#: Multipliers on every rating's annual default probability, run at the repo's
#: own t_dof. 1.0 (the shipped PDs) and 0.0 (no defaults) are already covered
#: by the two t-sweep arms, so only the interior points are listed here.
DEFAULT_PD_MULT_GRID = (0.01, 0.03, 0.1, 0.3)

#: Annualized TE budgets in bp. Both ends of the committed M6 grid: 5bp is
#: where M6 reports its largest overshoot, 50bp its smallest.
DEFAULT_TE_GRID = (5.0, 50.0)

#: Scenario counts for the Gaussian control. The point of the arm is the
#: TREND, so the grid has to span a factor of several.
DEFAULT_CONTROL_SCEN_GRID = (1000, 2000, 4000, 8000)

#: A budget counts as binding only if the constrained quantity landed within
#: this relative distance of it. Same rule and same threshold as
#: scripts/plot_te_mandate.py: far above any solver tolerance, far below the
#: slack a genuinely non-binding budget shows.
BINDING_REL_SLACK = 0.05

#: Fraction of the sample used by the Hill tail-index estimate. 5% is the
#: conventional starting point; see `hill_left` for why the number is reported
#: as a diagnostic and not as a headline.
HILL_TAIL_FRACTION = 0.05

#: t_dof at which the t's own population kurtosis stops existing.
FINITE_KURTOSIS_DOF = 4.0

#: The repo's own operating point, marked on the figure and cross-checked.
REPO_DOF = 5.0

RATINGS = ("AAA", "AA", "A", "BBB", "BB", "B")


@dataclass(frozen=True)
class TailPoint:
    """One scenario-generator configuration to be measured."""

    arm: str
    label: str
    t_dof: float
    pd_mult: float
    n_scen: int

    def config(self) -> CreditScenarioConfig:
        base = CreditScenarioConfig().pd_annual
        return CreditScenarioConfig(
            t_dof=self.t_dof,
            pd_annual={r: base[r] * self.pd_mult for r in RATINGS},
        )


def hill_left(a: np.ndarray, tail_fraction: float = HILL_TAIL_FRACTION
              ) -> tuple[float, int]:
    """Hill tail index of the LEFT tail of `a`, and the k it used.

    For a Student-t with `dof` degrees of freedom the tail index equals
    `dof`, so on the no-jump arm this is a second, independent read on whether
    the generator produced the tail that was asked for.

    Reported as a diagnostic, never as a headline, for two reasons a referee
    would raise first. The estimate is famously sensitive to `k` - there is no
    consensus rule and this one is a convention, not a choice defended here.
    And on the arms with defaults switched on, the left tail is not a power
    law at all: a default caps the loss at `(recovery - price) / price`, so
    what the estimator sees is a bounded jump wearing a Pareto costume and the
    number it returns describes the estimator, not the distribution.

    Returns (nan, 0) when the tail is too short or degenerate to estimate.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    losses = -(a - a.mean())
    losses = np.sort(losses[losses > 0.0])[::-1]     # largest loss first
    k = int(round(tail_fraction * a.size))
    if k < 10 or losses.size < k + 1 or losses[k] <= 0.0:
        return float("nan"), 0
    s = float(np.log(losses[:k] / losses[k]).mean())
    if not np.isfinite(s) or s <= 0.0:
        return float("nan"), k
    return 1.0 / s, k


def series_moments(a: np.ndarray, prefix: str) -> dict[str, float]:
    """Sample moments of one return series, prefixed for a flat CSV row.

    Excess kurtosis is the plain moment ratio (`m4 / m2^2 - 3`), not the
    small-sample-corrected estimator: the correction assumes normality to
    derive itself, which is the hypothesis under test here, and at S >= 1000
    the two differ by less than the seed-to-seed spread anyway.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    c = a - a.mean()
    m2 = float((c ** 2).mean())
    if m2 <= 0.0:
        nan = float("nan")
        return {f"{prefix}_sd": 0.0, f"{prefix}_skew": nan,
                f"{prefix}_exkurt": nan, f"{prefix}_hill_alpha": nan,
                f"{prefix}_hill_k": 0, f"{prefix}_min": 0.0}
    alpha, k = hill_left(a)
    return {
        f"{prefix}_sd": float(np.sqrt(m2)),
        f"{prefix}_skew": float((c ** 3).mean() / m2 ** 1.5),
        f"{prefix}_exkurt": float((c ** 4).mean() / m2 ** 2 - 3.0),
        f"{prefix}_hill_alpha": alpha,
        f"{prefix}_hill_k": k,
        f"{prefix}_min": float(a.min()),
    }


def solve_once(model, solver_name: str, time_limit: float
               ) -> tuple[float, str, float]:
    """One solve; returns (wall seconds, status, objective).

    Deliberately NOT the median of repeats that `run_te_mandate.solve_once`
    takes. That script is a timing benchmark and its numbers are the point;
    here the timing is incidental and the portfolio is the point, so paying
    3x the wall clock to stabilise a column nothing is concluded from would be
    a poor trade against the number of grid points this sweep needs.
    """
    import cvxpy as cp

    kind = SOLVERS[solver_name]["kind"]
    if kind == "clarabel":
        solver, opts = cp.CLARABEL, {"time_limit": time_limit}
    elif kind == "highs-ipm":
        solver = cp.HIGHS
        opts = {"highs_options": {"solver": "ipm", "run_crossover": "off",
                                  "time_limit": time_limit,
                                  "threads": 8, "random_seed": 0}}
    else:
        raise ValueError(kind)
    t0 = time.perf_counter()
    model.problem.solve(solver=solver, verbose=False, warm_start=False,
                        **opts)
    secs = time.perf_counter() - t0
    value = (float(model.problem.value)
             if model.problem.value is not None else float("nan"))
    return secs, str(model.problem.status), value


#: Statuses whose weight vector is a converged answer. Copied from
#: scripts/run_te_mandate.py: from anything else the recomputed tracking error
#: is a number with a unit and no meaning.
CONVERGED = ("optimal", "optimal_inaccurate")


def _blank_row() -> dict[str, object]:
    """Every column a row can carry, all empty.

    `src.bench.write_csv` takes its fieldnames from the FIRST row, so a row
    that fails early must still carry the full key set or the CSV silently
    loses columns for every later row.
    """
    nan = float("nan")
    row: dict[str, object] = {
        "arm": "", "point": "", "t_dof": nan, "pd_mult": nan,
        "n_scenarios": 0, "scen_seed": 0, "replicate": 0,
        "te_budget_bp": nan, "solver": "", "status": "", "wall_s": nan,
        "te_sigma_bp": nan, "te_mad_implied_bp": nan, "te_breach_bp": nan,
        "overshoot_x": nan, "sigma_over_mad_opt": nan, "binding": False,
        "objective": nan, "check_ok": False,
        "worst_rel_violation": nan, "worst_limit": "-",
        "purity_pass": False, "desk_pass": False, "gate_verdict": "",
        "realized_default_rate": nan, "target_default_rate": nan,
        "t_kurtosis_finite": False, "note": "",
    }
    for prefix in ("ref_active", "opt_active", "bench_ret"):
        row.update({f"{prefix}_sd": nan, f"{prefix}_skew": nan,
                    f"{prefix}_exkurt": nan, f"{prefix}_hill_alpha": nan,
                    f"{prefix}_hill_k": 0, f"{prefix}_min": nan})
    row["sigma_over_mad_ref"] = nan
    row["te_sigma_ref_bp"] = nan
    return row


def build_points(args) -> list[TailPoint]:
    """The full grid, in run order (cheapest arm first so failures show up
    early rather than an hour in)."""
    pts: list[TailPoint] = []
    for dof in args.dof_grid:
        pts.append(TailPoint("t-sweep-nojump", f"dof={dof:g},pd=0", dof, 0.0,
                             args.n_scenarios))
    for dof in args.dof_grid:
        pts.append(TailPoint("t-sweep-repo", f"dof={dof:g},pd=1", dof, 1.0,
                             args.n_scenarios))
    for m in args.pd_mult_grid:
        pts.append(TailPoint("jump-sweep", f"dof={REPO_DOF:g},pd={m:g}",
                             REPO_DOF, m, args.n_scenarios))
    return pts


def measure_point(uni, pt: TailPoint, seed: int, rep: int, args,
                  solver_name: str) -> list[dict[str, object]]:
    """Generate one scenario set and solve every budget on it."""
    R, mu, info = credit_scenarios(uni, n_scen=pt.n_scen, seed=seed,
                                   cfg=pt.config(), use_gpu=False,
                                   return_mode="excess")
    # The reference series: the INCUMBENT book's active returns. Chosen over
    # the optimized portfolio's because it does not depend on which budget is
    # being solved, so it is a property of the scenario set alone - the same
    # thing a desk can measure on its own tracking difference before any
    # optimizer runs. The optimized portfolio's own moments are recorded too,
    # per budget, because the optimizer manufactures tail shape and that is
    # part of the mechanism rather than a nuisance.
    ref_active = R @ (uni.w0 - uni.w_benchmark)
    bench_ret = R @ uni.w_benchmark
    te_ref = tracking_error_bp(uni, uni.w0, R, args.periods_per_year)
    common = _blank_row()
    common.update({
        "arm": pt.arm, "point": pt.label, "t_dof": pt.t_dof,
        "pd_mult": pt.pd_mult, "n_scenarios": pt.n_scen, "scen_seed": seed,
        "replicate": rep, "solver": solver_name,
        "realized_default_rate": info["realized_default_rate_per_bond"],
        "target_default_rate": info["target_default_rate_per_bond"],
        "t_kurtosis_finite": bool(pt.t_dof > FINITE_KURTOSIS_DOF),
        "sigma_over_mad_ref": te_ref["sigma_over_mad_implied"],
        "te_sigma_ref_bp": te_ref["te_sigma_bp"],
    })
    common.update(series_moments(ref_active, "ref_active"))
    common.update(series_moments(bench_ret, "bench_ret"))

    rows: list[dict[str, object]] = []
    for bp in args.te_grid:
        row = dict(common)
        row["te_budget_bp"] = bp
        limits = default_limits(uni, turnover=args.turnover, te_bp=bp,
                                te_mode="mad",
                                periods_per_year=args.periods_per_year)
        try:
            model = build_mean_cvar(
                R, mu, alpha=args.alpha, risk_aversion=args.risk_aversion,
                w_max=limits.w_max,
                constraint_builder=credit_constraint_builder(uni, limits,
                                                             R=R))
            secs, status, obj = solve_once(model, solver_name,
                                           args.time_limit)
            row.update({"wall_s": secs, "status": status, "objective": obj})
            if status not in CONVERGED or model.w.value is None:
                row["note"] = "unconverged; no portfolio to score"
                rows.append(row)
                print(f"    {pt.label:<18} bud={bp:<5g} {status:<12} "
                      f"{secs:6.1f}s  UNCONVERGED")
                continue
            w = model.weights()
            te = tracking_error_bp(uni, w, R, args.periods_per_year)
            opt_active = R @ (w - uni.w_benchmark)
            chk = check_solution(w=w, solver_objective=obj, R=R, mu=mu,
                                 alpha=args.alpha,
                                 risk_aversion=args.risk_aversion,
                                 w_max=limits.w_max)
            rep_df = exposure_report(uni, w, limits, R=R)
            viol, which = worst_violation(rep_df)
            gates = gate_verdicts(rep_df)
            slack_rel = 1.0 - te["te_mad_implied_bp"] / bp
            row.update(series_moments(opt_active, "opt_active"))
            row.update({
                "te_sigma_bp": te["te_sigma_bp"],
                "te_mad_implied_bp": te["te_mad_implied_bp"],
                "te_breach_bp": te["te_mad_implied_bp"] - bp,
                "overshoot_x": te["te_sigma_bp"] / bp,
                "sigma_over_mad_opt": te["sigma_over_mad_implied"],
                "binding": bool(slack_rel <= BINDING_REL_SLACK),
                "check_ok": bool(chk.ok),
                "worst_rel_violation": viol,
                "worst_limit": which if viol > 0 else "-",
                "purity_pass": gates.purity_pass,
                "desk_pass": gates.desk_pass,
                "gate_verdict": gates.verdict,
            })
            print(f"    {pt.label:<18} bud={bp:<5g} {status:<12} "
                  f"{secs:6.1f}s  exk_ref="
                  f"{row['ref_active_exkurt']:8.3f}  over="
                  f"{row['overshoot_x']:6.3f}x  "
                  f"{'bind' if row['binding'] else 'SLACK'}")
        except Exception as e:                                  # noqa: BLE001
            row["status"] = f"ERROR:{type(e).__name__}"
            row["note"] = str(e)[:200]
            print(f"    {pt.label:<18} bud={bp:<5g} FAILED "
                  f"{type(e).__name__}: {str(e)[:80]}")
        rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Sweep the scenario generator's tail parameters and "
                    "measure the MAD surrogate's under-delivery as a "
                    "function of MEASURED kurtosis (M9). CPU only.")
    ap.add_argument("--n-bonds", type=int, default=120)
    ap.add_argument("--n-scenarios", type=int, default=4000)
    ap.add_argument("--alpha", type=float, default=0.95)
    ap.add_argument("--risk-aversion", type=float, default=3.0)
    ap.add_argument("--turnover", type=float, default=0.25)
    ap.add_argument("--periods-per-year", type=float, default=12.0)
    ap.add_argument("--seed", type=int, default=7,
                    help="universe seed; held fixed across the whole sweep "
                         "so the only thing varying is the tail")
    ap.add_argument("--scen-seed", type=int, default=11,
                    help="first scenario seed; replicates use consecutive "
                         "values from here")
    ap.add_argument("--replicates", type=int, default=3,
                    help="scenario draws per grid point. The default-jump "
                         "tail is a rare-event statistic and one draw of it "
                         "is not a measurement")
    ap.add_argument("--dof-grid", type=float, nargs="*",
                    default=list(DEFAULT_DOF_GRID))
    ap.add_argument("--pd-mult-grid", type=float, nargs="*",
                    default=list(DEFAULT_PD_MULT_GRID))
    ap.add_argument("--te-grid", type=float, nargs="*",
                    default=list(DEFAULT_TE_GRID))
    ap.add_argument("--control-scen-grid", type=int, nargs="*",
                    default=list(DEFAULT_CONTROL_SCEN_GRID),
                    help="scenario counts for the Gaussian control arm; the "
                         "residual overshoot there is in-sample selection and "
                         "has to be shown shrinking, not asserted")
    ap.add_argument("--control-budget-bp", type=float, default=10.0)
    ap.add_argument("--solver", default="clarabel", choices=sorted(SOLVERS))
    ap.add_argument("--cross-check-solver", default="highs-ipm",
                    choices=sorted(SOLVERS) + [""],
                    help="re-solve the repo operating point with a second "
                         "solver; '' to skip")
    ap.add_argument("--time-limit", type=float, default=1200.0)
    ap.add_argument("--out", default="results/m9_tail_sweep.csv")
    args = ap.parse_args()

    if args.replicates < 1:
        ap.error("--replicates must be at least 1")

    pd.set_option("display.width", 170)
    pd.set_option("display.max_columns", 30)

    print(f"[1] Universe: {args.n_bonds} bonds (seed {args.seed})")
    uni = make_universe(n_bonds=args.n_bonds, seed=args.seed)

    points = build_points(args)
    seeds = [args.scen_seed + i for i in range(args.replicates)]
    n_solves = len(points) * len(seeds) * len(args.te_grid)
    print(f"[2] Grid: {len(points)} scenario configurations x "
          f"{len(seeds)} draws x {len(args.te_grid)} budgets "
          f"= {n_solves} solves, {args.solver}, "
          f"{args.n_scenarios:,} scenarios each")

    rows: list[dict[str, object]] = []
    t_start = time.perf_counter()
    for i, pt in enumerate(points, 1):
        print(f"  [{i}/{len(points)}] {pt.arm} {pt.label}")
        for rep, seed in enumerate(seeds):
            rows += measure_point(uni, pt, seed, rep, args, args.solver)

    # Cross-check: same instance, different algorithm family. The point is
    # not a second measurement, it is evidence that the first one is not a
    # property of Clarabel.
    if args.cross_check_solver:
        print(f"  cross-check with {args.cross_check_solver} at the repo "
              f"operating point")
        xpt = TailPoint("cross-check", f"dof={REPO_DOF:g},pd=1", REPO_DOF,
                        1.0, args.n_scenarios)
        rows += measure_point(uni, xpt, seeds[0], 0, args,
                              args.cross_check_solver)

    # Gaussian control. Runs last and at one budget: its job is the trend in
    # S, not another point on the kurtosis curve.
    print("  Gaussian control (t_dof=1e6, no defaults) vs scenario count")
    ctrl_args = argparse.Namespace(**vars(args))
    ctrl_args.te_grid = [args.control_budget_bp]
    for n_s in args.control_scen_grid:
        cpt = TailPoint("gaussian-control", f"dof=1e6,pd=0,S={n_s}", 1e6,
                        0.0, n_s)
        rows += measure_point(uni, cpt, seeds[0], 0, ctrl_args, args.solver)

    elapsed = time.perf_counter() - t_start
    df = pd.DataFrame(rows)
    print(f"\n[3] {len(df)} rows in {elapsed / 60:.1f} min")

    ok = df[df["status"].isin(CONVERGED) & df["binding"].astype(bool)]
    print("\n[4] The curve: median over replicates, binding rows only")
    if ok.empty:
        print("    NO binding converged rows - nothing to report")
    else:
        summary = (ok.groupby(["arm", "point", "t_dof", "pd_mult",
                               "n_scenarios", "te_budget_bp"])
                   .agg(n=("overshoot_x", "size"),
                        exkurt_ref=("ref_active_exkurt", "median"),
                        exkurt_opt=("opt_active_exkurt", "median"),
                        hill_ref=("ref_active_hill_alpha", "median"),
                        ratio_ref=("sigma_over_mad_ref", "median"),
                        overshoot=("overshoot_x", "median"),
                        overshoot_lo=("overshoot_x", "min"),
                        overshoot_hi=("overshoot_x", "max"),
                        defrate=("realized_default_rate", "median"))
                   .reset_index()
                   .sort_values(["te_budget_bp", "arm", "exkurt_ref"]))
        print(summary.to_string(index=False,
                                float_format=lambda v: f"{v:,.4g}"))

        ctrl = summary[summary["arm"].eq("gaussian-control")]
        print("\n[5] Harness validation: the Gaussian control must go to 1.0")
        if ctrl.empty:
            print("    control arm produced no binding rows - NOT validated")
        else:
            for _, r in ctrl.sort_values("n_scenarios").iterrows():
                print(f"    S={int(r['n_scenarios']):6,}  measured excess "
                      f"kurtosis {r['exkurt_ref']:+.3f}  "
                      f"incumbent sigma/MAD {r['ratio_ref']:.4f}  "
                      f"optimized overshoot {r['overshoot']:.4f}x")
            last = ctrl.sort_values("n_scenarios").iloc[-1]
            print(f"    -> at the largest S the fixed-portfolio ratio is "
                  f"{last['ratio_ref']:.4f} (exact answer 1.0) and the "
                  f"optimized overshoot is {last['overshoot']:.4f}x; the gap "
                  f"between them is in-sample selection over "
                  f"{args.n_bonds} weights.")

    non_binding = df[df["status"].isin(CONVERGED) & ~df["binding"]]
    if len(non_binding):
        print(f"\n    {len(non_binding)} converged row(s) where the budget "
              f"did NOT bind and which are therefore excluded from the "
              f"curve (the point measures some other limit):")
        print(non_binding[["arm", "point", "te_budget_bp",
                           "te_mad_implied_bp"]]
              .to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    bad = df[~df["status"].isin(CONVERGED)]
    if len(bad):
        print(f"\n    {len(bad)} row(s) did not converge:")
        print(bad[["arm", "point", "te_budget_bp", "status", "note"]]
              .to_string(index=False))
    failed_check = df[df["status"].isin(CONVERGED) & ~df["check_ok"]]
    if len(failed_check):
        print(f"\n    {len(failed_check)} row(s) failed the independent "
              f"solution check - treat their numbers as unverified.")

    env = capture_environment()
    env["script"] = "scripts/run_tail_sweep.py (M9 tail dose-response)"
    env["n_bonds"] = str(args.n_bonds)
    env["n_scenarios"] = str(args.n_scenarios)
    env["alpha"] = str(args.alpha)
    env["risk_aversion"] = str(args.risk_aversion)
    env["turnover"] = str(args.turnover)
    env["periods_per_year"] = str(args.periods_per_year)
    env["universe_seed"] = str(args.seed)
    env["scenario_seeds"] = ",".join(str(s) for s in seeds)
    env["dof_grid"] = ",".join(f"{v:g}" for v in args.dof_grid)
    env["pd_mult_grid"] = ",".join(f"{v:g}" for v in args.pd_mult_grid)
    env["te_grid_bp"] = ",".join(f"{v:g}" for v in args.te_grid)
    env["control_scen_grid"] = ",".join(str(v)
                                        for v in args.control_scen_grid)
    env["control_budget_bp"] = f"{args.control_budget_bp:g}"
    env["solver"] = args.solver
    env["cross_check_solver"] = args.cross_check_solver or "none"
    env["binding_rel_slack"] = str(BINDING_REL_SLACK)
    env["hill_tail_fraction"] = str(HILL_TAIL_FRACTION)
    env["repo_operating_dof"] = f"{REPO_DOF:g}"
    env["scenario_device"] = "CPU (NumPy)"
    env["kurtosis_axis"] = ("ref_active_exkurt: sample excess kurtosis of "
                            "the INCUMBENT book's active returns, R @ (w0 - "
                            "benchmark), measured on the same scenario draw "
                            "the solve used")
    env["literature_kurtosis"] = ("not encoded here on purpose; see the M9 "
                                  "notes for what the empirical literature "
                                  "reports and how heterogeneous it is")
    env["wall_minutes"] = f"{elapsed / 60:.1f}"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.out, df.to_dict("records"), env)
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Is "loose iterate OR pay for crossover" a real dichotomy? (M8)

The paper currently frames the practitioner's choice as binary: sign the fast
but slightly-infeasible interior point PDLP returns at 1e-4, or pay for
crossover and get a mandate-feasible basic solution that costs more than the CPU
simplex it was supposed to beat. A credit PM on the review panel rejected the
frame: no desk recognises it, because there is an obvious third option.

Every mandate limit in this problem except the tracking-error families lives
in WEIGHT space -- rating bands, sector caps, issuer caps, cell and total
duration matching, the two-sided DTS band, turnover, and the ||w - b||_1
active budget. None of them touches the S x N scenario block. So repairing a
loose point should be a projection onto an N-variable polytope, thousands of
times smaller than the LP that produced the point. This script measures
whether that is true, what it costs in objective, and -- the part that decides
whether it is honest -- what it CANNOT fix.

---

**(a) Which norm.** Both, reported side by side, because the choice is
economic rather than cosmetic and the contrast is the result.

- ``l2`` -- Euclidean projection. A QP. Spreads a small correction across
  every holding, so it moves almost no single name by a reportable amount and
  densifies the book.
- ``l1`` -- an LP. Concentrates the correction in few names, which is what a
  desk prefers to execute, and keeps the position count near the loose point's.
  It is also the norm the mandate itself already speaks: turnover is
  ``||w - w0||_1`` and the active budget is ``||w - b||_1``, so an L1 repair
  is measured in the same unit as two of the constraints it is restoring.

Neither is asserted to be better. The run reports L1 distance moved, names
moved by more than one basis point of weight, and position count before and
after, so the reader can price the difference.

**(b) Whether to preserve the objective.** Measured, not assumed, in two
further modes -- and the answer is no.

- ``l1_mu_lex`` -- lexicographic: take the minimum L1 distance ``d1``, then
  maximise ``mu'w`` inside that same ball. Still weight-space, still an LP,
  still cheap. It is the only objective-aware repair that keeps the cost
  advantage, because ``mu'w`` is the only part of the mean-CVaR objective that
  does not need the scenario block. On every CPU case run so far it made the
  TRUE objective WORSE than the plain L1 projection, by a factor of two to
  nine: chasing expected return inside the ball buys carry with tail risk that
  the weight-space surrogate cannot see. An objective-aware repair that can
  only see half the objective is worse than one that sees none of it.
- ``l1_ball_full`` -- the honest objective-aware repair: maximise the real
  ``mu'w - lambda * CVaR_alpha(w)`` over the full mandate inside the same L1
  ball. It recovers most of the objective, and it costs a scenario-block LP,
  i.e. the thing the repair existed to avoid. It can also come back
  INFEASIBLE, and does whenever a constraint the ball cannot see is violated
  at the ball's centre -- the minimum-L1 ball is by construction the tightest
  one that restores the weight-space limits, so it has no room left to fix a
  tail family. Both outcomes are measured numbers in the CSV rather than
  claims in a paragraph.

**(c) The tail families, and what the repair cannot restore.** The ex-ante
tracking-error families (``te_mode="sigma"``, a second-order cone, and
``te_mode="mad"``, its LP analogue) are defined on active RETURNS, so they
live in the S x N block and the projection is blind to them. Two consequences
this script refuses to paper over:

1. If a TE limit is binding, the projection cannot restore it -- and worse, it
   can BREAK one the loose point satisfied, because nothing in the small
   problem knows the constraint exists.
2. Therefore every repaired portfolio is re-scored against the FULL mandate
   with ``src.credit.exposure_report`` / ``gate_verdicts``, including the TE
   rows the repair never saw, plus the budget and box residuals that the
   exposure report does not cover (those come from ``src.check``). The CSV
   reports what the repair DELIVERED, not what it targeted, and carries
   ``families_repaired`` / ``families_invisible`` on every row so the two are
   never confused.

The CVaR objective is likewise invisible to the projection, but CVaR is an
objective term here rather than a limit, so it shows up as objective
degradation, not as a breach. Degradation is quoted against both the crossover/
basic solution and the exact simplex reference, so "the repair gives back more return
than crossover does" is a measurement in the table either way it falls.

**What the reported repair time is, exactly.** ``wall_s`` on a repair row is
CVXPY canonicalization plus the projection solve, median of ``--repair-runs``
timed repeats after a discarded warmup, with ``build_s`` broken out because a
desk pays canonicalization once and then re-solves against it. It excludes the
loose solve itself, which has its own row, and it excludes scoring the loose
point -- a desk repairing unconditionally never scores first, and scoring costs
one S x N matvec anyway. The reference rows time a full solve, so ``wall_kind``
says which of the two a row's ``wall_s`` means.

Read the sign of ``d_obj_vs_*_bp`` the way ``src/check.py`` reads its
reference gap: positive means worse than the reference, and NEGATIVE means the
row beat an exact optimum, which no feasible portfolio can do. A negative gap
is infeasibility wearing a good number, and on the sigma cases it appears on
exactly the rows the gate verdict has already called BREACH.

---

**Where the loose point comes from, and what a CPU run can and cannot say.**

``--loose-source ortools-pdlp`` (default) is the strongest CPU option and the
reason this script is worth running without a GPU. OR-Tools PDLP is the
reference CPU implementation of the same primal-dual hybrid gradient method
cuOpt runs on the GPU, so its early-stopped iterate is a genuine PDLP iterate
with genuine PDLP structure: bounds satisfied exactly (PDLP projects onto the
box every step), row constraints violated by an amount that reflects where the
restarted average happened to be when the termination test passed. That is not
a simulation of first-order infeasibility, it is first-order infeasibility.

It still is not cuOpt's. Two differences a reader should hold onto: the GPU
implementation differs in step-size rule, restart schedule and preconditioning,
so the *pattern* of which rows are left slack differs; and OR-Tools runs a GLOP
presolve by default (``presolve_options.use_glop``), which cuOpt does not, so
some structure in the returned point is simplex-derived. What the CPU run
establishes is the mechanism, the cost ratio and the sign of the objective
effect. What only the GPU session can establish is those numbers on cuOpt's
own iterates at production scale.

``--loose-source perturb`` manufactures a loose point instead: it walks the
exact reference solution along a seeded, budget-neutral direction, clipped to the box
(faithful, since PDLP satisfies variable bounds exactly and violates rows), and
bisects the step length until the worst desk-gate breach hits a target
multiple. It is controlled and reproducible, and it is NOT a first-order
iterate: its infeasibility has no algorithmic structure, and it is included as
a contrast to show how much of the result depends on that structure.

``--loose-source npz`` is the GPU-session hook and needs no code change: the
session saves cuOpt's loose weight vector (and, with ``--crossover-npz``, the
crossover/basic solution) and points this script at them with the same universe and
scenario seeds. Everything downstream is identical.

``--loose-source cuopt-pdlp`` is the published P0 path: the script solves the
full LP directly through installed cuOpt PDLP, with the same Stable2, PSLP
presolve and primal/dual/gap tolerances used by the TE benchmark. It emits the
raw 1e-4 and 1e-6 endpoints, repairs the 1e-4 raw endpoint, and measures the
1e-4 crossover/basic endpoint without OR-Tools or a hand-prepared NPZ.

Because the OR-Tools loose point requires libortools' symbols to bind before
highspy's, that solve runs in a subprocess of this same file (see
``--emit-loose-point`` and the note at the top of scripts/bench_worker.py).

Examples:
    # CPU end-to-end, no GPU needed
    python scripts/run_feasibility_repair.py --n-bonds 300 \\
        --n-scenarios 3000 --cpu-only --out results/m8_repair.csv

    # the same experiment with a TE limit the projection cannot see
    python scripts/run_feasibility_repair.py --cpu-only --te-grid 10 \\
        --te-mode mad

    # GPU session: cuOpt's own loose point and crossover/basic solution
    python scripts/run_feasibility_repair.py --loose-source npz \\
        --loose-npz cuopt_loose.npz --crossover-npz cuopt_xover.npz
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import pickle
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# MUST precede every import that reaches cvxpy, which loads highspy, whose
# HiGHS build exports C++ symbols that shadow the ones OR-Tools' bundled
# HiGHS provides -- libortools then dies on an undefined symbol. Loading
# libortools first binds its own symbols correctly and cvxpy simply comes up
# without the HIGHS backend, which the worker never uses. Sniffing argv is
# the only hook early enough; scripts/bench_worker.py solves the same problem
# by deferring all of its imports into main().
if "--emit-loose-point" in sys.argv:
    from ortools.pdlp.python import pdlp  # noqa: F401

import cvxpy as cp  # noqa: E402
import pandas as pd  # noqa: E402

from src.bench import (  # noqa: E402
    assert_gpu_idle, capture_environment, cuopt_pdlp_spec, write_csv,
)
from src.check import check_solution, empirical_cvar  # noqa: E402
from src.credit import (  # noqa: E402
    DESK_TOL_BY_UNIT, UNIT_WEIGHT, credit_constraint_builder, default_limits,
    exposure_report, gate_verdicts, tracking_error_bp,
)
from src.model import build_mean_cvar  # noqa: E402
from src.scenarios import credit_scenarios, warmup_gpu  # noqa: E402
from src.universe import make_universe  # noqa: E402

#: One basis point of portfolio weight -- the granularity a position is
#: reported at, and therefore the threshold at which a repair has actually
#: changed a name by more than a rounding digit.
NAME_MOVE_GATE = DESK_TOL_BY_UNIT[UNIT_WEIGHT]

#: Repair modes, in dependency order. `l1` runs unconditionally because the
#: two objective-aware modes are defined relative to ITS optimal distance --
#: an objective-aware repair inside an arbitrary ball would be answering a
#: different question, and the ball has to be the tightest one that restores
#: feasibility for the comparison to be about the objective alone.
REPAIR_MODES = ("l2", "l1", "l1_mu_lex", "l1_ball_full")

#: Statuses whose weight vector is an answer rather than an arbitrary iterate.
CONVERGED = ("optimal", "optimal_inaccurate")

#: Reference solves. `conic` marks configurations that can accept a
#: second-order cone constraint. On a sigma-TE case the LP-only rows are
#: recorded as UNSUPPORTED_BY_METHOD rather than failed: PDLP and simplex not being
#: able to express an SOC is a fact about the problem, not a measurement.
REFERENCE_CONFIGS = (
    ("highs-simplex", dict(kind="highs-simplex", conic=False,
                           role="reference-exact")),
    ("highs-ipm+xover", dict(kind="highs-xover", conic=False,
                             role="reference-crossover")),
    ("clarabel", dict(kind="clarabel", conic=True, role="reference-conic")),
)


# --------------------------------------------------------------------------
# The loose-point worker (own process; see the argv sniff above)
# --------------------------------------------------------------------------

def emit_loose_point(args: argparse.Namespace) -> int:
    """Solve the full credit program with OR-Tools PDLP and save the iterate.

    Runs as a subprocess of this file. The problem is passed in as files
    rather than regenerated from seeds for the same reason
    scripts/bench_worker.py does it: the parent may have generated its
    scenarios on the GPU, and the two paths do not share a random stream, so a
    regenerated matrix would be a different problem wearing the same seed.
    """
    from src.pdlp_compat import PatchedPDLP     # imports ortools internals
    from ortools.pdlp import solvers_pb2

    d = np.load(args.problem_npz)
    with open(args.universe_pkl, "rb") as f:
        uni, limits = pickle.load(f)
    R, mu = d["R"], d["mu"]

    model = build_mean_cvar(
        R, mu, alpha=float(d["alpha"]),
        risk_aversion=float(d["risk_aversion"]), w_max=limits.w_max,
        constraint_builder=credit_constraint_builder(uni, limits, R=R))

    tol = float(args.loose_tol[0])     # the parent passes exactly one
    params = solvers_pb2.PrimalDualHybridGradientParams()
    params.num_threads = args.threads
    crit = params.termination_criteria.simple_optimality_criteria
    crit.eps_optimal_absolute = tol
    crit.eps_optimal_relative = tol

    # Discarded warmup then one timed solve: the wall time recorded here is
    # the "fast loose solve" the whole dichotomy is about, so it must not
    # carry first-call canonicalization.
    model.problem.solve(solver=PatchedPDLP(), verbose=False,
                        warm_start=False, parameters_proto=params,
                        time_limit_sec=args.time_limit)
    t0 = time.perf_counter()
    model.problem.solve(solver=PatchedPDLP(), verbose=False,
                        warm_start=False, parameters_proto=params,
                        time_limit_sec=args.time_limit)
    wall = time.perf_counter() - t0

    np.savez(args.emit_loose_point, w=model.weights(), wall_s=wall,
             status=str(model.problem.status),
             label=f"ortools-pdlp@{tol:g}",
             config=" ".join(str(params).split()))
    print(f"    worker: {model.problem.status} in {wall:.3f}s")
    return 0


def run_pdlp_worker(uni, limits, R, mu, alpha: float, lam: float,
                    tol: float, threads: int, time_limit: float) -> dict:
    """Spawn the worker and return {w, wall_s, status, label, config}."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        npz, pkl, out = tmp / "prob.npz", tmp / "uni.pkl", tmp / "loose.npz"
        np.savez(npz, R=R, mu=mu, alpha=alpha, risk_aversion=lam)
        with open(pkl, "wb") as f:
            pickle.dump((uni, limits), f)
        cmd = [sys.executable, str(Path(__file__).resolve()),
               "--emit-loose-point", str(out), "--problem-npz", str(npz),
               "--universe-pkl", str(pkl), "--loose-tol", str(tol),
               "--threads", str(threads), "--time-limit", str(time_limit)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not out.exists():
            raise RuntimeError(
                f"PDLP loose-point worker failed (rc={proc.returncode}):\n"
                f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
        d = np.load(out)
        return {"w": np.asarray(d["w"]).ravel(),
                "wall_s": float(d["wall_s"]), "status": str(d["status"]),
                "label": str(d["label"]), "config": str(d["config"])}


# --------------------------------------------------------------------------
# Manufacturing a loose point without a first-order solver
# --------------------------------------------------------------------------

def perturb_to_target(uni, limits, R, w_ref: np.ndarray, seed: int,
                      target_mult: float) -> tuple[np.ndarray, dict]:
    """Walk `w_ref` off the mandate until the worst desk breach hits target.

    The direction is seeded, zero-sum before clipping, and normalized to unit
    L1 so the step length is interpretable as gross weight moved. The point is
    clipped into the box and NOT renormalized: PDLP satisfies variable bounds
    exactly and violates row constraints, and `sum(w) = 1` is a row, so budget
    drift belongs in a faithful imitation while a bound violation does not.

    Returns the point and a dict of what the search actually achieved --
    bisection on a non-monotone objective can miss, and the run must say so
    rather than report the target as if it had been hit.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(uni.n_bonds)
    d = z - z.mean()
    d = d / np.abs(d).sum()

    def at(step: float) -> tuple[np.ndarray, float]:
        w = np.clip(w_ref + step * d, 0.0, limits.w_max)
        gv = gate_verdicts(exposure_report(uni, w, limits, R=R))
        return w, gv.desk_worst_multiple

    hi = 1e-4
    for _ in range(40):
        _, m = at(hi)
        if m >= target_mult:
            break
        hi *= 2.0
    lo = 0.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        _, m = at(mid)
        if m < target_mult:
            lo = mid
        else:
            hi = mid
    w, achieved = at(hi)
    return w, {"step_l1": float(hi), "target_mult": float(target_mult),
               "achieved_mult": float(achieved), "seed": int(seed)}


# --------------------------------------------------------------------------
# The repair itself
# --------------------------------------------------------------------------

@dataclasses.dataclass
class RepairOut:
    w: np.ndarray
    status: str
    build_s: float
    solve_s: float
    l1_distance: float


@dataclasses.dataclass
class DirectEndpointOut:
    w: np.ndarray | None
    status: str
    first_solve_s: float
    steady_state_median_s: float
    steady_state_runs_s: tuple[float, ...]
    total_endpoint_s: float
    elapsed_is_lower_bound: bool
    solver_objective: float


def weight_space_limits(limits) -> tuple[object, list[str]]:
    """The limits the projection CAN see, and the names of the ones it cannot.

    Exactly one family is dropped: the risk-space tracking-error constraint,
    which is defined on `R @ (w - b)` and therefore lives in the scenario
    block. `active_l1` is NOT dropped -- ``||w - b||_1`` is a weight-space
    budget despite belonging to the same mandate paragraph, and dropping it
    would understate what the projection can restore.
    """
    ws = dataclasses.replace(limits, te_bp=np.inf)
    invisible: list[str] = []
    if np.isfinite(limits.te_bp):
        invisible.append("tracking_error_sigma_SOC"
                         if limits.te_mode == "sigma"
                         else "tracking_error_mad_LP")
    return ws, invisible


def build_polytope(uni, ws_limits) -> tuple[cp.Variable, list, list[str]]:
    """The weight-space mandate polytope, as (variable, constraints, names)."""
    v = cp.Variable(uni.n_bonds, nonneg=True)
    cons = [cp.sum(v) == 1.0, v <= ws_limits.w_max]
    names = ["budget", "w_max"]
    extra, extra_names = credit_constraint_builder(uni, ws_limits, R=None)(v)
    return v, cons + extra, names + extra_names


def _highs_opts(threads: int, time_limit: float) -> dict:
    return {"highs_options": {"solver": "simplex", "time_limit": time_limit,
                              "threads": threads, "parallel": "on",
                              "random_seed": 0}}


def repair_once(uni, limits, ws_limits, w_loose: np.ndarray, mode: str,
                R, mu, alpha: float, lam: float, threads: int,
                time_limit: float, d1: float, lex_slack: float,
                conic: bool) -> RepairOut:
    """One repair, timed with model construction and solve kept separate.

    `build_s` is CVXPY canonicalization, which a production desk pays once and
    then re-solves against; `solve_s` is what each subsequent loose point
    costs. Both are reported because collapsing them would let the headline be
    chosen after the fact.
    """
    if mode in ("l1_mu_lex", "l1_ball_full") and not np.isfinite(d1):
        raise ValueError(
            f"{mode} needs the L1 projection distance, which did not exist "
            f"(the l1 repair failed); no ball, no objective-aware repair")

    t0 = time.perf_counter()
    if mode == "l1_ball_full":
        # The one mode that is NOT a small problem: the true mean-CVaR
        # objective needs the scenario block back, and the full mandate is
        # re-imposed so the TE families are visible again. Its cost is the
        # measurement.
        full = credit_constraint_builder(uni, limits, R=R)

        def builder(w):
            cons, names = full(w)
            cons.append(cp.norm1(w - w_loose) <= d1 * (1.0 + lex_slack))
            return cons, names + ["repair_l1_ball"]

        model = build_mean_cvar(R, mu, alpha=alpha, risk_aversion=lam,
                                w_max=limits.w_max,
                                constraint_builder=builder)
        prob, var = model.problem, model.w
        if conic:
            solver, opts = cp.CLARABEL, {}
        else:
            solver, opts = cp.HIGHS, _highs_opts(threads, time_limit)
    else:
        var, cons, _ = build_polytope(uni, ws_limits)
        if mode == "l2":
            prob = cp.Problem(cp.Minimize(cp.sum_squares(var - w_loose)),
                              cons)
            solver, opts = cp.CLARABEL, {}
        elif mode == "l1":
            prob = cp.Problem(cp.Minimize(cp.norm1(var - w_loose)), cons)
            solver, opts = cp.HIGHS, _highs_opts(threads, time_limit)
        elif mode == "l1_mu_lex":
            prob = cp.Problem(
                cp.Maximize(mu @ var),
                cons + [cp.norm1(var - w_loose) <= d1 * (1.0 + lex_slack)])
            solver, opts = cp.HIGHS, _highs_opts(threads, time_limit)
        else:
            raise ValueError(f"unknown repair mode {mode!r}")
    build_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    prob.solve(solver=solver, verbose=False, warm_start=False, **opts)
    solve_s = time.perf_counter() - t0

    if prob.status not in CONVERGED or var.value is None:
        return RepairOut(w=np.full(uni.n_bonds, np.nan),
                         status=str(prob.status), build_s=build_s,
                         solve_s=solve_s, l1_distance=float("nan"))
    w = np.asarray(var.value).ravel()
    return RepairOut(w=w, status=str(prob.status), build_s=build_s,
                     solve_s=solve_s,
                     l1_distance=float(np.abs(w - w_loose).sum()))


def program_size(prob: cp.Problem, solver) -> tuple[float, float, float]:
    """(rows, cols, nnz) of the canonicalized constraint matrix, or NaNs.

    Measured rather than derived: the whole claim under test is that the
    repair program is orders of magnitude smaller than the LP, and a formula
    for the size would be assuming the answer.
    """
    try:
        data = prob.get_problem_data(solver)[0]
        A = data["A"]
        return float(A.shape[0]), float(A.shape[1]), float(A.nnz)
    except Exception:  # noqa: BLE001
        return float("nan"), float("nan"), float("nan")


# --------------------------------------------------------------------------
# Scoring: what did the portfolio actually deliver, against the FULL mandate
# --------------------------------------------------------------------------

def score_portfolio(uni, w: np.ndarray, limits, R, R_oos, mu,
                    alpha: float, lam: float) -> dict:
    """Every published number about one weight vector, recomputed from numpy.

    Two scorers, not one, because neither covers the whole mandate:
    `exposure_report` carries the six credit families and the TE rows but not
    the budget or the box, and `src.check.check_solution` carries those. A
    loose first-order point can be clean on every credit family and still be
    holding 99.996% of the book -- the CPU validation found exactly that -- so
    reporting only the credit gates would have published a pass the portfolio
    did not earn.
    """
    rep = exposure_report(uni, w, limits, R=R)
    gv = gate_verdicts(rep)
    # solver_objective is deliberately NaN: nothing here trusts a solver's
    # self-reported value, and `objective_recomputed` is the number used.
    chk = check_solution(w=w, solver_objective=float("nan"), R=R, mu=mu,
                         alpha=alpha, risk_aversion=lam, w_max=limits.w_max)
    te = tracking_error_bp(uni, w, R, limits.periods_per_year)
    cvar_oos, var_oos = empirical_cvar(-(R_oos @ w), alpha)
    expected = float(mu @ w)

    scored_te = (te["te_mad_implied_bp"] if limits.te_mode == "mad"
                 else te["te_sigma_bp"])
    te_breach = (scored_te - limits.te_bp if np.isfinite(limits.te_bp)
                 else float("nan"))

    out = dict(gv.as_row())
    out.update({
        "budget_signed": chk.budget_signed,
        "budget_residual": chk.budget_residual,
        "max_bound_violation": chk.max_bound_violation,
        "n_positions": chk.n_positions,
        "n_dust_positions": chk.n_dust_positions,
        "n_nonzero_weights": chk.n_nonzero_weights,
        "check_ok": bool(chk.ok),
        "check_notes": chk.notes or "ok",
        "objective_in": chk.objective_recomputed,
        "solver_objective": float("nan"),
        "objective_recomputed": chk.objective_recomputed,
        "objective_discrepancy": float("nan"),
        "objective_quality_target": float("nan"),
        "objective_consistent": False,
        "expected_excess_bp_monthly": expected * 1e4,
        "cvar_in": chk.cvar_empirical,
        "cvar_recomputed": chk.cvar_empirical,
        "cvar_oos": cvar_oos,
        "var_oos": var_oos,
        # mu is a model input, not a statistic of the evaluation draw, so the
        # out-of-sample objective swaps only the risk term.
        "objective_oos": expected - lam * cvar_oos,
        "mu_w": expected,
        "te_sigma_bp": te["te_sigma_bp"],
        "te_mad_implied_bp": te["te_mad_implied_bp"],
        "te_breach_bp": te_breach,
        "active_l1": te["active_l1"],
    })
    return out


ROW_TEMPLATE: dict[str, object] = {
    # identity
    "case": "", "te_budget_bp": np.nan, "te_mode": "", "portfolio": "",
    "role": "", "method_class": "", "endpoint": "", "status": "",
    "status_source": "", "in_class": True, "note": "",
    # timing
    "wall_s": np.nan, "wall_s_min": np.nan, "wall_s_max": np.nan,
    "build_s": np.nan, "first_solve_s": np.nan,
    "steady_state_median_s": np.nan, "total_endpoint_s": np.nan,
    "elapsed_is_lower_bound": False,
    "n_timed_runs": np.nan, "wall_kind": "",
    # program size
    "prog_rows": np.nan, "prog_cols": np.nan, "prog_nnz": np.nan,
    "full_lp_cols_est": np.nan, "full_lp_nnz_est": np.nan,
    "nnz_ratio_vs_full_lp": np.nan,
    # mandate, credit families
    "purity_pass": False, "purity_worst_rel": np.nan,
    "purity_worst_limit": "-", "desk_pass": False,
    "desk_worst_multiple": np.nan, "desk_worst_limit": "-",
    "desk_worst_bp_weight": np.nan, "desk_worst_usd": np.nan,
    "n_purity_breaches": np.nan, "n_desk_breaches": np.nan,
    "gate_verdict": "",
    # mandate, budget and box (not covered by the exposure report)
    "budget_signed": np.nan, "budget_residual": np.nan,
    "max_bound_violation": np.nan,
    # book shape
    "n_positions": np.nan, "n_dust_positions": np.nan,
    "n_nonzero_weights": np.nan,
    "check_ok": False, "check_notes": "",
    # economics
    "objective_in": np.nan, "solver_objective": np.nan,
    "objective_recomputed": np.nan, "objective_discrepancy": np.nan,
    "objective_quality_target": np.nan, "objective_consistent": False,
    "expected_excess_bp_monthly": np.nan, "cvar_in": np.nan,
    "cvar_recomputed": np.nan, "cvar_oos": np.nan, "var_oos": np.nan,
    "objective_oos": np.nan, "mu_w": np.nan,
    "te_sigma_bp": np.nan, "te_mad_implied_bp": np.nan,
    "te_breach_bp": np.nan, "active_l1": np.nan,
    # degradation vs the two reference endpoints the dichotomy is about, plus whichever
    # exact optimum exists for this case (an SOC case has no LP basic solution,
    # so without the third pair those rows would carry no comparison)
    "d_obj_vs_xover_bp": np.nan, "d_obj_vs_simplex_bp": np.nan,
    "d_cvar_oos_vs_xover_bp": np.nan, "d_cvar_oos_vs_simplex_bp": np.nan,
    "d_obj_oos_vs_xover_bp": np.nan, "d_obj_oos_vs_simplex_bp": np.nan,
    "d_obj_vs_exact_bp": np.nan, "d_obj_oos_vs_exact_bp": np.nan,
    "d_cvar_oos_vs_exact_bp": np.nan, "exact_reference": "",
    "d_expected_excess_vs_xover_bp": np.nan,
    "d_expected_excess_vs_simplex_bp": np.nan,
    "d_expected_excess_vs_exact_bp": np.nan,
    "d_cvar_in_vs_xover_bp": np.nan, "d_cvar_in_vs_simplex_bp": np.nan,
    "d_cvar_in_vs_exact_bp": np.nan,
    "source_reference": "", "d_obj_vs_source_bp": np.nan,
    "d_expected_excess_vs_source_bp": np.nan,
    "d_cvar_in_vs_source_bp": np.nan,
    # repair-specific
    "loose_spec": "", "repaired_from": "", "repair_l1_moved": np.nan,
    "repair_linf_moved": np.nan, "max_single_name_move": np.nan,
    "names_moved_gt_1bp": np.nan, "n_positions_delta": np.nan,
    "n_dust_delta": np.nan, "n_nonzero_delta": np.nan,
    "families_repaired": "", "families_invisible": "",
    "te_visible_to_repair": np.nan,
}


def make_row(**kw) -> dict:
    row = dict(ROW_TEMPLATE)
    row.update(kw)
    unknown = set(kw) - set(ROW_TEMPLATE)
    if unknown:
        # write_csv takes its fieldnames from the first row, so a key that
        # only some rows carry would silently vanish from the CSV.
        raise KeyError(f"column(s) not in ROW_TEMPLATE: {sorted(unknown)}")
    return row


def endpoint_total_seconds(raw_s: float, endpoint_s: float) -> float:
    """Total endpoint time when a downstream endpoint depends on a raw solve."""
    if not (math.isfinite(float(raw_s)) and math.isfinite(float(endpoint_s))):
        return float("nan")
    return float(raw_s) + float(endpoint_s)


def objective_consistency_columns(row: dict, solver_objective: float,
                                  quality_target=1e-4) -> dict:
    if quality_target is None:
        quality_target = 1e-4
    objective_tol = float(getattr(
        quality_target, "check_objective_tol", quality_target))
    row["objective_quality_target"] = objective_tol
    obj = float(solver_objective)
    row["solver_objective"] = obj
    if not math.isfinite(obj):
        row["objective_discrepancy"] = float("nan")
        row["objective_consistent"] = False
        return row
    row["objective_discrepancy"] = abs(row["objective_recomputed"] - obj)
    row["objective_consistent"] = bool(
        row["objective_discrepancy"] <= objective_tol)
    return row


def no_portfolio_columns(verdict: str) -> dict:
    """Columns for an endpoint row that produced no scoreable weight vector."""
    return {
        "purity_pass": False, "purity_worst_rel": np.nan,
        "purity_worst_limit": "-", "desk_pass": False,
        "desk_worst_multiple": np.nan, "desk_worst_limit": "-",
        "desk_worst_bp_weight": np.nan, "desk_worst_usd": np.nan,
        "n_purity_breaches": np.nan, "n_desk_breaches": np.nan,
        "gate_verdict": verdict,
        "check_ok": False, "check_notes": verdict,
    }


def make_unavailable_endpoint_row(base, *, portfolio: str, role: str,
                                  endpoint: str, method_class: str,
                                  status: str, status_source: str,
                                  note: str, **kw) -> dict:
    return base(
        portfolio=portfolio, role=role, endpoint=endpoint,
        method_class=method_class, status=status,
        status_source=status_source, note=note, **no_portfolio_columns(status),
        **kw,
    )


def make_reference_unsupported_row(base, *, portfolio: str, role: str,
                                   method_class: str, endpoint: str,
                                   note: str) -> dict:
    return make_unavailable_endpoint_row(
        base, portfolio=portfolio, role=role, endpoint=endpoint,
        method_class=method_class, status="UNSUPPORTED_BY_METHOD",
        status_source="author-classification", in_class=False,
        wall_kind="solve", note=note)


def make_reference_error_row(base, *, portfolio: str, role: str,
                             method_class: str, endpoint: str,
                             error: Exception) -> dict:
    return base(
        portfolio=portfolio, role=role, endpoint=endpoint,
        method_class=method_class, status=f"ERROR:{type(error).__name__}",
        status_source="runner-reference-error", wall_kind="solve",
        note=str(error)[:200], **no_portfolio_columns("ERROR"))


def add_source_loss_metrics(row: dict, source: dict | None,
                            source_label: str) -> dict:
    if source is None:
        return row
    row["source_reference"] = source_label
    row["d_obj_vs_source_bp"] = (
        source["objective_in"] - row["objective_in"]) * 1e4
    row["d_expected_excess_vs_source_bp"] = (
        source["mu_w"] - row["mu_w"]) * 1e4
    row["d_cvar_in_vs_source_bp"] = (
        row["cvar_in"] - source["cvar_in"]) * 1e4
    return row


def add_degradation(row: dict, refs: dict[str, dict],
                    exact_key: str) -> dict:
    """Objective and out-of-sample CVaR gaps vs the crossover/basic solution,
    the simplex exact reference and the case's exact optimum, in basis points of monthly
    return. Signed so that positive always means the row is WORSE: less
    objective, more tail loss."""
    row["exact_reference"] = exact_key if exact_key in refs else "-"
    for tag, key in (("xover", "highs-ipm+xover"),
                     ("simplex", "highs-simplex"),
                     ("exact", exact_key)):
        ref = refs.get(key)
        if ref is None:
            continue
        row[f"d_obj_vs_{tag}_bp"] = (ref["objective_in"]
                                     - row["objective_in"]) * 1e4
        row[f"d_obj_oos_vs_{tag}_bp"] = (ref["objective_oos"]
                                         - row["objective_oos"]) * 1e4
        row[f"d_cvar_oos_vs_{tag}_bp"] = (row["cvar_oos"]
                                          - ref["cvar_oos"]) * 1e4
        row[f"d_expected_excess_vs_{tag}_bp"] = (
            ref["mu_w"] - row["mu_w"]) * 1e4
        row[f"d_cvar_in_vs_{tag}_bp"] = (
            row["cvar_in"] - ref["cvar_in"]) * 1e4
    return row


# --------------------------------------------------------------------------
# Loose points, and the repair sweep over one of them
# --------------------------------------------------------------------------

def cuopt_pdlp_specs_for_tols(loose_tols: list[float] | tuple[float, ...],
                              time_limit: float):
    """Raw cuOpt PDLP specs plus the 1e-4 crossover/basic endpoint spec."""
    raw = [
        cuopt_pdlp_spec(float(tol), time_limit_s=time_limit, crossover=False)
        for tol in loose_tols
    ]
    basic = cuopt_pdlp_spec(1e-4, time_limit_s=time_limit, crossover=True)
    return raw, basic


def solve_direct_endpoint(model, spec, runs: int) -> DirectEndpointOut:
    """Solve one CVXPY endpoint directly; stop DNF/error rows after one cap."""
    t0 = time.perf_counter()
    try:
        model.problem.solve(solver=spec.cvxpy_solver, verbose=False,
                            warm_start=False, **spec.solver_opts)
        first_solve_s = time.perf_counter() - t0
        status = str(model.problem.status)
    except Exception as e:  # noqa: BLE001
        first_solve_s = time.perf_counter() - t0
        return DirectEndpointOut(
            w=None, status=f"ERROR:{type(e).__name__}",
            first_solve_s=first_solve_s,
            steady_state_median_s=float("nan"),
            steady_state_runs_s=(), total_endpoint_s=first_solve_s,
            elapsed_is_lower_bound=True, solver_objective=float("nan"),
        )

    if status not in CONVERGED:
        return DirectEndpointOut(
            w=None, status=status, first_solve_s=first_solve_s,
            steady_state_median_s=float("nan"), steady_state_runs_s=(),
            total_endpoint_s=first_solve_s, elapsed_is_lower_bound=True,
            solver_objective=float(model.problem.value or "nan"),
        )

    times: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        try:
            model.problem.solve(solver=spec.cvxpy_solver, verbose=False,
                                warm_start=False, **spec.solver_opts)
            status = str(model.problem.status)
        except Exception as e:  # noqa: BLE001
            status = f"ERROR:{type(e).__name__}"
        wall = time.perf_counter() - t0
        if status not in CONVERGED:
            return DirectEndpointOut(
                w=None, status=status, first_solve_s=first_solve_s,
                steady_state_median_s=float("nan"),
                steady_state_runs_s=tuple(times),
                total_endpoint_s=first_solve_s + sum(times) + wall,
                elapsed_is_lower_bound=True,
                solver_objective=float(model.problem.value or "nan"),
            )
        times.append(wall)

    return DirectEndpointOut(
        w=model.weights().copy(), status=status, first_solve_s=first_solve_s,
        steady_state_median_s=statistics.median(times) if times else
        float("nan"),
        steady_state_runs_s=tuple(times),
        total_endpoint_s=statistics.median(times) if times else float("nan"),
        elapsed_is_lower_bound=False,
        solver_objective=float(model.problem.value or "nan"),
    )


def unavailable_cuopt_points(args, status: str, status_source: str,
                             note: str) -> list[dict]:
    return [{
        "w": None,
        "label": f"cuopt-pdlp@{float(tol):g}",
        "spec": f"tol={float(tol):g}",
        "wall_s": float("nan"),
        "wall_s_min": float("nan"),
        "wall_s_max": float("nan"),
        "first_solve_s": float("nan"),
        "steady_state_median_s": float("nan"),
        "total_endpoint_s": float("nan"),
        "elapsed_is_lower_bound": False,
        "n_timed_runs": 0.0,
        "status": status,
        "status_source": status_source,
        "method_class": "LP",
        "endpoint": "raw",
        "note": note,
    } for tol in args.loose_tol]


def collect_loose_points(args, case_idx: int, case_name: str, conic: bool,
                         uni, limits, R, mu, refs: dict,
                         model=None) -> list[dict]:
    """Every loose point this case is to be repaired from.

    A list rather than a single point because how loose the point is IS the
    independent variable: a repair that is cheap and harmless at 1e-4 and
    ruinous at 1e-2 is a different recommendation from one that is cheap at
    both, and running one tolerance per invocation would pay for the
    expensive reference solves twice to find out.
    """
    if args.loose_source == "npz":
        d = np.load(args.loose_npz[case_idx])
        return [{
            "w": np.asarray(d["w"]).ravel(),
            "label": str(d["label"]) if "label" in d else "loose-npz",
            "spec": args.loose_npz[case_idx],
            "wall_s": float(d["wall_s"]) if "wall_s" in d else float("nan"),
            "wall_s_min": float("nan"),
            "wall_s_max": float("nan"),
                "first_solve_s": float("nan"),
                "steady_state_median_s": (
                    float(d["wall_s"]) if "wall_s" in d else float("nan")),
                "total_endpoint_s": (
                    float(d["wall_s"]) if "wall_s" in d else float("nan")),
                "elapsed_is_lower_bound": False,
                "n_timed_runs": float("nan"),
            "status": str(d["status"]) if "status" in d else "supplied",
            "status_source": "supplied",
            "method_class": "LP",
            "endpoint": "raw",
            "note": f"supplied by {args.loose_npz[case_idx]}",
        }]

    if args.loose_source == "cuopt-pdlp":
        if conic:
            return unavailable_cuopt_points(
                args, "UNSUPPORTED_BY_METHOD", "author-classification",
                f"case {case_name} is a second-order cone program and cuOpt "
                f"PDLP is an LP endpoint")
        if model is None:
            return unavailable_cuopt_points(
                args, "UNAVAILABLE", "runner-build-error",
                "internal error: no full LP model was supplied")
        try:
            raw_specs, _ = cuopt_pdlp_specs_for_tols(
                args.loose_tol, args.time_limit)
        except Exception as e:  # noqa: BLE001
            return unavailable_cuopt_points(
                args, "UNAVAILABLE", "runner-build-error", str(e)[:200])

        out = []
        for spec in raw_specs:
            print(f"    loose point: {spec.label} (direct cuOpt)")
            solved = solve_direct_endpoint(model, spec, args.ref_runs)
            walls = solved.steady_state_runs_s
            out.append({
                "w": solved.w,
                "label": spec.label,
                "spec": spec.tolerance_desc,
                "wall_s": solved.steady_state_median_s,
                "wall_s_min": min(walls) if walls else float("nan"),
                "wall_s_max": max(walls) if walls else float("nan"),
                "first_solve_s": solved.first_solve_s,
                "steady_state_median_s": solved.steady_state_median_s,
                "total_endpoint_s": solved.total_endpoint_s,
                "elapsed_is_lower_bound": solved.elapsed_is_lower_bound,
                "n_timed_runs": float(len(walls)),
                "status": solved.status,
                "status_source": "solver-returned",
                "solver_objective": solved.solver_objective,
                "solver_spec": spec,
                "method_class": "LP",
                "endpoint": "raw",
                "note": "direct cuOpt PDLP raw endpoint; no OR-Tools or NPZ",
            })
        return out

    if args.loose_source == "ortools-pdlp":
        if conic:
            raise SystemExit(
                f"case {case_name} is a second-order cone program and PDLP "
                f"cannot express it; use --loose-source perturb, or supply a "
                f"point with --loose-npz")
        out = []
        for tol in args.loose_tol:
            print(f"    loose point: OR-Tools PDLP @ {tol:g} (subprocess)")
            got = run_pdlp_worker(uni, limits, R, mu, args.alpha,
                                  args.risk_aversion, tol, args.threads,
                                  args.time_limit)
            out.append({
                "w": got["w"], "label": got["label"], "spec": f"tol={tol:g}",
                "wall_s": got["wall_s"], "status": got["status"],
                "wall_s_min": float("nan"), "wall_s_max": float("nan"),
                "first_solve_s": float("nan"),
                "steady_state_median_s": got["wall_s"],
                "total_endpoint_s": got["wall_s"],
                "elapsed_is_lower_bound": False,
                "n_timed_runs": 1.0, "status_source": "solver-returned",
                "method_class": "LP", "endpoint": "raw",
                "note": ("genuine first-order iterate; OR-Tools PDLP, not "
                         "cuOpt: different step-size rule, restart schedule "
                         "and preconditioning, plus a GLOP presolve"),
            })
        return out

    anchor = refs.get("highs-simplex") or refs.get("clarabel")
    if anchor is None:
        raise SystemExit("--loose-source perturb needs a converged exact "
                         "reference to walk away from, and this case has "
                         "none")
    out = []
    for target in args.perturb_target_mult:
        w, pinfo = perturb_to_target(uni, limits, R, anchor["w"],
                                     args.perturb_seed, target)
        out.append({
            "w": w, "label": f"perturbed@{target:g}x",
            "spec": f"target={target:g}x,seed={args.perturb_seed}",
            "wall_s": float("nan"), "status": "manufactured",
            "wall_s_min": float("nan"), "wall_s_max": float("nan"),
            "first_solve_s": float("nan"),
            "steady_state_median_s": float("nan"),
            "total_endpoint_s": float("nan"),
            "elapsed_is_lower_bound": False,
            "n_timed_runs": float("nan"), "status_source": "manufactured",
            "method_class": "manufactured", "endpoint": "raw",
            "note": (f"MANUFACTURED, not a first-order iterate: seeded "
                     f"zero-sum direction, L1 step {pinfo['step_l1']:.3e}, "
                     f"target {target:g}x desk tol, achieved "
                     f"{pinfo['achieved_mult']:.3g}x"),
        })
    return out


@dataclasses.dataclass
class CaseContext:
    """Everything the repair sweep needs that is fixed within one case."""

    args: argparse.Namespace
    uni: object
    limits: object
    ws_limits: object
    conic: bool
    R: np.ndarray
    R_oos: np.ndarray
    mu: np.ndarray
    refs: dict
    exact_key: str
    base: object          # callable producing a pre-filled row
    full_cols: int
    full_nnz: float


def repair_sweep(ctx: CaseContext, loose: dict, sc_loose: dict) -> list[dict]:
    """Run every requested repair mode against one loose point."""
    args, base = ctx.args, ctx.base
    w_loose, label = loose["w"], loose["label"]
    rows: list[dict] = []
    # `l1` runs whether or not it was requested: the two objective-aware
    # modes are defined relative to ITS optimal distance, and that distance
    # has to be re-derived for every loose point.
    d1 = float("nan")
    for mode in REPAIR_MODES:
        wanted = mode in args.modes
        if mode != "l1" and not wanted:
            continue
        if mode in ("l1_mu_lex", "l1_ball_full") and not np.isfinite(d1):
            if wanted:
                rows.append(make_unavailable_endpoint_row(
                    base, portfolio=f"repair:{mode}", role="repair",
                    endpoint="repaired-feasible", method_class="LP",
                    status="UNAVAILABLE",
                    status_source="author-classification",
                    repaired_from=label, loose_spec=loose["spec"],
                    wall_kind="repair",
                    note="dependent on a converged l1 repair distance; the "
                         "l1 repair family produced no feasible ball"))
            continue
        outs, builds, solves = [], [], []
        try:
            for k in range(args.repair_runs + 1):
                out = repair_once(
                    ctx.uni, ctx.limits, ctx.ws_limits, w_loose, mode, ctx.R,
                    ctx.mu, args.alpha, args.risk_aversion, args.threads,
                    args.time_limit, d1, args.lex_slack, ctx.conic)
                if k == 0:
                    continue              # discarded warmup, as everywhere
                outs.append(out)
                builds.append(out.build_s)
                solves.append(out.solve_s)
        except Exception as e:  # noqa: BLE001
            if wanted:
                rows.append(make_unavailable_endpoint_row(
                    base,
                    portfolio=f"repair:{mode}", role="repair",
                    endpoint="repaired-feasible", method_class="LP",
                    status=f"ERROR:{type(e).__name__}",
                    status_source="runner-repair-error",
                    repaired_from=label, loose_spec=loose["spec"],
                    wall_kind="repair",
                    note=str(e)[:200]))
            print(f"    repair:{mode:<13} FAILED {type(e).__name__}: "
                  f"{str(e)[:70]}")
            continue

        out = outs[-1]
        if mode == "l1":
            d1 = out.l1_distance
        if out.status not in CONVERGED:
            # Not a timing. An infeasible projection means no repaired
            # portfolio exists, which is a statement about the polytope --
            # and for `l1_ball_full` it is the expected one whenever a
            # constraint the L1 ball cannot see is violated at its centre.
            if wanted:
                rows.append(make_unavailable_endpoint_row(
                    base,
                    portfolio=f"repair:{mode}", role="repair",
                    endpoint="repaired-feasible", method_class="LP",
                    status=out.status, status_source="solver-returned",
                    repaired_from=label, loose_spec=loose["spec"],
                    wall_kind="repair",
                    note="repair program did not solve; no repaired "
                         "portfolio exists, which is a finding about the "
                         "feasible set rather than a timing"))
            print(f"    repair:{mode:<13} {out.status} - no portfolio")
            continue
        if not wanted:
            continue

        walls = [b + s for b, s in zip(builds, solves)]
        sc = score_portfolio(ctx.uni, out.w, ctx.limits, ctx.R, ctx.R_oos,
                             ctx.mu, args.alpha, args.risk_aversion)
        moved = np.abs(out.w - w_loose)
        # Size is measured on a freshly posed program, so the ratio in the
        # CSV describes the repair as it is stated rather than as cached.
        if mode == "l1_ball_full":
            pr, pc, pn = float("nan"), float(ctx.full_cols), ctx.full_nnz
        else:
            v, cons, _ = build_polytope(ctx.uni, ctx.ws_limits)
            probe = cp.Problem(cp.Minimize(cp.norm1(v - w_loose)), cons)
            pr, pc, pn = program_size(probe, cp.HIGHS)
        row = base(
            portfolio=f"repair:{mode}", role="repair", method_class="LP",
            endpoint="repaired-feasible", status=out.status,
            status_source="solver-returned",
            wall_s=statistics.median(walls), wall_s_min=min(walls),
            wall_s_max=max(walls), build_s=statistics.median(builds),
            steady_state_median_s=statistics.median(solves),
            total_endpoint_s=endpoint_total_seconds(
                loose["wall_s"], statistics.median(walls)),
            n_timed_runs=float(len(walls)), wall_kind="repair",
            prog_rows=pr, prog_cols=pc, prog_nnz=pn,
            nnz_ratio_vs_full_lp=(pn / ctx.full_nnz if ctx.full_nnz
                                  else np.nan),
            repaired_from=label, loose_spec=loose["spec"],
            repair_l1_moved=float(moved.sum()),
            repair_linf_moved=float(moved.max()),
            max_single_name_move=float(moved.max()),
            names_moved_gt_1bp=float((moved > NAME_MOVE_GATE).sum()),
            n_positions_delta=float(sc["n_positions"]
                                    - sc_loose["n_positions"]),
            n_dust_delta=float(sc["n_dust_positions"]
                               - sc_loose["n_dust_positions"]),
            n_nonzero_delta=float(sc["n_nonzero_weights"]
                                  - sc_loose["n_nonzero_weights"]),
            **sc)
        add_source_loss_metrics(row, sc_loose, label)
        rows.append(add_degradation(row, ctx.refs, ctx.exact_key))
        print(f"    repair:{mode:<13} {statistics.median(walls) * 1000:8.1f}"
              f"ms (solve {statistics.median(solves) * 1000:7.1f}ms)  "
              f"verdict={sc['gate_verdict']:<7} "
              f"desk={sc['desk_worst_multiple']:.3g}x  "
              f"L1={moved.sum():.5f} names>1bp="
              f"{int((moved > NAME_MOVE_GATE).sum())}  "
              f"positions {sc_loose['n_positions']}->{sc['n_positions']}")
    return rows


# --------------------------------------------------------------------------
# Reference solves
# --------------------------------------------------------------------------

def solve_reference(model, kind: str, threads: int, time_limit: float,
                    runs: int) -> tuple[np.ndarray, str, float, list[float]]:
    if kind == "highs-simplex":
        solver, opts = cp.HIGHS, _highs_opts(threads, time_limit)
    elif kind == "highs-xover":
        # The CPU stand-in for "pay for crossover": an interior-point solve
        # followed by crossover to a basic solution, which is the same two
        # steps cuOpt performs with crossover=1. Not the same implementation
        # and not the same cost curve -- the GPU session replaces this row.
        solver = cp.HIGHS
        opts = {"highs_options": {"solver": "ipm", "run_crossover": "on",
                                  "time_limit": time_limit,
                                  "threads": threads, "parallel": "on",
                                  "random_seed": 0}}
    elif kind == "clarabel":
        solver, opts = cp.CLARABEL, {}
    else:
        raise ValueError(kind)

    model.problem.solve(solver=solver, verbose=False, warm_start=False,
                        **opts)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model.problem.solve(solver=solver, verbose=False, warm_start=False,
                            **opts)
        times.append(time.perf_counter() - t0)
    return (model.weights().copy(), str(model.problem.status),
            statistics.median(times), times)


# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__.split("\n\n")[0])
    ap.add_argument("--n-bonds", type=int, default=500)
    ap.add_argument("--n-scenarios", type=int, default=20_000)
    ap.add_argument("--alpha", type=float, default=0.95)
    ap.add_argument("--risk-aversion", type=float, default=3.0)
    ap.add_argument("--turnover", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--scen-seed", type=int, default=11)
    ap.add_argument("--oos-seed", type=int, default=101)
    ap.add_argument("--te-grid", type=float, nargs="*", default=[10.0],
                    help="annualized TE budgets in bp; one mandate case per "
                         "value, plus an always-present no-TE case. A finite "
                         "budget is the case the projection CANNOT fully "
                         "repair, which is the point of running it")
    ap.add_argument("--te-mode", choices=["mad", "sigma"], default="mad",
                    help="mad keeps the full program an LP; sigma is the "
                         "faithful second-order cone reading, which no "
                         "LP-only solver can express")
    ap.add_argument("--active-l1", type=float, default=float("inf"),
                    help="||w - benchmark||_1 budget; weight-space, so the "
                         "projection CAN see it")
    ap.add_argument("--loose-source",
                    choices=["cuopt-pdlp", "ortools-pdlp", "perturb", "npz"],
                    default="cuopt-pdlp")
    ap.add_argument("--loose-tol", type=float, nargs="+",
                    default=[1e-4, 1e-6],
                    help="PDLP eps_optimal abs+rel; one loose point and one "
                         "repair sweep per value. 1e-4 is the paper's "
                         "operating point; 1e-2 is deliberately far looser, "
                         "because how loose the point is IS the independent "
                         "variable")
    ap.add_argument("--loose-npz", type=str, nargs="*", default=[],
                    help="one .npz per mandate case, each with key w (and "
                         "optionally wall_s, status, label). The GPU-session "
                         "hook")
    ap.add_argument("--crossover-npz", type=str, nargs="*", default=[],
                    help="one .npz per mandate case holding the crossover/"
                         "basic solution to compare against, same keys")
    ap.add_argument("--perturb-target-mult", type=float, nargs="+",
                    default=[10.0],
                    help="worst desk-gate breach the manufactured point is "
                         "bisected onto, as a multiple of desk tolerance; "
                         "one loose point per value")
    ap.add_argument("--perturb-seed", type=int, default=2027)
    ap.add_argument("--modes", nargs="*", default=["l1", "l1_mu_lex"],
                    choices=list(REPAIR_MODES))
    ap.add_argument("--repair-runs", type=int, default=5,
                    help="timed repeats per repair (median reported)")
    ap.add_argument("--ref-runs", type=int, default=3)
    ap.add_argument("--lex-slack", type=float, default=1e-9,
                    help="relative widening of the L1 ball for the two "
                         "objective-aware modes, so the ball is not "
                         "infeasible by rounding")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--time-limit", type=float, default=900.0)
    ap.add_argument("--cpu-only", action="store_true",
                    help="CPU scenario generation and no GPU-idle check")
    ap.add_argument("--allow-busy-gpu", action="store_true")
    ap.add_argument("--out", type=str, default="")
    # worker plumbing; not for interactive use
    ap.add_argument("--emit-loose-point", type=str, default="",
                    help=argparse.SUPPRESS)
    ap.add_argument("--problem-npz", type=str, default="",
                    help=argparse.SUPPRESS)
    ap.add_argument("--universe-pkl", type=str, default="",
                    help=argparse.SUPPRESS)
    return ap


def main() -> int:
    args = build_parser().parse_args()
    if args.emit_loose_point:
        return emit_loose_point(args)

    if not args.cpu_only:
        assert_gpu_idle(raise_on_busy=not args.allow_busy_gpu)

    pd.set_option("display.width", 170)
    pd.set_option("display.max_columns", 30)

    # The no-TE case pins `te_mode` explicitly even though the budget is
    # infinite: `default_limits` defaults it to "sigma", and an inert mode
    # name in the CSV would read as a mandate the run never imposed.
    cases: list[tuple[str, dict]] = [
        ("no-TE", dict(te_bp=np.inf, te_mode=args.te_mode))]
    cases += [(f"{args.te_mode}@{bp:g}bp",
               dict(te_bp=bp, te_mode=args.te_mode)) for bp in args.te_grid]

    for flag, vals in (("--loose-npz", args.loose_npz),
                       ("--crossover-npz", args.crossover_npz)):
        if vals and len(vals) != len(cases):
            raise SystemExit(
                f"{flag} takes one path per mandate case; got {len(vals)} "
                f"for {len(cases)} case(s) {[c for c, _ in cases]}")
    if args.loose_source == "npz" and not args.loose_npz:
        raise SystemExit("--loose-source npz requires --loose-npz")

    print(f"[1] Universe: {args.n_bonds} bonds (seed {args.seed})")
    uni = make_universe(n_bonds=args.n_bonds, seed=args.seed)

    print(f"[2] Scenarios: {args.n_scenarios:,} monthly excess-return paths")
    if not args.cpu_only:
        warmup_gpu(uni)
    R, mu, info = credit_scenarios(uni, n_scen=args.n_scenarios,
                                   seed=args.scen_seed,
                                   use_gpu=not args.cpu_only,
                                   return_mode="excess")
    R_oos, _, _ = credit_scenarios(uni, n_scen=args.n_scenarios,
                                   seed=args.oos_seed,
                                   use_gpu=not args.cpu_only,
                                   return_mode="excess")
    print(f"    {info['device']}: {info['seconds']:.3f}s; "
          f"evaluation set seed {args.oos_seed}")

    full_cols = args.n_bonds + args.n_scenarios + 1
    rows: list[dict] = []

    for case_idx, (case_name, te_kw) in enumerate(cases):
        limits = default_limits(uni, turnover=args.turnover,
                                active_l1=args.active_l1, **te_kw)
        conic = (limits.te_mode == "sigma" and np.isfinite(limits.te_bp))
        ws_limits, invisible = weight_space_limits(limits)
        _, _, repaired_names = build_polytope(uni, ws_limits)
        fam_rep = ",".join(repaired_names)
        fam_inv = ",".join(invisible) if invisible else "-"

        print(f"\n[3.{case_idx}] Case {case_name}: {limits.describe()}")
        print(f"    projection sees: {fam_rep}")
        print(f"    projection blind to: {fam_inv}"
              + ("" if invisible else "  (nothing: every limit is "
                                      "weight-space)"))

        model = build_mean_cvar(
            R, mu, alpha=args.alpha, risk_aversion=args.risk_aversion,
            w_max=limits.w_max,
            constraint_builder=credit_constraint_builder(uni, limits, R=R))
        full_nnz = float(model.nnz_estimate)

        def base(**kw) -> dict:
            return make_row(case=case_name, te_mode=limits.te_mode,
                            te_budget_bp=limits.te_bp,
                            families_repaired=fam_rep,
                            families_invisible=fam_inv,
                            te_visible_to_repair=not invisible,
                            full_lp_cols_est=float(full_cols),
                            full_lp_nnz_est=full_nnz, **kw)

        # ---- reference solves ----------------------------------------
        refs: dict[str, dict] = {}
        ref_rows: list[dict] = []
        # An SOC case has no LP basic solution, so the exact optimum it
        # degrades against is Clarabel's; an LP case has the simplex exact
        # reference.
        exact_key = "clarabel" if conic else "highs-simplex"
        for label, spec in REFERENCE_CONFIGS:
            if conic and not spec["conic"]:
                ref_rows.append(make_reference_unsupported_row(
                    base, portfolio=label, role=spec["role"],
                    method_class="LP", endpoint=(
                        "basic" if spec["kind"] == "highs-xover"
                        else "exact/reference"),
                    note="LP-only method cannot express the sigma TE SOC "
                         "constraint"))
                print(f"    {label:<16} UNSUPPORTED_BY_METHOD")
                continue
            if not conic and spec["kind"] == "clarabel":
                continue     # an LP has two exact references already
            if (args.loose_source == "cuopt-pdlp"
                    and spec["kind"] == "highs-xover"):
                continue     # published path uses cuOpt's own basic endpoint
            if args.crossover_npz and spec["kind"] == "highs-xover":
                # The CPU IPM+crossover row exists only as a stand-in for a
                # crossover the run could not perform. Once the GPU session
                # supplies the real one, running it too would put a second
                # "crossover" in the table and let the cost ratio be formed
                # against the wrong machine.
                continue
            try:
                w, st, secs, ts = solve_reference(
                    model, spec["kind"], args.threads, args.time_limit,
                    args.ref_runs)
            except Exception as e:  # noqa: BLE001
                ref_rows.append(make_reference_error_row(
                    base, portfolio=label, role=spec["role"],
                    method_class=("SOCP" if spec["conic"] else "LP"),
                    endpoint=("exact-conic" if spec["conic"]
                              else ("basic" if spec["kind"] == "highs-xover"
                                    else "exact/reference")),
                    error=e))
                print(f"    {label:<16} FAILED {type(e).__name__}")
                continue
            sc = score_portfolio(uni, w, limits, R, R_oos, mu, args.alpha,
                                 args.risk_aversion)
            if st in CONVERGED:
                refs[label] = dict(sc, w=w)
            ref_rows.append(base(
                portfolio=label, role=spec["role"], status=st,
                status_source="solver-returned",
                method_class=("SOCP" if spec["conic"] else "LP"),
                endpoint=("exact-conic" if spec["conic"]
                          else ("basic" if spec["kind"] == "highs-xover"
                                else "exact/reference")),
                wall_s=secs, wall_s_min=min(ts), wall_s_max=max(ts),
                steady_state_median_s=secs, total_endpoint_s=secs,
                n_timed_runs=float(len(ts)), wall_kind="solve", **sc))
            print(f"    {label:<16} {st:<9} {secs:7.3f}s  "
                  f"verdict={sc['gate_verdict']:<7} "
                  f"obj={sc['objective_in']:.6f} "
                  f"positions={sc['n_positions']}")

        if args.loose_source == "cuopt-pdlp":
            if conic:
                ref_rows.append(make_unavailable_endpoint_row(
                    base, portfolio="cuopt-pdlp+xover@0.0001",
                    role="reference-crossover", method_class="LP",
                    endpoint="basic", status="UNSUPPORTED_BY_METHOD",
                    status_source="author-classification", wall_kind="solve",
                    note="cuOpt PDLP basic endpoint cannot express an SOC "
                         "tracking-error constraint"))
            else:
                try:
                    _, basic_spec = cuopt_pdlp_specs_for_tols(
                        args.loose_tol, args.time_limit)
                    print(f"    {basic_spec.label:<16} direct cuOpt basic")
                    solved = solve_direct_endpoint(
                        model, basic_spec, args.ref_runs)
                    if solved.w is None:
                        ref_rows.append(make_unavailable_endpoint_row(
                            base, portfolio=basic_spec.label,
                            role="reference-crossover", method_class="LP",
                            endpoint="basic", status=solved.status,
                            status_source="solver-returned",
                            first_solve_s=solved.first_solve_s,
                            total_endpoint_s=solved.total_endpoint_s,
                            elapsed_is_lower_bound=(
                                solved.elapsed_is_lower_bound),
                            n_timed_runs=float(
                                len(solved.steady_state_runs_s)),
                            wall_kind="solve",
                            note="cuOpt PDLP basic endpoint produced no "
                                 "scoreable weight vector"))
                    else:
                        sc = score_portfolio(
                            uni, solved.w, limits, R, R_oos, mu, args.alpha,
                            args.risk_aversion)
                        objective_consistency_columns(
                            sc, solved.solver_objective, basic_spec)
                        refs["highs-ipm+xover"] = dict(sc, w=solved.w)
                        walls = solved.steady_state_runs_s
                        ref_rows.append(base(
                            portfolio=basic_spec.label,
                            role="reference-crossover", method_class="LP",
                            endpoint="basic", status=solved.status,
                            status_source="solver-returned",
                            wall_s=solved.steady_state_median_s,
                            wall_s_min=min(walls) if walls else np.nan,
                            wall_s_max=max(walls) if walls else np.nan,
                            first_solve_s=solved.first_solve_s,
                            steady_state_median_s=(
                                solved.steady_state_median_s),
                            total_endpoint_s=solved.total_endpoint_s,
                            elapsed_is_lower_bound=(
                                solved.elapsed_is_lower_bound),
                            n_timed_runs=float(len(walls)),
                            wall_kind="solve", **sc))
                        print(f"    {basic_spec.label:<16} {solved.status:<9} "
                              f"{solved.steady_state_median_s:7.3f}s  "
                              f"verdict={sc['gate_verdict']:<7} "
                              f"positions={sc['n_positions']}")
                except Exception as e:  # noqa: BLE001
                    ref_rows.append(make_unavailable_endpoint_row(
                        base, portfolio="cuopt-pdlp+xover@0.0001",
                        role="reference-crossover", method_class="LP",
                        endpoint="basic", status="UNAVAILABLE",
                        status_source="runner-build-error",
                        wall_kind="solve", note=str(e)[:200]))

        # A crossover/basic solution supplied by the GPU session overrides the
        # CPU stand-in, so the comparison is against cuOpt's own basic
        # endpoint.
        if args.crossover_npz:
            d = np.load(args.crossover_npz[case_idx])
            w = np.asarray(d["w"]).ravel()
            sc = score_portfolio(uni, w, limits, R, R_oos, mu, args.alpha,
                                 args.risk_aversion)
            refs["highs-ipm+xover"] = dict(sc, w=w)
            ref_rows.append(base(
                portfolio=str(d["label"]) if "label" in d else "xover-npz",
                role="reference-crossover", method_class="LP",
                endpoint="basic", status_source="supplied",
                status=str(d["status"]) if "status" in d else "supplied",
                wall_s=float(d["wall_s"]) if "wall_s" in d else np.nan,
                steady_state_median_s=(
                    float(d["wall_s"]) if "wall_s" in d else np.nan),
                total_endpoint_s=(
                    float(d["wall_s"]) if "wall_s" in d else np.nan),
                wall_kind="solve", note=f"supplied by "
                f"{args.crossover_npz[case_idx]}", **sc))

        for r in ref_rows:
            rows.append(add_degradation(r, refs, exact_key))

        # ---- the loose points ------------------------------------------
        loose_points = collect_loose_points(
            args, case_idx, case_name, conic, uni, limits, R, mu, refs,
            model=model)

        ctx = CaseContext(
            args=args, uni=uni, limits=limits, ws_limits=ws_limits,
            conic=conic, R=R, R_oos=R_oos, mu=mu, refs=refs,
            exact_key=exact_key, base=base, full_cols=full_cols,
            full_nnz=full_nnz)
        for loose in loose_points:
            if loose.get("w") is None:
                rows.append(make_unavailable_endpoint_row(
                    base, portfolio=loose["label"], role="loose",
                    endpoint=loose["endpoint"],
                    method_class=loose["method_class"],
                    status=loose["status"],
                    status_source=loose["status_source"],
                    note=loose["note"], loose_spec=loose["spec"],
                    wall_s=loose["wall_s"],
                    wall_s_min=loose["wall_s_min"],
                    wall_s_max=loose["wall_s_max"],
                    first_solve_s=loose["first_solve_s"],
                    steady_state_median_s=loose["steady_state_median_s"],
                    total_endpoint_s=loose["total_endpoint_s"],
                    elapsed_is_lower_bound=loose["elapsed_is_lower_bound"],
                    n_timed_runs=loose["n_timed_runs"],
                    wall_kind="solve"))
                print(f"    {loose['label']:<20} {loose['status']} - "
                      "no scoreable raw endpoint")
                continue
            sc_loose = score_portfolio(uni, loose["w"], limits, R, R_oos, mu,
                                       args.alpha, args.risk_aversion)
            if "solver_objective" in loose:
                objective_consistency_columns(
                    sc_loose, loose["solver_objective"],
                    loose.get("solver_spec"))
            rows.append(add_degradation(base(
                portfolio=loose["label"], role="loose",
                method_class=loose["method_class"],
                endpoint=loose["endpoint"], status=loose["status"],
                status_source=loose["status_source"], wall_s=loose["wall_s"],
                wall_s_min=loose["wall_s_min"],
                wall_s_max=loose["wall_s_max"],
                first_solve_s=loose["first_solve_s"],
                steady_state_median_s=loose["steady_state_median_s"],
                total_endpoint_s=loose["total_endpoint_s"],
                elapsed_is_lower_bound=loose["elapsed_is_lower_bound"],
                n_timed_runs=loose["n_timed_runs"],
                wall_kind="solve", note=loose["note"],
                loose_spec=loose["spec"], **sc_loose), refs, exact_key))
            print(f"    {loose['label']:<20} {loose['status']:<9} "
                  f"verdict={sc_loose['gate_verdict']} "
                  f"desk_worst={sc_loose['desk_worst_multiple']:.4g}x "
                  f"({sc_loose['desk_worst_limit']}) "
                  f"budget_resid={sc_loose['budget_residual']:.2e} "
                  f"positions={sc_loose['n_positions']}")
            rows.extend(repair_sweep(ctx, loose, sc_loose))

    df = pd.DataFrame(rows)

    print("\n[4] Result")
    show = ["case", "portfolio", "repaired_from", "status", "wall_s",
            "gate_verdict", "desk_worst_multiple", "desk_worst_limit",
            "budget_residual", "te_breach_bp", "d_obj_vs_xover_bp",
            "d_obj_vs_exact_bp", "repair_l1_moved", "names_moved_gt_1bp",
            "n_positions"]
    print(df[show].to_string(index=False,
                             float_format=lambda v: f"{v:,.4g}"))

    rep_df = df[df["role"] == "repair"]
    # Ratios are formed WITHIN a case. A repair from one mandate divided by a
    # crossover from another would be a headline made of two experiments.
    print("\n    Repair cost against the crossover it is supposed to "
          "replace, per case and loose point:")
    for case_name, _ in cases:
        sub = df[df["case"] == case_name]
        xo = sub[(sub["role"] == "reference-crossover")
                 & sub["wall_s"].notna()]
        if not len(xo):
            print(f"      {case_name}: no crossover/basic solution to compare with")
            continue
        t_xo = float(xo["wall_s"].iloc[0])
        for src in sub.loc[sub["role"] == "repair",
                           "repaired_from"].unique():
            reps = sub[(sub["role"] == "repair")
                       & (sub["repaired_from"] == src)
                       & sub["wall_s"].notna()]
            if not len(reps):
                continue
            best = reps.loc[reps["wall_s"].idxmin()]
            print(f"      {case_name} / {src}: {best['portfolio']} "
                  f"{best['wall_s'] * 1000:.1f}ms vs crossover "
                  f"{t_xo * 1000:.1f}ms -> "
                  f"{t_xo / max(float(best['wall_s']), 1e-12):,.0f}x cheaper")
    clean = rep_df[rep_df["gate_verdict"].isin(["OK", "DESK-OK"])]
    print(f"    {len(clean)}/{len(rep_df)} repaired portfolios pass the desk "
          f"gate on the FULL mandate")
    still = rep_df[rep_df["gate_verdict"] == "BREACH"]
    if len(still):
        print("    repairs that did NOT deliver a repaired-feasible endpoint "
              "(the limits the projection cannot see):")
        print(still[["case", "portfolio", "desk_worst_limit",
                     "desk_worst_multiple", "te_breach_bp"]]
              .to_string(index=False, float_format=lambda v: f"{v:,.4g}"))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        env = capture_environment()
        env["script"] = ("scripts/run_feasibility_repair.py "
                         "(M8 weight-space feasibility repair)")
        env["n_bonds"] = str(args.n_bonds)
        env["n_scenarios"] = str(args.n_scenarios)
        env["alpha"] = str(args.alpha)
        env["risk_aversion"] = str(args.risk_aversion)
        env["turnover"] = str(args.turnover)
        env["universe_seed"] = str(args.seed)
        env["scenario_seed"] = str(args.scen_seed)
        env["oos_seed"] = str(args.oos_seed)
        env["te_grid_bp"] = ",".join(f"{v:g}" for v in args.te_grid) or "none"
        env["te_mode"] = args.te_mode
        env["active_l1"] = str(args.active_l1)
        env["loose_source"] = args.loose_source
        env["loose_tol"] = ",".join(f"{v:g}" for v in args.loose_tol)
        env["loose_npz"] = ",".join(args.loose_npz) or "n/a"
        env["crossover_npz"] = ",".join(args.crossover_npz) or "n/a"
        env["perturb_target_mult"] = ",".join(
            f"{v:g}" for v in args.perturb_target_mult)
        env["perturb_seed"] = str(args.perturb_seed)
        env["repair_modes"] = ",".join(args.modes)
        env["repair_runs"] = str(args.repair_runs)
        env["ref_runs"] = str(args.ref_runs)
        env["lex_slack"] = str(args.lex_slack)
        env["threads"] = str(args.threads)
        env["time_limit_s"] = str(args.time_limit)
        env["scenario_device"] = info["device"]
        env["crossover_reference"] = (
            "supplied via --crossover-npz" if args.crossover_npz
            else "HiGHS IPM + crossover on CPU, a stand-in for cuOpt "
                 "crossover=1")
        env["name_move_gate"] = f"{NAME_MOVE_GATE:g} (1bp of weight)"
        env["objective_units"] = ("monthly excess return; *_bp columns are "
                                  "basis points of that")
        write_csv(args.out, df.to_dict("records"), env)
        print(f"\nWrote {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Constraint ablation: is crossover cost really driven by the constraint set?

THE QUESTION UNDER TEST
-----------------------
P0 asks a narrower question than the original cumulative ablation: does the
turnover sentence still explain the full-mandate crossover bill when L0,
full, and full-minus-turnover are regenerated as a paired experiment in the
same session? L0 is a control for closeness; the scientific contrast remains
full versus full-minus-turnover.

As stated in the old README that attribution was correlational, and weakly so.
The equity and credit problems differ in more than constraint count: different return
data, different conditioning, different degeneracy, a different benchmark, a
different incumbent. Any of those could produce the same inversion. Comparing
them and attributing the difference to constraint count is a single
uncontrolled comparison with several confounds riding along.

This script runs the controlled version. ONE universe, ONE scenario matrix,
ONE set of solver configurations, ONE tolerance. The only thing that varies is
which limit families are switched on, swept from the equity-equivalent
minimum (budget + box, which is exactly the equity problem's constraint set)
up to the full credit mandate:

    L0  budget + box only          <- the equity-equivalent case
    L1  + turnover budget
    L2  + rating bands
    L3  + sector caps
    L4  + issuer caps
    L5  + duration cells and the aggregate duration band
    L6  + DTS band (two-sided)     <- the full mandate

If crossover cost is governed by constraint complexity, the cuOpt+crossover
column should climb monotonically across the levels while plain PDLP stays
roughly flat, and the level at which cuOpt+crossover crosses HiGHS is the
answer to "how much mandate does it take to lose the GPU's advantage". If
instead crossover cost jumps at one family and ignores the rest, the
"more families of limits" story is wrong and the real driver is that one
family (turnover is the obvious suspect: it is ONE mandate statement but
canonicalizes to ~2N+1 LP rows and N auxiliary variables, more rows than
every other family combined).

WHAT THIS STILL DOES NOT CONTROL
--------------------------------
Adding a constraint family changes more than a row count. It moves the
optimal face, changes its dimension and degeneracy, and changes the row
conditioning of the canonicalized matrix. So this isolates constraint set as
the treatment on a fixed problem — which the equity-vs-credit comparison does
not — but it does not isolate "number of rows" from "geometry of the feasible
region". Two columns are recorded to help separate them: `mandate_rows` (the
count a PM would recognise: one statement per limit) and `lp_ineq_rows_added`
(what the solver actually receives after canonicalization). If cost tracks the
second and not the first, it is a linear-algebra story, not a mandate story.

THE CPU SIDE OF THE COMPARISON
------------------------------
Five configurations per level, not three: cuOpt PDLP with and without
crossover, HiGHS dual simplex, and HiGHS interior point with and without
crossover. The IPM pair is here because interior point is the CPU algorithm
class whose hardware story is genuinely ambiguous for this problem shape —
20-50 iterations, each dominated by one factorization of a normal-equations
system sized by the BOND count, dense and GEMM-shaped. Claiming a speedup
"against the best CPU option" while benchmarking only simplex and PDLP leaves
out the one CPU algorithm that might not lose.

`highs_ipm` and `highs_ipm+xover` are separated for the same reason the cuOpt
columns are: without crossover HiGHS stops at an interior point where every
weight is strictly positive, which is a dense portfolio, not a signable one.
`nonzero_weights` is the column that shows it — `positions` cannot, because on
a small CPU instance the interior point's extra weights sat at 1e-16, far
under the dust threshold, leaving `positions` identical to the vertex's.
Whether the extra weights are that small at full mandate size is unmeasured;
read `nonzero_weights` against `positions` per row rather than assuming.
Whether each row really ran the algorithm its label claims is recorded, per
row, in `algorithm_evidence`, read out of HiGHS's own iteration counters.

HOW TO READ THE OUTPUT (results/p0_ablation_loo.csv)
----------------------------------------------------
One row per (level, solver configuration). The columns that carry the argument:

    level, families            which limit families are on
    mandate_rows               limit statements, PM-level count
    lp_ineq_rows_added         canonical inequality rows added vs L0 (measured)
    wall_s_median              median of `--runs` timed solves, warmup discarded
    solver_internal_s          solver-reported time where exposed
    crossover_cost_s           (X+xover wall) - (X wall) at the same level, for
                               each solver X that has both variants
    crossover_multiple         the same, as a ratio
    algorithm_evidence         which algorithm the solver's own counters say
                               ran, and whether it ended on a valid basis
    nonzero_weights            strictly positive weights: the interior-vs-
                               vertex test, at any magnitude
    check_budget_resid         |sum(w) - 1| from the independent checker
    check_budget_signed        the same with its sign: >0 over-invested,
                               <0 under-invested. The magnitude gates; the
                               direction is what a reader is owed.
    max_bound_violation        worst box violation from the independent checker
    worst_rel_violation        worst breach of THIS level's limits (0 = clean)
    limits_verdict             "ok" or the limit that broke and by how much
    worst_rel_violation_full   the same portfolio scored against the FULL
                               mandate — at low levels this is expected to be
                               large, and it is the reason the constraints are
                               there in the first place

Read `crossover_cost_s` down the levels. That single column is the experiment.
`wall_s_median` for the fastest CPU configuration that ends on a valid
basis — highs-simplex or highs_ipm+xover, whichever wins the level — is the
line cuOpt+crossover has to stay under.

RUNNING IT (needs the GPU box — cuOpt and HiGHS both installed)
--------------------------------------------------------------
    python scripts/run_ablation.py --out results/p0_ablation_loo.csv

Defaults are the P0 working cell (500 bonds x 20,000 scenarios, alpha 0.95,
risk aversion 3.0, PDLP tolerance 1e-4, five timed repeats) and the paired
levels are L0_budget_box, L6_full_mandate and LOO_no_turnover. It checkpoints
after every configuration and `--resume` skips rows for the same session id.

The historical seven-level sweep is still available, but it must be requested
explicitly and should write to its historical artifact name:

    python scripts/run_ablation.py --levels all --out results/ablation.csv

    # cheaper first pass, to confirm the plumbing before spending the time
    python scripts/run_ablation.py --n-bonds 300 --n-scenarios 4000 \
        --runs 1 --out results/tmp/ablation_small.csv

Reuses the existing harness rather than reimplementing timing: process
isolation per solver configuration, warmup discard, median-of-N, independent
solution checking and the env-header CSV writer all come from src/bench.py.
The reduced limit sets are built here rather than in src/ because they are an
experiment, not a modelling feature — `reduce_limits()` only neutralizes
fields that `credit_constraint_builder` already treats as "family off"
(empty dict, infinite tolerance, zero DTS floor), so no modelling code
changes and no new code path is introduced into the thing being measured.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

MARKER = "RESULT_JSON:"

#: Cumulative limit-family levels, in the order a mandate accumulates them.
#: L0 is the equity-equivalent problem: budget + box + the CVaR tail block,
#: which is exactly what scripts/run_bench.py solves.
LEVELS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("L0_budget_box", ()),
    ("L1_turnover", ("turnover",)),
    ("L2_rating", ("turnover", "rating_bands")),
    ("L3_sector", ("turnover", "rating_bands", "sector_caps")),
    ("L4_issuer", ("turnover", "rating_bands", "sector_caps", "issuer_caps")),
    ("L5_duration", ("turnover", "rating_bands", "sector_caps", "issuer_caps",
                     "duration")),
    ("L6_full_mandate", ("turnover", "rating_bands", "sector_caps",
                         "issuer_caps", "duration", "dts")),
)

ALL_FAMILIES = LEVELS[-1][1]
LOO_LEVEL_NAME = "LOO_no_turnover"
P0_LOO_LEVELS: tuple[tuple[str, tuple[str, ...]], ...] = (
    LEVELS[0],
    LEVELS[-1],
    (LOO_LEVEL_NAME, tuple(f for f in ALL_FAMILIES if f != "turnover")),
)
P0_LEVEL_NAMES = tuple(name for name, _ in P0_LOO_LEVELS)


def default_output_path() -> str:
    """P0 writes a new artifact and never overwrites historical ablation.csv."""
    return "results/p0_ablation_loo.csv"


def select_levels(spec: str) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Resolve the public level selector.

    Empty means the P0 paired leave-one-out run.  The old cumulative sweep is
    still reachable as ``--levels all`` so historical reproduction stays one
    flag away without being the default write target.
    """
    token = (spec or "").strip()
    if not token or token.lower() in {"p0", "loo", "p0-loo"}:
        return P0_LOO_LEVELS
    if token.lower() in {"all", "legacy"}:
        return LEVELS

    known = {name: (name, fams) for name, fams in LEVELS}
    known[LOO_LEVEL_NAME] = P0_LOO_LEVELS[2]
    want = [s.strip() for s in token.split(",") if s.strip()]
    missing = [s for s in want if s not in known]
    if missing:
        raise ValueError(
            f"unknown level(s) {missing}; known: {sorted(known)}")
    return tuple(known[s] for s in want)


def is_legacy_level_run(levels: tuple[tuple[str, tuple[str, ...]], ...]) -> bool:
    """Historical cumulative sweep: preserve old resume rows as-is."""
    return tuple(name for name, _ in levels) == tuple(name for name, _ in LEVELS)


def level_design_id(levels: tuple[tuple[str, tuple[str, ...]], ...]) -> str:
    """Explicit identity for the chosen level design."""
    return "levels:" + "|".join(name for name, _ in levels)


def p0_pair_id(args) -> str:
    """Result-defining identity shared by both paired P0 levels."""
    return (
        f"p0-ablation-n{args.n_bonds}-s{args.n_scenarios}"
        f"-useed{args.seed}-sseed{args.scen_seed}"
        f"-alpha{args.alpha:g}-risk{args.risk_aversion:g}"
        f"-turnover{args.turnover:g}-tol{args.tol:g}"
    )


def ablation_session_id(
    args,
    levels: tuple[tuple[str, tuple[str, ...]], ...],
) -> str:
    """Data config plus level design; rows from different designs never mix."""
    return f"{p0_pair_id(args)}|{level_design_id(levels)}"


def resume_key_from_row(row: dict) -> tuple[str, str]:
    """The Task-4 checkpoint key remains scoped to (level, solver)."""
    return str(row.get("level", "")), str(row.get("solver", ""))


def _row_matches_identity(row: dict, identity: str) -> bool:
    if not identity:
        return True
    session = str(row.get("ablation_session_id", ""))
    if session:
        return session == identity
    return str(row.get("pair_id", "")) == identity


def completed_resume_keys(
    rows: list[dict],
    pair_id: str = "",
    legacy: bool = False,
) -> set[tuple[str, str]]:
    """Rows already valid for this exact paired run."""
    keys: set[tuple[str, str]] = set()
    for row in rows:
        if not legacy and not _row_matches_identity(row, pair_id):
            continue
        keys.add(resume_key_from_row(row))
    return keys


def resume_rows_for_current_pair(
    rows: list[dict],
    pair_id: str = "",
    legacy: bool = False,
) -> list[dict]:
    """Discard stale rows from a reused checkpoint file."""
    if legacy or not pair_id:
        return list(rows)
    return [row for row in rows if _row_matches_identity(row, pair_id)]


def _finite_float(value) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def loo_interpretation(
    l0_cost_s: float,
    loo_cost_s: float,
    full_cost_s: float,
    close_to_l0_rel: float = 0.20,
    close_to_full_rel: float = 0.10,
    material_drop_rel: float = 0.25,
) -> dict[str, object]:
    """Deterministic wording for the full-minus-turnover evidence.

    Dominance needs both halves of the planned test: removing turnover leaves
    the cost close to L0 and materially below the full mandate.  Without an L0
    row the helper may report a marginal reduction, but it will not claim
    dominance.
    """
    vals = [l0_cost_s, loo_cost_s, full_cost_s]
    l0, loo, full = (_finite_float(v) for v in vals)
    marginal = full - loo if math.isfinite(full) and math.isfinite(loo) else math.nan
    share = marginal / full if math.isfinite(marginal) and full > 0 else math.nan
    loo_over_full = loo / full if math.isfinite(loo) and full > 0 else math.nan
    loo_over_l0 = loo / l0 if math.isfinite(loo) and l0 > 0 else math.nan

    if not (math.isfinite(full) and math.isfinite(loo)):
        verdict = "not_supported"
        note = "missing paired full or no-turnover crossover cost"
    elif abs(full - loo) <= close_to_full_rel * max(abs(full), 1e-12):
        verdict = "not_supported"
        note = "no-turnover remains close to full; turnover dominance is not supported"
    elif math.isfinite(l0) and l0 >= 0:
        close_l0 = abs(loo - l0) <= close_to_l0_rel * max(abs(l0), 1.0)
        material = marginal >= material_drop_rel * max(abs(full), 1e-12)
        if close_l0 and material:
            verdict = "dominant"
            note = "no-turnover is close to L0 and materially below full"
        else:
            verdict = "partial"
            note = "turnover contributes, but the paired evidence is not dominant"
    else:
        verdict = "partial"
        note = "turnover contributes, but no same-session L0 row was measured"

    return {
        "turnover_marginal_cost_s": marginal,
        "turnover_marginal_share": share,
        "loo_over_full_cost": loo_over_full,
        "loo_over_l0_cost": loo_over_l0,
        "turnover_dominance_verdict": verdict,
        "turnover_dominance_note": note,
    }


def summarise_active_limits(rep) -> dict[str, object]:
    """Count finite in-force limit rows and near-binding rows."""
    live = rep[np.isfinite(rep["rel_slack"])]
    binding = live[live["binding"]]

    def _join(values) -> str:
        vals = sorted({str(v) for v in values if str(v)})
        return "|".join(vals) if vals else "-"

    return {
        "active_constraint_count": int(len(live)),
        "active_constraint_families": _join(live["family"]),
        "binding_constraint_count": int(len(binding)),
        "binding_constraint_families": _join(binding["family"]),
        "binding_constraints": _join(binding["item"]),
    }


def job_metadata(factory: str, kwargs: dict) -> dict[str, str]:
    """Classify a solver row by method and endpoint."""
    xover = bool(kwargs.get("crossover"))
    if factory == "cuopt_pdlp":
        return {
            "method_class": "LP first-order",
            "endpoint": "basic" if xover else "raw",
            "requested_endpoint": "basic" if xover else "raw",
        }
    if factory == "highs":
        return {
            "method_class": "simplex",
            "endpoint": "basic",
            "requested_endpoint": "basic",
        }
    if factory == "highs_ipm":
        return {
            "method_class": "interior-point",
            "endpoint": "basic" if xover else "raw",
            "requested_endpoint": "basic" if xover else "raw",
        }
    return {
        "method_class": factory,
        "endpoint": "raw",
        "requested_endpoint": "raw",
    }


def with_p0_annotations(rows: list[dict]) -> list[dict]:
    """Add paired leave-one-out accounting without changing raw measurements."""
    _derive_crossover_cost(rows)

    by_level_solver: dict[tuple[str, str], dict] = {}
    by_group_level_solver: dict[tuple[str, str, str, str], dict] = {}
    expected_design = level_design_id(P0_LOO_LEVELS)
    expected_levels = "|".join(P0_LEVEL_NAMES)
    for r in rows:
        level = str(r.get("level", ""))
        solver = str(r.get("solver", ""))
        by_level_solver[(level, solver)] = r
        session = str(r.get("ablation_session_id", ""))
        paired = str(r.get("paired_levels", ""))
        design = str(r.get("level_design_id", ""))
        if session and paired == expected_levels and design == expected_design:
            by_group_level_solver[(session, paired, level, solver)] = r

    for r in rows:
        r.setdefault("raw_wall_s_median", "")
        r.setdefault("basic_cost_s", "")
        r.setdefault("total_endpoint_s", "")
        r.setdefault("dnf_repeat_policy", "")
        if r.get("dnf") in (True, "True") or str(r.get("status", "")).lower() \
                not in {"optimal", "optimal_inaccurate"}:
            if r.get("dnf") in (True, "True"):
                r["dnf_repeat_policy"] = "one-cap-nonresult"
            continue

        wall = _finite_float(r.get("wall_s_median"))
        if not math.isfinite(wall):
            continue
        solver = str(r.get("solver", ""))
        endpoint = str(r.get("endpoint", ""))
        if endpoint == "raw":
            r["raw_wall_s_median"] = wall
            r["total_endpoint_s"] = wall
        elif endpoint == "basic" and solver.endswith("+xover"):
            level = str(r.get("level", ""))
            session = str(r.get("ablation_session_id", ""))
            paired = str(r.get("paired_levels", ""))
            design = str(r.get("level_design_id", ""))
            scoped = session and paired == expected_levels \
                and design == expected_design
            base = by_group_level_solver.get(
                (session, paired, level, solver.removesuffix("+xover")))
            if base is None and not scoped:
                base = by_level_solver.get((level, solver.removesuffix("+xover")))
            b = _finite_float(base.get("wall_s_median") if base else math.nan)
            if math.isfinite(b):
                r["raw_wall_s_median"] = b
                r["basic_cost_s"] = wall - b
            r["total_endpoint_s"] = wall
        elif endpoint == "basic":
            r["total_endpoint_s"] = wall

    for r in rows:
        session = str(r.get("ablation_session_id", ""))
        paired = str(r.get("paired_levels", ""))
        design = str(r.get("level_design_id", ""))

        def _cost(level: str, solver: str = "cuopt_pdlp+xover") -> float:
            if not (session and paired == expected_levels
                    and design == expected_design):
                return math.nan
            row = by_group_level_solver.get((session, paired, level, solver))
            return _finite_float(row.get("crossover_cost_s") if row else math.nan)

        interp = loo_interpretation(
            l0_cost_s=_cost("L0_budget_box"),
            loo_cost_s=_cost(LOO_LEVEL_NAME),
            full_cost_s=_cost("L6_full_mandate"),
        )
        for k, v in interp.items():
            if isinstance(v, float) and not math.isfinite(v):
                r[k] = ""
            else:
                r[k] = v
    return rows


# --------------------------------------------------------------------------
# Reduced limit sets
# --------------------------------------------------------------------------

def reduce_limits(full, families: tuple[str, ...]):
    """A copy of `full` with every family outside `families` switched off.

    Switched off means "set to the value `credit_constraint_builder` already
    interprets as absent" — an empty band dict, an infinite cap, a zero DTS
    floor — never a wide-but-finite bound. A wide finite bound is still a row
    in the LP, which is precisely the thing being counted here, so a sloppy
    "effectively infinite" value would silently destroy the experiment.

    `w_max` and `bench_duration` are carried through unchanged at every level:
    the box is part of the L0 baseline and must not move, and
    `bench_duration` is only read when the duration family is on.
    """
    unknown = set(families) - set(ALL_FAMILIES)
    if unknown:
        raise ValueError(f"unknown limit families: {sorted(unknown)}")

    off = {}
    if "rating_bands" not in families:
        off["rating_bands"] = {}
    if "sector_caps" not in families:
        off["sector_caps"] = {}
    if "issuer_caps" not in families:
        off["issuer_cap"] = np.inf
    if "duration" not in families:
        off["cell_duration_tol"] = np.inf
        off["total_duration_tol"] = np.inf
    if "dts" not in families:
        off["dts_max"] = np.inf
        off["dts_min"] = 0.0
    if "turnover" not in families:
        off["turnover"] = np.inf
    return replace(full, **off)


def mandate_rows(uni, families: tuple[str, ...]) -> int:
    """Limit statements as a PM would count them, not as the LP sees them.

    Deliberately counts turnover as ONE statement even though it canonicalizes
    to ~2N+1 rows. The gap between this column and `lp_ineq_rows_added` is
    informative, so the two must not be defined to agree.
    """
    from src.universe import MATURITY_CELL_NAMES
    df = uni.bonds
    n = 0
    if "rating_bands" in families:
        n += 2 * df["rating"].nunique()
    if "sector_caps" in families:
        n += df["sector"].nunique()
    if "issuer_caps" in families:
        n += df["issuer_id"].nunique()
    if "duration" in families:
        n += 2 * len(MATURITY_CELL_NAMES) + 2
    if "dts" in families:
        n += 2
    if "turnover" in families:
        n += 1
    return n


def measure_canonical_rows(uni, full_limits, alpha: float,
                           risk_aversion: float,
                           levels: tuple[tuple[str, tuple[str, ...]], ...] = LEVELS
                           ) -> tuple[dict, str]:
    """Canonical (rows, vars) per level, measured on a 2-scenario probe.

    The scenario block contributes the same rows and variables at every level,
    so the DIFFERENCE between levels measured on a tiny problem is exactly the
    difference on the real one — and canonicalizing the real 500x20,000 model
    seven extra times would cost more memory than the whole experiment.

    Returns ({level: {...}}, description-of-how-it-was-measured). Falls back to
    an empty dict if no solver chain resolves; the counts are documentation,
    not results, and must never take the run down with them.
    """
    import cvxpy as cp

    from src.credit import credit_constraint_builder
    from src.model import build_mean_cvar

    rng = np.random.default_rng(0)
    R = rng.normal(0.0, 0.01, size=(2, uni.n_bonds))
    mu = R.mean(axis=0)

    installed = set(cp.installed_solvers())
    chain = next((s for s in ("HIGHS", "CUOPT", "SCIPY", "CLARABEL")
                  if s in installed), None)
    if chain is None:
        return {}, "unavailable (no cvxpy solver chain installed)"

    out: dict[str, dict] = {}
    try:
        probe_levels = list(LEVELS[:1])
        for lv in levels:
            if lv[0] not in {name for name, _ in probe_levels}:
                probe_levels.append(lv)
        for name, families in probe_levels:
            lim = reduce_limits(full_limits, families)
            builder = (credit_constraint_builder(uni, lim)
                       if families else None)
            m = build_mean_cvar(R, mu, alpha=alpha,
                                risk_aversion=risk_aversion,
                                w_max=full_limits.w_max,
                                constraint_builder=builder)
            data, _, _ = m.problem.get_problem_data(chain)
            G = data.get("G")
            A = data.get("A")
            n_ineq = int(G.shape[0]) if G is not None else 0
            n_eq = int(A.shape[0]) if A is not None else 0
            n_var = int((G if G is not None else A).shape[1])
            out[name] = {"lp_ineq_rows": n_ineq, "lp_eq_rows": n_eq,
                         "lp_vars": n_var}
    except Exception as e:  # noqa: BLE001
        return {}, f"unavailable ({type(e).__name__}: {e})"

    base = out[LEVELS[0][0]]
    for v in out.values():
        v["lp_ineq_rows_added"] = v["lp_ineq_rows"] - base["lp_ineq_rows"]
        v["lp_eq_rows_added"] = v["lp_eq_rows"] - base["lp_eq_rows"]
        v["lp_vars_added"] = v["lp_vars"] - base["lp_vars"]
        v["canonical_rows_added"] = (
            v["lp_ineq_rows_added"] + v["lp_eq_rows_added"])
        v["auxiliary_variables_added"] = v["lp_vars_added"]
    return out, (f"cvxpy {chain} chain, 2-scenario probe; the scenario block "
                 f"is level-invariant so the ADDED counts are exact for the "
                 f"full-size problem, the absolute counts are not")


# --------------------------------------------------------------------------
# Worker mode — one solver configuration, one clean process
# --------------------------------------------------------------------------


def run_spec_one_cap(model, spec, n_runs: int = 5, solve_once=None):
    """Run warmup plus timed repeats, but record the first DNF immediately.

    The shared benchmark runner retries a failed warmup to separate startup
    cost from solver convergence. P0's runbook asks for a stricter accounting
    rule: a DNF/ERROR consumes one cap and becomes a non-result, not another
    timed attempt.
    """
    import cvxpy as cp

    from src.bench import BenchRecord, _probe_algorithm, _solve_once
    from src.check import check_solution

    solve_once = solve_once or _solve_once
    ok_statuses = {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}

    wall, internal, status = solve_once(model, spec)
    compile_s = float(getattr(model.problem, "compilation_time", None)
                      or math.nan)
    if status not in ok_statuses:
        return BenchRecord(
            spec_label=spec.label, hardware=spec.hardware,
            tolerance_desc=spec.tolerance_desc, threads=spec.threads,
            n_assets=model.n_assets, n_scenarios=model.n_scenarios,
            nnz_estimate=model.nnz_estimate, status=status, dnf=True,
            n_runs=0, wall_s_median=math.nan, wall_s_all=[wall],
            solver_internal_s_median=math.nan, compile_s=compile_s,
            objective=math.nan, check=None,
            config_full=spec.config_full,
            algorithm_evidence=_probe_algorithm(model, spec),
            notes=f"DNF on first attempt, no retry (status={status})",
        )

    walls: list[float] = []
    internals: list[float] = []
    statuses: list[str] = []
    for _ in range(n_runs):
        wall, internal, status = solve_once(model, spec)
        statuses.append(status)
        if status not in ok_statuses:
            return BenchRecord(
                spec_label=spec.label, hardware=spec.hardware,
                tolerance_desc=spec.tolerance_desc, threads=spec.threads,
                n_assets=model.n_assets, n_scenarios=model.n_scenarios,
                nnz_estimate=model.nnz_estimate, status=status, dnf=True,
                n_runs=len(walls), wall_s_median=math.nan,
                wall_s_all=walls + [wall],
                solver_internal_s_median=math.nan, compile_s=compile_s,
                objective=math.nan, check=None,
                config_full=spec.config_full,
                algorithm_evidence=_probe_algorithm(model, spec),
                notes=f"DNF on timed run, no retry (status={status})",
            )
        walls.append(wall)
        if not math.isnan(internal):
            internals.append(internal)

    evidence = _probe_algorithm(model, spec)
    weights = model.weights()
    objective = float(model.problem.value)
    chk = check_solution(
        w=weights, solver_objective=objective, R=model.R, mu=model.mu,
        alpha=model.alpha, risk_aversion=model.risk_aversion,
        w_max=model.w_max, feasibility_tol=spec.check_feasibility_tol,
        objective_tol=spec.check_objective_tol,
    )
    notes = "" if chk.ok else f"CHECK FAILED: {chk.notes}"
    n_inacc = sum(1 for st in statuses if st == cp.OPTIMAL_INACCURATE)
    if n_inacc:
        notes = (notes + "; " if notes else "") + \
            f"solver flagged inaccurate on {n_inacc}/{len(statuses)} runs"

    return BenchRecord(
        spec_label=spec.label, hardware=spec.hardware,
        tolerance_desc=spec.tolerance_desc, threads=spec.threads,
        n_assets=model.n_assets, n_scenarios=model.n_scenarios,
        nnz_estimate=model.nnz_estimate, status=status, dnf=False,
        n_runs=len(walls), wall_s_median=statistics.median(walls),
        wall_s_all=walls,
        solver_internal_s_median=(
            statistics.median(internals) if internals else math.nan
        ),
        compile_s=compile_s, objective=objective, check=chk,
        config_full=spec.config_full, algorithm_evidence=evidence,
        notes=notes,
    )


def worker_main(args) -> int:
    """Run one spec on one level and print RESULT_JSON.

    Same isolation rationale as scripts/bench_worker.py: highspy and cuOpt
    each want their own process, CUDA context and allocator state stay
    contained, and a crash costs one configuration rather than the run. This
    script spawns ITSELF rather than reusing bench_worker.py only because the
    ablation additionally needs the exposure report scored against two limit
    sets, which bench_worker.py has no reason to carry. The timing itself is
    src.bench.run_spec, unchanged.
    """
    from src.bench import SPEC_FACTORIES
    from src.credit import credit_constraint_builder
    from src.model import build_mean_cvar

    d = np.load(args.data)
    R, mu = d["R"], d["mu"]
    alpha = float(d["alpha"])
    risk_aversion = float(d["risk_aversion"])
    w_max = float(d["w_max"])

    with open(args.universe, "rb") as f:
        uni, level_limits, full_limits, families = pickle.load(f)

    builder = (credit_constraint_builder(uni, level_limits)
               if families else None)
    model = build_mean_cvar(R, mu, alpha=alpha, risk_aversion=risk_aversion,
                            w_max=w_max, constraint_builder=builder)

    spec = SPEC_FACTORIES[args.factory](**json.loads(args.kwargs))
    rec = run_spec_one_cap(model, spec, n_runs=args.runs)
    out = rec.row()
    out["wall_s_all"] = [round(w, 4) for w in rec.wall_s_all]
    out["max_bound_violation"] = (
        f"{rec.check.max_bound_violation:.2e}" if rec.check else "")

    if rec.dnf:
        out.update({"worst_rel_violation": "", "limits_verdict": "DNF",
                    "worst_rel_violation_full": "",
                    "worst_limit_full": ""})
        print(MARKER + json.dumps(out))
        return 0

    w = model.weights()
    viol, which, level_summary = _score_with_summary(uni, w, level_limits)
    viol_f, which_f, full_summary = _score_with_summary(uni, w, full_limits)
    out["worst_rel_violation"] = f"{viol:.3e}"
    out["limits_verdict"] = (
        "ok" if viol <= 1e-6 else f"VIOLATED {which} by {viol:.2%} of its scale")
    out["worst_rel_violation_full"] = f"{viol_f:.3e}"
    out["worst_limit_full"] = which_f if viol_f > 1e-6 else "-"
    out.update(level_summary)
    out.update({f"full_{k}": v for k, v in full_summary.items()})
    print(MARKER + json.dumps(out))
    return 0


def _score(uni, w, limits) -> tuple[float, str]:
    """Worst relative limit breach under `limits` (0.0 if clean).

    Rows for switched-off families carry infinite or NaN relative slack (an
    infinite cap minus a finite exposure, over an infinite scale) and are
    dropped before ranking — otherwise the minimum is a NaN artifact of a
    limit that is not in force at this level.
    """
    from src.credit import exposure_report, worst_violation

    rep = exposure_report(uni, w, limits)
    live = rep[np.isfinite(rep["rel_slack"])]
    if live.empty:
        return 0.0, "no limits in force at this level"
    return worst_violation(live)


def _score_with_summary(uni, w, limits) -> tuple[float, str, dict[str, object]]:
    from src.credit import exposure_report, worst_violation

    rep = exposure_report(uni, w, limits)
    live = rep[np.isfinite(rep["rel_slack"])]
    if live.empty:
        return 0.0, "no limits in force at this level", summarise_active_limits(rep)
    viol, which = worst_violation(live)
    return viol, which, summarise_active_limits(rep)


# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------

def worker_timeout_s(time_limit_s: float, runs: int, overhead_s: float = 300.0
                     ) -> float:
    """Parent-process envelope for the first attempt, not all repeats.

    The solver itself owns the per-attempt time limit. If that first attempt
    hangs or fails to respect the solver cap, the parent records one lower
    bound and moves on. Successful workers then run repeats after convergence;
    P0's large-cell expectation is that those solves are far below the cap.
    """
    _ = runs  # part of the public signature; repeats do not enlarge the cap.
    return float(time_limit_s) + float(overhead_s)


def run_worker(
    argv: list[str],
    timeout_s: float,
    run_subprocess=subprocess.run,
    perf_counter=time.perf_counter,
) -> tuple[dict | None, str]:
    cmd = [sys.executable, str(Path(__file__).resolve())] + argv
    t0 = perf_counter()
    try:
        proc = run_subprocess(cmd, capture_output=True, text=True,
                              timeout=timeout_s)
    except subprocess.TimeoutExpired:
        elapsed = perf_counter() - t0
        elapsed = max(float(elapsed), float(timeout_s))
        return {
            "solver": "",
            "hardware": "",
            "status": "worker_timeout",
            "status_source": "runner-timeout",
            "dnf": True,
            "n_runs": 0,
            "wall_s_median": "",
            "wall_s_min": "",
            "wall_s_max": "",
            "wall_s_all": f"[{elapsed:.1f}]",
            "elapsed_s": round(elapsed, 4),
            "lower_bound_s": round(elapsed, 4),
            "dnf_repeat_policy": "one-cap-nonresult",
            "limits_verdict": "DNF",
            "notes": (
                f"worker process timeout after {elapsed:.1f}s "
                f"(parent envelope {timeout_s:.1f}s)"
            ),
        }, ""
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(MARKER):
            return json.loads(line[len(MARKER):]), ""
    tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-6:]
    return None, f"worker exit {proc.returncode}: " + " | ".join(tail)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-bonds", type=int, default=500)
    ap.add_argument("--n-scenarios", type=int, default=20_000)
    ap.add_argument("--alpha", type=float, default=0.95)
    ap.add_argument("--risk-aversion", type=float, default=3.0)
    ap.add_argument("--turnover", type=float, default=0.25)
    ap.add_argument("--tol", type=float, default=1e-4,
                    help="PDLP tolerance, held FIXED across levels — this is "
                         "the experiment's control, not a knob to tune")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--time-limit", type=float, default=300.0)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--scen-seed", type=int, default=11)
    ap.add_argument("--levels", default="",
                    help="comma-separated level names; default is the P0 "
                         "paired L0_budget_box/L6_full_mandate/"
                         "LOO_no_turnover run; use 'all' for the historical "
                         "cumulative sweep")
    ap.add_argument("--allow-busy-gpu", action="store_true",
                    help="proceed even if another workload is using the GPU "
                         "(results will not be publishable)")
    ap.add_argument("--resume", action="store_true",
                    help="skip (level, solver) pairs already present in --out")
    ap.add_argument("--out", type=str, default=default_output_path())
    # Worker mode (internal): this script spawns itself for process isolation.
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--data", default="", help=argparse.SUPPRESS)
    ap.add_argument("--universe", default="", help=argparse.SUPPRESS)
    ap.add_argument("--factory", default="", help=argparse.SUPPRESS)
    ap.add_argument("--kwargs", default="{}", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker:
        return worker_main(args)

    from src.bench import assert_gpu_idle, capture_environment, write_csv
    from src.credit import default_limits
    from src.scenarios import credit_scenarios, warmup_gpu
    from src.universe import make_universe

    # Hours of solves are worth nothing if the card is busy. Check first.
    assert_gpu_idle(raise_on_busy=not args.allow_busy_gpu)

    try:
        levels = select_levels(args.levels)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1
    pair_id = p0_pair_id(args)
    design_id = level_design_id(levels)
    session_id = ablation_session_id(args, levels)
    legacy_resume = is_legacy_level_run(levels)

    print(f"Constraint ablation: {args.n_bonds} bonds x "
          f"{args.n_scenarios:,} scenarios, {len(levels)} levels, "
          f"{args.runs} timed runs per configuration", flush=True)

    # ---- one universe, one scenario matrix, for every level ---------------
    uni = make_universe(n_bonds=args.n_bonds, seed=args.seed)
    full_limits = default_limits(uni, turnover=args.turnover)
    print(f"  limits (L6): {full_limits.describe()}", flush=True)
    warmup_gpu(uni)
    R, mu, sinfo = credit_scenarios(uni, n_scen=args.n_scenarios,
                                    seed=args.scen_seed, use_gpu=True)
    print(f"  scenarios: {sinfo['seconds']:.2f}s on {sinfo['device']}; "
          f"matrix {R.nbytes / 1e9:.2f} GB", flush=True)

    shapes, shape_note = measure_canonical_rows(
        uni, full_limits, args.alpha, args.risk_aversion, levels=levels)
    print(f"  canonical row probe: {shape_note}", flush=True)
    for name, _fams in levels:
        s = shapes.get(name, {})
        if s:
            print(f"    {name:<18} +{s['lp_ineq_rows_added']:>6,} ineq rows, "
                  f"+{s['lp_vars_added']:>6,} vars", flush=True)

    tmp = Path(tempfile.mkdtemp(prefix="ablation_"))
    data_npz = str(tmp / "data.npz")
    np.savez(data_npz, R=R, mu=mu, alpha=args.alpha,
             risk_aversion=args.risk_aversion, w_max=full_limits.w_max)
    del R, mu   # the workers own the memory from here

    # ---- resume ----------------------------------------------------------
    rows: list[dict] = []
    done: set[tuple[str, str]] = set()
    if args.resume and Path(args.out).exists():
        import csv as _csv
        with open(args.out) as f:
            body = [ln for ln in f if not ln.startswith("#")]
        for r in _csv.DictReader(body):
            rows.append(dict(r))
        rows = resume_rows_for_current_pair(
            rows, pair_id=session_id, legacy=legacy_resume)
        done = completed_resume_keys(
            rows, pair_id=session_id, legacy=legacy_resume)
        print(f"  resuming: {len(done)} configuration(s) already measured",
              flush=True)

    jobs = [
        ("cuopt_pdlp", {"tol": args.tol, "time_limit_s": args.time_limit},
         "GPU"),
        ("cuopt_pdlp", {"tol": args.tol, "time_limit_s": args.time_limit,
                        "crossover": True}, "GPU"),
        ("highs", {"time_limit_s": args.time_limit,
                   "threads": args.threads}, "CPU"),
        # The interior-point CPU baseline, in the same two flavours as the GPU
        # column. Tolerances are left at the HiGHS defaults on purpose: this
        # is the vertex-accuracy branch of the comparison, and the ablation's
        # controlled variable is the constraint set, not the stopping rule.
        ("highs_ipm", {"time_limit_s": args.time_limit,
                       "threads": args.threads}, "CPU"),
        ("highs_ipm", {"time_limit_s": args.time_limit,
                       "threads": args.threads, "crossover": True}, "CPU"),
    ]

    t_start = time.perf_counter()
    for li, (name, families) in enumerate(levels, 1):
        lim = reduce_limits(full_limits, families)
        m_rows = mandate_rows(uni, families)
        print(f"\n[level {li}/{len(levels)}] {name}  "
              f"families={list(families) or ['(none)']}  "
              f"mandate_rows={m_rows}", flush=True)

        uni_pkl = str(tmp / f"uni_{name}.pkl")
        with open(uni_pkl, "wb") as f:
            pickle.dump((uni, lim, full_limits, families), f)

        for factory, kwargs, hardware in jobs:
            label = factory + ("+xover" if kwargs.get("crossover") else "")
            meta = job_metadata(factory, kwargs)
            if (name, label) in done:
                print(f"  {label:<22}already measured", flush=True)
                continue
            print(f"  {label:<22}", end="", flush=True)
            budget = worker_timeout_s(args.time_limit, args.runs)
            rec, diag = run_worker(
                ["--worker", "--data", data_npz, "--universe", uni_pkl,
                 "--factory", factory, "--kwargs", json.dumps(kwargs),
                 "--runs", str(args.runs)],
                timeout_s=budget)
            if rec is None:
                rec = {"solver": label, "hardware": hardware,
                       "status": "worker_failure", "dnf": True,
                       "status_source": "runner-worker-error",
                       "elapsed_s": "",
                       "lower_bound_s": "",
                       "wall_s_all": "",
                       "wall_s_median": "", "limits_verdict": "DNF",
                       "notes": diag}
                print(f"DNF ({diag[:60]})", flush=True)
            else:
                rec["solver"] = label
                mark = ("DNF" if rec["dnf"]
                        else f"{float(rec['wall_s_median']):.2f}s")
                print(f"{mark}   limits={rec.get('limits_verdict', '')[:44]}",
                      flush=True)
            rec.update({
                "level": name,
                "families": "|".join(families) or "(none)",
                "n_families": len(families),
                "mandate_rows": m_rows,
                "n_bonds": args.n_bonds, "n_scenarios": args.n_scenarios,
                "nnz": args.n_scenarios * (args.n_bonds + 2),
                "scenario_gen_s": round(sinfo["seconds"], 3),
                "pair_id": pair_id,
                "level_design_id": design_id,
                "ablation_session_id": session_id,
                "paired_levels": "|".join(n for n, _ in levels),
                "same_session_pair": set(n for n, _ in levels)
                == {n for n, _ in P0_LOO_LEVELS},
                "p0_default_run": levels == P0_LOO_LEVELS,
                "pdlp_tolerance": args.tol,
                "runs_requested": args.runs,
                "time_limit_s": args.time_limit,
                "dnf_repeat_policy": (
                    "one-cap-nonresult" if rec.get("dnf") is True else ""),
            })
            rec.update(meta)
            rec.update({k: v for k, v in shapes.get(name, {}).items()})
            rows.append(rec)
            # Checkpoint after every configuration: this run is long, and a
            # crash in level 5 must not cost levels 1-4.
            _write(args, rows, sinfo, shape_note)

        Path(uni_pkl).unlink(missing_ok=True)

    Path(data_npz).unlink(missing_ok=True)
    with_p0_annotations(rows)
    _write(args, rows, sinfo, shape_note)
    _print_table(rows)
    print(f"\nAblation finished in "
          f"{(time.perf_counter() - t_start) / 60:.1f} min")
    print(f"Wrote {args.out}")
    return 0


def _crossover_scope_key(row: dict) -> tuple[str, ...]:
    """Scope raw/xover pairing.

    P0 rows carry explicit session and level-design identity. Historical
    ablation rows do not, so they fall back to the old level-only grouping.
    Partially identified rows get their own scope instead of silently falling
    into legacy and contaminating old artifacts.
    """
    session = str(row.get("ablation_session_id", ""))
    design = str(row.get("level_design_id", ""))
    paired = str(row.get("paired_levels", ""))
    level = str(row.get("level", ""))
    if session and design and paired:
        return ("scoped", session, design, paired, level)
    if not (session or design or paired):
        return ("legacy", level)
    return ("partial", session, design, paired, level)


def _derive_crossover_cost(rows: list[dict]) -> None:
    """Fill crossover_cost_s / crossover_multiple per level, per solver family.

    Computed for EVERY "<solver>" / "<solver>+xover" pair present, not just
    the GPU one. The same subtraction asked of cuOpt PDLP and of HiGHS
    interior point is the same question — what does a signable vertex cost on
    top of an interior point — put to two algorithm classes on two kinds of
    hardware, and reporting it for only one of them answers half of it.

    Left blank wherever either side is a DNF — an incomparable pair produces
    no number rather than a misleading one.
    """
    suffix = "+xover"
    by_scope: dict[tuple[str, ...], dict[str, dict]] = {}
    for r in rows:
        by_scope.setdefault(_crossover_scope_key(r), {})[
            str(r.get("solver", ""))] = r
    for lvl in by_scope.values():
        for r in lvl.values():
            r.setdefault("crossover_cost_s", "")
            r.setdefault("crossover_multiple", "")
        for name, xo in lvl.items():
            if not name.endswith(suffix):
                continue
            base = lvl.get(name[: -len(suffix)])
            if not base:
                continue
            try:
                b = float(base["wall_s_median"])
                x = float(xo["wall_s_median"])
            except (TypeError, ValueError, KeyError):
                continue
            if (base.get("dnf") in (True, "True")
                    or xo.get("dnf") in (True, "True")):
                continue
            xo["crossover_cost_s"] = round(x - b, 4)
            xo["crossover_multiple"] = round(x / b, 3) if b > 0 else ""


def _print_table(rows: list[dict]) -> None:
    print("\n" + "=" * 100)
    print(f"{'level':<18} {'solver':<22} {'wall_med(s)':>11} "
          f"{'xover_cost':>10} {'mand_rows':>9} {'limits':<20}")
    print("-" * 100)
    for r in rows:
        print(f"{str(r.get('level', '')):<18} {str(r.get('solver', '')):<22} "
              f"{str(r.get('wall_s_median', '')):>11} "
              f"{str(r.get('crossover_cost_s', '')):>10} "
              f"{str(r.get('mandate_rows', '')):>9} "
              f"{str(r.get('limits_verdict', ''))[:20]:<20}")
    print("=" * 100)


def _write(args, rows: list[dict], sinfo: dict, shape_note: str) -> None:
    from src.bench import capture_environment, write_csv

    with_p0_annotations(rows)
    env = capture_environment()
    env["experiment"] = ("constraint ablation on ONE credit problem: the "
                         "limit families are the only thing that varies")
    env["hypothesis"] = ("crossover cost is driven by the size/complexity of "
                         "the constraint set, not by credit-vs-equity data")
    try:
        levels = select_levels(args.levels)
    except ValueError:
        levels = LEVELS
    env["levels"] = ", ".join(n for n, _ in levels)
    env["level_selector"] = args.levels or "p0-loo-default"
    env["default_output"] = default_output_path()
    env["pair_id"] = p0_pair_id(args)
    env["level_design_id"] = level_design_id(levels)
    env["ablation_session_id"] = ablation_session_id(args, levels)
    env["n_bonds"] = str(args.n_bonds)
    env["n_scenarios"] = str(args.n_scenarios)
    env["pdlp_tolerance"] = str(args.tol)
    env["alpha"] = str(args.alpha)
    env["risk_aversion"] = str(args.risk_aversion)
    env["turnover_one_way"] = str(args.turnover)
    env["universe_seed"] = str(args.seed)
    env["scenario_seed"] = str(args.scen_seed)
    env["scenario_device"] = str(sinfo.get("device", ""))
    env["threads_cpu_solvers"] = str(args.threads)
    env["time_limit_s"] = str(args.time_limit)
    env["runs_per_config"] = str(args.runs)
    env["row_count_probe"] = shape_note
    env["isolation"] = "one clean subprocess per (level, solver configuration)"
    env["dnf_policy"] = "DNF/ERROR is recorded once and not repeated"
    env["cpu_baselines"] = (
        "HiGHS dual simplex plus HiGHS interior point with and without "
        "crossover, all at HiGHS defaults; the algorithm_evidence column "
        "records what each row's own solver counters say actually ran, so an "
        "IPM label is checkable rather than asserted")
    env["controls"] = ("identical universe, scenario matrix, tolerance, "
                       "solver configs and box across levels; only the limit "
                       "families change")
    env["not_controlled"] = ("adding a family also changes the optimal face, "
                             "its degeneracy and the row conditioning — this "
                             "isolates the constraint SET, not row count "
                             "alone")

    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    norm = [{k: r.get(k, "") for k in keys} for r in rows]
    for r in norm:
        if isinstance(r.get("wall_s_all"), list):
            r["wall_s_all"] = json.dumps(r["wall_s_all"])
    if norm:
        write_csv(args.out, norm, env)


if __name__ == "__main__":
    sys.exit(main())

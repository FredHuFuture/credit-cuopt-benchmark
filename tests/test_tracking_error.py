"""The tracking-error family, as assertions rather than printed output.

Four claims `src/credit.py` and `scripts/run_te_mandate.py` make about family
7. All CPU-only, no GPU, no cuOpt, small enough to run in a few seconds.

The fourth is the one that matters: the LP-representable MAD form of a
tracking-error mandate is NOT the mandate, and the gap is not a rounding
error — the optimizer actively widens it, because minimizing mean absolute
deviation over fat-tailed active returns is cheapest exactly where the
standard deviation is worst. Any claim that a Mean-CVaR LP "respects a 10bp
tracking-error mandate" has to say which of the two numbers it means.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import cvxpy as cp  # noqa: E402

from src.credit import (  # noqa: E402
    MAD_TO_SIGMA, credit_constraint_builder, default_limits,
    exposure_report, tracking_error_bp, worst_violation,
)
from src.model import build_mean_cvar  # noqa: E402
from src.scenarios import credit_scenarios  # noqa: E402
from src.universe import make_universe  # noqa: E402

N_BONDS, N_SCEN = 80, 800
ALPHA, RISK_AVERSION = 0.95, 3.0


@pytest.fixture(scope="module")
def problem_data():
    uni = make_universe(n_bonds=N_BONDS, seed=7)
    R, mu, _ = credit_scenarios(uni, n_scen=N_SCEN, seed=11, use_gpu=False,
                                return_mode="excess")
    return uni, R, mu


def _solve(uni, R, mu, limits):
    model = build_mean_cvar(
        R, mu, alpha=ALPHA, risk_aversion=RISK_AVERSION, w_max=limits.w_max,
        constraint_builder=credit_constraint_builder(uni, limits, R=R))
    model.problem.solve(solver=cp.CLARABEL, verbose=False, warm_start=False)
    assert model.problem.status == "optimal", model.problem.status
    return model


def test_recomputation_matches_definition(problem_data):
    """`tracking_error_bp` is the textbook formula, not an LP by-product."""
    uni, R, _ = problem_data
    w = uni.w0
    te = tracking_error_bp(uni, w, R, periods_per_year=12.0)

    a = w - uni.w_benchmark
    A = R @ a
    expect_sigma = A.std(ddof=0) * np.sqrt(12.0) * 1e4
    expect_mad = np.abs(A - A.mean()).mean() * MAD_TO_SIGMA * np.sqrt(12.0) * 1e4

    assert te["te_sigma_bp"] == pytest.approx(expect_sigma, rel=1e-12)
    assert te["te_mad_implied_bp"] == pytest.approx(expect_mad, rel=1e-12)
    # The benchmark tracks itself perfectly, by definition.
    assert tracking_error_bp(uni, uni.w_benchmark, R)["te_sigma_bp"] == \
        pytest.approx(0.0, abs=1e-9)


def test_sigma_mode_is_a_cone_and_mad_mode_is_not(problem_data):
    """The mandate's FORM decides the problem class, which decides whether
    cuOpt's PDLP can solve it at all."""
    uni, R, mu = problem_data
    dims = {}
    for mode in ("mad", "sigma"):
        limits = default_limits(uni, te_bp=10.0, te_mode=mode)
        model = build_mean_cvar(
            R, mu, alpha=ALPHA, risk_aversion=RISK_AVERSION,
            w_max=limits.w_max,
            constraint_builder=credit_constraint_builder(uni, limits, R=R))
        data = model.problem.get_problem_data(cp.CLARABEL)[0]
        dims[mode] = data["dims"]

    assert list(dims["mad"].soc) == [], \
        "mad mode must stay in the LP class for PDLP to apply"
    assert list(dims["sigma"].soc) == [N_SCEN + 1], \
        "sigma mode must produce exactly one SOC of dimension S+1"


@pytest.mark.parametrize("mode,budget_bp", [("mad", 10.0), ("sigma", 10.0)])
def test_constraint_binds_at_its_own_budget(problem_data, mode, budget_bp):
    """Each mode is enforced in the units it is written in."""
    uni, R, mu = problem_data
    limits = default_limits(uni, te_bp=budget_bp, te_mode=mode)
    model = _solve(uni, R, mu, limits)
    te = tracking_error_bp(uni, model.weights(), R, limits.periods_per_year)
    scored = te["te_mad_implied_bp"] if mode == "mad" else te["te_sigma_bp"]
    # Binding from below only: a looser outcome is legitimate (the budget may
    # simply not bind), an exceeded one is a formulation error.
    assert scored <= budget_bp * (1 + 1e-6), \
        f"{mode} mode returned {scored:.3f}bp against a {budget_bp}bp budget"


def test_mad_budget_does_not_deliver_the_sigma_mandate(problem_data):
    """The load-bearing negative result.

    Constrain the LP-representable MAD form at 10bp and the portfolio's
    ACTUAL ex-ante tracking error — the quantity Bloomberg 34100 writes down
    — lands far above 10bp. This is a modelling gap, not a solver-tolerance
    gap: it reproduces with an exact interior-point solve at default
    tolerances.
    """
    uni, R, mu = problem_data
    budget = 10.0
    limits = default_limits(uni, te_bp=budget, te_mode="mad")
    model = _solve(uni, R, mu, limits)
    te = tracking_error_bp(uni, model.weights(), R, limits.periods_per_year)

    assert te["te_mad_implied_bp"] <= budget * (1 + 1e-6)
    # The optimizer does not merely fail to deliver sigma <= 10bp; it lands
    # well outside, because minimizing MAD over fat-tailed active returns is
    # cheapest where sigma is worst. 1.5x is a deliberately conservative
    # floor - the observed ratio on the M6 run was ~3x.
    assert te["te_sigma_bp"] > 1.5 * budget, (
        f"expected the MAD surrogate to badly understate sigma-TE; got "
        f"sigma={te['te_sigma_bp']:.2f}bp vs budget {budget}bp")


def test_exposure_report_scores_the_constrained_mode_only(problem_data):
    """`worst_violation` must not fail a solve on a basis the solver was
    never given. Under mad mode the sigma row is reported but unconstrained."""
    uni, R, mu = problem_data
    limits = default_limits(uni, te_bp=10.0, te_mode="mad")
    model = _solve(uni, R, mu, limits)
    rep = exposure_report(uni, model.weights(), limits, R=R)

    te_rows = rep[rep["family"] == "tracking_error"]
    assert len(te_rows) == 3, "expected sigma, MAD-implied and active-L1 rows"
    assert (te_rows["limit"] == "unconstrained").sum() == 2, \
        "under mad mode only the MAD row carries a finite budget"
    viol, _ = worst_violation(rep)
    assert viol <= 1e-6, "a feasible solve must not register a violation"


def test_te_family_is_off_by_default(problem_data):
    """Every committed M1-M3 result must be unaffected by this addition."""
    uni, _, _ = problem_data
    limits = default_limits(uni)
    assert not np.isfinite(limits.te_bp)
    assert not np.isfinite(limits.active_l1)
    # ...and the builder must not demand a scenario matrix it does not need.
    build = credit_constraint_builder(uni, limits)
    cons, names = build(cp.Variable(uni.n_bonds, nonneg=True))
    assert not any("tracking_error" in n for n in names)

"""The M7 QP assembly, as assertions rather than trust.

`scripts/run_qp_branch.py` compares a dense-covariance QP on the GPU against
the same QP on a CPU interior-point solver against the low-rank-plus-diagonal
factored formulation. That comparison is only worth reading if three things
are true, and none of them is obvious from looking at the code:

1. **The quadratic form is symmetric and positive semi-definite.** The dense
   arm uses `cp.psd_wrap`, which tells CVXPY to SKIP the convexity check —
   necessary, because that check is an O(N^3) eigendecomposition that would
   cost more than every solve in the sweep combined at N = 10,000. Skipping a
   check is only safe if something else performs it. This file is that
   something else.

2. **The factored and dense formulations are the same problem.** Under the
   default `--sigma-basis factor` the dense arms receive
   `B diag(f) B' + diag(d)` materialized, and the factored arms receive the
   three pieces unmultiplied. If those two disagree, the headline three-way
   comparison is measuring a modelling difference while claiming to measure a
   representation difference.

3. **The QP is handed the SAME limit sheet the LP branch solves against.**
   This is the experiment's whole validity condition. The entire premise of
   M7 is "one universe, one mandate, only the objective changes" — if the QP
   quietly gets a different or smaller constraint set, every comparison to
   the committed M1-M3 credit results is void and the mandate-compliance
   headline is meaningless. So it is tested explicitly, constraint by
   constraint, by residual and not just by name.

Plus the Q-convention calibration constants, verified against numpy rather
than against the arithmetic in the docstring that produced them. A silently
doubled risk aversion is the failure mode this whole study is most exposed
to: it does not crash, it does not warn, and it attaches every timing and
every mandate verdict to a problem nobody asked for.

CPU-only, no solver, no GPU, no cuOpt. Nothing here calls `.solve()`.

    pytest tests/test_qp_model.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import cvxpy as cp  # noqa: E402

from run_qp_branch import (  # noqa: E402
    CALIBRATION_LAMBDA, CALIBRATION_MU, CALIBRATION_OBJ_IF_DOUBLED,
    CALIBRATION_OBJ_STAR, CALIBRATION_P_STAR, CALIBRATION_S,
    FACTOR_EIG_FLOOR, analytic_exposures, build_qp_dense, build_qp_factored,
    factor_fit_diagnostics, fit_factor_model,
)
from src.credit import credit_constraint_builder, default_limits  # noqa: E402
from src.model import build_mean_cvar  # noqa: E402
from src.scenarios import CreditScenarioConfig, credit_scenarios  # noqa: E402
from src.universe import make_universe  # noqa: E402

N_BONDS, N_SCEN = 150, 1_200
RISK_AVERSION = 3.0


@pytest.fixture(scope="module")
def problem_data():
    uni = make_universe(n_bonds=N_BONDS, seed=7)
    R, mu, _ = credit_scenarios(uni, n_scen=N_SCEN, seed=11, use_gpu=False,
                                return_mode="excess")
    fm = fit_factor_model(uni, R)
    limits = default_limits(uni, turnover=0.25)
    return uni, R, mu, fm, limits


def _random_weights(n: int, seed: int = 5, k: int = 8) -> np.ndarray:
    """A few budget-feasible weight vectors. Feasibility against the full
    mandate is NOT required: these probe algebraic identities, which must hold
    everywhere, not only on the feasible set."""
    rng = np.random.default_rng(seed)
    return rng.dirichlet(np.ones(n), k)


# --------------------------------------------------------------------------
# 1. The quadratic form
# --------------------------------------------------------------------------

def test_sigma_is_exactly_symmetric(problem_data):
    """Not symmetric to floating-point noise — exactly symmetric.

    cuOpt internally forms Q + Q^T for the Hessian. A matrix that is symmetric
    only to 1e-17 would therefore be handed to the GPU as a marginally
    different problem than the CPU arms receive, which is precisely the kind
    of difference a benchmark must not contain.
    """
    _, _, _, fm, _ = problem_data
    Sigma = fm.dense()
    assert np.array_equal(Sigma, Sigma.T), \
        "Sigma must be bit-identical to its transpose, not merely close"


def test_sigma_is_positive_semi_definite(problem_data):
    """The check `cp.psd_wrap` skips in the hot path, performed here instead.

    Positive DEFINITE, in fact: `fit_factor_model` floors the specific
    variances above zero on purpose, so the optimum is a point rather than a
    face and the arms can be compared on their weight vectors as well as on
    their objectives.
    """
    _, _, _, fm, _ = problem_data
    Sigma = fm.dense()
    ev = np.linalg.eigvalsh(Sigma)
    assert ev.min() > 0.0, f"Sigma is not positive definite: min eig {ev.min():.3e}"
    # And no negative eigenvalue hiding at the noise floor either.
    assert ev.min() > -1e-14 * max(ev.max(), 1.0)

    assert np.all(fm.f >= 0.0), "factor variances must be nonnegative"
    assert np.all(fm.d > 0.0), "specific variances must be strictly positive"


def test_zero_variance_directions_are_dropped_exactly(problem_data):
    """The analytic exposure matrix is rank-deficient by construction.

    The rating dummy columns sum to the systematic column, and so do the
    sector columns, so `B` has exactly two redundant directions. They are
    removed after the eigendecomposition rather than by hand-dropping a
    reference category, which keeps the reported factor count honest AND is
    exact: a direction carrying zero variance contributes nothing to Sigma.
    """
    uni, R, _, fm, _ = problem_data
    B_raw, labels = analytic_exposures(uni)
    assert B_raw.shape[1] == len(labels)
    assert fm.n_dropped >= 1, \
        "expected the collinear rating/sector blocks to produce null directions"
    assert fm.n_factors == B_raw.shape[1] - fm.n_dropped
    assert fm.f.min() > FACTOR_EIG_FLOOR * fm.f.max()
    # Sorted descending, so a reader can truncate the model by eye.
    assert np.all(np.diff(fm.f) <= 0.0)


def test_analytic_exposures_match_the_scenario_generator(problem_data):
    """B is read off `src/scenarios.py`, not fitted.

    The generator builds `ds_rel = (credit_sys + rating + sector + idio) *
    hy_mult` and the return as `carry - SD * OAS * ds_rel + convexity`, so the
    loading on every systematic block is `-SD * OAS * hy_mult`. If that
    identity ever drifts, the factor model stops being the risk model and
    becomes an unmotivated compression.
    """
    uni, _, _, _, _ = problem_data
    B, labels = analytic_exposures(uni)
    cfg = CreditScenarioConfig()
    ratings = uni.bonds["rating"].to_numpy()
    hy = np.where(np.isin(ratings, ("BB", "B")), cfg.hy_vol_mult, 1.0)
    beta = -uni.spread_duration * uni.oas * hy

    assert labels[0] == "credit_sys"
    np.testing.assert_allclose(B[:, 0], beta, rtol=0, atol=0)

    # Each rating block carries the same beta, on that rating's bonds only.
    for j, lab in enumerate(labels):
        if not lab.startswith("rating:"):
            continue
        m = ratings == lab.split(":", 1)[1]
        np.testing.assert_allclose(B[m, j], beta[m], rtol=0, atol=0)
        assert np.all(B[~m, j] == 0.0)


# --------------------------------------------------------------------------
# 2. Factored == dense
# --------------------------------------------------------------------------

def test_factored_quadratic_matches_dense_numerically(problem_data):
    """w'(B diag(f) B' + diag(d))w computed both ways, on many portfolios."""
    _, _, _, fm, _ = problem_data
    Sigma = fm.dense()
    for w in _random_weights(fm.n_assets):
        dense_q = float(w @ (Sigma @ w))
        fact_q = fm.portfolio_variance(w)
        assert fact_q == pytest.approx(dense_q, rel=1e-11, abs=1e-24)


def test_factored_and_dense_cvxpy_objectives_agree(problem_data):
    """The identity has to survive the CVXPY assembly, not just my numpy.

    Both models are built through the functions the experiment actually calls,
    the same weight vector is written into both, and the OBJECTIVE EXPRESSIONS
    are evaluated. No solver runs: this is about whether the two builders
    encode the same function.
    """
    uni, _, mu, fm, limits = problem_data
    Sigma = fm.dense()
    builder = credit_constraint_builder(uni, limits, R=None)

    m_dense, _ = build_qp_dense(Sigma, mu, RISK_AVERSION, limits.w_max,
                                constraint_builder=builder)
    m_fact, _ = build_qp_factored(fm, mu, RISK_AVERSION, limits.w_max,
                                  constraint_builder=builder)

    y = next(v for v in m_fact.problem.variables() if v.name() == "y")
    for w in _random_weights(fm.n_assets, seed=17, k=5):
        m_dense.w.value = w
        m_fact.w.value = w
        # y is a free variable pinned by the factor-definition equality; at a
        # solution it equals B'w, so that is what it is set to here.
        y.value = fm.B.T @ w
        v_dense = float(m_dense.problem.objective.value)
        v_fact = float(m_fact.problem.objective.value)
        assert v_fact == pytest.approx(v_dense, rel=1e-10, abs=1e-18)
        # ...and both must agree with the model's own recomputation, which is
        # what `check_qp` uses to score every row in the CSV.
        assert m_dense.objective_value(w) == pytest.approx(v_dense, rel=1e-10)
        assert m_fact.objective_value(w) == pytest.approx(v_fact, rel=1e-10)


def test_factored_hessian_is_strictly_diagonal(problem_data):
    """The property the whole factored branch turns on.

    With F diagonalized into B and D diagonal, substituting y = B'w makes the
    Hessian in (w, y) strictly diagonal. That is not a cosmetic win: a
    diagonal Hessian is the only quadratic objective OR-Tools' PDLP accepts,
    so the factored MVO is natively solvable by a first-order method while the
    dense one is solvable by no PDLP at all, cuOpt's included.

    Canonicalization only — `get_problem_data` does not solve.
    """
    uni, _, mu, fm, limits = problem_data
    builder = credit_constraint_builder(uni, limits, R=None)
    installed = set(cp.installed_solvers())
    chain = next((s for s in ("CLARABEL", "SCS", "OSQP") if s in installed),
                 None)
    if chain is None:
        pytest.skip("no QP-capable cvxpy chain installed to canonicalize with")

    m_fact, _ = build_qp_factored(fm, mu, RISK_AVERSION, limits.w_max,
                                  constraint_builder=builder)
    data = m_fact.problem.get_problem_data(chain)[0]
    P = data.get("P")
    assert P is not None, f"{chain} chain did not receive a quadratic objective"
    P = P.tocoo()
    assert np.array_equal(P.row, P.col), \
        "the factored Hessian must have no off-diagonal nonzeros"

    m_dense, _ = build_qp_dense(fm.dense(), mu, RISK_AVERSION, limits.w_max,
                                constraint_builder=builder)
    Pd = m_dense.problem.get_problem_data(chain)[0].get("P")
    assert Pd is not None
    # The contrast is the point: dense is O(N^2) nonzeros, factored is O(N+K).
    assert Pd.nnz > 100 * P.nnz
    assert m_dense.quad_nnz == fm.n_assets ** 2
    assert m_fact.quad_nnz == fm.n_assets + fm.n_factors


def test_factor_fit_diagnostics_are_zero_against_their_own_sigma(problem_data):
    """Under `--sigma-basis factor` the arms solve the identical problem.

    The error columns exist to quantify the `sample` basis's confound. Fed the
    factored model's OWN dense matrix they must report exactly zero, otherwise
    the columns cannot be trusted to report a real error either.
    """
    _, _, _, fm, _ = problem_data
    diag = factor_fit_diagnostics(fm, fm.dense(), n_portfolios=40)
    assert diag["sigma_rel_fro_err"] == pytest.approx(0.0, abs=1e-13)
    assert diag["portvol_rel_err_median"] == pytest.approx(0.0, abs=1e-12)
    assert diag["portvol_rel_err_max"] == pytest.approx(0.0, abs=1e-10)


def test_factor_model_approximates_the_sample_covariance(problem_data):
    """A sanity band on the `sample` basis, not a precision claim.

    The factor model cannot carry the convexity term (quadratic in the shocks)
    or the issuer default jumps (which REPLACE the return rather than perturb
    it), so a gap is expected and is the thing the confound columns report.
    This asserts only that the gap is a modelling residual and not a bug: the
    factored matrix must track the sample covariance far better than a
    diagonal-only model would.
    """
    _, R, _, fm, _ = problem_data
    Sigma_sample = np.cov(R, rowvar=False)
    diag = factor_fit_diagnostics(fm, Sigma_sample, n_portfolios=100)

    diag_only = np.diag(np.diag(Sigma_sample))
    den = np.linalg.norm(Sigma_sample, "fro")
    naive = np.linalg.norm(diag_only - Sigma_sample, "fro") / den

    assert diag["sigma_rel_fro_err"] < 0.5
    assert diag["sigma_rel_fro_err"] < naive, (
        f"the factor model ({diag['sigma_rel_fro_err']:.3f}) must beat a "
        f"diagonal-only covariance ({naive:.3f}) or it is not earning its "
        f"keep")
    assert diag["portvol_rel_err_median"] < 0.25


# --------------------------------------------------------------------------
# 3. The limit sheet — the experiment's validity condition
# --------------------------------------------------------------------------

def test_qp_gets_the_identical_limit_sheet_as_the_lp(problem_data):
    """The whole premise: one universe, one mandate, only the objective moves.

    Checked three ways, because a name-only check would pass a builder that
    emitted the right labels against the wrong numbers:

      a. the extra constraint NAMES match the LP branch's, in order;
      b. the base box is the same `limits.w_max` in both;
      c. every extra constraint's RESIDUAL is bit-identical on the same
         weight vector.

    (c) is the one that matters. It compares what the solvers actually
    receive, not what the code says it built.
    """
    uni, R, mu, fm, limits = problem_data

    lp = build_mean_cvar(
        R, mu, alpha=0.95, risk_aversion=RISK_AVERSION, w_max=limits.w_max,
        constraint_builder=credit_constraint_builder(uni, limits, R=None))
    qp, qp_names = build_qp_dense(
        fm.dense(), mu, RISK_AVERSION, limits.w_max,
        constraint_builder=credit_constraint_builder(uni, limits, R=None))

    # (a) The LP's base block is budget + w_max + cvar_tail; the QP has no
    # scenario block, so its base is budget + w_max. Everything after that is
    # the credit mandate and must be identical.
    assert lp.extra_constraint_names[:3] == ["budget", "w_max", "cvar_tail"]
    assert qp_names[:2] == ["budget", "w_max"]
    lp_extra = lp.extra_constraint_names[3:]
    qp_extra = qp_names[2:]
    assert qp_extra == lp_extra, (
        f"the QP's mandate differs from the LP's:\n  LP {lp_extra}\n"
        f"  QP {qp_extra}")
    assert lp_extra, "the mandate must not be empty, or this proves nothing"

    # (b) Same box.
    assert qp.w_max == limits.w_max == lp.w_max

    # (c) Same residuals on the same weights.
    lp_cons = lp.problem.constraints[3:]
    qp_cons = qp.problem.constraints[2:]
    assert len(lp_cons) == len(qp_cons) == len(lp_extra)

    for w in _random_weights(uni.n_bonds, seed=23, k=3):
        lp.w.value = w
        qp.w.value = w
        for name, c_lp, c_qp in zip(lp_extra, lp_cons, qp_cons):
            v_lp = np.atleast_1d(np.asarray(c_lp.violation(), dtype=float))
            v_qp = np.atleast_1d(np.asarray(c_qp.violation(), dtype=float))
            assert v_lp.shape == v_qp.shape, name
            np.testing.assert_allclose(
                v_qp, v_lp, rtol=0, atol=0,
                err_msg=f"constraint {name!r} differs between the LP and QP")


def test_qp_mandate_needs_no_scenario_matrix(problem_data):
    """Families 1-6 are pure weight-space linear constraints.

    Only the tracking-error family is scenario-dependent, and it is off by
    default, so the QP can be handed `R=None` — which is what lets the QP
    branch be scenario-free while carrying the same mandate. If this ever
    stops holding, `score_limits` in the runner starts raising instead of
    scoring.
    """
    uni, _, mu, fm, limits = problem_data
    assert not np.isfinite(limits.te_bp)

    build = credit_constraint_builder(uni, limits, R=None)
    cons, names = build(cp.Variable(uni.n_bonds, nonneg=True))
    assert cons and names
    assert not any("tracking_error" in n for n in names)

    # ...and the exposure report the mandate verdict comes from must score a
    # portfolio with no scenario matrix at all.
    from src.credit import exposure_report, worst_violation
    rep = exposure_report(uni, uni.w_benchmark, limits, R=None)
    live = rep[np.isfinite(rep["rel_slack"])]
    assert not live.empty
    assert set(live["family"]) >= {"rating", "sector", "issuer", "duration",
                                   "dts", "turnover"}

    # `default_limits` derives every band from benchmark exposures, so the
    # index is feasible on every family EXCEPT turnover — which measures
    # distance from the INCUMBENT book, not from the index, and is therefore
    # a property of w0 rather than of the limit sheet. Asserted the way the
    # source docstring states it, so the test pins the documented behaviour
    # instead of a looser version of it.
    non_turnover = live[live["family"] != "turnover"]
    viol, which = worst_violation(non_turnover)
    assert viol <= 1e-9, (
        f"the benchmark must satisfy every index-derived limit; {which} "
        f"breached by {viol:.3%} of its scale")

    # And the turnover row breaches by exactly the incumbent's distance to the
    # index, over the budget. This is what makes the exclusion above a
    # documented fact rather than a convenient carve-out.
    to_row = live[live["family"] == "turnover"].iloc[0]
    assert float(to_row["portfolio"]) == pytest.approx(
        uni.one_way_distance_to_benchmark(), rel=1e-12)


def test_factored_model_carries_the_same_mandate(problem_data):
    """The factored arm is not allowed a cheaper constraint set either.

    Its only extra rows are the K factor definitions y = B'w, which are part
    of the objective's representation, not of the mandate.
    """
    uni, _, mu, fm, limits = problem_data
    builder = credit_constraint_builder(uni, limits, R=None)
    _, dense_names = build_qp_dense(fm.dense(), mu, RISK_AVERSION,
                                    limits.w_max, constraint_builder=builder)
    _, fact_names = build_qp_factored(fm, mu, RISK_AVERSION, limits.w_max,
                                      constraint_builder=builder)
    assert fact_names[:3] == ["budget", "w_max", "factor_definition"]
    assert fact_names[3:] == dense_names[2:]


# --------------------------------------------------------------------------
# 4. The Q-convention calibration constants
# --------------------------------------------------------------------------

def test_calibration_optimum_is_what_the_docstring_claims():
    """Verified against numpy, not against the algebra that produced it.

    The 2-variable calibration is the only defence against a silently doubled
    risk aversion, so its target must not itself be an unchecked hand
    derivation.
    """
    p = np.linspace(0.0, 1.0, 2_000_001)
    X = np.column_stack([p, 1.0 - p])
    quad = np.einsum("ij,jk,ik->i", X, CALIBRATION_S, X)
    obj = X @ CALIBRATION_MU - 0.5 * CALIBRATION_LAMBDA * quad
    i = int(np.argmax(obj))

    assert p[i] == pytest.approx(CALIBRATION_P_STAR, abs=1e-6)
    assert obj[i] == pytest.approx(CALIBRATION_OBJ_STAR, abs=1e-9)
    # The optimum is interior, so neither the box nor nonnegativity is active
    # and the calibration tests the OBJECTIVE rather than the constraints.
    assert 0.0 < p[i] < 1.0

    # And the doubled-risk-aversion value the failure message quotes.
    obj2 = X @ CALIBRATION_MU - 0.5 * (2.0 * CALIBRATION_LAMBDA) * quad
    assert obj2.max() == pytest.approx(CALIBRATION_OBJ_IF_DOUBLED, abs=1e-9)
    # It has to be far enough from the correct value that a 1e-6 gate cannot
    # miss it. It is, by a factor of more than two.
    assert abs(CALIBRATION_OBJ_IF_DOUBLED - CALIBRATION_OBJ_STAR) > 0.5


def test_calibration_matrix_is_psd_and_the_problem_is_convex():
    ev = np.linalg.eigvalsh(CALIBRATION_S)
    assert np.array_equal(CALIBRATION_S, CALIBRATION_S.T)
    assert ev.min() > 0.0


def test_calibration_builds_through_the_same_cvxpy_path():
    """Same `quad_form` + `psd_wrap` construction the sweep uses.

    Evaluated, not solved: if the expression the calibration builds does not
    equal the closed form at p*, the gate would be testing the wrong thing.
    """
    w = cp.Variable(2, nonneg=True)
    obj = cp.Maximize(
        CALIBRATION_MU @ w
        - 0.5 * CALIBRATION_LAMBDA * cp.quad_form(w, cp.psd_wrap(CALIBRATION_S)))
    prob = cp.Problem(obj, [cp.sum(w) == 1.0, w <= 1.0])
    w.value = np.array([CALIBRATION_P_STAR, 1.0 - CALIBRATION_P_STAR])
    assert float(prob.objective.value) == pytest.approx(
        CALIBRATION_OBJ_STAR, abs=1e-12)

"""The attribution baseline is a face, not a portfolio.

`min_risk_baseline` minimises DTS over a limit set that already contains a
DTS floor, so the optimal value is forced and the optimal face is
high-dimensional: the solver returns an arbitrary vertex of it. Every
attribution number the writeup quotes off that vertex — the "two thirds is
mandate-corner de-risking" split, the two-row dominance comparison — is a
property of the vertex rather than of the min-DTS portfolio as a class.

Five claims about `min_risk_face`, the function that turns the point into a
range. All CPU-only, HiGHS only, a few seconds:

1. On a limit set stripped down to `sum(w)=1, w>=0, dts'w >= D`, the face is
   a slice of the simplex and its expected-return band can be computed by
   exhaustive two-bond enumeration. The LP band must match it. This is the
   test that the range is the RIGHT range and not merely a range.
2. The committed vertex sits strictly inside that band — the flaw, in one
   assertion.
3. The band does not come from the width of the pin: shrink the pin 100x and
   the band does not move.
4. The face is pinned at the ACHIEVED DTS, not at the floor. With a floor of
   zero, pinning at the floor would describe the whole polytope.
5. Minimising CVaR over the face is a different optimisation from
   extremising return, it is exact, and the vertex it returns is flagged as
   selected on the evaluation set — because scoring it on that same set is
   hindsight, not an achievable rule.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.credit import (  # noqa: E402
    CreditLimits, default_limits, min_risk_baseline, min_risk_face,
)
from src.scenarios import credit_scenarios  # noqa: E402
from src.universe import make_universe  # noqa: E402

N_BONDS, N_SCEN = 40, 300
ALPHA = 0.95


@pytest.fixture(scope="module")
def problem_data():
    uni = make_universe(n_bonds=N_BONDS, seed=7)
    R, mu, _ = credit_scenarios(uni, n_scen=N_SCEN, seed=11, use_gpu=False,
                                return_mode="excess")
    R_oos, _, _ = credit_scenarios(uni, n_scen=N_SCEN, seed=101,
                                   use_gpu=False, return_mode="excess")
    return uni, R, mu, R_oos


def _simplex_limits(dts_floor: float) -> CreditLimits:
    """Everything off except the DTS floor.

    The feasible set collapses to `{w >= 0, sum(w) = 1, dts'w >= floor}`,
    whose min-DTS face is a slice of the simplex — the one geometry where the
    right answer can be written down without an LP.
    """
    return CreditLimits(
        rating_bands={}, sector_caps={}, issuer_cap=np.inf, w_max=1.0,
        cell_duration_tol=np.inf, total_duration_tol=np.inf,
        bench_duration=np.nan, dts_max=np.inf, dts_min=dts_floor,
        turnover=np.inf,
    )


def _two_bond_envelope(dts: np.ndarray, mu: np.ndarray, target: float,
                       maximize: bool) -> float:
    """Exact optimum of max/min mu'w over `sum(w)=1, dts'w=target, w>=0`.

    Two equality rows, so every basic feasible solution has at most two
    nonzero weights and enumerating pairs is exhaustive. Deliberately not an
    LP: an independent check has to come from different machinery than the
    thing it checks.
    """
    best = -np.inf if maximize else np.inf
    n = len(dts)
    for i in range(n):
        if abs(dts[i] - target) < 1e-9:
            best = max(best, mu[i]) if maximize else min(best, mu[i])
        for j in range(i + 1, n):
            di, dj = float(dts[i]), float(dts[j])
            if abs(di - dj) < 1e-12:
                continue
            theta = (target - dj) / (di - dj)
            if not -1e-12 <= theta <= 1.0 + 1e-12:
                continue
            val = theta * mu[i] + (1.0 - theta) * mu[j]
            best = max(best, val) if maximize else min(best, val)
    return float(best)


def test_return_band_matches_exhaustive_enumeration(problem_data):
    """The band is the right band, checked without an LP."""
    uni, _, mu, _ = problem_data
    target = float(uni.w_benchmark @ uni.dts)      # interior of the DTS range
    limits = _simplex_limits(target)

    face = min_risk_face(uni, limits, mu)
    assert face.dts_target == pytest.approx(target, rel=1e-9)
    assert face.floor_binds

    hi = _two_bond_envelope(uni.dts, mu, target, maximize=True)
    lo = _two_bond_envelope(uni.dts, mu, target, maximize=False)
    assert face.exp_return_hi == pytest.approx(hi, rel=1e-7)
    assert face.exp_return_lo == pytest.approx(lo, rel=1e-7)
    # ...and the band is a real one, not a rounding artifact.
    assert face.exp_return_range > 0.1 * abs(hi)

    # Every recovered vertex is actually ON the face, within the stated pin.
    for v in face.vertices:
        assert abs(v.dts - face.dts_target) <= face.dts_band * 1.000001, \
            f"vertex {v.label} left the face: DTS {v.dts}"


def test_the_committed_vertex_is_one_point_of_that_band(problem_data):
    """The defect, as an assertion.

    `min_risk_baseline` returns exactly the vertex this face's baseline solve
    returns (same program, same solver), and that vertex's expected return
    sits strictly inside the band — so quoting it as "the min-DTS portfolio's
    expected return" quotes one arbitrary point of a set.
    """
    uni, _, mu, _ = problem_data
    limits = _simplex_limits(float(uni.w_benchmark @ uni.dts))

    face = min_risk_face(uni, limits, mu)
    w_base = min_risk_baseline(uni, limits)
    np.testing.assert_allclose(w_base, face.vertex("baseline_vertex").w,
                               rtol=0, atol=1e-12)

    committed = float(mu @ w_base)
    assert face.exp_return_lo < committed < face.exp_return_hi
    assert not face.looks_like_a_point
    assert face.max_vertex_l1 > 1e-3


def test_the_band_is_not_an_artifact_of_the_pin_width(problem_data):
    """Shrink the pin 100x; the band must not move.

    The face is pinned as a two-sided band rather than an equality, which
    makes the reported range a slight OUTER approximation. This measures how
    slight instead of asserting it.
    """
    uni, _, mu, _ = problem_data
    limits = _simplex_limits(float(uni.w_benchmark @ uni.dts))

    wide = min_risk_face(uni, limits, mu, face_rel_tol=1e-9)
    tight = min_risk_face(uni, limits, mu, face_rel_tol=1e-11)

    assert tight.dts_band == pytest.approx(wide.dts_band / 100.0, rel=1e-9)
    assert tight.exp_return_hi == pytest.approx(wide.exp_return_hi, rel=1e-9)
    assert tight.exp_return_lo == pytest.approx(wide.exp_return_lo, rel=1e-9)


def test_pinned_at_the_achieved_dts_not_at_the_floor(problem_data):
    """With no DTS floor, pinning at the floor would be pinning at zero.

    The box `w <= 1/N` plus the budget forces the equal-weight portfolio, so
    the feasible set — and therefore the face — is a single point at a DTS
    far above `dts_min = 0`. A function that pinned at the floor would have
    described the whole polytope instead, and would not report a point here.
    """
    uni, _, mu, _ = problem_data
    n = uni.n_bonds
    limits = CreditLimits(
        rating_bands={}, sector_caps={}, issuer_cap=np.inf, w_max=1.0 / n,
        cell_duration_tol=np.inf, total_duration_tol=np.inf,
        bench_duration=np.nan, dts_max=np.inf, dts_min=0.0, turnover=np.inf,
    )
    equal_weight_dts = float(uni.dts.mean())

    face = min_risk_face(uni, limits, mu)
    assert not face.floor_binds
    assert face.dts_target == pytest.approx(equal_weight_dts, rel=1e-9)

    # A genuine single-point face: the band collapses and says so. Not to
    # exactly zero — three separate LP solves land on the same point to their
    # own feasibility tolerance, which on 40 weights and a ~1e-3 mu leaves a
    # few 1e-12 of width. Asserted relatively, because "zero" here means
    # "nothing a solver could distinguish", not "the same float".
    assert face.looks_like_a_point
    assert face.max_vertex_l1 <= 1e-6
    assert face.exp_return_range <= 1e-6 * abs(face.exp_return_lo)
    assert face.exp_return_lo == pytest.approx(float(mu.mean()), rel=1e-7)
    band = face.as_band_row()
    assert band["face_looks_like_a_point"] is True
    assert band["face_exp_return_range"] <= 1e-6 * abs(face.exp_return_lo)


def test_min_cvar_over_the_face_is_a_different_and_flagged_solve(problem_data):
    """CVaR and return are extremised in different directions, and only one
    of the two CVaR ends is computable at all.

    Minimising CVaR over the face is an LP and is exact. MAXIMISING it is a
    convex maximisation over a polytope and is not attempted, so the high end
    is only ever the worst vertex observed. And the min-CVaR vertex is chosen
    using the evaluation set, which makes its out-of-sample CVaR an in-sample
    number for that set — flagged, and excluded from the ex-ante band.
    """
    uni, _, mu, R_oos = problem_data
    limits = default_limits(uni)

    face = min_risk_face(uni, limits, mu, R_eval=R_oos, alpha=ALPHA,
                         minimize_cvar_over_face=True)
    labels = [v.label for v in face.vertices]
    assert labels == ["baseline_vertex", "max_return", "min_return",
                      "min_cvar"]

    v_cvar = face.vertex("min_cvar")
    assert v_cvar.selected_using_eval_set is True
    assert all(not face.vertex(lbl).selected_using_eval_set
               for lbl in labels if lbl != "min_cvar")

    # Exact: no other point recovered from the face can beat it.
    assert face.cvar_lo_is_exact
    assert face.cvar_lo == pytest.approx(v_cvar.cvar, rel=1e-12)
    assert all(v.cvar >= face.cvar_lo - 1e-12 for v in face.vertices)

    # The direction that minimises CVaR is NOT the direction that extremises
    # return: if it were, this whole distinction would be decorative.
    assert v_cvar.exp_return > face.exp_return_lo
    assert v_cvar.exp_return < face.exp_return_hi

    # The ex-ante band excludes it; the hindsight band includes it and is
    # therefore weakly wider on the de-risking end.
    ex_lo, ex_hi = face.cvar_band(include_eval_selected=False)
    hind_lo, hind_hi = face.cvar_band(include_eval_selected=True)
    assert hind_lo <= ex_lo
    assert hind_hi == pytest.approx(ex_hi, rel=1e-12)
    assert face.cvar_hi_observed == pytest.approx(hind_hi, rel=1e-12)
    assert face.as_band_row()["face_cvar_hi_is_lower_bound"] is True


def test_face_needs_no_evaluation_set_to_produce_a_return_band(problem_data):
    """The return band is the cheap half: three small LPs, no scenarios.

    Whatever the GPU session decides about the expensive CVaR solve, the
    return band must always be emittable.
    """
    uni, _, mu, _ = problem_data
    face = min_risk_face(uni, default_limits(uni), mu)

    assert np.isfinite(face.exp_return_lo) and np.isfinite(face.exp_return_hi)
    assert all(np.isnan(v.cvar) for v in face.vertices)
    assert np.isnan(face.cvar_lo) and not face.cvar_lo_is_exact
    assert len(face.as_rows()) == 3

    with pytest.raises(ValueError, match="needs R_eval"):
        min_risk_face(uni, default_limits(uni), mu,
                      minimize_cvar_over_face=True)

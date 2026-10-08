"""P0 TE mandate runner behavior.

These tests are deliberately CPU-only and mostly pure: the P0 runner's
published rows must be auditable without asking a solver to reproduce every
branch on demand.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import cvxpy as cp
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from run_te_mandate import (  # noqa: E402
    DEFAULT_P0_MAD_GRID_BP,
    SOLVERS,
    build_bisection_budgets,
    independent_portfolio_metrics,
    make_error_row,
    make_unsupported_row,
    select_recalibration_candidate,
    solver_call_config,
)


def test_solver_ids_expose_native_socp_and_backward_aliases():
    """A missing native SOCP method or stale solver ID makes P0 mislabel rows."""
    assert set(SOLVERS) >= {
        "cuopt-pdlp-1e-4",
        "cuopt-pdlp-1e-6",
        "cuopt-pdlp-xover",
        "cuopt-barrier-socp",
        "clarabel-socp",
        "highs-simplex",
    }
    assert SOLVERS["cuopt-barrier-socp"]["conic"] is True
    assert SOLVERS["cuopt-barrier-socp"]["method_class"] == "SOCP"

    cfg = solver_call_config("cuopt-barrier-socp", time_limit=12.5)
    assert cfg.solver == getattr(cp, "CUOPT", "CUOPT")
    assert cfg.opts["solver_method"] == "Barrier"
    assert "pdlp_solver_mode" not in cfg.opts

    assert solver_call_config("clarabel", 1.0).canonical_id == "clarabel-socp"
    assert solver_call_config("cuopt@1e-4", 1.0).canonical_id == \
        "cuopt-pdlp-1e-4"


def test_sigma_rows_mark_lp_only_methods_as_author_unsupported():
    """LP-only methods on sigma TE are classification rows, not failed solves."""
    row = make_unsupported_row(
        case="sigma@10bp", te_mode="sigma", te_budget_bp=10.0,
        solver_id="highs-simplex",
    )

    assert row["status"] == "UNSUPPORTED_BY_METHOD"
    assert row["status_source"] == "author-classification"
    assert row["method_class"] == "LP"
    assert row["endpoint"] == "simplex"
    assert row["check_ok"] is False
    assert math.isnan(row["build_s"])
    assert "SOC" in row["note"]


def test_legacy_solver_request_is_visible_beside_canonical_id():
    """Historical M6 labels must round-trip visibly under P0 canonical IDs."""
    row = make_unsupported_row(
        case="sigma@10bp", te_mode="sigma", te_budget_bp=10.0,
        solver_id="cuopt@1e-4",
    )

    assert row["solver"] == "cuopt-pdlp-1e-4"
    assert row["solver_requested"] == "cuopt@1e-4"
    assert row["solver_legacy_id"] == "cuopt@1e-4"


def test_selector_uses_eligibility_and_highest_independent_objective():
    """The recalibration pick is by feasible quality, never grid order."""
    rows = [
        {
            "case": "mad@1bp", "te_budget_bp": 1.0, "status": "optimal",
            "te_mode": "mad", "status_source": "solver-returned",
            "check_ok": True,
            "purity_pass": True, "desk_pass": True,
            "te_sigma_bp": 9.8, "objective_recomputed": 0.10,
        },
        {
            "case": "mad@2bp", "te_budget_bp": 2.0, "status": "optimal",
            "te_mode": "mad", "status_source": "solver-returned",
            "check_ok": True,
            "purity_pass": True, "desk_pass": True,
            "te_sigma_bp": 10.4, "objective_recomputed": 0.50,
        },
        {
            "case": "mad@3bp", "te_budget_bp": 3.0, "status": "optimal",
            "te_mode": "mad", "status_source": "solver-returned",
            "check_ok": True,
            "purity_pass": False, "desk_pass": True,
            "te_sigma_bp": 9.9, "objective_recomputed": 0.80,
        },
        {
            "case": "mad@5bp", "te_budget_bp": 5.0, "status": "optimal",
            "te_mode": "mad", "status_source": "solver-returned",
            "check_ok": True,
            "purity_pass": True, "desk_pass": True,
            "te_sigma_bp": 10.0, "objective_recomputed": 0.30,
        },
    ]

    assert select_recalibration_candidate(rows, sigma_target_bp=10.0)["case"] \
        == "mad@5bp"


def test_selector_ignores_higher_objective_non_mad_rows():
    """Only MAD candidates are selectable; sigma and validation are scoring."""
    rows = [
        {
            "case": "mad@5bp", "te_mode": "mad", "te_budget_bp": 5.0,
            "status": "optimal", "status_source": "solver-returned",
            "check_ok": True, "purity_pass": True, "desk_pass": True,
            "te_sigma_bp": 9.9, "objective_recomputed": 0.30,
        },
        {
            "case": "sigma@10bp", "te_mode": "sigma", "te_budget_bp": 10.0,
            "status": "optimal", "status_source": "solver-returned",
            "check_ok": True, "purity_pass": True, "desk_pass": True,
            "te_sigma_bp": 10.0, "objective_recomputed": 99.0,
        },
        {
            "case": "validation-summary", "te_mode": "sigma-validation",
            "te_budget_bp": 10.0, "status": "VALIDATION_SUMMARY",
            "status_source": "author-classification", "check_ok": True,
            "purity_pass": True, "desk_pass": True, "te_sigma_bp": 1.0,
            "objective_recomputed": 100.0,
        },
    ]

    assert select_recalibration_candidate(rows, sigma_target_bp=10.0)["case"] \
        == "mad@5bp"


def test_selector_returns_none_when_no_candidate_is_feasible():
    """No feasible candidate must stay explicit instead of picking a fallback."""
    rows = [{
        "case": "mad@1bp", "te_mode": "mad", "status": "optimal",
        "check_ok": True, "purity_pass": True, "desk_pass": True,
        "te_sigma_bp": 10.2, "objective_recomputed": 1.0,
    }]

    assert select_recalibration_candidate(rows, sigma_target_bp=10.0) is None


def test_bisection_increasing_stops_on_budget_width_not_sigma_gap():
    """A near-target midpoint must not stop while budget width is still wide."""
    budgets = build_bisection_budgets(
        initial_budgets=[1.0, 5.0],
        sigma_at_budget=lambda budget: budget,
        sigma_target_bp=3.005,
        max_steps=8,
        stop_width_bp=0.1,
    )

    assert budgets[:2] == [1.0, 5.0]
    assert len(budgets) > 3
    assert len(budgets) <= 10
    assert min(abs(b - 3.005) for b in budgets) <= 0.1


def test_bisection_decreasing_bracket_is_refined_by_budget_width():
    """Descending sigma-vs-budget relationships still bracket correctly."""
    budgets = build_bisection_budgets(
        initial_budgets=[1.0, 5.0],
        sigma_at_budget=lambda budget: 6.0 - budget,
        sigma_target_bp=2.995,
        max_steps=8,
        stop_width_bp=0.1,
    )

    assert budgets[:2] == [1.0, 5.0]
    assert len(budgets) > 3
    assert len(budgets) <= 10
    assert min(abs(b - 3.005) for b in budgets) <= 0.1


def test_p0_grid_is_preserved_before_bisection_candidates():
    """Bisection should refine the bracket without replacing fixed-grid runs."""
    budgets = build_bisection_budgets(
        initial_budgets=DEFAULT_P0_MAD_GRID_BP,
        sigma_at_budget=lambda budget: budget,
        sigma_target_bp=4.2,
        max_steps=8,
        stop_width_bp=0.1,
    )

    assert budgets[:5] == [1.0, 2.0, 3.0, 5.0, 10.0]
    assert len(budgets) <= 13


def test_build_and_check_errors_are_not_solver_returned_statuses():
    """Only errors raised by a solver call may use solver-returned source."""
    build_row = make_error_row(
        case="mad@1bp", te_mode="mad", te_budget_bp=1.0,
        solver_id="clarabel-socp", error=ValueError("bad build"),
        error_stage="build",
    )
    check_row = make_error_row(
        case="mad@1bp", te_mode="mad", te_budget_bp=1.0,
        solver_id="clarabel-socp", error=AssertionError("bad check"),
        error_stage="check",
    )

    assert build_row["status_source"] == "runner-build-error"
    assert check_row["status_source"] == "runner-check-error"
    assert build_row["status"].startswith("ERROR:ValueError")


def test_independent_objective_expected_excess_and_cvar_are_recomputed():
    """Published economics come from weights and scenarios, not solver text."""
    R = np.array([
        [-0.01, -0.03],
        [0.02, 0.00],
        [0.00, -0.01],
        [0.01, 0.03],
    ])
    mu = np.array([0.002, 0.004])
    w = np.array([0.25, 0.75])

    metrics = independent_portfolio_metrics(
        w=w, R=R, mu=mu, alpha=0.5, risk_aversion=2.0,
        solver_objective=-0.030,
    )

    assert metrics["expected_excess_bp_monthly"] == pytest.approx(35.0)
    assert metrics["cvar_recomputed"] == pytest.approx(0.01625)
    assert metrics["objective_recomputed"] == pytest.approx(-0.029)
    assert metrics["objective_discrepancy"] == pytest.approx(0.001)

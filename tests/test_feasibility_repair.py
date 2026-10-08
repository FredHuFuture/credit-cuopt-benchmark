"""The weight-space repair: what it may drop, and what it must not.

Three things can quietly invalidate the M8 experiment, and none of them shows
up as an exception:

1. **Dropping the wrong family.** The projection is allowed to ignore exactly
   one thing -- the risk-space tracking-error constraint -- because that is
   the only family defined on `R @ (w - b)`. If it also dropped `active_l1`,
   which is a weight-space budget living in the same mandate paragraph, the
   run would understate what the repair restores and the reported polytope
   would not be the mandate polytope.

2. **A column that only some rows carry.** `src.bench.write_csv` takes its
   fieldnames from the FIRST row, so a key emitted by `score_portfolio` but
   missing from `ROW_TEMPLATE` would vanish from every CSV silently. The
   template is therefore asserted to be a superset of what the scorer emits.

3. **A projection that does not land in the set.** The whole claim is that
   the repaired point satisfies the weight-space mandate. That is checked
   here against `exposure_report`, i.e. against the same independent scorer
   the runner publishes from, not against the solver's own status.

The solves here are deliberately tiny -- 60 bonds, 200 scenarios -- because
these are statements about formulation, not about performance.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import cvxpy as cp  # noqa: E402
import run_feasibility_repair as rfr  # noqa: E402

from run_feasibility_repair import (  # noqa: E402
    ROW_TEMPLATE, CaseContext, RepairOut, add_source_loss_metrics,
    build_polytope, cuopt_pdlp_specs_for_tols, endpoint_total_seconds,
    make_reference_error_row, make_reference_unsupported_row, make_row,
    make_unavailable_endpoint_row, objective_consistency_columns,
    perturb_to_target, repair_once, repair_sweep, score_portfolio,
    solve_direct_endpoint,
    weight_space_limits,
)
from src.credit import (  # noqa: E402
    credit_constraint_builder, default_limits, exposure_report,
    gate_verdicts,
)
from src.scenarios import credit_scenarios  # noqa: E402
from src.universe import make_universe  # noqa: E402

ALPHA, LAM = 0.95, 3.0


@pytest.fixture(scope="module")
def problem():
    uni = make_universe(n_bonds=60, seed=7)
    R, mu, _ = credit_scenarios(uni, n_scen=200, seed=11, use_gpu=False,
                                return_mode="excess")
    R_oos, _, _ = credit_scenarios(uni, n_scen=200, seed=101, use_gpu=False,
                                   return_mode="excess")
    return uni, R, R_oos, mu


def test_only_the_risk_space_family_is_dropped(problem):
    uni, R, _, _ = problem
    limits = default_limits(uni, te_bp=10.0, te_mode="mad", active_l1=0.4)
    ws, invisible = weight_space_limits(limits)

    w = cp.Variable(uni.n_bonds, nonneg=True)
    full = credit_constraint_builder(uni, limits, R=R)(w)[1]
    kept = credit_constraint_builder(uni, ws, R=None)(w)[1]

    assert invisible == ["tracking_error_mad_LP"]
    assert set(full) - set(kept) == {"tracking_error_mad_LP"}
    # The weight-space active budget belongs to the same mandate paragraph as
    # the TE constraint and must survive the split.
    assert "active_weight_l1" in kept
    assert ws.active_l1 == limits.active_l1


def test_sigma_mode_names_the_cone_it_drops(problem):
    uni, _, _, _ = problem
    limits = default_limits(uni, te_bp=10.0, te_mode="sigma")
    _, invisible = weight_space_limits(limits)
    assert invisible == ["tracking_error_sigma_SOC"]


def test_no_te_budget_leaves_nothing_invisible(problem):
    uni, _, _, _ = problem
    _, invisible = weight_space_limits(default_limits(uni))
    assert invisible == []


def test_row_template_covers_every_scored_column(problem):
    uni, R, R_oos, mu = problem
    limits = default_limits(uni)
    sc = score_portfolio(uni, uni.w_benchmark, limits, R, R_oos, mu, ALPHA,
                         LAM)
    missing = sorted(set(sc) - set(ROW_TEMPLATE))
    assert not missing, (
        f"score_portfolio emits {missing}, which write_csv would drop "
        f"because it takes its fieldnames from the first row")


def test_score_portfolio_exports_independent_checker_and_objective_agreement(
        problem):
    uni, R, R_oos, mu = problem
    limits = default_limits(uni)
    sc = score_portfolio(uni, uni.w_benchmark, limits, R, R_oos, mu, ALPHA,
                         LAM)

    assert sc["check_ok"] is True
    assert sc["check_notes"] == "ok"
    assert sc["objective_consistent"] is False
    assert np.isnan(sc["objective_discrepancy"])


def test_make_row_rejects_an_unknown_column():
    with pytest.raises(KeyError):
        make_row(portfolio="x", not_a_column=1.0)


def test_l1_projection_lands_inside_the_weight_space_mandate(problem):
    uni, R, _, mu = problem
    limits = default_limits(uni)
    ws, _ = weight_space_limits(limits)
    # A point well outside the mandate: the whole book pushed toward the
    # longest-duration names, which breaks duration matching and turnover at
    # once.
    order = np.argsort(uni.duration)[::-1]
    w_bad = np.zeros(uni.n_bonds)
    w_bad[order[:40]] = 1.0 / 40.0
    assert gate_verdicts(exposure_report(uni, w_bad, limits, R=R)).verdict \
        == "BREACH"

    out = repair_once(uni, limits, ws, w_bad, "l1", R, mu, ALPHA, LAM,
                      threads=2, time_limit=120.0, d1=float("nan"),
                      lex_slack=1e-9, conic=False)
    assert isinstance(out, RepairOut)
    assert out.status == "optimal"
    gv = gate_verdicts(exposure_report(uni, out.w, limits, R=R))
    assert gv.desk_pass
    assert abs(out.w.sum() - 1.0) < 1e-9
    assert out.w.min() > -1e-9
    assert out.w.max() <= limits.w_max + 1e-9


def test_l2_projection_is_never_further_than_l1_in_l2_norm(problem):
    uni, R, _, mu = problem
    limits = default_limits(uni)
    ws, _ = weight_space_limits(limits)
    rng = np.random.default_rng(3)
    w_bad = np.abs(rng.random(uni.n_bonds))
    w_bad /= w_bad.sum()

    kw = dict(threads=2, time_limit=120.0, lex_slack=1e-9, conic=False)
    l2 = repair_once(uni, limits, ws, w_bad, "l2", R, mu, ALPHA, LAM,
                     d1=float("nan"), **kw)
    l1 = repair_once(uni, limits, ws, w_bad, "l1", R, mu, ALPHA, LAM,
                     d1=float("nan"), **kw)
    assert l2.status == "optimal" and l1.status == "optimal"
    # Each projection is optimal in its OWN norm; the cross-comparison is the
    # only assertion that holds without assuming which one a desk prefers.
    assert (np.linalg.norm(l2.w - w_bad)
            <= np.linalg.norm(l1.w - w_bad) + 1e-9)
    assert l1.l1_distance <= float(np.abs(l2.w - w_bad).sum()) + 1e-9


def test_objective_aware_modes_refuse_to_run_without_a_ball(problem):
    uni, R, _, mu = problem
    limits = default_limits(uni)
    ws, _ = weight_space_limits(limits)
    with pytest.raises(ValueError, match="L1 projection distance"):
        repair_once(uni, limits, ws, uni.w_benchmark, "l1_mu_lex", R, mu,
                    ALPHA, LAM, threads=2, time_limit=120.0,
                    d1=float("nan"), lex_slack=1e-9, conic=False)


def test_polytope_carries_the_budget_and_the_box(problem):
    uni, _, _, _ = problem
    ws, _ = weight_space_limits(default_limits(uni))
    _, cons, names = build_polytope(uni, ws)
    # Neither is part of the credit constraint builder, and a projection
    # without them would return an unnormalized book.
    assert names[:2] == ["budget", "w_max"]
    assert len(cons) == len(names)


def test_perturbation_is_seeded_and_stays_in_the_box(problem):
    uni, R, _, _ = problem
    limits = default_limits(uni)
    a, info_a = perturb_to_target(uni, limits, R, uni.w_benchmark, 11, 20.0)
    b, info_b = perturb_to_target(uni, limits, R, uni.w_benchmark, 11, 20.0)
    assert np.array_equal(a, b)
    assert info_a["step_l1"] == info_b["step_l1"]
    # PDLP satisfies variable bounds exactly and violates rows; a faithful
    # imitation must do the same, so the box holds and the budget drifts.
    assert a.min() >= 0.0
    assert a.max() <= limits.w_max + 1e-12
    assert info_a["achieved_mult"] > 1.0


def test_cuopt_loose_source_uses_te_benchmark_pdlp_options(monkeypatch):
    calls = []

    def fake_spec(tol, time_limit_s, crossover=False):
        calls.append((tol, time_limit_s, crossover))
        return SimpleNamespace(
            label=f"cuopt-pdlp{'+xover' if crossover else ''}@{tol:g}",
            solver_opts={
                "solver_method": "PDLP",
                "pdlp_solver_mode": "Stable2",
                "presolve": 2,
                "absolute_primal_tolerance": tol,
                "relative_primal_tolerance": tol,
                "absolute_dual_tolerance": tol,
                "relative_dual_tolerance": tol,
                "absolute_gap_tolerance": tol,
                "relative_gap_tolerance": tol,
                "time_limit": time_limit_s,
                "crossover": 1 if crossover else 0,
            },
        )

    monkeypatch.setattr(rfr, "cuopt_pdlp_spec", fake_spec)

    raw, basic = cuopt_pdlp_specs_for_tols([1e-4, 1e-6], 12.5)

    assert calls == [(1e-4, 12.5, False), (1e-6, 12.5, False),
                     (1e-4, 12.5, True)]
    assert [s.label for s in raw] == ["cuopt-pdlp@0.0001",
                                      "cuopt-pdlp@1e-06"]
    assert raw[0].solver_opts["presolve"] == 2
    assert raw[0].solver_opts["pdlp_solver_mode"] == "Stable2"
    assert raw[0].solver_opts["absolute_gap_tolerance"] == pytest.approx(1e-4)
    assert basic.solver_opts["crossover"] == 1


def test_total_endpoint_time_adds_raw_and_repair_components():
    assert endpoint_total_seconds(2.0, 0.35) == pytest.approx(2.35)
    assert np.isnan(endpoint_total_seconds(float("nan"), 0.35))


def test_source_loss_metrics_are_against_the_row_source_point():
    row = make_row(objective_in=0.010, mu_w=0.018, cvar_in=0.002)
    source = {"objective_in": 0.012, "mu_w": 0.020, "cvar_in": 0.0015}

    add_source_loss_metrics(row, source, "cuopt-pdlp@0.0001")

    assert row["source_reference"] == "cuopt-pdlp@0.0001"
    assert row["d_obj_vs_source_bp"] == pytest.approx(20.0)
    assert row["d_expected_excess_vs_source_bp"] == pytest.approx(20.0)
    assert row["d_cvar_in_vs_source_bp"] == pytest.approx(5.0)


def test_failed_l1_family_emits_rows_for_dependent_repairs(monkeypatch):
    uni = SimpleNamespace(n_bonds=2)
    args = SimpleNamespace(
        modes=["l1", "l1_mu_lex"], repair_runs=1, alpha=ALPHA,
        risk_aversion=LAM, threads=1, time_limit=1.0, lex_slack=1e-9,
    )
    ctx = CaseContext(
        args=args, uni=uni, limits=object(), ws_limits=object(), conic=False,
        R=np.zeros((1, 2)), R_oos=np.zeros((1, 2)), mu=np.zeros(2), refs={},
        exact_key="highs-simplex", base=lambda **kw: make_row(case="no-TE",
                                                              **kw),
        full_cols=3, full_nnz=4.0,
    )

    def fake_repair_once(*args, **kwargs):
        return RepairOut(
            w=np.array([0.5, 0.5]), status="infeasible", build_s=0.0,
            solve_s=0.1, l1_distance=float("nan"),
        )

    monkeypatch.setattr(rfr, "repair_once", fake_repair_once)

    rows = repair_sweep(
        ctx,
        {"w": np.array([0.6, 0.4]), "label": "raw", "spec": "tol=0.0001",
         "wall_s": 2.0},
        {"n_positions": 2, "n_nonzero_weights": 2},
    )

    assert [(r["portfolio"], r["status"]) for r in rows] == [
        ("repair:l1", "infeasible"),
        ("repair:l1_mu_lex", "UNAVAILABLE"),
    ]
    assert rows[1]["status_source"] == "author-classification"
    assert rows[1]["endpoint"] == "repaired-feasible"


def test_unavailable_endpoint_row_preserves_published_schema():
    row = make_unavailable_endpoint_row(
        lambda **kw: make_row(case="mad@10bp", **kw),
        portfolio="cuopt-pdlp@0.0001",
        role="loose",
        endpoint="raw",
        method_class="LP",
        status="UNAVAILABLE",
        status_source="runner-build-error",
        note="cuOpt is not installed",
    )

    assert set(row) == set(ROW_TEMPLATE)
    assert row["portfolio"] == "cuopt-pdlp@0.0001"
    assert row["endpoint"] == "raw"
    assert row["status"] == "UNAVAILABLE"
    assert row["gate_verdict"] == "UNAVAILABLE"
    assert np.isnan(row["total_endpoint_s"])


class _FakeProblem:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.status = ""
        self.value = 0.0
        self.calls = 0

    def solve(self, **_kwargs):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        self.status, self.value = outcome


class _FakeModel:
    def __init__(self, outcomes):
        self.problem = _FakeProblem(outcomes)

    def weights(self):
        return np.array([0.25, 0.75])


def test_direct_endpoint_nonconverged_preserves_elapsed_lower_bound():
    model = _FakeModel([("user_limit", 12.0)])
    spec = SimpleNamespace(cvxpy_solver="FAKE", solver_opts={})

    out = solve_direct_endpoint(model, spec, runs=3)

    assert out.status == "user_limit"
    assert out.w is None
    assert out.first_solve_s >= 0.0
    assert out.total_endpoint_s == pytest.approx(out.first_solve_s)
    assert out.elapsed_is_lower_bound is True
    assert model.problem.calls == 1


def test_direct_endpoint_exception_preserves_elapsed_lower_bound():
    model = _FakeModel([RuntimeError("boom")])
    spec = SimpleNamespace(cvxpy_solver="FAKE", solver_opts={})

    out = solve_direct_endpoint(model, spec, runs=3)

    assert out.status == "ERROR:RuntimeError"
    assert out.first_solve_s >= 0.0
    assert out.total_endpoint_s == pytest.approx(out.first_solve_s)
    assert out.elapsed_is_lower_bound is True
    assert model.problem.calls == 1


def test_direct_endpoint_success_total_is_representative_latency(monkeypatch):
    model = _FakeModel([
        ("optimal", 12.0),
        ("optimal", 12.0),
        ("optimal", 12.0),
        ("optimal", 12.0),
    ])
    spec = SimpleNamespace(cvxpy_solver="FAKE", solver_opts={})
    ticks = iter([0.0, 10.0, 10.0, 12.0, 12.0, 16.0, 16.0, 26.0])
    monkeypatch.setattr(rfr.time, "perf_counter", lambda: next(ticks))

    out = solve_direct_endpoint(model, spec, runs=3)

    assert out.first_solve_s == pytest.approx(10.0)
    assert out.steady_state_runs_s == pytest.approx((2.0, 4.0, 10.0))
    assert out.steady_state_median_s == pytest.approx(4.0)
    assert out.total_endpoint_s == pytest.approx(4.0)
    assert out.elapsed_is_lower_bound is False


def test_objective_consistency_uses_solver_spec_quality_target():
    loose = SimpleNamespace(check_objective_tol=1e-3)
    tight = SimpleNamespace(check_objective_tol=1e-5)

    loose_row = make_row(objective_recomputed=1.0)
    tight_row = make_row(objective_recomputed=1.0)
    objective_consistency_columns(loose_row, 1.0005, loose)
    objective_consistency_columns(tight_row, 1.0005, tight)

    assert loose_row["objective_quality_target"] == pytest.approx(1e-3)
    assert loose_row["objective_discrepancy"] == pytest.approx(5e-4)
    assert loose_row["objective_consistent"] is True
    assert tight_row["objective_quality_target"] == pytest.approx(1e-5)
    assert tight_row["objective_discrepancy"] == pytest.approx(5e-4)
    assert tight_row["objective_consistent"] is False


def test_reference_unsupported_rows_are_method_scoped_for_sigma():
    row = make_reference_unsupported_row(
        lambda **kw: make_row(case="sigma@10bp", **kw),
        portfolio="highs-simplex",
        role="reference-exact",
        method_class="LP",
        endpoint="exact/reference",
        note="LP-only solver cannot express an SOC constraint",
    )

    assert row["status"] == "UNSUPPORTED_BY_METHOD"
    assert row["status_source"] == "author-classification"
    assert row["method_class"] == "LP"
    assert row["endpoint"] == "exact/reference"
    assert row["gate_verdict"] == "UNSUPPORTED_BY_METHOD"


def test_clarabel_soc_reference_metadata_is_exact_conic_not_lp():
    row = make_row(
        portfolio="clarabel", role="reference-exact",
        status="optimal", method_class="SOCP", endpoint="exact-conic",
    )

    assert row["method_class"] == "SOCP"
    assert row["endpoint"] == "exact-conic"


def test_reference_error_row_preserves_status_source_and_schema():
    row = make_reference_error_row(
        lambda **kw: make_row(case="sigma@10bp", **kw),
        portfolio="clarabel",
        role="reference-exact",
        method_class="SOCP",
        endpoint="exact-conic",
        error=RuntimeError("bad cone"),
    )

    assert set(row) == set(ROW_TEMPLATE)
    assert row["status"] == "ERROR:RuntimeError"
    assert row["status_source"] == "runner-reference-error"
    assert row["gate_verdict"] == "ERROR"

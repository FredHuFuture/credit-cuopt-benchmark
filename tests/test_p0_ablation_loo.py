"""P0 full-minus-turnover ablation behavior.

The expensive run is a three-level paired experiment: L0 control, full
mandate, and full-minus-turnover. These tests keep that contract solver-free:
the only thing removed from the full mandate is turnover, the rows carry
enough accounting to interpret the pair later, and the dominance helper never
recreates the old causal-share claim.
"""

from __future__ import annotations

import math
import subprocess
import sys
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from run_ablation import (  # noqa: E402
    LEVELS,
    LOO_LEVEL_NAME,
    P0_LOO_LEVELS,
    ablation_session_id,
    completed_resume_keys,
    default_output_path,
    is_legacy_level_run,
    level_design_id,
    loo_interpretation,
    p0_pair_id,
    reduce_limits,
    resume_rows_for_current_pair,
    run_worker,
    run_spec_one_cap,
    resume_key_from_row,
    select_levels,
    summarise_active_limits,
    with_p0_annotations,
    worker_timeout_s,
)
from src.credit import default_limits, exposure_report  # noqa: E402
from src.universe import make_universe  # noqa: E402


def _args(**overrides):
    class Args:
        n_bonds = 500
        n_scenarios = 20_000
        alpha = 0.95
        risk_aversion = 3.0
        turnover = 0.25
        tol = 1e-4
        runs = 5
        time_limit = 300.0
        threads = 16
        seed = 7
        scen_seed = 11
        levels = ""
        out = "results/p0_ablation_loo.csv"

    for k, v in overrides.items():
        setattr(Args, k, v)
    return Args


def _row(**overrides):
    base = {
        "level": "L6_full_mandate",
        "solver": "cuopt_pdlp",
        "status": "optimal",
        "dnf": False,
        "wall_s_median": "2.0",
        "endpoint": "raw",
        "pair_id": "p0-ablation-n500-s20000-useed7-sseed11-tol0.0001",
        "paired_levels": "L0_budget_box|L6_full_mandate|LOO_no_turnover",
        "level_design_id": "levels:L0_budget_box|L6_full_mandate|LOO_no_turnover",
        "ablation_session_id": (
            "p0-ablation-n500-s20000-useed7-sseed11-tol0.0001"
            "|levels:L0_budget_box|L6_full_mandate|LOO_no_turnover"
        ),
    }
    base.update(overrides)
    return base


def test_p0_defaults_are_the_paired_leave_one_out_run():
    """A plain P0 run must not overwrite historical results/ablation.csv."""
    assert default_output_path() == "results/p0_ablation_loo.csv"
    assert _args().runs == 5
    assert [name for name, _ in select_levels("")] == [
        "L0_budget_box",
        "L6_full_mandate",
        LOO_LEVEL_NAME,
    ]
    assert select_levels("all") != P0_LOO_LEVELS
    assert is_legacy_level_run(select_levels("all"))
    assert not is_legacy_level_run(select_levels(""))


def test_explicit_p0_selector_preserves_leave_one_out_families():
    """The runbook's explicit selector must not alias LOO back to full."""
    selected = select_levels(
        "L0_budget_box,L6_full_mandate,LOO_no_turnover")

    assert selected == P0_LOO_LEVELS
    assert "turnover" in dict(selected)["L6_full_mandate"]
    assert "turnover" not in dict(selected)[LOO_LEVEL_NAME]


def test_loo_no_turnover_removes_only_turnover_field():
    """The leave-one-out treatment is one field, not a second mandate."""
    uni = make_universe(n_bonds=80, seed=3)
    full = default_limits(uni, turnover=0.25)

    l6 = reduce_limits(full, dict(P0_LOO_LEVELS)["L6_full_mandate"])
    loo = reduce_limits(full, dict(P0_LOO_LEVELS)[LOO_LEVEL_NAME])

    changed = []
    for f in fields(full):
        a = getattr(l6, f.name)
        b = getattr(loo, f.name)
        same = a == b
        if isinstance(same, np.ndarray):
            same = bool(same.all())
        if not same:
            changed.append(f.name)

    assert changed == ["turnover"]
    assert l6.turnover == pytest.approx(0.25)
    assert math.isinf(loo.turnover)


def test_pair_id_is_identical_across_levels_and_changes_with_inputs():
    base = p0_pair_id(_args())
    assert base == p0_pair_id(_args(levels="L6_full_mandate"))
    assert base != p0_pair_id(_args(scen_seed=12))
    assert base != p0_pair_id(_args(tol=1e-6))

    assert level_design_id(P0_LOO_LEVELS) != level_design_id(LEVELS)
    assert ablation_session_id(_args(), P0_LOO_LEVELS) != \
        ablation_session_id(_args(levels="L6_full_mandate"),
                            select_levels("L6_full_mandate"))


def test_resume_key_stays_level_solver_scoped_and_current_pair_filtered():
    rows = [
        _row(level="L6_full_mandate", solver="cuopt_pdlp"),
        _row(level=LOO_LEVEL_NAME, solver="cuopt_pdlp+xover"),
        _row(level=LOO_LEVEL_NAME, solver="cuopt_pdlp",
             pair_id="older-config", ablation_session_id="older-session"),
    ]

    assert resume_key_from_row(rows[0]) == ("L6_full_mandate", "cuopt_pdlp")
    assert completed_resume_keys(rows, pair_id=rows[0]["ablation_session_id"]) == {
        ("L6_full_mandate", "cuopt_pdlp"),
        (LOO_LEVEL_NAME, "cuopt_pdlp+xover"),
    }
    assert resume_rows_for_current_pair(rows, pair_id=rows[0]["ablation_session_id"]) == \
        rows[:2]


def test_legacy_resume_preserves_old_rows_without_pair_id():
    old_rows = [
        {"level": "L0_budget_box", "solver": "cuopt_pdlp",
         "wall_s_median": "1.0"},
        {"level": "L6_full_mandate", "solver": "highs",
         "wall_s_median": "2.0"},
    ]

    assert completed_resume_keys(
        old_rows, pair_id="new-p0", legacy=True) == {
            ("L0_budget_box", "cuopt_pdlp"),
            ("L6_full_mandate", "highs"),
        }
    assert resume_rows_for_current_pair(
        old_rows, pair_id="new-p0", legacy=True) == old_rows


def test_p0_annotations_add_endpoint_costs_and_turnover_marginal_evidence():
    rows = [
        _row(level="L0_budget_box", solver="cuopt_pdlp",
             wall_s_median="1.0", endpoint="raw"),
        _row(level="L0_budget_box", solver="cuopt_pdlp+xover",
             wall_s_median="2.0", endpoint="basic"),
        _row(level="L6_full_mandate", solver="cuopt_pdlp",
             wall_s_median="2.0", endpoint="raw"),
        _row(level="L6_full_mandate", solver="cuopt_pdlp+xover",
             wall_s_median="20.0", endpoint="basic"),
        _row(level=LOO_LEVEL_NAME, solver="cuopt_pdlp",
             wall_s_median="2.5", endpoint="raw"),
        _row(level=LOO_LEVEL_NAME, solver="cuopt_pdlp+xover",
             wall_s_median="5.0", endpoint="basic"),
    ]

    out = with_p0_annotations(rows)
    by_key = {(r["level"], r["solver"]): r for r in out}
    full_xo = by_key[("L6_full_mandate", "cuopt_pdlp+xover")]
    loo_xo = by_key[(LOO_LEVEL_NAME, "cuopt_pdlp+xover")]

    assert full_xo["crossover_cost_s"] == pytest.approx(18.0)
    assert full_xo["raw_wall_s_median"] == pytest.approx(2.0)
    assert full_xo["basic_cost_s"] == pytest.approx(18.0)
    assert full_xo["total_endpoint_s"] == pytest.approx(20.0)
    assert full_xo["turnover_marginal_cost_s"] == pytest.approx(15.5)
    assert loo_xo["turnover_marginal_cost_s"] == pytest.approx(15.5)
    assert full_xo["turnover_dominance_verdict"] == "partial"
    assert "87.8" not in str(full_xo)


def test_interpretation_ignores_l0_from_different_session_or_level_design():
    rows = [
        _row(level="L6_full_mandate", solver="cuopt_pdlp",
             wall_s_median="2.0", endpoint="raw"),
        _row(level="L6_full_mandate", solver="cuopt_pdlp+xover",
             wall_s_median="20.0", endpoint="basic"),
        _row(level=LOO_LEVEL_NAME, solver="cuopt_pdlp",
             wall_s_median="2.5", endpoint="raw"),
        _row(level=LOO_LEVEL_NAME, solver="cuopt_pdlp+xover",
             wall_s_median="3.5", endpoint="basic"),
        _row(level="L0_budget_box", solver="cuopt_pdlp",
             wall_s_median="1.0", endpoint="raw",
             ablation_session_id="other-session"),
        _row(level="L0_budget_box", solver="cuopt_pdlp+xover",
             wall_s_median="2.0", endpoint="basic",
             ablation_session_id="other-session"),
    ]

    full_xo = next(
        r for r in with_p0_annotations(rows)
        if r["level"] == "L6_full_mandate" and r["solver"] == "cuopt_pdlp+xover"
    )

    assert full_xo["turnover_dominance_verdict"] == "partial"
    assert "no same-session L0" in full_xo["turnover_dominance_note"]


def test_crossover_cost_does_not_pair_against_other_session_raw_row():
    rows = [
        _row(level="L0_budget_box", solver="cuopt_pdlp",
             wall_s_median="1.0", endpoint="raw"),
        _row(level="L0_budget_box", solver="cuopt_pdlp+xover",
             wall_s_median="2.0", endpoint="basic"),
        _row(level="L6_full_mandate", solver="cuopt_pdlp",
             wall_s_median="2.0", endpoint="raw"),
        _row(level="L6_full_mandate", solver="cuopt_pdlp+xover",
             wall_s_median="20.0", endpoint="basic"),
        _row(level=LOO_LEVEL_NAME, solver="cuopt_pdlp",
             wall_s_median="2.5", endpoint="raw"),
        _row(level=LOO_LEVEL_NAME, solver="cuopt_pdlp+xover",
             wall_s_median="5.0", endpoint="basic"),
        _row(level="L6_full_mandate", solver="cuopt_pdlp",
             wall_s_median="99.0", endpoint="raw",
             ablation_session_id="other-session",
             level_design_id="levels:L0_budget_box|L6_full_mandate|LOO_no_turnover",
             paired_levels="L0_budget_box|L6_full_mandate|LOO_no_turnover"),
    ]

    out = with_p0_annotations(rows)
    current = next(
        r for r in out
        if r["level"] == "L6_full_mandate"
        and r["solver"] == "cuopt_pdlp+xover"
        and r["ablation_session_id"] != "other-session"
    )

    assert current["crossover_cost_s"] == pytest.approx(18.0)
    assert current["basic_cost_s"] == pytest.approx(18.0)
    assert current["turnover_marginal_cost_s"] == pytest.approx(15.5)


def test_loo_interpretation_boundaries_are_deterministic():
    dominant = loo_interpretation(l0_cost_s=1.0, loo_cost_s=1.08,
                                  full_cost_s=10.0)
    partial = loo_interpretation(l0_cost_s=1.0, loo_cost_s=4.0,
                                 full_cost_s=10.0)
    retained = loo_interpretation(l0_cost_s=1.0, loo_cost_s=9.7,
                                  full_cost_s=10.0)
    unknown = loo_interpretation(l0_cost_s=math.nan, loo_cost_s=4.0,
                                 full_cost_s=10.0)

    assert dominant["turnover_dominance_verdict"] == "dominant"
    assert partial["turnover_dominance_verdict"] == "partial"
    assert retained["turnover_dominance_verdict"] == "not_supported"
    assert unknown["turnover_dominance_verdict"] == "partial"
    for verdict in (dominant, partial, retained, unknown):
        assert "87.8" not in verdict["turnover_dominance_note"]


def test_active_limit_summary_counts_only_finite_live_rows():
    uni = make_universe(n_bonds=60, seed=9)
    full = default_limits(uni, turnover=0.25)
    loo = reduce_limits(full, dict(P0_LOO_LEVELS)[LOO_LEVEL_NAME])
    rep = exposure_report(uni, uni.w_benchmark, loo)

    summary = summarise_active_limits(rep)

    assert summary["active_constraint_count"] > 0
    assert "turnover" not in summary["active_constraint_families"].split("|")


def test_dnf_rows_keep_one_cap_policy_and_printable_accounting():
    rows = with_p0_annotations([
        _row(status="user_limit", dnf=True, wall_s_median="",
             wall_s_all="[300.0]", endpoint="basic", notes="time cap"),
    ])

    row = rows[0]
    assert row["dnf_repeat_policy"] == "one-cap-nonresult"
    assert row["total_endpoint_s"] == ""
    assert row["basic_cost_s"] == ""
    assert row["turnover_dominance_verdict"] in {"partial", "not_supported"}


def test_one_cap_runner_does_not_retry_a_dnf_warmup():
    calls = []

    def fake_solve_once(_model, _spec):
        calls.append("solve")
        return 123.0, math.nan, "user_limit"

    model = SimpleNamespace(
        problem=SimpleNamespace(compilation_time=0.25),
        n_assets=2,
        n_scenarios=3,
        nnz_estimate=12,
    )
    spec = SimpleNamespace(
        label="cuopt_pdlp",
        hardware="GPU",
        tolerance_desc="tol=1e-4",
        threads=None,
        internal_time_exposed=True,
        config_full="{}",
        check_feasibility_tol=1e-6,
        check_objective_tol=1e-4,
        algorithm_probe=None,
    )

    rec = run_spec_one_cap(model, spec, n_runs=5, solve_once=fake_solve_once)

    assert calls == ["solve"]
    assert rec.dnf is True
    assert rec.status == "user_limit"
    assert rec.wall_s_all == [123.0]
    assert rec.wall_s_median != rec.wall_s_median
    assert "no retry" in rec.notes


def test_parent_worker_timeout_is_one_cap_envelope_not_repeat_budget():
    assert worker_timeout_s(time_limit_s=1800.0, runs=5) == pytest.approx(2100.0)
    assert worker_timeout_s(time_limit_s=1800.0, runs=5) < \
        1800.0 * (5 + 1) + 300.0


def test_worker_process_timeout_returns_printable_lower_bound_row():
    calls = []
    ticks = iter([100.0, 1912.5])

    def fake_run(cmd, capture_output, text, timeout):
        calls.append((cmd, capture_output, text, timeout))
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout)

    rec, diag = run_worker(
        ["--worker"], timeout_s=1810.0,
        run_subprocess=fake_run, perf_counter=lambda: next(ticks),
    )

    assert diag == ""
    assert rec["status"] == "worker_timeout"
    assert rec["status_source"] == "runner-timeout"
    assert rec["dnf"] is True
    assert rec["dnf_repeat_policy"] == "one-cap-nonresult"
    assert rec["elapsed_s"] == pytest.approx(1812.5)
    assert rec["lower_bound_s"] == pytest.approx(1812.5)
    assert rec["wall_s_all"] == "[1812.5]"
    assert calls[0][3] == pytest.approx(1810.0)

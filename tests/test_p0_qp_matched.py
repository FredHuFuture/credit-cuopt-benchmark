"""P0 QP matched-quality accounting.

These tests are intentionally solver-free.  Task 3 is mostly about what rows
are allowed to become evidence after the expensive solves finish: strict
Clarabel settings, common endpoint/quality gates, objective recomputation, and
ratio exclusion.  None of that should depend on having a GPU.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import run_qp_branch as qp  # noqa: E402

from run_qp_branch import (  # noqa: E402
    ARMS,
    annotate_matched_quality,
    achieved_quality_target,
    grid_for_gate_mode,
    eligibility_reason,
    math_instance_id,
    normalise_rows_for_write,
    resume_key_for_config,
    resume_key_from_row,
    should_repair_strict_clarabel_row,
    solver_options,
)


def _row(**overrides):
    base = {
        "n_bonds": 500,
        "objective_kind": "mv",
        "sigma_basis": "factor",
        "form": "factored",
        "hardware": "CPU",
        "solver": "CLARABEL",
        "arm": "clarabel_strict_factored",
        "status": "optimal",
        "dnf": False,
        "check_ok": True,
        "respects_mandate": True,
        "obj_rel_gap": "1e-10",
        "objective": "0.0100",
        "obj_recomputed": "0.0100",
        "wall_s_median": "2.0",
        "repair_s": "",
        "endpoint": "raw",
        "risk_aversion": "3.0",
        "turnover": "0.25",
        "universe_seed": "7",
        "scenario_seed": "11",
        "n_scenarios_for_sigma": "20000",
        "n_runs_requested": "3",
        "skip_calibration": "False",
        "quality_target": "strict-1e-9",
        "solver_config_id": "clarabel-strict-1e-9",
        "achieved_quality_target": "p0-independent-gate-v1",
    }
    base.update(overrides)
    return base


def test_strict_clarabel_arms_use_one_e_minus_nine_tolerances():
    assert "clarabel_strict_dense" in ARMS
    assert "clarabel_strict_factored" in ARMS

    opts = solver_options("clarabel_strict_factored", 123.0, 8)
    assert opts["time_limit"] == 123.0
    assert opts["tol_feas"] == pytest.approx(1e-9)
    assert opts["tol_gap_abs"] == pytest.approx(1e-9)
    assert opts["tol_gap_rel"] == pytest.approx(1e-9)
    assert opts["max_iter"] >= 500


def test_failed_gate_row_is_excluded_with_specific_reason():
    row = _row(respects_mandate=False, worst_limit="issuer cap")

    ok, reason = eligibility_reason(row, endpoint="raw",
                                    quality_target="strict-1e-9")

    assert not ok
    assert "respects_mandate=False" in reason
    assert "issuer cap" in reason


def test_strict_quality_excludes_optimal_inaccurate_status():
    row = _row(status="optimal_inaccurate")

    ok, reason = eligibility_reason(row)

    assert not ok
    assert "expected optimal" in reason


def test_best_feasible_objective_and_relative_gap_are_grouped_by_instance():
    rows = [
        _row(arm="clarabel_strict_factored", hardware="CPU",
             obj_recomputed="10.0", objective="10.0", wall_s_median="2.0"),
        _row(arm="cuopt_barrier_factored", hardware="GPU", solver="CUOPT",
             obj_recomputed="9.0", objective="9.0", wall_s_median="1.0"),
        _row(arm="bad", hardware="CPU", obj_recomputed="99.0",
             objective="99.0", respects_mandate=False),
        _row(n_bonds=1000, arm="other-instance", obj_recomputed="5.0",
             objective="5.0"),
    ]

    out = annotate_matched_quality(rows)
    by_arm = {r["arm"]: r for r in out}

    assert by_arm["clarabel_strict_factored"]["best_feasible_objective"] \
        == pytest.approx(10.0)
    assert by_arm["cuopt_barrier_factored"]["best_feasible_objective"] \
        == pytest.approx(10.0)
    assert by_arm["cuopt_barrier_factored"]["relative_best_objective_gap"] \
        == pytest.approx(0.1)
    assert by_arm["bad"]["best_feasible_objective"] == pytest.approx(10.0)
    assert by_arm["bad"]["ratio_eligible"] is False
    assert "respects_mandate=False" in by_arm["bad"]["ratio_exclusion_reason"]
    assert by_arm["other-instance"]["best_feasible_objective"] \
        == pytest.approx(5.0)


def test_sample_basis_dense_and_factored_are_different_math_instances():
    dense = _row(sigma_basis="sample", form="dense",
                 arm="clarabel_strict_dense", obj_recomputed="10.0",
                 objective="10.0")
    factored = _row(sigma_basis="sample", form="factored",
                    arm="clarabel_strict_factored", obj_recomputed="5.0",
                    objective="5.0")

    assert math_instance_id(dense) != math_instance_id(factored)

    out = {r["arm"]: r for r in annotate_matched_quality([dense, factored])}
    assert out["clarabel_strict_dense"]["best_feasible_objective"] \
        == pytest.approx(10.0)
    assert out["clarabel_strict_factored"]["best_feasible_objective"] \
        == pytest.approx(5.0)


def test_factor_basis_dense_and_factored_share_math_instance():
    dense = _row(sigma_basis="factor", form="dense",
                 arm="clarabel_strict_dense")
    factored = _row(sigma_basis="factor", form="factored",
                    arm="clarabel_strict_factored")

    assert math_instance_id(dense) == math_instance_id(factored)


def test_repaired_endpoint_uses_solve_plus_repair_latency():
    rows = [
        _row(arm="clarabel_strict_factored", endpoint="repaired",
             wall_s_median="3.0", repair_s="0.7",
             objective="9.5", obj_recomputed="9.5"),
    ]

    out = annotate_matched_quality(rows)[0]

    assert out["common_quality_time_s"] == pytest.approx(3.7)
    assert out["endpoint"] == "repaired"
    assert out["ratio_eligible"] is True


def test_checker_failure_is_not_a_repair_eligible_raw_row():
    row = _row(check_ok=False, check_notes="objective mismatch",
               endpoint="repaired", repair_s="0.1")

    ok, reason = eligibility_reason(row)

    assert not ok
    assert "check_ok=False" in reason
    assert "objective mismatch" in reason


def test_eligibility_recomputes_objective_gap_and_ignores_stale_field():
    row = _row(objective="1.0", obj_recomputed="0.5", obj_rel_gap="0.0")

    ok, reason = eligibility_reason(row)

    assert not ok
    assert "objective_gap" in reason


def test_strict_repair_trigger_covers_every_size_with_purity_gate_failure():
    assert should_repair_strict_clarabel_row(
        "clarabel_strict_factored", 500, True, "optimal", True, 2e-6)
    assert not should_repair_strict_clarabel_row(
        "clarabel_strict_factored", 500, True, "optimal", False, 2e-6)
    assert not should_repair_strict_clarabel_row(
        "clarabel_strict_factored", 500, True, "optimal_inaccurate", True, 2e-6)
    assert should_repair_strict_clarabel_row(
        "clarabel_strict_factored", 300, True, "optimal", True, 2e-6)
    assert not should_repair_strict_clarabel_row(
        "cuopt_barrier_factored", 500, True, "optimal", True, 2e-6)
    assert not should_repair_strict_clarabel_row(
        "clarabel_strict_factored", 500, True, "optimal", True, 5e-7)


def test_solver_stack_ratio_is_nan_for_mismatched_endpoint_pair():
    rows = [
        _row(arm="cuopt_barrier_factored", hardware="GPU", solver="CUOPT",
             endpoint="raw", quality_target="default", wall_s_median="1.0",
             objective="9.9", obj_recomputed="9.9"),
        _row(arm="clarabel_strict_factored", hardware="CPU",
             endpoint="repaired", quality_target="strict-1e-9",
             wall_s_median="2.0", repair_s="0.5",
             objective="10.0", obj_recomputed="10.0"),
    ]

    out = annotate_matched_quality(rows)
    gpu = next(r for r in out if r["hardware"] == "GPU")

    assert math.isnan(gpu["solver_stack_ratio"])
    assert "no matched CPU row" in gpu["solver_stack_ratio_exclusion_reason"]
    assert "endpoint/quality" in gpu["solver_stack_ratio_exclusion_reason"]


def test_solver_stack_ratio_uses_common_achieved_quality_not_solver_config_label():
    rows = [
        _row(arm="cuopt_barrier_factored", hardware="GPU", solver="CUOPT",
             endpoint="raw", quality_target="default",
             solver_config_id="cuopt-barrier-default", wall_s_median="1.5",
             objective="9.9", obj_recomputed="9.9"),
        _row(arm="clarabel_strict_factored", hardware="CPU", solver="CLARABEL",
             endpoint="raw", quality_target="strict-1e-9",
             solver_config_id="clarabel-strict-1e-9", wall_s_median="3.0",
             objective="10.0", obj_recomputed="10.0"),
    ]

    out = annotate_matched_quality(rows)
    gpu = next(r for r in out if r["hardware"] == "GPU")
    cpu = next(r for r in out if r["hardware"] == "CPU")

    assert gpu["solver_stack_ratio"] == pytest.approx(0.5)
    assert gpu["solver_stack_ratio_peer"] == "clarabel_strict_factored"
    assert cpu["solver_stack_ratio"] == pytest.approx(2.0)
    assert cpu["solver_stack_ratio_peer"] == "cuopt_barrier_factored"
    assert gpu["achieved_quality_target"] == "p0-independent-gate-v1"
    assert cpu["achieved_quality_target"] == "p0-independent-gate-v1"


def test_achieved_quality_target_records_failed_independent_gate():
    assert achieved_quality_target(_row()) == "p0-independent-gate-v1"
    assert achieved_quality_target(_row(respects_mandate=False)) \
        == "failed-independent-gate"


class _Args:
    sigma_basis = "factor"
    objective = "mv"
    seed = 7
    scen_seed = 11
    n_scenarios = 20_000
    risk_aversion = 3.0
    turnover = 0.25
    runs = 3
    skip_calibration = False
    repair_strict_clarabel = False
    repair_n = 500


def test_resume_key_distinguishes_raw_from_optional_repair_mode():
    raw = resume_key_for_config(_Args, 500, "clarabel_strict_factored")

    class RepairArgs(_Args):
        repair_strict_clarabel = True

    repaired = resume_key_for_config(RepairArgs, 500,
                                     "clarabel_strict_factored")
    assert raw != repaired

    old_raw_row = _row(repair_mode="none")
    assert resume_key_from_row(old_raw_row) == raw


def test_repaired_completion_skips_same_repair_enabled_config():
    class RepairArgs(_Args):
        repair_strict_clarabel = True

    completed = _row(endpoint="repaired",
                     repair_mode="strict-clarabel-repair-all-sizes",
                     requested_endpoint="repair-if-needed")

    assert resume_key_from_row(completed) == resume_key_for_config(
        RepairArgs, 500, "clarabel_strict_factored")


def test_raw_strict_row_that_passed_repair_gate_skips_same_config():
    class RepairArgs(_Args):
        repair_strict_clarabel = True

    completed = _row(endpoint="raw",
                     repair_mode="strict-clarabel-repair-all-sizes",
                     requested_endpoint="repair-if-needed",
                     respects_mandate=True,
                     worst_rel_violation="0.0")

    assert resume_key_from_row(completed) == resume_key_for_config(
        RepairArgs, 500, "clarabel_strict_factored")


def test_legacy_raw_row_does_not_skip_later_repair_enabled_config():
    class RepairArgs(_Args):
        repair_strict_clarabel = True

    legacy_raw = _row(endpoint="raw", repair_mode="none",
                      respects_mandate=False, worst_rel_violation="2.0e-6")

    assert resume_key_from_row(legacy_raw) != resume_key_for_config(
        RepairArgs, 500, "clarabel_strict_factored")


def test_resume_key_includes_result_defining_config():
    base = resume_key_for_config(_Args, 500, "clarabel_strict_factored")

    class SampleArgs(_Args):
        sigma_basis = "sample"

    class DifferentSeedArgs(_Args):
        scen_seed = 12

    class DifferentRiskArgs(_Args):
        risk_aversion = 4.0

    assert resume_key_for_config(SampleArgs, 500,
                                 "clarabel_strict_factored") != base
    assert resume_key_for_config(DifferentSeedArgs, 500,
                                 "clarabel_strict_factored") != base
    assert resume_key_for_config(DifferentRiskArgs, 500,
                                 "clarabel_strict_factored") != base


def test_gate_mode_restricts_grid_to_the_n500_gate():
    assert grid_for_gate_mode((300, 500, 1000), True, repair_n=500) == (500,)
    assert grid_for_gate_mode((300, 500, 1000), False, repair_n=500) \
        == (300, 500, 1000)


def test_write_normalisation_exposes_full_p0_schema_on_non_results():
    rows = normalise_rows_for_write([
        {
            "n_bonds": 10000,
            "arm": "clarabel_strict_dense",
            "form": "dense",
            "hardware": "CPU",
            "solver": "CLARABEL",
            "status": "skipped_host_memory",
            "dnf": True,
            "notes": "host guard",
        }
    ])

    row = rows[0]
    for key in (
        "endpoint", "quality_target", "method_class", "ratio_eligible",
        "ratio_exclusion_reason", "best_feasible_objective",
        "relative_best_objective_gap", "solver_stack_ratio",
        "common_quality_time_s", "check_ok", "respects_mandate",
        "objective_source", "requested_endpoint",
    ):
        assert key in row
    assert row["endpoint"] == "raw"
    assert row["quality_target"] == "strict-1e-9"
    assert row["ratio_eligible"] is False
    assert "status=skipped_host_memory" in row["ratio_exclusion_reason"]


def test_scenario_generation_oom_becomes_one_recorded_row_per_pending_arm():
    make_rows = getattr(qp, "scenario_generation_nonresults", None)
    assert callable(make_rows), "runner needs an OOM-to-row boundary"

    args = SimpleNamespace(
        sigma_basis="factor", objective="mv", seed=7, scen_seed=11,
        n_scenarios=20_000, risk_aversion=3.0, turnover=0.25, runs=3,
        skip_calibration=False, repair_strict_clarabel=True, repair_n=500,
    )
    pending = [
        ("clarabel_strict_factored", 0.2),
        ("cuopt_barrier_factored", 0.2),
    ]

    rows = make_rows(args, 29_222, pending, "failed to allocate 4.7 GiB")

    assert [r["arm"] for r in rows] == [a for a, _ in pending]
    assert {r["status"] for r in rows} == {"scenario_generation_oom"}
    assert all(r["dnf"] and not r["check_ok"] for r in rows)
    assert all(r["status_source"] == "author-classification" for r in rows)
    assert all("4.7 GiB" in r["notes"] for r in rows)
    assert rows[0]["requested_endpoint"] == "repair-if-needed"
    assert rows[1]["requested_endpoint"] == "raw"

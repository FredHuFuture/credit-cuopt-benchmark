"""Request timing contracts, exercised with real small CPU solves."""

from __future__ import annotations

import csv
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


@pytest.fixture(scope="module", autouse=True)
def independent_highs_scheduler():
    """Other suites request 16/default threads; this lane requests one.

    HiGHS keeps a process-global scheduler. Reset real native state at this
    suite boundary, without changing production behavior or faking a solve.
    """
    import highspy
    highspy.Highs.resetGlobalScheduler(True)
    yield
    highspy.Highs.resetGlobalScheduler(True)


def timing_api():
    # A missing implementation is an assertion failure, not a collection error.
    assert importlib.util.find_spec("src.request_timing") is not None, \
        "the independent request timing lane has not been implemented"
    return importlib.import_module("src.request_timing")


def tiny_instance():
    return timing_api().prepare_instance(n_bonds=12, n_scenarios=32,
                                   universe_seed=7, scenario_seed=11)


@pytest.mark.parametrize("solver,mode,endpoint", [
    ("highs-simplex", "mad", "basic"),
    ("clarabel-socp", "sigma", "interior"),
])
def test_real_cpu_requests_separate_cold_from_cached(solver, mode, endpoint):
    """Catches batch totals, implicit compilation, and mislabeled warm reuse."""
    timing, instance = timing_api(), tiny_instance()
    session = timing.RequestSession(instance, timing.RequestConfig(
        solver=solver, te_mode=mode, te_budget_bp=1000, sigma_target_bp=1000))
    cold = session.run("cold")
    model = session.model
    cold_weights = model.weights().copy()
    cached = session.run("cached", warm_start=True)
    assert session.model is model
    for row, weights in ((cold, cold_weights), (cached, model.weights())):
        assert row["accepted"], row
        assert row["status"] == "optimal"
        assert row["status_source"] == "solver-returned"
        assert row["solver_status"] == "optimal"
        np.testing.assert_array_equal(json.loads(row["weights_json"]), weights)
        assert row["endpoint"] == endpoint
        assert row["repair_s"] == 0
        assert row["canonicalize_s"] > 0
        assert row["solve_call_s"] > 0
        assert row["validation_s"] > 0
        assert row["total_request_s"] >= sum(row[k] for k in (
            "build_s", "canonicalize_s", "solve_call_s", "repair_s", "validation_s"))
        assert row["total_request_s"] < 10
        assert row["timing_complete"]
        assert row["solver_reuse_guaranteed"] is False
        assert "adapter" in row["solve_call_scope"]
        assert "snapshot" in row["resource_note"]
        assert "not peak" in row["resource_note"]
        assert json.loads(row["versions_json"])["cvxpy"]
        assert json.loads(row["thread_config_json"])["solver_threads_requested"] == 1
        assert row["scenario_generation_s"] > 0
        assert row["scenario_seed"] == 11
        assert len(row["model_hash"]) == 64
    assert cold["warm_start_requested"] is False
    assert cold["build_s"] > 0
    assert cached["warm_start_requested"] is True
    assert cached["build_s"] == 0
    assert cached["request_id"] != cold["request_id"]
    assert cached["model_hash"] == cold["model_hash"]
    assert cold["request_kind"] == "fresh_model_cold"
    assert cached["request_kind"] == "same_problem_cached"
    session.run("cold")
    assert session.model is not model


def test_clock_measures_one_request_wall_including_overhead():
    """Catches replacing elapsed wall time with a sum or historical batch total."""
    timing, instance = timing_api(), tiny_instance()
    ticks = iter(range(100))
    session = timing.RequestSession(instance, timing.RequestConfig(
        solver="highs-simplex", te_mode="mad", te_budget_bp=1000,
        sigma_target_bp=1000, time_limit_s=1000))
    row = session.run("cold", clock=lambda: float(next(ticks)))
    assert row["accepted"]
    assert row["build_s"] == 1
    assert row["canonicalize_s"] == 1
    assert row["solve_call_s"] == 1
    assert row["validation_s"] == 1
    assert row["total_request_s"] > 4  # real request boundary includes overhead
    assert row["repair_s"] == 0


def test_hash_covers_data_constraints_and_objective():
    """Catches grouping different seeded inputs, mandates or objectives together."""
    timing, instance = timing_api(), tiny_instance()
    base = timing.RequestConfig(solver="highs-simplex", te_mode="mad")
    from dataclasses import replace
    fingerprint = timing.RequestSession(instance, base).model_hash
    same = timing.prepare_instance(n_bonds=12, n_scenarios=32,
                                  universe_seed=7, scenario_seed=11)
    assert timing.RequestSession(same, base).model_hash == fingerprint
    for cfg in (replace(base, te_budget_bp=20), replace(base, risk_aversion=2),
                replace(base, turnover=0.1), replace(base, te_mode="sigma")):
        assert timing.RequestSession(instance, cfg).model_hash != fingerprint
    different = timing.prepare_instance(n_bonds=12, n_scenarios=32,
                                       universe_seed=7, scenario_seed=12)
    assert timing.RequestSession(different, base).model_hash != fingerprint


def test_acceptance_rejects_status_objective_budget_box_and_purity():
    """Catches trusting Optimal, NaN metrics, or the desk gate alone."""
    timing, instance = timing_api(), tiny_instance()
    session = timing.RequestSession(instance, timing.RequestConfig(
        solver="highs-simplex", te_mode="mad", te_budget_bp=1000,
        sigma_target_bp=1000))
    w = instance.uni.w_benchmark.copy()
    from src.check import empirical_cvar
    obj = float(instance.mu @ w) - 3 * empirical_cvar(-(instance.R @ w), 0.95)[0]
    valid = session.validate(w, "optimal", obj)
    assert valid["accepted"]
    assert not session.validate(w, "user_limit", obj)["accepted"]
    assert not session.validate(w, "optimal", obj + 0.01)["accepted"]
    assert not session.validate(w, "optimal", float("nan"))["accepted"]
    assert not session.validate(w * 1.1, "optimal", obj)["budget_box_pass"]
    concentrated = np.zeros(12)
    concentrated[0] = 1
    assert not session.validate(concentrated, "optimal", obj)["budget_box_pass"]
    session.limits.turnover = 0.0
    assert not session.validate(w, "optimal", obj)["purity_pass"]
    assert not session.validate(np.full(12, np.nan), "optimal", obj)["accepted"]


def test_true_sigma_gate_rejects_mad_feasible_portfolio():
    """Catches accepting the MAD surrogate as proof of true sigma feasibility."""
    timing, instance = timing_api(), tiny_instance()
    session = timing.RequestSession(instance, timing.RequestConfig(
        solver="highs-simplex", te_mode="mad", te_budget_bp=1000,
        sigma_target_bp=0, sigma_tolerance_bp=0))
    row = session.run("cold")
    assert row["status"] == "optimal"
    assert row["objective_consistent"]
    assert row["budget_box_pass"]
    assert row["purity_pass"]
    assert row["te_sigma_bp"] > 0
    assert not row["sigma_pass"]
    assert not row["accepted"]


def test_feasibility_is_separate_from_reported_optimality():
    """Catches conflating a feasible capped iterate with a converged optimum."""
    timing, instance = timing_api(), tiny_instance()
    session = timing.RequestSession(instance, timing.RequestConfig(
        solver="highs-simplex", te_mode="mad", te_budget_bp=1000,
        sigma_target_bp=1000))
    w = instance.uni.w_benchmark
    from src.check import empirical_cvar
    objective = float(instance.mu @ w) - 3 * empirical_cvar(-(instance.R @ w), 0.95)[0]
    result = session.validate(w, "user_limit", objective)
    assert result["portfolio_feasible"]
    assert result["objective_consistent"]
    assert not result["status_accepted"]
    assert not result["optimality_reported"]
    assert not result["accepted"]


def test_invalid_model_is_a_printable_error_without_retry():
    """Catches uncaught build/canonicalization failures and stale cache reuse."""
    timing, instance = timing_api(), tiny_instance()
    from dataclasses import replace
    invalid = replace(instance, mu=np.zeros(2))
    session = timing.RequestSession(invalid, timing.RequestConfig(solver="clarabel-socp"))
    row = session.run("cold")
    assert row["status"] == "ERROR:ValueError"
    assert row["status_source"] == "author-classification"
    assert row["solver_status"] == ""
    assert row["error_stage"] == "build_s"
    assert not row["accepted"]
    assert not row["repeat_eligible"]
    assert "ERROR:ValueError" in timing.format_record(row)


def test_unsupported_and_expired_requests_are_printable_nonresults():
    """Catches timeout iterates being scored or a failed request being repeated."""
    timing, instance = timing_api(), tiny_instance()
    bad = timing.RequestSession(instance, timing.RequestConfig(
        solver="highs-simplex", te_mode="sigma"))
    row = bad.run("cold")
    assert row["status"] == "UNSUPPORTED_BY_METHOD"
    assert row["status_source"] == "author-classification"
    assert not row["accepted"]
    assert not row["repeat_eligible"]
    assert "UNSUPPORTED_BY_METHOD" in timing.format_record(row)
    capped = timing.RequestSession(instance, timing.RequestConfig(
        solver="clarabel-socp", time_limit_s=1e-12))
    row = capped.run("cold")
    assert row["status"] == "TIME_LIMIT"
    assert row["status_source"] == "author-classification"
    assert row["solver_status"] == ""
    assert not row["accepted"]
    assert not row["repeat_eligible"]
    with pytest.raises(ValueError, match="successful"):
        capped.run("cached")


def test_infeasible_is_nonresult_not_cached_retry():
    """Catches re-solving a known infeasible request as a warm timing sample."""
    timing, instance = timing_api(), tiny_instance()
    session = timing.RequestSession(instance, timing.RequestConfig(
        solver="clarabel-socp", te_mode="sigma", te_budget_bp=0,
        turnover=0, time_limit_s=5))
    row = session.run("cold")
    assert row["status"] in ("infeasible", "infeasible_inaccurate")
    assert row["status_source"] == "solver-returned"
    assert row["solver_status"] == row["status"]
    assert not row["accepted"]
    assert not row["repeat_eligible"]
    assert math.isnan(row["objective_recomputed"])
    with pytest.raises(ValueError, match="successful"):
        session.run("cached")


def test_ratios_require_same_problem_endpoint_and_symmetric_reference_quality():
    """Catches ratio laundering across formulations, endpoints, and poor objectives."""
    timing = timing_api()
    reference = dict(accepted=True, status="optimal", model_hash="abc",
                     formulation="mean_cvar_mad_LP", endpoint="basic",
                     request_kind="fresh_model_cold", timing_complete=True,
                     schema_version="request-v1", objective_recomputed=0.02,
                     warm_start_requested=False, objective_quality_target_bp=1.0,
                     thread_config_json='{"solver_threads_requested":1}',
                     total_request_s=4.0, acceptance_config_json="same")
    candidate = {**reference, "total_request_s": 2.0}
    assert timing.matched_request_ratio(candidate, reference, 1.0) == 2.0
    for update in (
        {"accepted": False}, {"model_hash": "different"},
        {"formulation": "mean_cvar_sigma_SOCP"}, {"endpoint": "interior"},
        {"request_kind": "same_problem_cached"}, {"timing_complete": False},
        {"acceptance_config_json": "looser"}, {"status": "user_limit"},
        {"objective_recomputed": 0.019}, {"objective_recomputed": 0.021},
        {"objective_recomputed": float("nan")}, {"total_request_s": 0.0},
    ):
        assert timing.matched_request_ratio({**candidate, **update}, reference, 1.0) is None
    assert timing.matched_request_ratio(candidate, {**reference, "accepted": False}, 1.0) is None


@pytest.mark.parametrize("field,value", [
    ("warm_start_requested", True),
    ("warm_start_requested", None),
    ("objective_quality_target_bp", 2.0),
    ("objective_quality_target_bp", math.nan),
    ("objective_quality_target_bp", None),
    ("thread_config_json", '{"solver_threads_requested":2}'),
    ("thread_config_json", None),
])
@pytest.mark.parametrize("side", ["candidate", "reference"])
def test_ratios_require_explicit_matching_warm_quality_and_threads(field, value, side):
    """Catches comparable-looking rows with different reuse or quality contracts."""
    timing = timing_api()
    rows = {name: dict(accepted=True, status="optimal", model_hash="abc",
        formulation="mean_cvar_mad_LP", endpoint="basic", request_kind="same_problem_cached",
        timing_complete=True, schema_version="request-v1", objective_recomputed=0.02,
        total_request_s=seconds, acceptance_config_json="same", warm_start_requested=False,
        objective_quality_target_bp=1.0, thread_config_json='{"solver_threads_requested":1}')
        for name, seconds in (("candidate", 2.0), ("reference", 4.0))}
    if value is None:
        rows[side].pop(field)
    else:
        rows[side][field] = value
    assert timing.matched_request_ratio(rows["candidate"], rows["reference"], 1.0) is None


def test_ratio_target_cannot_override_both_rows_declared_quality():
    """Catches a caller silently changing an otherwise matching pair's quality target."""
    timing = timing_api()
    reference = dict(accepted=True, status="optimal", model_hash="abc",
        formulation="mean_cvar_mad_LP", endpoint="basic", request_kind="same_problem_cached",
        timing_complete=True, schema_version="request-v1", objective_recomputed=0.02,
        total_request_s=4.0, acceptance_config_json="same", warm_start_requested=True,
        objective_quality_target_bp=2.0, thread_config_json='{"solver_threads_requested":1}')
    candidate = {**reference, "total_request_s": 2.0}
    assert timing.matched_request_ratio(candidate, reference, 1.0) is None
    assert timing.matched_request_ratio(candidate, reference, 2.0) == 2.0


def test_worker_launch_oserror_is_checkpointed_and_next_solver_runs(monkeypatch, tmp_path):
    """Catches escaped launch errors, leaked real pipe ends and joining an unstarted child."""
    from scripts import run_request_bench as runner
    context = runner.mp.get_context("spawn")
    real_start = context.Process.start
    real_pipe = context.Pipe
    connections, attempted = [], []

    def tracked_pipe(*args, **kwargs):
        pair = real_pipe(*args, **kwargs)
        connections.extend(pair)
        return pair

    def fail_first_launch(process):
        attempted.append(process)
        if len(attempted) == 1:
            raise OSError("injected launch failure")
        return real_start(process)

    # Inject only the OS launch boundary. Pipes, process lifecycle, checkpoint
    # and the subsequent Clarabel solve remain real.
    monkeypatch.setattr(context, "Pipe", tracked_pipe)
    monkeypatch.setattr(context.Process, "start", fail_first_launch)
    out = tmp_path / "launch-failure.csv"
    assert runner.main(["--n-bonds", "12", "--n-scenarios", "32",
                        "--te-budget-bp", "1000", "--sigma-target-bp", "1000",
                        "--cached-requests", "0", "--out", str(out)]) == 0
    with out.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    failed, completed = rows
    assert failed["solver"] == "highs-simplex"
    assert failed["status"] == "ERROR:WorkerStartup"
    assert failed["status_source"] == "author-classification"
    assert failed["solver_status"] == ""
    assert failed["timing_complete"] == "False"
    assert failed["accepted"] == "False"
    for key in ("build_s", "canonicalize_s", "solve_call_s", "validation_s", "total_request_s"):
        assert math.isnan(float(failed[key]))
    assert completed["solver"] == "clarabel-socp"
    assert completed["accepted"] == "True"
    assert len(attempted) == 2
    assert len(connections) == 4
    assert all(connection.closed for connection in connections)


def test_checkpoint_flushes_each_row_and_never_overwrites(tmp_path):
    """Catches batch-only persistence and accidental destruction of prior results."""
    timing = timing_api()
    out = tmp_path / "checkpoint.csv"
    with timing.RequestCheckpoint(out) as checkpoint:
        checkpoint.append({"request_id": "first", "accepted": False})
        with out.open(newline="") as stream:
            assert list(csv.DictReader(stream)) == [{"request_id": "first", "accepted": "False"}]
        checkpoint.append({"request_id": "second", "accepted": True})
    before = out.read_bytes()
    with pytest.raises(FileExistsError):
        with timing.RequestCheckpoint(out):
            pass
    assert out.read_bytes() == before


def run_cli(*args):
    # Whole-CLI test watchdog: the default two-solver smoke allows
    # 2 * (30s worker readiness + 2 * 30s requests) = 180s.
    # Allow extra time for imports, preparation, IPC and cleanup.
    # Production request limits and acceptance gates remain unchanged.
    return subprocess.run([sys.executable, str(REPO / "scripts/run_request_bench.py"),
                           *map(str, args)], cwd=REPO, text=True, capture_output=True,
                          timeout=240, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})


def test_cli_cpu_smoke_checkpoints_without_cross_endpoint_ratios(tmp_path):
    """Catches unsafe defaults, unlabeled endpoints, or missing per-request output."""
    timing_api()
    out = tmp_path / "tiny.csv"
    result = run_cli("--n-bonds", 12, "--n-scenarios", 32,
                     "--te-budget-bp", 1000, "--sigma-target-bp", 1000, "--out", out)
    assert result.returncode == 0, result.stdout + result.stderr
    with out.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4
    assert {r["solver"] for r in rows} == {"highs-simplex", "clarabel-socp"}
    assert {r["endpoint"] for r in rows} == {"basic", "interior"}
    assert all(r["accepted"] == "True" for r in rows)
    assert all(float(r["risk_aversion"]) == 3 for r in rows)
    assert all(r["request_ratio"] == "" for r in rows)
    assert all(len(json.loads(r["weights_json"])) == 12 for r in rows)
    assert all(r["status_source"] == "solver-returned" for r in rows)
    before = out.read_bytes()
    again = run_cli("--out", out)
    assert again.returncode != 0
    assert out.read_bytes() == before


def test_cli_gpu_requires_opt_in_before_output(tmp_path):
    """Catches an accidental GPU run from a solver name alone."""
    timing_api()
    out = tmp_path / "forbidden.csv"
    result = run_cli("--solvers", "cuopt-pdlp-1e-6", "--out", out)
    assert result.returncode != 0
    assert "--allow-gpu" in result.stderr
    assert not out.exists()


def test_cli_cap_writes_one_nonresult_per_solver_without_retries(tmp_path):
    """Catches unbounded native work and repeated DNF warm samples."""
    timing_api()
    out = tmp_path / "capped.csv"
    result = run_cli("--n-bonds", 12, "--n-scenarios", 32,
                     "--time-limit", 0.000001, "--cached-requests", 3, "--out", out)
    assert result.returncode == 0, result.stdout + result.stderr
    with out.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert all(r["status"] == "TIME_LIMIT" for r in rows)
    assert all(r["accepted"] == "False" for r in rows)
    assert all(r["status_source"] == "author-classification" for r in rows)
    assert all(json.loads(r["weights_json"]) == [] for r in rows)
    assert all(math.isnan(float(r["cvar_empirical"])) for r in rows)
    assert "TIME_LIMIT" in result.stdout


def test_too_small_universe_is_rejected_before_output(tmp_path):
    """Catches leaking the universe generator's minimum holding size failure."""
    timing = timing_api()
    with pytest.raises(ValueError, match="10"):
        timing.prepare_instance(n_bonds=9, n_scenarios=32)
    out = tmp_path / "invalid.csv"
    result = run_cli("--n-bonds", 9, "--out", out)
    assert result.returncode != 0
    assert "10" in result.stderr
    assert not out.exists()


def test_audit_metrics_and_weights_are_recomputed_for_rejected_portfolio():
    """Catches blank/wrong economics or weights hidden by a failed gate."""
    from dataclasses import replace
    timing, instance = timing_api(), tiny_instance()
    R = np.zeros((4, 12))
    R[:, :2] = [[-0.01, -0.03], [0.02, 0], [0, -0.01], [0.01, 0.03]]
    mu = np.zeros(12)
    mu[:2] = [0.002, 0.004]
    w = np.zeros(12)
    w[:2] = [0.25, 0.75]
    session = timing.RequestSession(replace(instance, R=R, mu=mu),
        timing.RequestConfig(alpha=0.6))
    result = session.validate(w, "user_limit", -0.0518125)
    assert result["cvar_empirical"] == pytest.approx(0.0184375)
    assert result["var_empirical"] == pytest.approx(0.0075)
    assert result["expected_excess_bp_monthly"] == pytest.approx(35)
    assert result["objective_recomputed"] == pytest.approx(-0.0518125)
    assert json.loads(result["weights_json"]) == [0.25, 0.75] + [0.0] * 10
    assert not result["accepted"]


def test_desk_metrics_preserve_a_purity_only_turnover_breach():
    """Catches using the purity boolean as the desk decision or hiding its limit."""
    from dataclasses import replace
    timing, instance = timing_api(), tiny_instance()
    instance = replace(instance, R=np.zeros((4, 12)), mu=np.zeros(12))
    distance = instance.uni.one_way_distance_to_benchmark()
    session = timing.RequestSession(instance, timing.RequestConfig(turnover=distance - 5e-6))
    result = session.validate(instance.uni.w_benchmark, "optimal", 0.0)
    assert result["purity_worst_limit"] == "one-way turnover"
    assert not result["purity_pass"]
    assert result["desk_pass"]
    assert result["desk_worst_multiple"] == pytest.approx(0.05)


def test_nonresults_have_explicit_empty_audit_fields():
    """Catches absent/default-zero fields impersonating a scored portfolio."""
    timing, instance = timing_api(), tiny_instance()
    session = timing.RequestSession(instance, timing.RequestConfig(
        solver="highs-simplex", te_mode="sigma"))
    row = session.run("cold")
    for field in ("cvar_empirical", "var_empirical", "expected_excess_bp_monthly",
                  "desk_worst_multiple"):
        assert math.isnan(row[field])
    assert row["purity_worst_limit"] == ""
    assert row["desk_pass"] is False
    assert json.loads(row["weights_json"]) == []


def test_source_manifest_and_author_config_are_frozen_at_session_setup():
    """Catches unhashed model dependencies or environment substituted for config."""
    timing, instance = timing_api(), tiny_instance()
    config = timing.RequestConfig(solver="highs-simplex", te_mode="mad", risk_aversion=2)
    session = timing.RequestSession(instance, config)
    row = session.base_record("cold")
    sources = json.loads(row["source_sha256_json"])
    required = {"scripts/run_request_bench.py", "scripts/run_te_mandate.py",
                "src/request_timing.py", "src/model.py", "src/credit.py",
                "src/check.py", "src/scenarios.py", "src/universe.py"}
    assert set(sources) == required
    for name in required:
        assert sources[name] == hashlib.sha256((REPO / name).read_bytes()).hexdigest()
    author = json.loads(row["author_config_json"])
    assert author["solver"] == "highs-simplex"
    assert author["risk_aversion"] == 2
    assert author["te_mode"] == "mad"
    assert "environment" not in author
    assert session.base_record("cached")["source_sha256_json"] == row["source_sha256_json"]


def test_author_time_cap_preserves_solver_status_after_real_solve():
    """Catches relabeling author wall limits as solver-returned timeouts."""
    timing, instance = timing_api(), tiny_instance()
    session = timing.RequestSession(instance, timing.RequestConfig(
        te_budget_bp=1000, sigma_target_bp=1000, time_limit_s=1000))
    # All stage clock differences are zero; only the final wall boundary expires.
    # The native solve, canonicalization and validation are real.
    ticks = iter([0.0] * 17 + [1001.0])
    row = session.run("cold", clock=lambda: next(ticks))
    assert row["status"] == "TIME_LIMIT"
    assert row["status_source"] == "author-classification"
    assert row["solver_status"] == "optimal"
    assert len(json.loads(row["weights_json"])) == 12

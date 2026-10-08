"""Single-request measurements, independent of historical batch timing rows.

Preparation and worker startup are outside the request. A cold request creates
a fresh CVXPY model, but does not promise a cold process/device/library. A cached
request uses that same Problem; warm_start is a request, not a reuse guarantee.
The CLI supplies a process watchdog; the in-process API uses solver time limits
and checks elapsed wall time between stages (it cannot interrupt native code).
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time
from uuid import uuid4

import numpy as np
import psutil

from scripts.run_te_mandate import SOLVERS, solver_call_config
from .check import check_solution
from .credit import (PURITY_REL_GATE, credit_constraint_builder, default_limits,
                     exposure_report, gate_verdicts, tracking_error_bp)
from .model import build_mean_cvar
from .scenarios import CreditScenarioConfig, credit_scenarios
from .universe import BondUniverse, make_universe

CONVERGED = {"optimal", "optimal_inaccurate"}
STAGES = ("build_s", "canonicalize_s", "solve_call_s", "repair_s", "validation_s")
SOURCE_FILES = (
    "scripts/run_request_bench.py", "scripts/run_te_mandate.py",
    "src/request_timing.py", "src/model.py", "src/credit.py",
    "src/check.py", "src/scenarios.py", "src/universe.py",
)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class RequestConfig:
    solver: str = "clarabel-socp"
    te_mode: str = "sigma"
    te_budget_bp: float = 10.0
    sigma_target_bp: float = 10.0
    sigma_tolerance_bp: float = 0.00001
    turnover: float = 0.25
    alpha: float = 0.95
    risk_aversion: float = 3.0
    feasibility_tol: float = 1e-6
    objective_tol: float = 1e-6
    time_limit_s: float = 30.0
    threads: int = 1
    allow_gpu: bool = False

    def __post_init__(self):
        if self.solver not in SOLVERS:
            raise ValueError(f"unknown solver {self.solver!r}")
        if self.solver.startswith("cuopt") and not self.allow_gpu:
            raise ValueError("cuOpt requires explicit --allow-gpu")
        if self.te_mode not in {"mad", "sigma"}:
            raise ValueError("te_mode must be mad or sigma")
        if not 0 < self.alpha < 1 or self.threads < 1:
            raise ValueError("alpha must be in (0, 1); threads must be positive")
        for name in ("te_budget_bp", "sigma_target_bp", "sigma_tolerance_bp",
                     "turnover", "risk_aversion", "feasibility_tol", "objective_tol",
                     "time_limit_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.time_limit_s == 0 or self.risk_aversion == 0:
            raise ValueError("time_limit_s and risk_aversion must be positive")


@dataclass(frozen=True)
class PreparedInstance:
    uni: BondUniverse
    R: np.ndarray
    mu: np.ndarray
    scenario_seed: int
    universe_generation_s: float
    scenario_generation_s: float
    instance_hash: str


def prepare_instance(n_bonds=100, n_scenarios=1000, universe_seed=7,
                     scenario_seed=11) -> PreparedInstance:
    """Generate one shared CPU scenario draw, timed outside all requests."""
    if n_bonds < 10 or n_scenarios < 2:
        raise ValueError("at least 10 bonds and two scenarios are required")
    start = time.perf_counter()
    uni = make_universe(n_bonds=n_bonds, seed=universe_seed)
    universe_s = time.perf_counter() - start
    start = time.perf_counter()
    R, mu, _ = credit_scenarios(uni, n_scen=n_scenarios, seed=scenario_seed,
                               use_gpu=False, return_mode="excess")
    scenario_s = time.perf_counter() - start
    digest = hashlib.sha256(b"credit-request-inputs-v1")
    # Include credit group labels, durations, incumbent and benchmark, not just R.
    digest.update(uni.bonds.to_json(orient="split", double_precision=15).encode())
    for array in (R, mu, uni.w0, uni.w_benchmark):
        data = np.ascontiguousarray(array, dtype=np.float64)
        digest.update(_json(data.shape).encode())
        digest.update(data.tobytes())
    return PreparedInstance(uni, R, mu, scenario_seed, universe_s, scenario_s,
                            digest.hexdigest())


def _versions():
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("numpy", "scipy", "pandas", "cvxpy", "highspy", "clarabel",
                 "psutil", "cuopt", "cuopt-cu12", "cuopt-cu13", "cupy-cuda12x",
                 "cupy-cuda13x"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def _snapshot(gpu):
    """Instantaneous process/host/device observations; never peak estimates."""
    snap = {}
    try:
        process = psutil.Process()
        snap.update(pid=process.pid, process_rss_bytes=process.memory_info().rss,
                    process_threads=process.num_threads(),
                    host_available_bytes=psutil.virtual_memory().available)
    except psutil.Error as exc:
        snap["host_error"] = type(exc).__name__
    if gpu:
        try:
            result = subprocess.run([
                "nvidia-smi", "--query-gpu=uuid,name,driver_version,utilization.gpu,memory.used",
                "--format=csv,noheader,nounits"], text=True, capture_output=True,
                timeout=1, check=True)
            snap["gpu_uuid_name_driver_util_pct_memory_mib"] = result.stdout.strip()
        except (OSError, subprocess.SubprocessError) as exc:
            snap["gpu_error"] = type(exc).__name__
    return _json(snap)


def _empty_validation():
    return dict(accepted=False, objective_consistent=False, budget_box_pass=False,
                portfolio_feasible=False, status_accepted=False,
                optimality_reported=False, validation_performed=False,
                purity_pass=False, sigma_pass=False, objective_recomputed=math.nan,
                solver_objective=math.nan, objective_discrepancy=math.nan,
                cvar_empirical=math.nan, var_empirical=math.nan,
                expected_excess_bp_monthly=math.nan, weights_json="[]",
                budget_residual=math.nan, max_bound_violation=math.nan,
                purity_worst_rel=math.nan, purity_worst_limit="",
                desk_pass=False, desk_worst_multiple=math.nan, te_sigma_bp=math.nan,
                te_mad_implied_bp=math.nan)


class RequestSession:
    """One fixed input/solver configuration, with explicitly requested reuse."""

    def __init__(self, instance: PreparedInstance, config: RequestConfig):
        self.instance, self.config = instance, config
        self.limits = default_limits(instance.uni, te_bp=config.te_budget_bp,
                                     te_mode=config.te_mode, turnover=config.turnover)
        self.constraints_json = _json(asdict(self.limits))
        model_spec = _json(dict(version="mean-cvar-credit-v1", alpha=config.alpha,
                               risk_aversion=config.risk_aversion,
                               constraints=asdict(self.limits)))
        self.model_hash = hashlib.sha256(
            (instance.instance_hash + model_spec).encode()).hexdigest()
        self.model = None
        self.repeat_eligible = False
        self.versions_json = _json(_versions())
        # Freeze explicit author choices and a scoped on-disk source manifest
        # outside request timing. Environment observations remain separate.
        self.author_config_json = _json(asdict(config))
        root = Path(__file__).resolve().parents[1]
        self.source_sha256_json = _json({name: hashlib.sha256(
            (root / name).read_bytes()).hexdigest() for name in SOURCE_FILES})

    def base_record(self, kind, warm_start=False):
        cfg, ins = self.config, self.instance
        spec = SOLVERS[cfg.solver]
        endpoint = ("basic" if cfg.solver == "highs-simplex" or spec.get("crossover")
                    else "interior" if spec["conic"] else "nonbasic_first_order")
        acceptance = dict(feasibility_tol=cfg.feasibility_tol,
                          objective_consistency_tol=cfg.objective_tol,
                          purity_rel_tol=PURITY_REL_GATE,
                          sigma_target_bp=cfg.sigma_target_bp,
                          sigma_tolerance_bp=cfg.sigma_tolerance_bp,
                          statuses=sorted(CONVERGED))
        return dict(
            schema_version="request-v1", request_id=uuid4().hex,
            request_kind="fresh_model_cold" if kind == "cold" else "same_problem_cached",
            solver=cfg.solver, method=spec["endpoint"], endpoint=endpoint,
            endpoint_note="configured method endpoint; no independent basis certificate",
            formulation=f"mean_cvar_{cfg.te_mode}_{'SOCP' if cfg.te_mode == 'sigma' else 'LP'}",
            model_hash=self.model_hash, instance_hash=ins.instance_hash,
            n_bonds=ins.uni.n_bonds, n_scenarios=ins.R.shape[0],
            universe_seed=ins.uni.seed, scenario_seed=ins.scenario_seed,
            scenario_device="CPU", scenario_return_mode="excess",
            scenario_config_json=_json(asdict(CreditScenarioConfig())),
            universe_generation_s=ins.universe_generation_s,
            scenario_generation_s=ins.scenario_generation_s,
            alpha=cfg.alpha, risk_aversion=cfg.risk_aversion,
            te_mode=cfg.te_mode, te_budget_bp=cfg.te_budget_bp,
            sigma_target_bp=cfg.sigma_target_bp, constraints_json=self.constraints_json,
            acceptance_config_json=_json(acceptance), versions_json=self.versions_json,
            author_config_json=self.author_config_json,
            source_sha256_json=self.source_sha256_json,
            source_hash_scope="on-disk source bytes captured at session setup",
            thread_config_json=_json(dict(solver_threads_requested=cfg.threads,
                thread_limit_scope="HiGHS/Clarabel; cuOpt uses its existing method defaults",
                environment={name: os.environ.get(name, "unset") for name in (
                    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")})),
            warm_start_requested=bool(warm_start), solver_reuse_guaranteed=False,
            cold_scope="fresh model; process/library/device may already be warm",
            solve_call_scope="CVXPY adapter + transfer + solver + result unpack; not pure device",
            total_request_scope="single request wall; excludes preparation, startup and checkpoint IO",
            time_limit_s=cfg.time_limit_s, solver_options_json="{}",
            status="NOT_RUN", status_source="author-classification",
            solver_status="", error_stage="", note="",
            **{key: 0.0 for key in STAGES}, total_request_s=math.nan,
            supervisor_observed_s=math.nan, timing_complete=False,
            repair_applied=False, repeat_eligible=False,
            **_empty_validation(), resource_before_json="{}", resource_after_json="{}",
            resource_note="instantaneous snapshots, not peak; GPU telemetry is device-wide",
            reference_request_id="", reference_objective=math.nan,
            objective_quality_target_bp=math.nan, request_ratio="",
            ratio_note="no matched accepted reference",
        )

    def validate(self, w, status, solver_objective):
        """Score returned weights independently, including the true sigma gate."""
        out = _empty_validation()
        out.update(status_accepted=status in CONVERGED,
                   optimality_reported=status == "optimal")
        w = np.asarray(w, dtype=float).ravel()
        if w.shape != self.instance.mu.shape or not np.isfinite(w).all():
            return out
        out["weights_json"] = _json(w.tolist())
        if not math.isfinite(solver_objective):
            return out
        cfg, ins = self.config, self.instance
        check = check_solution(w, solver_objective, ins.R, ins.mu, cfg.alpha,
                               cfg.risk_aversion, self.limits.w_max,
                               feasibility_tol=cfg.feasibility_tol,
                               objective_tol=cfg.objective_tol)
        report = exposure_report(ins.uni, w, self.limits, R=ins.R)
        gates = gate_verdicts(report)
        te = tracking_error_bp(ins.uni, w, ins.R, self.limits.periods_per_year)
        out.update(
            validation_performed=True,
            solver_objective=solver_objective, objective_recomputed=check.objective_recomputed,
            cvar_empirical=check.cvar_empirical, var_empirical=check.var_empirical,
            expected_excess_bp_monthly=float(1e4 * (ins.mu @ w)),
            objective_discrepancy=check.objective_abs_gap,
            objective_consistent=bool(np.isfinite(check.objective_abs_gap)
                                     and check.objective_abs_gap <= cfg.objective_tol),
            budget_residual=check.budget_residual,
            max_bound_violation=check.max_bound_violation,
            budget_box_pass=bool(check.budget_residual <= cfg.feasibility_tol
                                and check.max_bound_violation <= cfg.feasibility_tol),
            # Unconstrained report rows deliberately carry +inf slack.
            purity_pass=bool(gates.purity_pass and report["rel_slack"].notna().all()),
            purity_worst_rel=gates.purity_worst_rel,
            purity_worst_limit=gates.purity_worst_limit,
            desk_pass=bool(gates.desk_pass), desk_worst_multiple=gates.desk_worst_multiple,
            te_sigma_bp=te["te_sigma_bp"], te_mad_implied_bp=te["te_mad_implied_bp"],
            sigma_pass=bool(np.isfinite(te["te_sigma_bp"]) and te["te_sigma_bp"]
                            <= cfg.sigma_target_bp + cfg.sigma_tolerance_bp),
        )
        out["portfolio_feasible"] = all(out[k] for k in (
            "budget_box_pass", "purity_pass", "sigma_pass"))
        out["accepted"] = all(out[k] for k in (
            "status_accepted", "objective_consistent", "portfolio_feasible"))
        return out

    def run(self, kind="cold", warm_start=False, *, clock=time.perf_counter):
        if kind not in {"cold", "cached"}:
            raise ValueError("kind must be cold or cached")
        if kind == "cached" and (self.model is None or not self.repeat_eligible):
            raise ValueError("cached request needs a successful prior solve on this Problem")
        start = clock()
        row = self.base_record(kind, warm_start)
        self.repeat_eligible = False
        stage = "setup"

        def remaining():
            left = self.config.time_limit_s - (clock() - start)
            if left <= 0:
                raise TimeoutError("single-request wall time cap reached")
            return left

        def measure(name, operation):
            nonlocal stage
            stage = name
            remaining()
            tick = clock()
            try:
                return operation()
            finally:
                row[name] = clock() - tick

        try:
            remaining()
            row["resource_before_json"] = _snapshot(self.config.allow_gpu)
            if self.config.te_mode == "sigma" and not SOLVERS[self.config.solver]["conic"]:
                row.update(status="UNSUPPORTED_BY_METHOD", note="LP method cannot express sigma SOC")
            else:
                if kind == "cold":
                    self.model = measure("build_s", lambda: build_mean_cvar(
                        self.instance.R, self.instance.mu, alpha=self.config.alpha,
                        risk_aversion=self.config.risk_aversion, w_max=self.limits.w_max,
                        constraint_builder=credit_constraint_builder(
                            self.instance.uni, self.limits, self.instance.R)))
                cfg = solver_call_config(self.config.solver, remaining())
                opts = cfg.opts
                if cfg.solver == "HIGHS":
                    opts["highs_options"].update(threads=self.config.threads, parallel="off",
                        primal_feasibility_tolerance=1e-9, dual_feasibility_tolerance=1e-9)
                elif cfg.solver == "CLARABEL":
                    opts.update(max_threads=self.config.threads, tol_feas=1e-10,
                                tol_gap_abs=1e-10, tol_gap_rel=1e-10)
                data, chain, inverse = measure("canonicalize_s", lambda:
                    self.model.problem.get_problem_data(solver=cfg.solver, solver_opts=opts))
                left = remaining()
                (opts["highs_options"] if cfg.solver == "HIGHS" else opts)["time_limit"] = left
                row["solver_options_json"] = _json(opts)

                def solve():
                    solution = chain.solve_via_data(self.model.problem, data,
                        warm_start=warm_start, verbose=False, solver_opts=opts)
                    self.model.problem.unpack_results(solution, chain, inverse)

                measure("solve_call_s", solve)
                row["solver_status"] = self.model.problem.status
                row["status_source"] = "solver-returned"
                row["status"] = ("TIME_LIMIT" if row["solver_status"] == "user_limit"
                                 else row["solver_status"])
                if self.model.w.value is not None:
                    # Retain the returned iterate even if validation is capped.
                    # JSON null denotes a nonfinite entry, never a usable weight.
                    row["weights_json"] = _json([
                        float(v) if np.isfinite(v) else None for v in self.model.weights()])
                remaining()
                if self.model.w.value is not None and self.model.problem.value is not None:
                    row.update(measure("validation_s", lambda: self.validate(
                        self.model.weights(), row["solver_status"], float(self.model.problem.value))))
                    row["repeat_eligible"] = row["accepted"]
                # No repair is performed in this lane: repair_s stays exactly zero.
        except TimeoutError as exc:
            row.update(status="TIME_LIMIT", status_source="author-classification",
                       error_stage=stage, note=str(exc), accepted=False,
                       repeat_eligible=False)
        except Exception as exc:
            row.update(status=f"ERROR:{type(exc).__name__}", status_source="author-classification",
                       error_stage=stage,
                       note=str(exc)[:500], accepted=False, repeat_eligible=False)
        row["resource_after_json"] = _snapshot(self.config.allow_gpu)
        row["total_request_s"] = clock() - start
        row["timing_complete"] = True
        if row["total_request_s"] > self.config.time_limit_s:
            row.update(status="TIME_LIMIT", status_source="author-classification",
                       accepted=False, repeat_eligible=False,
                       note="single-request wall time cap reached")
        self.repeat_eligible = row["repeat_eligible"]
        return row


def matched_request_ratio(candidate, reference, objective_quality_target_bp):
    """Reference/candidate single-request wall ratio, or None if incomparable.

    The accepted reference's independently recomputed objective is the explicit
    quality target, tested symmetrically. It is not a certificate of optimality.
    Request class, warm-start request and thread configuration must match.
    Both rows must explicitly declare the supplied objective quality target.
    """
    if not math.isfinite(objective_quality_target_bp) or objective_quality_target_bp < 0:
        raise ValueError("objective quality target must be finite and nonnegative")
    keys = ("schema_version", "model_hash", "formulation", "endpoint", "request_kind",
            "acceptance_config_json", "thread_config_json")
    if any(not candidate.get(k) or candidate.get(k) != reference.get(k) for k in keys):
        return None
    for row in (candidate, reference):
        if (type(row.get("warm_start_requested")) is not bool
                or row["warm_start_requested"] != reference.get("warm_start_requested")
                or row.get("objective_quality_target_bp") != objective_quality_target_bp):
            return None
        if (row.get("accepted") is not True or row.get("timing_complete") is not True
                or row.get("status") not in CONVERGED):
            return None
        if not all(math.isfinite(row.get(k, math.nan)) for k in (
                "objective_recomputed", "total_request_s")) or row["total_request_s"] <= 0:
            return None
    if abs(candidate["objective_recomputed"] - reference["objective_recomputed"]) * 1e4 \
            > objective_quality_target_bp:
        return None
    return reference["total_request_s"] / candidate["total_request_s"]


class RequestCheckpoint:
    """Exclusive new CSV; fsync every appended request, never resume/overwrite."""

    def __init__(self, path):
        self.path = Path(path)
        self.stream = None
        self.writer = None

    def __enter__(self):
        self.stream = self.path.open("x", newline="", encoding="utf-8")
        return self

    def append(self, row):
        if self.writer is None:
            self.writer = csv.DictWriter(self.stream, fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(row)
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def __exit__(self, *_):
        self.stream.close()


def format_record(row):
    return (f"{row['solver']} {row['request_kind']} {row['endpoint']} "
            f"status={row['status']} accepted={row['accepted']} "
            f"request_s={row['total_request_s']:.6g} {row['note']}")

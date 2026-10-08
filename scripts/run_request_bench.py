#!/usr/bin/env python
"""Bounded small request lane. Default: 100 bonds x 1000 CPU scenarios.

One fresh-model cold request and one same-Problem cached request per solver.
Only the cached request asks for warm_start by default. No solver reuse promise.
Default methods have different formulations/endpoints, so receive no ratio.
Outputs are exclusively new files; use a new --out path for another run.
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.request_timing import (  # noqa: E402
    SOLVERS, STAGES, RequestCheckpoint, RequestConfig, RequestSession,
    format_record, matched_request_ratio, prepare_instance,
)


def _worker(connection, instance, config):
    try:
        session = RequestSession(instance, config)
        connection.send("ready")
        while True:
            request = connection.recv()
            if request is None:
                break
            kind, warm_start = request
            connection.send(session.run(kind, warm_start))
    finally:
        connection.close()


def bounded_requests(instance, config, cached_requests=1, warm_start=True):
    """Keep one worker's Problem alive, kill it on a cap/crash, never retry DNF.

    The watchdog includes IPC/scheduling; completed rows use the child's precise
    request clock. If killed, stage/total times are unavailable, and the parent's
    observed elapsed time is a separate field, never substituted for a solve.
    Startup has its own 30s cap outside the request.
    """
    template = RequestSession(instance, config)
    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    process = None
    try:
        process = context.Process(target=_worker, args=(child, instance, config))
        process.start()
        child.close()
        if not parent.poll(30):
            row = template.base_record("cold")
            row.update(status="ERROR:WorkerStartupTimeout", error_stage="worker_startup",
                       note="worker did not become ready within 30 seconds")
            for key in STAGES:
                row[key] = math.nan if key != "repair_s" else 0.0
            yield row
            return
        parent.recv()
        for index in range(1 + cached_requests):
            kind = "cold" if index == 0 else "cached"
            requested_warm = warm_start if index else False
            row = template.base_record(kind, requested_warm)
            start = time.perf_counter()
            try:
                parent.send((kind, requested_warm))
                if not parent.poll(config.time_limit_s):
                    process.terminate()
                    process.join(timeout=1)
                    if process.is_alive():
                        process.kill()
                        process.join(timeout=1)
                    row.update(status="TIME_LIMIT", error_stage="worker_watchdog",
                               note="worker cap reached; stage and total times unavailable")
                else:
                    row = parent.recv()
            except (EOFError, BrokenPipeError, OSError) as exc:
                row.update(status="ERROR:WorkerExit", error_stage="worker",
                           note=f"worker exited: {type(exc).__name__}")
            row["supervisor_observed_s"] = time.perf_counter() - start
            if not row["timing_complete"]:
                for key in STAGES:
                    row[key] = math.nan if key != "repair_s" else 0.0
            yield row
            if not row["repeat_eligible"]:
                break
    except (EOFError, BrokenPipeError, OSError) as exc:
        row = template.base_record("cold")
        row.update(status="ERROR:WorkerStartup", error_stage="worker_startup", note=str(exc))
        for key in STAGES:
            row[key] = math.nan if key != "repair_s" else 0.0
        yield row
    finally:
        try:
            if process is not None:
                # start() can fail before there is a child to join.
                if process.pid is not None:
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=1)
                    if process.is_alive():
                        process.kill()
                        process.join(timeout=1)
                if not process.is_alive():
                    process.close()
        finally:
            child.close()
            parent.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-bonds", type=int, default=100)
    parser.add_argument("--n-scenarios", type=int, default=1000)
    parser.add_argument("--universe-seed", type=int, default=7)
    parser.add_argument("--scenario-seed", type=int, default=11)
    parser.add_argument("--solvers", nargs="+", choices=sorted(SOLVERS),
                        default=["highs-simplex", "clarabel-socp"])
    parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument("--te-mode", choices=("auto", "mad", "sigma"), default="auto",
                        help="auto: LP methods use MAD; conic methods use sigma")
    parser.add_argument("--te-budget-bp", type=float, default=10)
    parser.add_argument("--sigma-target-bp", type=float, default=10)
    parser.add_argument("--sigma-tolerance-bp", type=float, default=0.00001)
    parser.add_argument("--turnover", type=float, default=0.25)
    parser.add_argument("--risk-aversion", type=float, default=3.0,
                        help="objective is mu'w - risk_aversion * CVaR (default: 3)")
    parser.add_argument("--time-limit", type=float, default=30,
                        help="single-request wall cap in seconds, including validation")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--cached-requests", type=int, default=1)
    parser.add_argument("--warm-start", action=argparse.BooleanOptionalAction, default=True,
                        help="ask for warm start on cached requests only")
    parser.add_argument("--objective-quality-target-bp", type=float, default=1.0,
                        help="symmetric gap to a matched accepted reference for ratios")
    parser.add_argument("--out", type=Path, default=REPO / "results/p1_request_timing_cpu.csv")
    args = parser.parse_args(argv)
    if not 0 <= args.cached_requests <= 10:
        parser.error("--cached-requests must be between 0 and 10")
    if args.n_bonds < 10 or args.n_scenarios < 2:
        parser.error("at least 10 bonds and two scenarios are required")
    if (not math.isfinite(args.objective_quality_target_bp)
            or args.objective_quality_target_bp < 0):
        parser.error("objective quality target must be finite and nonnegative")
    configs = []
    try:
        for solver in dict.fromkeys(args.solvers):
            mode = ("sigma" if SOLVERS[solver]["conic"] else "mad") \
                if args.te_mode == "auto" else args.te_mode
            configs.append(RequestConfig(solver=solver, te_mode=mode,
                te_budget_bp=args.te_budget_bp, sigma_target_bp=args.sigma_target_bp,
                sigma_tolerance_bp=args.sigma_tolerance_bp, turnover=args.turnover,
                risk_aversion=args.risk_aversion, time_limit_s=args.time_limit,
                threads=args.threads, allow_gpu=args.allow_gpu))
    except ValueError as exc:
        parser.error(str(exc))
    try:
        with RequestCheckpoint(args.out) as checkpoint:
            instance = prepare_instance(args.n_bonds, args.n_scenarios,
                                        args.universe_seed, args.scenario_seed)
            print(f"preparation: universe_s={instance.universe_generation_s:.6g} "
                  f"scenario_s={instance.scenario_generation_s:.6g} (outside requests)", flush=True)
            references = []
            for config in configs:
                for row in bounded_requests(instance, config, args.cached_requests, args.warm_start):
                    row["objective_quality_target_bp"] = args.objective_quality_target_bp
                    # Only a different method in the same formulation/endpoint/class
                    # can become a reference. Never synthesize ratios from batch totals.
                    for reference in references:
                        if row["solver"] == reference["solver"]:
                            continue
                        ratio = matched_request_ratio(row, reference, args.objective_quality_target_bp)
                        if ratio is not None:
                            row.update(request_ratio=ratio, reference_request_id=reference["request_id"],
                                       reference_objective=reference["objective_recomputed"],
                                       ratio_note="matched accepted reference wall / candidate wall")
                            break
                    checkpoint.append(row)
                    print(format_record(row), flush=True)
                    if row["accepted"]:
                        references.append(row)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"request benchmark: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

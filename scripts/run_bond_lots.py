#!/usr/bin/env python3
"""Small CPU-only next-version experiment. Never overwrites an existing output."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.bond_lots import (LotConfig, prepare_lot_book, round_weights, repair_lots,
                           score_lot_portfolio, solve_continuous)
from src.credit import default_limits
from src.scenarios import credit_scenarios
from src.universe import make_universe

FIELDS = """endpoint nav_usd n_bonds n_scen universe_seed scenario_seed status status_source
solver objective_kind objective_solver solver_objective_consistent objective_abs_gap
objective_recomputed expected_excess_bp_monthly cvar var sigma_target_bp te_sigma_bp
te_mad_implied_bp sigma_over_mad_implied active_l1 mean_active_return_bp sigma_pass
sigma_constrained_in_milp linear_purity_pass linear_desk_pass linear_worst_constraint
linear_worst_relative_breach integrality_violations holding_violations trade_violations
quantity_pass n_positions n_trades independent_integer_feasible optimality_proven accepted_integer_portfolio accepted_endpoint ratio_eligible
cash_weight cash_usd initial_cash_weight one_way_turnover_including_cash
l1_distance_from_continuous objective_loss_bp expected_excess_loss_bp_monthly
build_s canonicalize_s solve_call_s validation_s stage_wall_s source_request_s total_request_s
data_generation_s minimum_holding_par increment_par minimum_trade_par cash_max
mip_gap mip_nodes error validation_status scenario_sha256""".split()


def classify_endpoint(endpoint, row):
    """Keep a feasible capped incumbent distinct from a converged endpoint."""
    continuous_feasible = bool(row["linear_purity_pass"] and row["sigma_pass"])
    integer_feasible = bool(continuous_feasible and row["quantity_pass"])
    converged = row["status"] in ("optimal", "optimal_inaccurate", "ROUNDED")
    accepted = bool(converged and row["solver_objective_consistent"] and continuous_feasible
                    and (endpoint == "continuous-socp" or integer_feasible))
    return dict(independent_integer_feasible=integer_feasible,
                optimality_proven=row["status"] == "optimal", accepted_endpoint=accepted,
                accepted_integer_portfolio=bool(endpoint != "continuous-socp" and accepted))


def run_case(config: LotConfig, n_bonds=60, n_scen=500, seed=7, scenario_seed=11,
             time_limit=30., sigma_target_bp=10.):
    """Yield each completed endpoint immediately, including its audit payload."""
    config.validate()
    if not (10 <= n_bonds <= 500 and 10 <= n_scen <= 5000):
        raise ValueError("bounded pilot requires 10..500 bonds and 10..5000 scenarios")
    if not np.isfinite(time_limit) or time_limit <= 0 or not np.isfinite(sigma_target_bp) or sigma_target_bp <= 0:
        raise ValueError("positive finite time limit and sigma target required")
    data_start = time.perf_counter()
    book = prepare_lot_book(make_universe(n_bonds, seed=seed), config)
    limits = default_limits(book.uni)
    R, mu, _ = credit_scenarios(book.uni, n_scen=n_scen, seed=scenario_seed, use_gpu=False)
    data_s = time.perf_counter() - data_start
    fingerprint = hashlib.sha256(R.tobytes() + mu.tobytes()).hexdigest()
    base = dict(nav_usd=config.nav_usd, n_bonds=n_bonds, n_scen=n_scen, universe_seed=seed,
                scenario_seed=scenario_seed, initial_cash_weight=book.cash0, data_generation_s=data_s,
                ratio_eligible=False, scenario_sha256=fingerprint, **{
                    k: v for k, v in asdict(config).items() if k != "nav_usd"})
    source_w = None
    source_cash = None
    source_row = None
    for endpoint in ("continuous-socp", "naive-rounding", "lot-milp-repair"):
        stage_start = time.perf_counter()
        row = {k: "" for k in FIELDS}
        row.update(base, endpoint=endpoint, accepted_endpoint=False, accepted_integer_portfolio=False)
        payload = dict(endpoint=endpoint, nav_usd=config.nav_usd, limits=asdict(limits),
                       assumptions=asdict(config), incumbent_lots=book.incumbent_lots.tolist(),
                       weight_per_lot=book.weight_per_lot.tolist())
        if endpoint != "continuous-socp" and source_w is None:
            row.update(status="SKIPPED_SOURCE_NOT_ACCEPTED", status_source="author-classification")
            yield row, payload
            continue
        if endpoint == "continuous-socp":
            result = solve_continuous(book, limits, R, mu, sigma_target_bp=sigma_target_bp, time_limit=time_limit)
            w, cash = result.w, result.cash
            row.update(solver="CLARABEL", objective_kind="mean_minus_3_cvar", source_request_s=0.)
        elif endpoint == "lot-milp-repair":
            result = repair_lots(book, limits, source_w, source_cash, time_limit=time_limit)
            w, cash = result.w, result.cash
            row.update(solver="HIGHS", objective_kind="l1_distance_not_portfolio_objective",
                       source_request_s=source_row["total_request_s"])
        else:
            result = None
            w, cash = round_weights(source_w, book)
            row.update(status="ROUNDED", status_source="author-classification", solver="none",
                       objective_kind="no_solver_objective", source_request_s=source_row["total_request_s"],
                       build_s=0., canonicalize_s=0., solve_call_s=0., solver_objective_consistent=True)
        if result is not None:
            row.update(status=result.status, status_source="solver-returned", objective_solver=result.objective_solver,
                       build_s=result.build_s, canonicalize_s=result.canonicalize_s, solve_call_s=result.solve_call_s,
                       mip_gap=result.mip_gap, mip_nodes=result.mip_nodes, error=result.error)
        check_start = time.perf_counter()
        if w is not None:
            try:
                scored, residuals = score_lot_portfolio(book, limits, w, cash, R, mu, sigma_target_bp=sigma_target_bp)
                row["validation_status"] = "SCORED"
            except (ValueError, FloatingPointError) as exc:
                row.update(validation_status="ERROR:"+type(exc).__name__, error=str(exc),
                           linear_purity_pass=False, sigma_pass=False, quantity_pass=False,
                           independent_integer_feasible=False, solver_objective_consistent=False)
                w = None
        else:
            row["validation_status"] = "NO_PORTFOLIO"
        if w is not None:
            row.update(scored)
            if result is not None:
                recomputed = (scored["objective_recomputed"] if endpoint == "continuous-socp" else
                              float(np.abs(w-source_w).sum()) + abs(cash-source_cash))
                gap = abs(recomputed-result.objective_solver)
                row.update(objective_abs_gap=gap, solver_objective_consistent=bool(np.isfinite(gap) and gap <= 1e-7))
            row.update(classify_endpoint(endpoint, row))
            payload.update(weights=w.tolist(), cash=cash,
                           quantities_par=(w/book.weight_per_lot*config.increment_par).tolist(),
                           residuals=residuals.to_dict(orient="records"))
            if endpoint != "continuous-socp":
                row.update(l1_distance_from_continuous=float(np.abs(w-source_w).sum()) + abs(cash-source_cash),
                           objective_loss_bp=(source_row["objective_recomputed"]-scored["objective_recomputed"])*1e4,
                           expected_excess_loss_bp_monthly=source_row["expected_excess_bp_monthly"]-scored["expected_excess_bp_monthly"])
        row["validation_s"] = time.perf_counter()-check_start
        row["stage_wall_s"] = time.perf_counter()-stage_start
        row["total_request_s"] = row["source_request_s"] + row["stage_wall_s"]
        if endpoint == "continuous-socp":
            source_row = row.copy()
            if row["accepted_endpoint"]:
                source_w, source_cash = w, cash
        yield row, payload


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_experiment(out: Path, records, metadata):
    """Exclusive output creation and per-endpoint flush, never mutate old results."""
    paths = (out, out.with_suffix(".portfolios.jsonl"), out.with_suffix(".metadata.json"))
    for path in paths:
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}; choose a new --out")
    out.parent.mkdir(parents=True, exist_ok=True)
    with paths[2].open("x", encoding="utf-8") as meta:
        json.dump(_json_safe(metadata), meta, indent=2, allow_nan=False)
        meta.write("\n")
    with out.open("x", newline="", encoding="utf-8") as summary, paths[1].open("x", encoding="utf-8") as audit:
        writer = csv.DictWriter(summary, fieldnames=FIELDS)
        writer.writeheader()
        summary.flush()
        for row, payload in records:
            # Payload first: every completed CSV row has an auditable portfolio.
            audit.write(json.dumps(_json_safe(payload), allow_nan=False) + "\n")
            audit.flush()
            writer.writerow(row)
            summary.flush()
            print(f"NAV={row['nav_usd']:,.0f} {row['endpoint']}: {row['status']} "
                  f"accepted={row['accepted_endpoint']} sigma={row.get('te_sigma_bp', '')} "
                  f"total_request_s={row.get('total_request_s', '')}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT / "results/p1_bond_lots_cpu.csv")
    p.add_argument("--nav", type=float, nargs="+", default=[1e6, 1e7, 1e8])
    p.add_argument("--bonds", type=int, default=60)
    p.add_argument("--scenarios", type=int, default=500)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--scenario-seed", type=int, default=11)
    p.add_argument("--time-limit", type=float, default=30)
    p.add_argument("--sigma-target-bp", type=float, default=10)
    p.add_argument("--increment-par", type=float, default=1000)
    p.add_argument("--minimum-holding-par", type=float, default=10000)
    p.add_argument("--minimum-trade-par", type=float, default=5000)
    p.add_argument("--cash-max", type=float, default=.02)
    args = p.parse_args()
    configs = [LotConfig(v, args.increment_par, args.minimum_holding_par, args.minimum_trade_par, args.cash_max) for v in args.nav]
    for config in configs:
        config.validate()
    if len(set(args.nav)) != len(args.nav):
        p.error("duplicate NAV cases are not independent repeats")
    meta = dict(schema="p1-bond-lots-v1", utc=datetime.now(timezone.utc).isoformat(),
                platform=platform.platform(), python=platform.python_version(),
                packages={v: importlib.metadata.version(v) for v in ("numpy", "cvxpy", "clarabel", "highspy")},
                git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                source_sha256={str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
                               for f in (Path(__file__), ROOT/"src/bond_lots.py", ROOT/"src/credit.py", ROOT/"src/scenarios.py", ROOT/"src/universe.py")},
                command=sys.argv, assumptions="hypothetical uniform lots; synthetic settlement-price proxy; accrued=0; cash excess return=0",
                timing="one observation per endpoint/NAV; single request excluding scenario generation, imports and output I/O; repair includes source request; no speed ratios",
                threads="HiGHS automatic; BLAS environment controls apply; no thread-matched performance claim",
                scope="continuous SOCP reference; MILP restores linear constraints only; sigma post-check, NOT MISOCP")
    def records():
        for config in configs:
            yield from run_case(config, args.bonds, args.scenarios, args.seed, args.scenario_seed,
                                args.time_limit, args.sigma_target_bp)
    write_experiment(args.out, records(), meta)


if __name__ == "__main__":
    main()

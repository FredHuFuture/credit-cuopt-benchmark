"""Quantity checks must not be replaced by continuous-weight feasibility."""
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.universe import make_universe
from src.credit import default_limits
from src.bond_lots import (
    LotConfig, prepare_lot_book, round_weights, quantity_check,
    cash_turnover, repair_lots, score_lot_portfolio, solve_continuous,
)


@pytest.fixture(scope="module", autouse=True)
def isolated_highs_scheduler():
    # HiGHS thread pools are process-global; older suites request two threads.
    import highspy
    highspy.Highs.resetGlobalScheduler(True)
    yield
    highspy.Highs.resetGlobalScheduler(True)


def simple_book():
    uni = make_universe(20, seed=7)
    uni.bonds = uni.bonds.iloc[:6].copy()
    uni.w_benchmark = np.full(6, 1 / 6)
    uni.bonds.loc[:, "price"] = 100.0
    uni.w0 = np.array([0.5, 0.5, 0, 0, 0, 0])
    book = prepare_lot_book(uni, LotConfig(100_000, 1_000, 2_000, 5_000, 0.1))
    limits = replace(default_limits(book.uni), rating_bands={}, sector_caps={},
                     issuer_cap=1.0, w_max=1.0, cell_duration_tol=100,
                     total_duration_tol=100, dts_max=1e9, dts_min=0,
                     turnover=1.0, te_bp=np.inf)
    return book, limits


def test_rounding_preserves_cash_and_does_not_renormalize():
    book, _ = simple_book()
    w, cash = round_weights(np.array([0.503, 0.473, 0.024, 0, 0, 0]), book)
    np.testing.assert_allclose(w, [0.50, 0.47, 0.02, 0, 0, 0], atol=1e-14)
    assert cash == pytest.approx(0.01)
    assert w.sum() == pytest.approx(0.99)


def test_cash_leg_counts_in_one_way_turnover():
    assert cash_turnover(np.array([0.4, 0.4]), np.array([0.5, 0.5]), 0.2, 0) == pytest.approx(0.2)


def test_holding_minimum_is_not_trade_minimum():
    book, _ = simple_book()
    # Holding 47k is fine; the 3k sale violates the hypothetical 5k minimum trade.
    check = quantity_check(np.array([0.50, 0.47, 0.02, 0, 0, 0]), book)
    assert check["holding_violations"] == 0
    assert check["trade_violations"] == 2
    assert not check["quantity_pass"]


def test_off_lattice_and_too_small_holdings_fail_independently():
    book, _ = simple_book()
    check = quantity_check(np.array([0.501, 0.48, 0.01, 0, 0, 0]), book)
    assert check["integrality_violations"] == 1
    assert check["holding_violations"] == 1


@pytest.mark.parametrize("field,value", [("nav_usd", 0), ("increment_par", -1),
                                          ("minimum_holding_par", 2500),
                                          ("cash_max", 1.1)])
def test_invalid_quantity_assumptions_rejected(field, value):
    cfg = LotConfig(100_000, 1_000, 2_000, 5_000, 0.1)
    with pytest.raises(ValueError):
        prepare_lot_book(make_universe(20), replace(cfg, **{field: value}))


def test_milp_repairs_trade_size_and_distance_is_recomputed():
    book, limits = simple_book()
    target = np.array([0.50, 0.47, 0.03, 0, 0, 0])
    out = repair_lots(book, limits, target, 0.0, time_limit=10)
    assert out.status == "optimal"
    assert quantity_check(out.w, book)["quantity_pass"]
    # Closest permissible trade is a 5k transfer, costing .02+.02 in L1.
    np.testing.assert_allclose(out.w, [0.50, 0.45, 0.05, 0, 0, 0], atol=1e-7)
    assert out.objective_solver == pytest.approx(0.04, abs=1e-7)
    assert out.total_request_s >= out.build_s + out.canonicalize_s + out.solve_call_s


def test_milp_infeasible_is_nonresult_not_success():
    book, limits = simple_book()
    out = repair_lots(book, replace(limits, w_max=0.01), book.uni.w0, 0, time_limit=10)
    assert out.status == "infeasible"
    assert out.w is None


def test_independent_score_catches_true_sigma_even_when_linear_limits_pass():
    book, limits = simple_book()
    R = np.tile(np.array([[.02, -.02, 0, 0, 0, 0], [-.02, .02, 0, 0, 0, 0]]), (10, 1))
    w = np.array([1., 0, 0, 0, 0, 0])
    row, residuals = score_lot_portfolio(book, limits, w, 0, R, np.zeros(6), sigma_target_bp=1)
    assert row["linear_purity_pass"]
    assert row["quantity_pass"]
    assert not row["sigma_pass"]
    assert not row["accepted_integer_portfolio"]
    assert "one-way turnover including cash" in set(residuals["item"])


def test_cash_bound_and_actual_nav_breach_dollars_are_scored():
    book, limits = simple_book()
    R = np.zeros((10, 6))
    row, residuals = score_lot_portfolio(book, limits, np.array([0.6, 0.5, 0, 0, 0, 0]), -.1, R, np.zeros(6))
    assert not row["linear_purity_pass"]
    assert residuals.loc[residuals.item == "cash range", "breach_usd"].item() == pytest.approx(10_000)


def test_continuous_and_integer_endpoints_share_linear_mandate():
    uni = make_universe(20, seed=7)
    book = prepare_lot_book(uni, LotConfig(10_000_000, 1_000, 10_000, 5_000, .02))
    limits = default_limits(book.uni)
    from src.scenarios import credit_scenarios
    R, mu, _ = credit_scenarios(book.uni, n_scen=100, seed=11, use_gpu=False)
    continuous = solve_continuous(book, limits, R, mu, sigma_target_bp=10, time_limit=15)
    assert continuous.status == "optimal"
    source, _ = score_lot_portfolio(book, limits, continuous.w, continuous.cash, R, mu, sigma_target_bp=10)
    assert source["linear_purity_pass"] and source["sigma_pass"]
    assert abs(source["objective_recomputed"] - continuous.objective_solver) < 1e-7
    repaired = repair_lots(book, limits, continuous.w, continuous.cash, time_limit=15)
    row, _ = score_lot_portfolio(book, limits, repaired.w, repaired.cash, R, mu, sigma_target_bp=10)
    assert row["linear_purity_pass"] and row["quantity_pass"]
    # Sigma is only checked, never silently enforced or assumed for this MILP.
    assert row["sigma_constrained_in_milp"] is False


def test_runner_reports_all_arms_and_includes_source_in_repair_latency(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_bond_lots import run_case, write_experiment
    cfg = LotConfig(10_000_000, 1_000, 10_000, 5_000, .02)
    records = list(run_case(cfg, n_bonds=20, n_scen=100, time_limit=10))
    assert [r[0]["endpoint"] for r in records] == ["continuous-socp", "naive-rounding", "lot-milp-repair"]
    raw, rounded, repaired = [r[0] for r in records]
    assert repaired["total_request_s"] == pytest.approx(raw["total_request_s"] + repaired["stage_wall_s"])
    assert rounded["objective_kind"] == "no_solver_objective"
    assert repaired["objective_kind"] == "l1_distance_not_portfolio_objective"
    assert repaired["solver_objective_consistent"]
    assert repaired["ratio_eligible"] is False  # different feasible sets and no repeat study
    out = tmp_path / "new.csv"
    write_experiment(out, records, {"purpose": "test"})
    assert out.exists() and out.with_suffix(".portfolios.jsonl").exists()
    before = out.read_bytes()
    with pytest.raises(FileExistsError):
        write_experiment(out, records, {})
    assert out.read_bytes() == before


def test_feasibility_and_proof_are_separate_for_time_limit_incumbent():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_bond_lots import classify_endpoint
    row = dict(linear_purity_pass=True, sigma_pass=True, quantity_pass=True,
               solver_objective_consistent=True, status="user_limit")
    verdict = classify_endpoint("lot-milp-repair", row)
    assert verdict["independent_integer_feasible"] is True
    assert verdict["accepted_endpoint"] is False
    assert verdict["optimality_proven"] is False


def test_malformed_portfolio_shapes_cannot_broadcast_into_another_problem():
    book, limits = simple_book()
    with pytest.raises(ValueError, match="shape"):
        prepare_lot_book(replace(book.uni, w0=np.array([1.])), book.config)
    with pytest.raises(ValueError, match="shape"):
        quantity_check(np.array([.1]), book)
    with pytest.raises(ValueError, match="shape"):
        repair_lots(book, limits, book.uni.w0[:, None], 0)


def test_nan_mandate_cannot_disable_an_enabled_constraint():
    book, limits = simple_book()
    bad = replace(limits, turnover=np.nan)
    with pytest.raises(ValueError, match="turnover"):
        repair_lots(book, bad, book.uni.w0, book.cash0)
    with pytest.raises(ValueError, match="turnover"):
        score_lot_portfolio(book, bad, book.uni.w0, book.cash0, np.zeros((10, 6)), np.zeros(6))


def test_corrupt_solver_output_checkpoints_failure_and_skips_dependencies(monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_bond_lots
    from src.bond_lots import LotSolve
    # Native solver boundary only: real validation and runner must handle NaNs.
    monkeypatch.setattr(run_bond_lots, "solve_continuous", lambda *a, **k:
                        LotSolve(np.full(20, np.nan), 0., "user_limit", np.nan, 0., 0., 0., 0.))
    records = list(run_bond_lots.run_case(LotConfig(), n_bonds=20, n_scen=100))
    first = records[0][0]
    assert first["status"] == "user_limit"
    assert first["validation_status"] == "ERROR:ValueError"
    assert first["accepted_endpoint"] is False
    assert all(r[0]["status"] == "SKIPPED_SOURCE_NOT_ACCEPTED" for r in records[1:])

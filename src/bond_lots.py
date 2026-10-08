"""Illustrative quantity layer; no exchange, issuer or liquidity validation.

Par increments and minimum sizes are hypothetical inputs, not N-PORT fields.
Synthetic prices are settlement-price proxies with zero accrued interest.
Cash has zero *excess* return. No rounding endpoint renormalizes bond weights.
The MILP restores the linear mandate, NOT the sigma cone. Always score both.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import time

import cvxpy as cp
import numpy as np
import pandas as pd

from .check import empirical_cvar
from .credit import (CreditLimits, PURITY_REL_GATE, UNIT_WEIGHT,
                     credit_constraint_builder, exposure_report, gate_verdicts,
                     score_breaches, tracking_error_bp)
from .universe import BondUniverse


@dataclass(frozen=True)
class LotConfig:
    nav_usd: float = 10_000_000
    increment_par: float = 1_000
    minimum_holding_par: float = 10_000
    minimum_trade_par: float = 5_000
    cash_max: float = .02

    def validate(self):
        for name in ("nav_usd", "increment_par", "minimum_holding_par", "minimum_trade_par"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(self.cash_max) or not 0 <= self.cash_max < 1:
            raise ValueError("cash_max must be in [0, 1)")
        for value in (self.minimum_holding_par, self.minimum_trade_par):
            if not np.isclose(value / self.increment_par, round(value / self.increment_par), rtol=0, atol=1e-9):
                raise ValueError("this experiment requires minima to be multiples of increment_par")


@dataclass(frozen=True)
class LotBook:
    uni: BondUniverse
    config: LotConfig
    weight_per_lot: np.ndarray
    incumbent_lots: np.ndarray
    cash0: float


def _vector(value, n, name):
    vector = np.asarray(value, dtype=float)
    if vector.shape != (n,):
        raise ValueError(f"{name} shape must be ({n},), got {vector.shape}")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must be finite")
    return vector


def _validate_limits(limits):
    # Explicit +inf disables supported limits; NaN is never a disabling flag.
    for key, value in asdict(limits).items():
        if key == "te_mode":
            continue
        values = list(value.values()) if isinstance(value, dict) else value
        numbers = np.asarray(values, dtype=float)
        if np.isnan(numbers).any() or np.isneginf(numbers).any() or (numbers < 0).any():
            raise ValueError(f"invalid mandate {key}: use nonnegative values or explicit +inf")
    if not np.isfinite(limits.periods_per_year) or limits.periods_per_year <= 0:
        raise ValueError("periods_per_year must be finite and positive")
    if any(lo > hi for lo, hi in limits.rating_bands.values()) or limits.dts_min > limits.dts_max:
        raise ValueError("mandate lower bound exceeds upper bound")


def _risk_inputs(R, mu, n, alpha, risk_aversion, sigma_target_bp):
    R = np.asarray(R, dtype=float)
    if R.ndim != 2 or R.shape[1] != n or R.shape[0] < 2 or not np.isfinite(R).all():
        raise ValueError("scenario matrix must be finite, with shape (S >= 2, n_bonds)")
    _vector(mu, n, "mu")
    if not 0 < alpha < 1 or not np.isfinite(risk_aversion) or risk_aversion < 0:
        raise ValueError("invalid alpha or risk_aversion")
    if np.isnan(sigma_target_bp) or sigma_target_bp < 0:
        raise ValueError("invalid sigma target")


def prepare_lot_book(uni: BondUniverse, config: LotConfig) -> LotBook:
    """Create a lattice-valid incumbent once, shared by all comparison arms.

    Floor original holdings to lots, drop sub-minimum positions, leave cash.
    This changes the incumbent from historical runs; it is a NEW experiment.
    The initial book need not satisfy the final mandate or final cash cap.
    """
    config.validate()
    prices = uni.bonds["price"].to_numpy(dtype=float)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("prices must be positive settlement-price proxies")
    w0 = _vector(uni.w0, uni.n_bonds, "incumbent")
    if (not np.isfinite(w0).all() or (w0 < 0).any()
            or abs(w0.sum() - 1) > 1e-6):
        raise ValueError("original incumbent must be finite, nonnegative and fully invested")
    unit = config.increment_par * prices / (100 * config.nav_usd)
    lots = np.floor(w0 / unit + 1e-10).astype(np.int64)
    lots[lots < round(config.minimum_holding_par / config.increment_par)] = 0
    adjusted = lots * unit
    return LotBook(replace(uni, w0=adjusted), config, unit, lots, float(1 - adjusted.sum()))


def round_weights(w: np.ndarray, book: LotBook) -> tuple[np.ndarray, float]:
    """Naive nearest-lot rounding; intentionally does not repair overspending."""
    w = np.asarray(w, dtype=float)
    if w.shape != book.weight_per_lot.shape or not np.isfinite(w).all():
        raise ValueError("weights must be a finite aligned vector")
    lots = np.floor(np.maximum(w, 0) / book.weight_per_lot + .5)
    lots[lots < round(book.config.minimum_holding_par / book.config.increment_par)] = 0
    rounded = lots * book.weight_per_lot
    return rounded, float(1 - rounded.sum())


def cash_turnover(w, w0, cash: float, cash0: float) -> float:
    return .5 * (float(np.abs(np.asarray(w) - w0).sum()) + abs(cash - cash0))


def quantity_check(w: np.ndarray, book: LotBook) -> dict:
    counts = _vector(w, book.uni.n_bonds, "weights") / book.weight_per_lot
    tol = 1e-6  # in units of lots, not USD or NAV
    minimum = book.config.minimum_holding_par / book.config.increment_par
    trade_min = book.config.minimum_trade_par / book.config.increment_par
    trades = np.abs(counts - book.incumbent_lots)
    integrality = int((np.abs(counts - np.rint(counts)) > tol).sum())
    holdings = int((((counts > tol) & (counts < minimum - tol)) | (counts < -tol)).sum())
    trade_bad = int(((trades > tol) & (trades < trade_min - tol)).sum())
    return dict(integrality_violations=integrality, holding_violations=holdings,
                trade_violations=trade_bad, quantity_pass=not (integrality or holdings or trade_bad),
                n_positions=int((counts > tol).sum()), n_trades=int((trades > tol).sum()))


def _linear_constraints(book, limits, w, cash):
    _validate_limits(limits)
    # The shared builder's turnover omits cash; replace it, do not duplicate it.
    linear = replace(limits, turnover=np.inf, te_bp=np.inf)
    extra, _ = credit_constraint_builder(book.uni, linear)(w)
    cons = [w >= 0, w <= limits.w_max, cash >= 0,
            cash <= book.config.cash_max, cp.sum(w) + cash == 1] + extra
    if np.isfinite(limits.turnover):
        cons.append(cp.norm1(w - book.uni.w0) + cp.abs(cash - book.cash0) <= 2 * limits.turnover)
    return cons


@dataclass
class LotSolve:
    w: np.ndarray | None
    cash: float | None
    status: str
    objective_solver: float
    build_s: float
    canonicalize_s: float
    solve_call_s: float
    total_request_s: float
    mip_gap: float = np.nan
    mip_nodes: float = np.nan
    error: str = ""


def _finish_solve(start, problem, w, cash, solver, opts) -> LotSolve:
    built = time.perf_counter()
    canonical = built
    finished = built
    try:
        problem.get_problem_data(solver)
        canonical = time.perf_counter()
        problem.solve(solver=solver, warm_start=False, verbose=False, **opts)
        finished = time.perf_counter()
        status = problem.status
        # A time-limited incumbent is not an optimal timing observation.
        has_weights = w.value is not None and cash.value is not None
        weights = np.asarray(w.value).ravel().copy() if has_weights else None
        cash_value = float(cash.value) if has_weights else None
        info = problem.solver_stats.extra_stats
        return LotSolve(weights, cash_value, status, float(problem.value), built-start,
                        canonical-built, finished-canonical, time.perf_counter()-start,
                        float(getattr(info, "mip_gap", np.nan)),
                        float(getattr(info, "mip_node_count", np.nan)))
    except (cp.error.SolverError, MemoryError) as exc:
        finished = time.perf_counter()
        return LotSolve(None, None, "ERROR:" + type(exc).__name__, np.nan,
                        built-start, max(0, canonical-built), finished-canonical,
                        finished-start, error=str(exc))


def solve_continuous(book: LotBook, limits: CreditLimits, R, mu,
                     sigma_target_bp=10., alpha=.95, risk_aversion=3., time_limit=60.) -> LotSolve:
    """Cash-aware continuous SOCP reference for the SAME lot-adjusted incumbent."""
    start = time.perf_counter()
    _risk_inputs(R, mu, book.uni.n_bonds, alpha, risk_aversion, sigma_target_bp)
    w = cp.Variable(book.uni.n_bonds)
    cash, t = cp.Variable(), cp.Variable()
    u = cp.Variable(len(R), nonneg=True)
    cons = _linear_constraints(book, limits, w, cash)
    cons += [u >= -R @ w - t]
    if np.isfinite(sigma_target_bp):
        active = R @ (w - book.uni.w_benchmark)
        centered = active - cp.sum(active) / len(R)
        cons += [cp.norm(centered) <= sigma_target_bp / (1e4 * np.sqrt(limits.periods_per_year)) * np.sqrt(len(R))]
    obj = cp.Maximize(mu @ w - risk_aversion * (t + cp.sum(u) / ((1-alpha)*len(R))))
    return _finish_solve(start, cp.Problem(obj, cons), w, cash, cp.CLARABEL,
                         dict(tol_feas=1e-10, tol_gap_abs=1e-10, tol_gap_rel=1e-10,
                              max_iter=300, time_limit=time_limit))


def repair_lots(book: LotBook, limits: CreditLimits, target_w, target_cash,
                time_limit=30.) -> LotSolve:
    """Minimum L1 bond+cash displacement under quantity and linear constraints.

    A sell and a buy cannot be netted to fabricate a qualifying minimum trade.
    No sigma constraint, cardinality cap, transaction costs or market liquidity.
    """
    start = time.perf_counter()
    n = book.uni.n_bonds
    target_w = _vector(target_w, n, "target weights")
    if not np.isscalar(target_cash) or not np.isfinite(target_cash):
        raise ValueError("target cash must be a finite scalar")
    if not np.isfinite(limits.w_max) or limits.w_max <= 0:
        raise ValueError("a finite positive single-bond cap is required for tight big-M bounds")
    counts = cp.Variable(n, integer=True)
    held = cp.Variable(n, boolean=True)
    buys, sells = cp.Variable(n, boolean=True), cp.Variable(n, boolean=True)
    cash = cp.Variable()
    w = cp.multiply(book.weight_per_lot, counts)
    max_lots = np.floor(limits.w_max / book.weight_per_lot + 1e-10)
    max_trade = np.maximum(max_lots, book.incumbent_lots)
    min_hold = round(book.config.minimum_holding_par / book.config.increment_par)
    min_trade = round(book.config.minimum_trade_par / book.config.increment_par)
    difference = counts - book.incumbent_lots
    cons = _linear_constraints(book, limits, w, cash) + [
        counts >= min_hold * held, counts <= cp.multiply(max_lots, held),
        buys + sells <= 1,
        difference >= min_trade * buys - cp.multiply(max_trade, sells),
        difference <= cp.multiply(max_trade, buys) - min_trade * sells,
    ]
    objective = cp.Minimize(cp.norm1(w - target_w) + cp.abs(cash - target_cash))
    return _finish_solve(start, cp.Problem(objective, cons), w, cash, cp.HIGHS,
                         dict(highs_options=dict(time_limit=time_limit, mip_rel_gap=1e-6,
                                                  mip_abs_gap=1e-9, mip_feasibility_tolerance=1e-8,
                                                  primal_feasibility_tolerance=1e-9,
                                                  random_seed=0)))


def score_lot_portfolio(book: LotBook, limits: CreditLimits, w, cash, R, mu,
                        sigma_target_bp=np.inf, alpha=.95, risk_aversion=3.):
    """Independent NumPy scoring, including cash, actual NAV and true sigma."""
    _validate_limits(limits)
    _risk_inputs(R, mu, book.uni.n_bonds, alpha, risk_aversion, sigma_target_bp)
    w = _vector(w, book.uni.n_bonds, "weights")
    if w.shape != book.uni.w0.shape or not np.isfinite(w).all() or not np.isfinite(cash):
        raise ValueError("non-finite or misaligned portfolio is not scoreable")
    rep = exposure_report(book.uni, w, replace(limits, turnover=np.inf, te_bp=np.inf), R=R)
    rep = rep.loc[rep.family != "turnover"].copy()
    turnover = cash_turnover(w, book.uni.w0, cash, book.cash0)
    extra = [
        ("cash_budget", "bond plus cash budget", -abs(float(w.sum())+cash-1), 1.),
        ("bounds", "single-bond bounds", min(float(w.min()), float(limits.w_max-w.max())), max(limits.w_max, 1e-6)),
        ("cash", "cash range", min(cash, book.config.cash_max-cash), max(book.config.cash_max, 1e-6)),
        ("turnover", "one-way turnover including cash", limits.turnover-turnover, max(limits.turnover, 1e-6)),
    ]
    added = pd.DataFrame([dict(family=f, item=i, slack=s, scale=k, unit=UNIT_WEIGHT)
                          for f, i, s, k in extra])
    rep = pd.concat([rep, added], ignore_index=True)
    rep["rel_slack"] = rep.slack / rep.scale
    enabled = np.isfinite(rep.slack)
    if (rep.slack.isna().any() or np.isneginf(rep.slack).any()
            or not np.isfinite(rep.loc[enabled, "rel_slack"]).all()):
        raise ValueError("invalid residual for an enabled mandate constraint")
    rep = score_breaches(rep)
    # Legacy exposure reports use a fixed illustrative NAV. This experiment varies it.
    rep["breach_usd"] = rep.breach_bp_weight * book.config.nav_usd / 1e4
    gates = gate_verdicts(rep)
    te = tracking_error_bp(book.uni, w, R, limits.periods_per_year)
    cvar, var = empirical_cvar(-(R @ w), alpha)
    sigma_pass = bool(te["te_sigma_bp"] <= sigma_target_bp * (1+PURITY_REL_GATE))
    qty = quantity_check(w, book)
    row = dict(**qty, **te, linear_purity_pass=gates.purity_pass,
               linear_desk_pass=gates.desk_pass, linear_worst_constraint=gates.purity_worst_limit,
               linear_worst_relative_breach=gates.purity_worst_rel,
               sigma_pass=sigma_pass, sigma_target_bp=sigma_target_bp,
               sigma_constrained_in_milp=False, cash_weight=cash,
               cash_usd=cash*book.config.nav_usd, one_way_turnover_including_cash=turnover,
               objective_recomputed=float(mu @ w)-risk_aversion*cvar,
               expected_excess_bp_monthly=float(mu @ w)*1e4, cvar=cvar, var=var,
               accepted_integer_portfolio=bool(gates.purity_pass and sigma_pass and qty["quantity_pass"]))
    return row, rep

"""Interactive re-optimization: the credit LP driven directly through cuOpt.

This is the part of the project that is not a benchmark. A portfolio manager
does not want one optimal portfolio; they want to ask *what if* — tighten the
issuer limit, lift the DTS floor, halve the turnover budget — and see the
answer while the thought is still fresh. That only works if a re-solve costs
about as long as moving a slider.

Two design decisions make it work:

1. **The problem shape never changes.** `set_pdlp_warm_start_data` requires an
   identical variable and constraint count, so every tunable limit is a *row
   bound*, not a row that gets added or removed. Turning a constraint "off"
   means widening its bound to something inert, never deleting it.
2. **The matrix is built once.** CVXPY is the right tool for a model you solve
   a handful of times; for a slider it is the wrong one. Here the constraint
   matrix is assembled once in CSR and handed to cuOpt's `DataModel`; a slider
   then rewrites a few entries of the bound vectors and re-solves.

Warm start turns out NOT to be the win it looks like — the negative result
-------------------------------------------------------------------------
cuOpt's PDLP warm start requires presolve to be disabled, because warm-start
state is sized to the problem the solver actually iterated on. That makes the
real choice "warm without presolve" against "cold with presolve", and on this
problem (500 bonds x 20,000 scenarios, ~10M nonzeros) the second wins
comfortably: warm start saves about a quarter against a cold solve on equal
terms, but presolve on its own is worth roughly an order of magnitude. So
`solve()` defaults to `warm=False`, and the warm path is kept because
measuring the trade is more useful than assuming it.

The other trap, worth more than either: **crossover defaults to on** through
this API. Left alone it was costing ~8x per solve — an interactive view wants
the interior point immediately, and the limit badge on screen is what tells
the user whether that point is good enough.

The LP is the same one `src/model.py` and `src/credit.py` describe — written
out explicitly here because the row layout has to be known to mutate it:

    variables : w (N)  t (1)  u (S)  d (N, turnover auxiliaries)
    rows      : budget | CVaR tail (S) | rating bands | sector caps
                | issuer caps | duration cells | total duration | DTS band
                | turnover | d >= w - w0 (N) | d >= w0 - w (N)

Minimized:  -mu'w + lam * ( t + 1/((1-alpha) S) * sum(u) )
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from .credit import CreditLimits
from .universe import BondUniverse


@dataclass
class Knobs:
    """Everything a slider may change. All map to bounds or objective terms."""

    risk_aversion: float = 3.0
    alpha: float = 0.95
    sector_cap: float = 0.30          # absolute cap, all sectors
    issuer_cap: float = 0.03
    dts_min_mult: float = 0.90        # x benchmark DTS
    dts_max_mult: float = 1.10
    turnover: float = 0.25            # one-way


@dataclass
class InteractiveModel:
    """A built credit LP plus the row/column bookkeeping needed to mutate it."""

    uni: BondUniverse
    limits: CreditLimits
    R: np.ndarray
    mu: np.ndarray
    A: sp.csr_matrix              # already row-scaled — see row_scale
    row_lb: np.ndarray            # RAW bounds; divided by row_scale at solve
    row_ub: np.ndarray
    var_lb: np.ndarray
    var_ub: np.ndarray
    n_bonds: int
    n_scen: int
    #: Per-row normalisation. A DTS row carries coefficients around 900
    #: (bp-years) while a CVaR row carries about 0.01 — five orders of
    #: magnitude apart in the same matrix. First-order methods are punished
    #: hard by that: unscaled, a 10M-nonzero solve took 61s; scaled, seconds.
    #: Rows and their bounds are divided by this, which leaves the feasible
    #: set identical and the conditioning far better.
    row_scale: np.ndarray = field(default_factory=lambda: np.ones(0))
    rows: dict[str, slice | int] = field(default_factory=dict)
    idx: dict[str, slice | int] = field(default_factory=dict)
    bench_dts: float = 0.0
    _warm: object = None
    _last_knobs: Knobs | None = None

    def scaled_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Row bounds in the same scaled units as `A`."""
        with np.errstate(invalid="ignore"):
            return self.row_lb / self.row_scale, self.row_ub / self.row_scale

    # ---- objective -----------------------------------------------------
    def objective(self, k: Knobs) -> np.ndarray:
        """Minimized objective: -mu'w + lam*t + lam/((1-a)S) * sum(u)."""
        c = np.zeros(self.A.shape[1])
        c[self.idx["w"]] = -self.mu
        c[self.idx["t"]] = k.risk_aversion
        c[self.idx["u"]] = k.risk_aversion / ((1.0 - k.alpha) * self.n_scen)
        return c

    # ---- bounds --------------------------------------------------------
    def apply_knobs(self, k: Knobs) -> None:
        """Rewrite the row bounds a slider controls. Shape never changes."""
        self.row_ub[self.rows["sector"]] = k.sector_cap
        self.row_ub[self.rows["issuer"]] = k.issuer_cap
        self.row_lb[self.rows["dts"]] = k.dts_min_mult * self.bench_dts
        self.row_ub[self.rows["dts"]] = k.dts_max_mult * self.bench_dts
        self.row_ub[self.rows["turnover"]] = 2.0 * k.turnover   # L1 counts both sides


def build_interactive(uni: BondUniverse, R: np.ndarray, mu: np.ndarray,
                      limits: CreditLimits, k: Knobs,
                      scale_rows: bool = True) -> InteractiveModel:
    """Assemble the CSR constraint matrix once.

    `scale_rows=False` leaves the raw units in place, which is worth
    measuring: cuOpt equilibrates internally, so pre-scaling can help or can
    fight the solver's own heuristic.
    """
    N, S = uni.n_bonds, R.shape[0]
    w0 = uni.w0

    G_rating, rating_labels = uni.group_matrix("rating")
    G_sector, _ = uni.group_matrix("sector")
    G_issuer, _ = uni.group_matrix("issuer_id")
    D_cell, _ = uni.duration_contribution_matrix()
    n_r, n_s, n_i, n_c = (G_rating.shape[0], G_sector.shape[0],
                          G_issuer.shape[0], D_cell.shape[0])

    n_var = N + 1 + S + N
    iw, it = slice(0, N), N
    iu, id_ = slice(N + 1, N + 1 + S), slice(N + 1 + S, n_var)

    def pad(M, cols_before=0, cols_after=0):
        blocks = []
        if cols_before:
            blocks.append(sp.csr_matrix((M.shape[0], cols_before)))
        blocks.append(M)
        if cols_after:
            blocks.append(sp.csr_matrix((M.shape[0], cols_after)))
        return sp.hstack(blocks, format="csr")

    tail = 1 + S + N            # columns after w
    blocks, lbs, ubs, rows = [], [], [], {}
    r = 0

    def add(M, lb, ub, name):
        nonlocal r
        blocks.append(M)
        lbs.append(np.atleast_1d(lb))
        ubs.append(np.atleast_1d(ub))
        rows[name] = slice(r, r + M.shape[0]) if M.shape[0] > 1 else r
        r += M.shape[0]

    # budget
    add(pad(sp.csr_matrix(np.ones((1, N))), 0, tail), 1.0, 1.0, "budget")

    # CVaR tail: R w + t + u >= 0
    cvar = sp.hstack([sp.csr_matrix(R),
                      sp.csr_matrix(np.ones((S, 1))),
                      sp.identity(S, format="csr"),
                      sp.csr_matrix((S, N))], format="csr")
    add(cvar, np.zeros(S), np.full(S, np.inf), "cvar")

    # rating bands (two-sided in a single row each)
    lo = np.array([limits.rating_bands.get(x, (0.0, 1.0))[0]
                   for x in rating_labels])
    hi = np.array([limits.rating_bands.get(x, (0.0, 1.0))[1]
                   for x in rating_labels])
    add(pad(sp.csr_matrix(G_rating), 0, tail), lo, hi, "rating")

    add(pad(sp.csr_matrix(G_sector), 0, tail),
        np.zeros(n_s), np.full(n_s, k.sector_cap), "sector")
    add(pad(sp.csr_matrix(G_issuer), 0, tail),
        np.zeros(n_i), np.full(n_i, k.issuer_cap), "issuer")

    bench_cell = D_cell @ uni.w_benchmark
    add(pad(sp.csr_matrix(D_cell), 0, tail),
        bench_cell - limits.cell_duration_tol,
        bench_cell + limits.cell_duration_tol, "cell_duration")

    bench_dur = float(uni.w_benchmark @ uni.duration)
    add(pad(sp.csr_matrix(uni.duration.reshape(1, -1)), 0, tail),
        bench_dur - limits.total_duration_tol,
        bench_dur + limits.total_duration_tol, "total_duration")

    bench_dts = float(uni.w_benchmark @ uni.dts)
    add(pad(sp.csr_matrix(uni.dts.reshape(1, -1)), 0, tail),
        k.dts_min_mult * bench_dts, k.dts_max_mult * bench_dts, "dts")

    # turnover: sum(d) <= 2 * one-way budget
    turn = sp.hstack([sp.csr_matrix((1, N + 1 + S)),
                      sp.csr_matrix(np.ones((1, N)))], format="csr")
    add(turn, -np.inf, 2.0 * k.turnover, "turnover")

    # d >= w - w0   and   d >= w0 - w
    eye = sp.identity(N, format="csr")
    zeros_mid = sp.csr_matrix((N, 1 + S))
    add(sp.hstack([-eye, zeros_mid, eye], format="csr"),
        -w0, np.full(N, np.inf), "turn_pos")
    add(sp.hstack([eye, zeros_mid, eye], format="csr"),
        w0, np.full(N, np.inf), "turn_neg")

    A = sp.vstack(blocks, format="csr")
    row_lb, row_ub = np.concatenate(lbs), np.concatenate(ubs)

    # Row equilibration: divide every row (and its bounds) by its largest
    # coefficient. Purely a change of units per constraint; the feasible set
    # is untouched.
    if scale_rows:
        absA = abs(A)
        row_scale = np.maximum(absA.max(axis=1).toarray().ravel(), 1e-12)
        A = sp.csr_matrix(sp.diags(1.0 / row_scale) @ A)
    else:
        row_scale = np.ones(A.shape[0])

    var_lb = np.zeros(n_var)
    var_ub = np.empty(n_var)
    var_ub[iw] = limits.w_max
    var_lb[it], var_ub[it] = -np.inf, np.inf          # VaR level is free
    var_ub[iu] = np.inf
    var_ub[id_] = np.inf

    return InteractiveModel(
        uni=uni, limits=limits, R=R, mu=mu, A=A, row_lb=row_lb, row_ub=row_ub,
        var_lb=var_lb, var_ub=var_ub, n_bonds=N, n_scen=S,
        row_scale=row_scale, rows=rows,
        idx={"w": iw, "t": it, "u": iu, "d": id_}, bench_dts=bench_dts,
    )


def solve(m: InteractiveModel, k: Knobs, warm: bool = False,
          tol: float = 1e-4, presolve: bool | None = None) -> dict:
    """Solve at the current knob settings. Returns weights, timing, status.

    `warm=True` reuses the previous iterate — and forces presolve off, because
    warm-start state is sized to the problem cuOpt actually iterated on. On
    this problem that trade is a bad one (see the module docstring), so warm
    is opt-in rather than the default.
    """
    from cuopt.linear_programming.data_model import DataModel
    from cuopt.linear_programming.solver import Solve
    from cuopt.linear_programming.solver_settings import (
        PDLPSolverMode, SolverMethod, SolverSettings,
    )

    # Timer starts HERE, above the DataModel rebuild, not just around Solve().
    # The comparison this number ends up in (scripts/check_demo_latency.py)
    # puts the CVXPY path's model build inside its own timer, so excluding the
    # rebuild here would time a shorter span for one path than the other and
    # flatter this one. What a slider actually costs is everything from "knob
    # moved" to "weights back", which is what is measured now.
    t0 = time.perf_counter()
    m.apply_knobs(k)
    c = m.objective(k)
    lb, ub = m.scaled_bounds()

    dm = DataModel()
    dm.set_csr_constraint_matrix(m.A.data, m.A.indices, m.A.indptr)
    dm.set_objective_coefficients(c)
    dm.set_constraint_lower_bounds(lb)
    dm.set_constraint_upper_bounds(ub)
    dm.set_variable_lower_bounds(m.var_lb)
    dm.set_variable_upper_bounds(m.var_ub)

    ss = SolverSettings()
    ss.set_parameter("method", SolverMethod.PDLP)
    ss.set_parameter("pdlp_solver_mode", PDLPSolverMode.Stable2)
    ss.set_parameter("log_to_console", 0)
    ss.set_parameter("time_limit", 120.0)
    # Crossover OFF, explicitly. Left at its default this path was spending
    # ~60s per solve against ~2.5s for the same model through CVXPY. Roughly
    # 8x of that gap was an unrequested crossover running on every slider
    # move; the remainder is unattributed and most likely row conditioning
    # (see scripts/diag_lp_shapes.py, which is what found the crossover half).
    # An interactive view wants the interior point fast; the limit badge on
    # screen is what tells the user whether that point is good enough, and
    # the tolerance control is how they buy a cleaner one.
    ss.set_parameter("crossover", 0)
    # Warm start and presolve are mutually exclusive here: warm-start state is
    # sized to the problem cuOpt actually iterated on, so a presolved solve
    # produces state a non-presolved re-solve rejects outright ("same amount
    # of constraints"). Presolve therefore goes off exactly when warm start
    # goes on — and measuring that trade is the point of the A/B.
    use_presolve = (not warm) if presolve is None else presolve
    ss.set_parameter("presolve", 1 if use_presolve else 0)
    for p in ("absolute_primal_tolerance", "relative_primal_tolerance",
              "absolute_dual_tolerance", "relative_dual_tolerance",
              "absolute_gap_tolerance", "relative_gap_tolerance"):
        ss.set_parameter(p, tol)

    used_warm = False
    if warm and m._warm is not None and not use_presolve:
        try:
            ss.set_pdlp_warm_start_data(m._warm)
            used_warm = True
        except Exception:  # noqa: BLE001 - fall back to a cold solve
            used_warm = False

    res = Solve(dm, ss)
    elapsed = time.perf_counter() - t0

    primal = res.get_primal_solution()
    status = res.get_termination_status()
    if not use_presolve:      # only unpresolved state is reusable
        try:
            m._warm = res.get_pdlp_warm_start_data()
        except Exception:  # noqa: BLE001 - warm start unavailable
            m._warm = None

    w = np.asarray(primal[:m.n_bonds]) if primal is not None else None
    m._last_knobs = k
    return {
        "w": w,
        "seconds": elapsed,
        "warm": used_warm,
        "status": getattr(status, "name", str(status)),
        "objective": float(res.get_primal_objective()),
    }

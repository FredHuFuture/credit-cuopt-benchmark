"""The credit constraint set — what makes this a *credit* optimizer.

Six families, all linear, all recognisable to anyone who has run a corporate
bond book:

1. **Rating bands** — weight in each rating bucket within a range. Ratings are
   how credit mandates are written and how risk is delegated.
2. **Sector caps** — no more than X% in Energy, and so on.
3. **Per-issuer concentration** — the binding one in practice. Defaults happen
   at the *issuer*, and this universe's scenario engine defaults issuers, not
   bonds, so this limit is what stops five bonds of one name becoming five
   simultaneous losses.
4. **Duration matching vs the benchmark, by maturity cell AND in aggregate** —
   the portfolio must carry the index's rate exposure cell by cell, or a
   barbell masquerades as an index match; and the cell tolerances must be
   backed by an aggregate band, or four ±0.25y cells silently permit a ±1.00y
   total-duration bet.
5. **DTS band, two-sided** — duration times spread (Ben Dor et al., 2007).
   Spread volatility scales with spread level, so DTS, not duration, is the
   natural sizing measure for credit risk. Bounded *below* as well as above:
   a one-sided cap lets the optimizer collapse credit beta toward zero, which
   for a benchmarked mandate is an enormous unstated tracking-error position —
   and it leaves the cap itself inert. This is the one limit here with no
   open-source precedent we could find (July 2026).
6. **Turnover budget** — you cannot trade the whole book to chase a
   marginally better portfolio.
7. **Ex-ante tracking error vs the benchmark** — added in M6. A DTS band
   controls systematic credit beta; it does not control TE, and every real
   benchmarked mandate carries both. See the long note below: this is the one
   family whose *form* decides what problem class the optimizer is solving.

Families 1-6 are linear in w, so the Rockafellar-Uryasev objective stays a
pure LP. Family 7 is linear only in the `mad` and `active_l1` forms.

Units trap, stated once: `turnover` below is ONE-WAY (sells, equivalently
buys). The LP constrains ||w - w0||_1, which counts both sides, hence the
factor of 2. Getting this wrong silently doubles or halves the budget.

---

**On tracking error, and why it gets three forms rather than one.**

Bloomberg's optimized fixed income index methodology 34100 carries ex-ante
tracking error both as an objective term and as a hard constraint at
**<= 10bp**, relaxed in 10bp increments only when the problem proves
infeasible. That is the mandate this project should be judged against, and it
is tighter than the tolerance a first-order solver is run at. Whether that
matters is an empirical question, so the constraint is implemented in the
three forms the industry actually uses and the runner reports all three:

- ``te_mode="sigma"`` — the faithful reading. Ex-ante TE is the standard
  deviation of active return, so the constraint is ``||A - mean(A)||_2 <=
  tau * sqrt(S)``: a **second-order cone** constraint. This takes the problem
  out of the LP class entirely, which means cuOpt's PDLP cannot solve it at
  all. That is not a bug to route around; it is the single most important
  thing this family measures.
- ``te_mode="mad"`` — the LP-preserving analogue: mean absolute deviation of
  active return (Konno-Yamazaki applied to the active portfolio), rescaled by
  ``sqrt(pi/2)`` so the budget number means the same thing in both modes
  **under a Gaussian**. Active credit returns are not Gaussian — that is the
  whole premise of using CVaR — so treat the two budgets as comparable in
  intent, not in value. MAD is reported alongside sigma by
  `tracking_error_bp` precisely so the reader can see the gap rather than
  take the conversion on faith.
- ``active_l1`` — the weight-space budget, ``||w - b||_1 <= c``. Bloomberg
  index methodology 38847 (Global Corporate PAB Select) minimizes exactly
  this and carries no tracking-error term at all, so this is not a
  simplification of practice; it *is* practice, at the same vendor, on the
  same date as 34100. Cheapest of the three: ~2N rows and no scenario block.

Cost note, because it changes the benchmark: `sigma` and `mad` both introduce
a second S x N block (``R @ w`` appears in the CVaR tail constraint and again
here), roughly doubling the nonzeros of the dominant block. CVXPY does not
share that subexpression. Introducing an explicit ``y = R @ w`` auxiliary
would keep one block at the cost of S equality rows; that is deliberately not
done here because it would change the baseline model and break comparability
with the committed M1-M3 results.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np
import pandas as pd

from .universe import BondUniverse

#: E|X| = sigma * sqrt(2/pi) for zero-mean Gaussian X, so sigma = MAD *
#: sqrt(pi/2). Used ONLY to express a MAD budget on the same scale as a
#: sigma budget. Credit active returns are not Gaussian; this constant makes
#: the two modes comparable in intent, not equal in value.
MAD_TO_SIGMA = float(np.sqrt(np.pi / 2.0))

#: Bloomberg optimized fixed income index methodology 34100 constrains ex-ante
#: tracking error at 10bp (relaxed in 10bp increments only if infeasible).
BLOOMBERG_34100_TE_BP = 10.0

# --------------------------------------------------------------------------
# Breach units and the two gates a breach is scored against
#
# One gate was never enough. The repo's original 1e-6 relative gate is a
# NUMERICAL statement: "this point is on the constraint to within the
# arithmetic". A desk asks a different question — "would anyone notice?" —
# and answers it at about 1bp of portfolio weight, because that is the
# precision positions are reported at and lot rounding does orders of
# magnitude more violence than 1e-6. Both are legitimate; they disagree; so
# both are reported and neither is quietly picked.
# --------------------------------------------------------------------------

#: Book size every dollar figure in the exposure report is quoted on. Defined
#: once so a reader on a different book rescales the whole CSV by one ratio,
#: and so no function invents its own notional.
NOTIONAL_USD = 1_000_000_000.0

#: Gate 1, "purity": the repo's original threshold, applied to `rel_slack`
#: (slack normalized by each limit's own characteristic scale). Unchanged in
#: value and in meaning — every committed result was scored against it.
PURITY_REL_GATE = 1e-6

#: Gate 2, "desk materiality", for the weight families: basis points of
#: portfolio weight.
DESK_WEIGHT_GATE_BP = 1.0

#: Natural unit of each limit family's slack. The report mixes weight
#: fractions, years of duration, bp-years of DTS and bp of annualized active
#: return; `rel_slack` exists because those are not comparable raw, and this
#: column exists so a breach can also be quoted in the unit a PM thinks in.
UNIT_WEIGHT = "weight_fraction"
UNIT_WEIGHT_L1 = "weight_l1_fraction"
UNIT_YEARS = "duration_years"
UNIT_DTS = "dts_bp_years"
UNIT_RETURN_BP = "active_return_bp_ann"

#: Desk-materiality tolerance per unit, expressed IN that unit. Only the
#: first two are "1bp" of anything. The rest are the precision at which each
#: quantity is reported to a credit PM — the only defensible way to extend a
#: "would anyone notice?" gate to units in which basis points of weight do
#: not exist. These are this repo's choices, not a vendor standard, and they
#: live in one place so they can be argued with rather than hunted for.
DESK_TOL_BY_UNIT: dict[str, float] = {
    # Weights are reported at ~1bp granularity; lot rounding moves them more.
    UNIT_WEIGHT: DESK_WEIGHT_GATE_BP / 1e4,
    UNIT_WEIGHT_L1: DESK_WEIGHT_GATE_BP / 1e4,
    # Duration is quoted to 0.01y, so half a tick rounds away.
    UNIT_YEARS: 0.005,
    # DTS is quoted in whole bp-years (the M2 limit sheet prints 829), so
    # half a bp-year rounds away.
    UNIT_DTS: 0.5,
    # Ex-ante TE is quoted to 0.1bp annualized; half a tick rounds away.
    UNIT_RETURN_BP: 0.05,
}

#: Whether "this breach in dollars" is a sentence that means anything, per
#: unit — and if not, why not. A dollar figure is reported ONLY where the
#: breach is a fraction of capital. Turning years of duration or bp-years of
#: DTS into dollars requires an assumed rate or spread move, i.e. inventing a
#: scenario and attaching it to a compliance number; that is a category
#: error, not a conversion, so the cell is left empty and the reason is
#: carried in the row.
USD_BASIS_BY_UNIT: dict[str, str] = {
    UNIT_WEIGHT: f"weight x ${NOTIONAL_USD / 1e9:.0f}bn book",
    UNIT_WEIGHT_L1: (f"gross active weight x ${NOTIONAL_USD / 1e9:.0f}bn "
                     f"book (counts overweights and underweights)"),
    UNIT_YEARS: "n/a: years of duration, dollars need an assumed yield move",
    UNIT_DTS: "n/a: bp-years of DTS, dollars need an assumed spread move",
    UNIT_RETURN_BP: "n/a: bp of active RETURN, not a fraction of capital",
}

#: The units in which a breach IS a fraction of capital.
_WEIGHT_UNITS = frozenset({UNIT_WEIGHT, UNIT_WEIGHT_L1})


@dataclass
class CreditLimits:
    """Desk limits. Use `default_limits(uni)` for a feasible starting set."""

    rating_bands: dict[str, tuple[float, float]] = field(default_factory=dict)
    sector_caps: dict[str, float] = field(default_factory=dict)
    issuer_cap: float = 0.02
    w_max: float = 0.01
    cell_duration_tol: float = 0.25        # years, per maturity cell
    total_duration_tol: float = 0.10       # years, portfolio aggregate
    bench_duration: float = np.nan         # target for the aggregate band
    dts_max: float = np.inf                # bp-years
    dts_min: float = 0.0                   # bp-years — see note below
    turnover: float = 0.25                 # ONE-WAY fraction of the portfolio

    # --- family 7: tracking error vs the benchmark ------------------------
    #: Annualized ex-ante TE budget in BASIS POINTS. inf disables the family,
    #: which is the default so every committed M1-M3 result is unaffected.
    te_bp: float = np.inf
    #: "sigma" -> second-order cone (leaves the LP class); "mad" -> stays LP.
    te_mode: str = "sigma"
    #: Scenario horizon -> annualization. src/scenarios.py is monthly, so 12.
    periods_per_year: float = 12.0
    #: Weight-space active budget ||w - benchmark||_1. Separate family, LP.
    active_l1: float = np.inf

    def describe(self) -> str:
        s = (
            f"issuer_cap={self.issuer_cap:.2%}, w_max={self.w_max:.2%}, "
            f"cell_dur_tol=±{self.cell_duration_tol:.2f}y, "
            f"total_dur={self.bench_duration:.2f}±{self.total_duration_tol:.2f}y, "
            f"DTS∈[{self.dts_min:,.0f}, {self.dts_max:,.0f}]bp-y, "
            f"turnover={self.turnover:.0%} one-way"
        )
        if np.isfinite(self.te_bp):
            s += f", TE<={self.te_bp:.0f}bp ann. ({self.te_mode})"
        if np.isfinite(self.active_l1):
            s += f", ||w-b||_1<={self.active_l1:.2f}"
        return s

    def te_period_budget(self) -> float:
        """The TE budget in per-period return units (not bp, not annual)."""
        if not np.isfinite(self.te_bp):
            return np.inf
        return (self.te_bp / 1e4) / np.sqrt(self.periods_per_year)


def default_limits(
    uni: BondUniverse,
    rating_band_slack: float = 0.10,
    sector_cap_slack: float = 0.05,
    dts_max_mult: float = 1.10,
    dts_min_mult: float = 0.90,
    cell_duration_tol: float = 0.25,
    total_duration_tol: float = 0.10,
    turnover: float = 0.25,
    issuer_cap_mult: float = 1.50,
    w_max_mult: float = 3.0,
    te_bp: float = np.inf,
    te_mode: str = "sigma",
    periods_per_year: float = 12.0,
    active_l1: float = np.inf,
) -> CreditLimits:
    """Limits derived from the benchmark, so the benchmark itself is feasible
    on every family except turnover (which measures distance from the
    incumbent, not from the index — check
    `uni.one_way_distance_to_benchmark()` before assuming the index is
    reachable).

    Deriving limits from benchmark exposures rather than hardcoding levels
    keeps the problem feasible across universe sizes and seeds, and makes the
    slack parameters — how far the portfolio may stray from the index — the
    explicit, tunable quantities they should be.

    Two of these deserve their names said out loud:

    - **DTS is bounded on BOTH sides.** A one-sided cap lets the optimizer
      collapse credit beta towards zero, which for a benchmarked credit
      mandate is an enormous unstated tracking-error position — and it makes
      the cap itself inert. The floor is what forces the optimizer to express
      a *credit selection* view rather than simply de-risking.
    - **Rating bands are multiplicative on the downside.** An additive ±10%
      band on a 4.7% single-B weight is not a band, it is permission to zero
      the bucket. The rule is `min(additive slack, 0.5 x benchmark weight)`,
      i.e. **no rating bucket may fall below half its benchmark weight** —
      the multiplicative branch binds for every bucket under 20% benchmark
      weight, which here is four of six including BB.
    """
    wb = uni.w_benchmark
    df = uni.bonds

    rating_w = df.assign(w=wb).groupby("rating", observed=True)["w"].sum()
    rating_bands = {}
    for r, v in rating_w.items():
        v = float(v)
        down = min(rating_band_slack, 0.5 * v)     # never a free zeroing
        rating_bands[r] = (max(0.0, v - down),
                           min(1.0, v + rating_band_slack))

    sector_w = df.assign(w=wb).groupby("sector", observed=True)["w"].sum()
    sector_caps = {s: min(1.0, float(v) + sector_cap_slack)
                   for s, v in sector_w.items()}

    issuer_w = df.assign(w=wb).groupby("issuer_id", observed=True)["w"].sum()
    issuer_cap = max(0.01, issuer_cap_mult * float(issuer_w.max()))
    w_max = max(0.005, w_max_mult * float(wb.max()))

    bench_dts = float(wb @ uni.dts)
    bench_dur = float(wb @ uni.duration)

    return CreditLimits(
        rating_bands=rating_bands, sector_caps=sector_caps,
        issuer_cap=issuer_cap, w_max=w_max,
        cell_duration_tol=cell_duration_tol,
        total_duration_tol=total_duration_tol, bench_duration=bench_dur,
        dts_max=dts_max_mult * bench_dts, dts_min=dts_min_mult * bench_dts,
        turnover=turnover,
        te_bp=te_bp, te_mode=te_mode, periods_per_year=periods_per_year,
        active_l1=active_l1,
    )


def credit_constraint_builder(uni: BondUniverse, limits: CreditLimits,
                              R: np.ndarray | None = None):
    """Return a callable `w -> (constraints, names)` for `build_mean_cvar`.

    The optimisation variable is created inside the model builder, so the
    constraint set has to be supplied as a callback rather than a list.

    `R` is the (S, N) scenario return matrix, required only when
    `limits.te_bp` is finite — the risk-space tracking-error families are
    defined on active *returns*, not on weights, so they cannot be built from
    the universe alone. Pass the same matrix the model is built on; passing a
    different draw would constrain TE under scenarios the objective never
    sees.
    """
    G_rating, rating_labels = uni.group_matrix("rating")
    G_sector, sector_labels = uni.group_matrix("sector")
    G_issuer, _ = uni.group_matrix("issuer_id")
    D_cell, cell_labels = uni.duration_contribution_matrix()

    bench_cell_dur = D_cell @ uni.w_benchmark
    dts_vec = uni.dts
    dur_vec = uni.duration
    w0 = uni.w0
    w_bench = uni.w_benchmark

    lo = np.array([limits.rating_bands.get(r, (0.0, 1.0))[0]
                   for r in rating_labels])
    hi = np.array([limits.rating_bands.get(r, (0.0, 1.0))[1]
                   for r in rating_labels])
    sector_cap = np.array([limits.sector_caps.get(s, 1.0)
                           for s in sector_labels])

    def build(w: cp.Variable):
        cons: list[cp.Constraint] = []
        names: list[str] = []

        if limits.rating_bands:
            cons += [G_rating @ w >= lo, G_rating @ w <= hi]
            names += ["rating_bands_lo", "rating_bands_hi"]
        if limits.sector_caps:
            cons.append(G_sector @ w <= sector_cap)
            names.append("sector_caps")
        if np.isfinite(limits.issuer_cap):
            cons.append(G_issuer @ w <= limits.issuer_cap)
            names.append("issuer_caps")
        if np.isfinite(limits.cell_duration_tol):
            cons.append(cp.abs(D_cell @ w - bench_cell_dur)
                        <= limits.cell_duration_tol)
            names.append("cell_duration_match")
        if np.isfinite(limits.total_duration_tol) \
                and np.isfinite(limits.bench_duration):
            # Aggregate band: without it, the per-cell tolerances sum to a
            # much wider total-duration bet than the mandate intends.
            cons.append(cp.abs(dur_vec @ w - limits.bench_duration)
                        <= limits.total_duration_tol)
            names.append("total_duration_match")
        if np.isfinite(limits.dts_max):
            cons.append(dts_vec @ w <= limits.dts_max)
            names.append("dts_cap")
        if limits.dts_min > 0:
            cons.append(dts_vec @ w >= limits.dts_min)
            names.append("dts_floor")
        if np.isfinite(limits.turnover):
            # ||w - w0||_1 counts buys AND sells; one-way budget => 2x.
            cons.append(cp.norm1(w - w0) <= 2.0 * limits.turnover)
            names.append("turnover")

        # --- family 7 --------------------------------------------------
        # Active exposure is measured against the BENCHMARK (w_bench), not
        # against the incumbent w0. Turnover is distance from what you hold;
        # tracking error is distance from what you are judged against. Using
        # w0 here would silently turn the TE budget into a second turnover
        # budget.
        if np.isfinite(limits.active_l1):
            cons.append(cp.norm1(w - w_bench) <= limits.active_l1)
            names.append("active_weight_l1")

        if np.isfinite(limits.te_bp):
            if R is None:
                raise ValueError(
                    "limits.te_bp is finite but no scenario matrix R was "
                    "passed to credit_constraint_builder(); the risk-space "
                    "TE families are defined on active returns")
            S = R.shape[0]
            tau = limits.te_period_budget()          # per-period return units
            A = R @ (w - w_bench)                    # active return, S-vector
            A_centered = A - cp.sum(A) / S
            if limits.te_mode == "sigma":
                # sqrt(sum(A_c^2)/S) <= tau  <=>  ||A_c||_2 <= tau*sqrt(S).
                # SECOND-ORDER CONE: this leaves the LP class. PDLP cannot
                # solve it; that outcome is the measurement, not a failure.
                cons.append(cp.norm(A_centered, 2) <= tau * np.sqrt(S))
                names.append("tracking_error_sigma_SOC")
            elif limits.te_mode == "mad":
                # (sum|A_c|/S) * sqrt(pi/2) <= tau. Stays linear.
                cons.append(cp.sum(cp.abs(A_centered)) * MAD_TO_SIGMA
                            <= tau * S)
                names.append("tracking_error_mad_LP")
            else:
                raise ValueError(
                    f"unknown te_mode {limits.te_mode!r}; "
                    f"expected 'sigma' or 'mad'")
        return cons, names

    return build


# --------------------------------------------------------------------------
# Diagnostics — does the portfolio make credit sense, and what is binding?
# --------------------------------------------------------------------------

def tracking_error_bp(uni: BondUniverse, w: np.ndarray, R: np.ndarray,
                      periods_per_year: float = 12.0) -> dict[str, float]:
    """Recompute tracking error from the scenario matrix, in annualized bp.

    Deliberately independent of the LP: no auxiliary variables, no solver
    output, no CVXPY expression reuse — just numpy on the weight vector the
    solver returned. This is the number that decides whether a solve
    respected the mandate, and a solver that reports "Optimal" inside its own
    residual tolerance can still fail it. Same discipline as `src/check.py`.

    Returns sigma and MAD-implied TE side by side on purpose. They agree only
    under a Gaussian; the gap between them on credit active returns is itself
    reportable, and quoting the MAD number alone would launder that away.
    """
    w = np.asarray(w, dtype=np.float64).ravel()
    a = w - uni.w_benchmark
    A = R @ a
    A_c = A - A.mean()
    ann = np.sqrt(periods_per_year) * 1e4

    sigma_bp = float(np.sqrt((A_c ** 2).mean()) * ann)
    mad_bp = float(np.abs(A_c).mean() * MAD_TO_SIGMA * ann)
    return {
        "te_sigma_bp": sigma_bp,
        "te_mad_implied_bp": mad_bp,
        # >1 means fatter-than-Gaussian active returns, so a MAD budget is
        # LOOSER than the sigma budget it was calibrated to match.
        "sigma_over_mad_implied": sigma_bp / mad_bp if mad_bp > 0 else np.nan,
        "active_l1": float(np.abs(a).sum()),
        "mean_active_return_bp": float(A.mean() * 1e4),
    }


def exposure_report(uni: BondUniverse, w: np.ndarray,
                    limits: CreditLimits,
                    R: np.ndarray | None = None) -> pd.DataFrame:
    """Every limit, its value under `w`, how much room is left, and — if the
    room is negative — the breach in both of the units this project scores
    breaches in.

    Slacks come in four different units — weight fractions, years of
    duration, bp-years of DTS, and bp of annualized active return — so a raw
    `min(slack)` compares a 0.5 bp-year DTS rounding artifact against a 0.5%
    turnover breach and calls the former worse. `rel_slack` normalizes each
    slack by the characteristic scale of its own limit; that is the column to
    rank and threshold on for the numerical gate.

    **Extended rather than duplicated.** The breach columns are added here
    instead of in a parallel function because every caller in the repo already
    builds this frame, and a second scoring function would let a runner write
    one verdict while a checker computed the other. Adding columns cannot
    break a caller that reads by name; changing `worst_violation`'s 2-tuple
    return would break six call sites, so that signature is left alone and
    `gate_verdicts()` is the companion that returns both verdicts at once.

    Columns added, all per row:

    - ``unit`` — the natural unit of this family's slack.
    - ``breach`` — ``max(0, -slack)``, in that natural unit.
    - ``breach_bp_weight`` / ``breach_usd`` — the same breach as basis points
      of portfolio weight and as dollars on `NOTIONAL_USD`. Populated ONLY
      where the breach is a fraction of capital; ``usd_basis`` says why not
      where it is empty (see `USD_BASIS_BY_UNIT` — turning years or bp-years
      into dollars needs an assumed market move, which would be a fabricated
      scenario wearing a compliance number).
    - ``breach_desk_multiple`` — the breach over that unit's desk tolerance,
      the desk gate's analogue of ``rel_slack`` and the column to rank on when
      asking "would the desk notice?".
    - ``breaches_purity_gate`` / ``breaches_desk_gate`` — the two verdicts,
      per row.
    """
    rows = []
    df = uni.bonds

    rating_w = df.assign(x=w).groupby("rating", observed=True)["x"].sum()
    bench_r = df.assign(x=uni.w_benchmark).groupby(
        "rating", observed=True)["x"].sum()
    for r, v in rating_w.items():
        lo, hi = limits.rating_bands.get(r, (0.0, 1.0))
        bw = float(bench_r.get(r, np.nan))
        # Scale = the band's own half-width, so a rating breach ranks on the
        # same footing as a duration or DTS breach. Scaling by 1.0 (portfolio
        # weight) would understate rating breaches ~10-20x versus every other
        # family — in the family most likely to bind.
        half = max(0.5 * (hi - lo), 1e-6)
        rows.append(("rating", r, float(v), bw, f"[{lo:.1%}, {hi:.1%}]",
                     min(float(v) - lo, hi - float(v)), half, UNIT_WEIGHT))

    sector_w = df.assign(x=w).groupby("sector", observed=True)["x"].sum()
    bench_s = df.assign(x=uni.w_benchmark).groupby(
        "sector", observed=True)["x"].sum()
    for s, v in sector_w.items():
        cap = limits.sector_caps.get(s, 1.0)
        bw = float(bench_s.get(s, np.nan))
        rows.append(("sector", s, float(v), bw, f"<= {cap:.1%}",
                     cap - float(v), max(cap - bw, 1e-6), UNIT_WEIGHT))

    issuer_w = df.assign(x=w).groupby("issuer_id", observed=True)["x"].sum()
    rows.append(("issuer", "max issuer weight", float(issuer_w.max()),
                 float(df.assign(x=uni.w_benchmark).groupby(
                     "issuer_id", observed=True)["x"].sum().max()),
                 f"<= {limits.issuer_cap:.2%}",
                 limits.issuer_cap - float(issuer_w.max()),
                 max(limits.issuer_cap, 1e-6), UNIT_WEIGHT))

    D, labels = uni.duration_contribution_matrix()
    port_d, bench_d = D @ w, D @ uni.w_benchmark
    for lab, pv, bv in zip(labels, port_d, bench_d):
        rows.append(("duration", f"{lab} contribution", float(pv), float(bv),
                     f"±{limits.cell_duration_tol:.2f}y of bench",
                     limits.cell_duration_tol - abs(float(pv) - float(bv)),
                     max(limits.cell_duration_tol, 1e-6), UNIT_YEARS))

    dur_p = float(w @ uni.duration)
    if np.isfinite(limits.bench_duration):
        rows.append(("duration", "total duration", dur_p,
                     limits.bench_duration,
                     f"±{limits.total_duration_tol:.2f}y of bench",
                     limits.total_duration_tol
                     - abs(dur_p - limits.bench_duration),
                     max(limits.total_duration_tol, 1e-6), UNIT_YEARS))

    dts_p = float(w @ uni.dts)
    dts_b = float(uni.w_benchmark @ uni.dts)
    rows.append(("dts", "DTS vs cap (bp-y)", dts_p, dts_b,
                 f"<= {limits.dts_max:,.0f}", limits.dts_max - dts_p,
                 max(dts_b, 1e-6), UNIT_DTS))
    if limits.dts_min > 0:
        rows.append(("dts", "DTS vs floor (bp-y)", dts_p, dts_b,
                     f">= {limits.dts_min:,.0f}", dts_p - limits.dts_min,
                     max(dts_b, 1e-6), UNIT_DTS))

    one_way = 0.5 * float(np.abs(w - uni.w0).sum())
    rows.append(("turnover", "one-way turnover", one_way, 0.0,
                 f"<= {limits.turnover:.1%}", limits.turnover - one_way,
                 max(limits.turnover, 1e-6), UNIT_WEIGHT))

    # Family 7. Reported whenever R is available, even if the budget is
    # infinite — an unconstrained TE is a number a credit PM wants to see,
    # and it is what makes the "GPU is fast but breaks the mandate" claim
    # checkable rather than asserted.
    if R is not None:
        te = tracking_error_bp(uni, w, R, limits.periods_per_year)
        # The mode that was CONSTRAINED is the one scored against the budget;
        # the other is reported with an infinite limit so it never silently
        # decides pass/fail on a basis the solver was not given.
        scored = ("te_mad_implied_bp" if limits.te_mode == "mad"
                  else "te_sigma_bp")
        for key, label in (("te_sigma_bp", "TE sigma (bp ann.)"),
                           ("te_mad_implied_bp", "TE MAD-implied (bp ann.)")):
            budget = limits.te_bp if key == scored else np.inf
            rows.append(("tracking_error", label, te[key], 0.0,
                         (f"<= {budget:.1f}bp" if np.isfinite(budget)
                          else "unconstrained"),
                         budget - te[key] if np.isfinite(budget) else np.inf,
                         max(budget if np.isfinite(budget) else te[key],
                             1e-6), UNIT_RETURN_BP))
        rows.append(("tracking_error", "active weight L1", te["active_l1"],
                     0.0,
                     (f"<= {limits.active_l1:.3f}"
                      if np.isfinite(limits.active_l1) else "unconstrained"),
                     (limits.active_l1 - te["active_l1"]
                      if np.isfinite(limits.active_l1) else np.inf),
                     max(limits.active_l1 if np.isfinite(limits.active_l1)
                         else max(te["active_l1"], 1e-6), 1e-6),
                     UNIT_WEIGHT_L1))

    rep = pd.DataFrame(rows, columns=[
        "family", "item", "portfolio", "benchmark", "limit", "slack", "scale",
        "unit"])
    rep["rel_slack"] = rep["slack"] / rep["scale"]
    rep["binding"] = rep["rel_slack"] < PURITY_REL_GATE
    return score_breaches(rep)


def score_breaches(rep: pd.DataFrame) -> pd.DataFrame:
    """Add the breach columns and the two per-row gate verdicts, in place.

    Split out of `exposure_report` so a hand-built frame can be scored in a
    test without a universe and a solve, which is the only way to exercise the
    boundary between the two gates deliberately rather than hoping a solver
    lands on it.

    An unconstrained row carries `slack = +inf` and therefore breach 0; that
    is not a pass earned by the portfolio, it is a limit that was never set,
    and `rel_slack` is +inf there so neither gate can be fooled by it.

    Raises `ValueError` on a unit with no entry in `DESK_TOL_BY_UNIT`.
    """
    unit = rep["unit"].astype(str)
    unknown = sorted(set(unit) - set(DESK_TOL_BY_UNIT))
    if unknown:
        # Loud, not NaN. An unmapped unit means a limit family was added
        # without deciding how a desk reads its breach, and a NaN multiple
        # would quietly pass the desk gate forever — a silent default is the
        # one outcome this pair of gates exists to prevent.
        raise ValueError(
            f"no desk tolerance for unit(s) {unknown}; add them to "
            f"DESK_TOL_BY_UNIT and USD_BASIS_BY_UNIT with a stated reason")

    slack = rep["slack"].to_numpy(dtype=float)
    # `slack < 0` rather than `max(0, -slack)`: the latter turns an exactly
    # binding limit into -0.0, which then prints as a negative breach.
    breach = np.where(np.isfinite(slack) & (slack < 0.0), -slack, 0.0)
    desk_tol = unit.map(DESK_TOL_BY_UNIT).to_numpy(dtype=float)
    is_weight = unit.isin(_WEIGHT_UNITS).to_numpy()

    rep = rep.copy()
    rep["breach"] = breach
    rep["desk_tol"] = desk_tol
    rep["breach_desk_multiple"] = breach / desk_tol
    rep["breach_bp_weight"] = np.where(is_weight, breach * 1e4, np.nan)
    rep["breach_usd"] = np.where(is_weight, breach * NOTIONAL_USD, np.nan)
    rep["usd_basis"] = unit.map(USD_BASIS_BY_UNIT)
    rep["breaches_purity_gate"] = rep["rel_slack"] < -PURITY_REL_GATE
    rep["breaches_desk_gate"] = rep["breach_desk_multiple"] > 1.0
    return rep


def worst_violation(rep: pd.DataFrame) -> tuple[float, str]:
    """Largest relative limit breach and which limit it is (0.0 if clean).

    This is the PURITY gate's ranking: relative to each limit's own scale.
    Unchanged — six call sites unpack this 2-tuple, and every committed
    result was produced by it. `worst_desk_violation` is its desk-materiality
    twin and `gate_verdicts` returns both.
    """
    row = rep.loc[rep["rel_slack"].idxmin()]
    return max(0.0, -float(row["rel_slack"])), str(row["item"])


def worst_desk_violation(rep: pd.DataFrame) -> tuple[float, str]:
    """Largest breach as a multiple of its unit's desk tolerance, and which.

    Ranked on `breach_desk_multiple` rather than `rel_slack` because the two
    genuinely disagree: `rel_slack` says a DTS breach of 1 bp-year on a 921
    bp-year book is 1e-3 of scale, while the desk says it is two ticks of the
    number it reads. Returns (0.0, "-") on a frame with no breach at all, so
    the caller does not have to distinguish "smallest breach" from "no
    breach" the way `worst_violation` forces.
    """
    if rep.empty or not rep["breach_desk_multiple"].gt(0).any():
        return 0.0, "-"
    row = rep.loc[rep["breach_desk_multiple"].idxmax()]
    return float(row["breach_desk_multiple"]), str(row["item"])


@dataclass(frozen=True)
class GateVerdicts:
    """Both mandate verdicts on one portfolio, as scalars a runner can write.

    The paper reports both because they disagree on real solves and picking
    one would be picking a headline. Named explicitly:

    - **purity** — `rel_slack >= -PURITY_REL_GATE` (1e-6) on every row, the
      repo's original numerical gate. Every committed result was scored on
      it, and its value and meaning are unchanged here.
    - **desk** — the breach must survive the precision the quantity is
      reported to a credit PM at: 1bp of portfolio weight for the weight
      families, and the reporting granularity of the unit elsewhere (see
      `DESK_TOL_BY_UNIT`). This gate exists because a senior credit PM's
      review of this work made the point that no compliance system is pure to
      1e-6, and under it a 2.3e-6 breach — enough to fail the purity gate —
      is a portfolio the desk would trade.

    `verdict` collapses the pair for a table column, and the pair is kept
    alongside it so nothing has to be re-derived from the string.
    """

    purity_pass: bool
    purity_worst_rel: float
    purity_worst_limit: str
    desk_pass: bool
    desk_worst_multiple: float
    desk_worst_limit: str
    desk_worst_bp_weight: float
    desk_worst_usd: float
    n_purity_breaches: int
    n_desk_breaches: int
    verdict: str

    def as_row(self) -> dict[str, object]:
        """Flat dict, ready to merge into a CSV row."""
        return {
            "purity_pass": self.purity_pass,
            "purity_worst_rel": self.purity_worst_rel,
            "purity_worst_limit": self.purity_worst_limit,
            "desk_pass": self.desk_pass,
            "desk_worst_multiple": self.desk_worst_multiple,
            "desk_worst_limit": self.desk_worst_limit,
            "desk_worst_bp_weight": self.desk_worst_bp_weight,
            "desk_worst_usd": self.desk_worst_usd,
            "n_purity_breaches": self.n_purity_breaches,
            "n_desk_breaches": self.n_desk_breaches,
            "gate_verdict": self.verdict,
        }


#: Verdict vocabulary. "DESK-OK" is the interesting one: the portfolio fails
#: the numerical gate and would still be traded.
VERDICT_OK = "OK"                 # inside both gates
VERDICT_DESK_ONLY = "DESK-OK"     # breaches 1e-6, immaterial to the desk
VERDICT_BREACH = "BREACH"         # breaches the desk gate too


def gate_verdicts(rep: pd.DataFrame) -> GateVerdicts:
    """Score one exposure report against both gates, side by side.

    The dollar and bp figures reported are those of the worst DESK breach,
    and they are NaN when that row is not weight-denominated — a duration or
    DTS breach has no dollar value without an assumed market move, and this
    function will not invent one to fill a column.
    """
    purity_rel, purity_item = worst_violation(rep)
    desk_mult, desk_item = worst_desk_violation(rep)
    purity_pass = purity_rel <= PURITY_REL_GATE
    desk_pass = desk_mult <= 1.0

    bp = usd = float("nan")
    if desk_mult > 0.0:
        row = rep.loc[rep["breach_desk_multiple"].idxmax()]
        bp = float(row["breach_bp_weight"])
        usd = float(row["breach_usd"])

    if not desk_pass:
        verdict = VERDICT_BREACH
    elif purity_pass:
        verdict = VERDICT_OK
    else:
        verdict = VERDICT_DESK_ONLY

    return GateVerdicts(
        purity_pass=purity_pass,
        purity_worst_rel=purity_rel,
        purity_worst_limit=purity_item if purity_rel > 0 else "-",
        desk_pass=desk_pass,
        desk_worst_multiple=desk_mult,
        desk_worst_limit=desk_item,
        desk_worst_bp_weight=bp,
        desk_worst_usd=usd,
        n_purity_breaches=int(rep["breaches_purity_gate"].sum()),
        n_desk_breaches=int(rep["breaches_desk_gate"].sum()),
        verdict=verdict,
    )


def _baseline_program(uni: BondUniverse, limits: CreditLimits,
                      R: np.ndarray | None = None
                      ) -> tuple[cp.Variable, list[cp.Constraint]]:
    """The baseline's variable and its feasible set, built once.

    Shared by `min_risk_baseline` and `min_risk_face` so the face is provably
    a face of the SAME polytope. If the two ever drifted apart, the "range
    around the committed vertex" claim would be a range around a different
    problem's vertex, which is worse than no range at all.
    """
    w = cp.Variable(uni.n_bonds, nonneg=True)
    cons: list[cp.Constraint] = [cp.sum(w) == 1.0, w <= limits.w_max]
    extra, _ = credit_constraint_builder(uni, limits, R=R)(w)
    return w, cons + extra


def _solve_for_weights(objective: cp.Minimize | cp.Maximize,
                       cons: list[cp.Constraint], w: cp.Variable,
                       solver, label: str) -> tuple[np.ndarray, str]:
    """Solve, and return the weight vector only — never a solver objective.

    Same discipline as `src/check.py`: every number this module reports about
    a portfolio is recomputed from the weights with numpy, so a solver that
    reports a value inside its own tolerance cannot decide a published band.
    """
    prob = cp.Problem(objective, cons)
    prob.solve(solver=solver, verbose=False, warm_start=False)
    if w.value is None:
        raise RuntimeError(f"min-DTS face solve {label!r} failed: "
                           f"{prob.status}")
    return np.asarray(w.value).ravel(), str(prob.status)


def min_risk_baseline(uni: BondUniverse, limits: CreditLimits,
                      R: np.ndarray | None = None) -> np.ndarray:
    """The scenario-free portfolio: minimise DTS under the identical limits.

    This is the attribution baseline, and it exists to keep the headline
    honest. A mandate whose bands are wide enough will let *any* de-risking
    rule walk to the low-risk corner of every limit, and that alone reduces
    tail risk — with no scenarios, no CVaR, no LP over the return matrix.
    Reporting a CVaR reduction without it cannot distinguish "the optimizer
    picked better bonds" from "the mandate permitted de-risking".

    Whatever CVaR reduction this portfolio achieves is the part of the
    headline that the scenario model did NOT earn.

    **Read `min_risk_face` before quoting a number from this.** The limit set
    passed in already contains a DTS *floor*, so this objective is bounded
    below by a constraint on the same quantity: the optimal value is forced
    and the optimal face is high-dimensional. What comes back is one arbitrary
    vertex of that face, chosen by the solver's pivoting, and any expected
    return or out-of-sample CVaR read off it is a property of that vertex
    rather than of the min-DTS portfolio as a class. This function is kept
    unchanged because every committed result was produced by it.

    `R` is only needed when `limits.te_bp` is finite (the risk-space TE
    families are defined on active returns); the default None reproduces the
    original behaviour exactly.
    """
    w, cons = _baseline_program(uni, limits, R=R)
    w_opt, _ = _solve_for_weights(cp.Minimize(uni.dts @ w), cons, w,
                                  cp.HIGHS, "min_dts")
    return w_opt


@dataclass(frozen=True)
class FaceVertex:
    """One recovered point of the min-DTS optimal face.

    `objective` records what the solve that produced it extremized, because
    the entire point of this object is that "the min-DTS portfolio" is not a
    portfolio, and a reader has to be able to see which direction produced
    which number before quoting it.
    """

    label: str
    objective: str
    w: np.ndarray
    dts: float
    duration: float
    exp_return: float
    cvar: float            # nan when no evaluation set was supplied
    l1_vs_baseline: float
    #: True when the DIRECTION that produced this vertex was computed on the
    #: evaluation set. Its out-of-sample CVaR is then an in-sample number for
    #: that set — a true statement about what the face contains, and NOT a
    #: portfolio any scenario-free rule could have picked in advance. The
    #: distinction decides whether a number belongs in an attribution band or
    #: in a hindsight bound, so it travels with the vertex.
    selected_using_eval_set: bool
    status: str

    def as_row(self) -> dict[str, object]:
        """Flat dict for a CSV; the weight vector stays out of it."""
        return {
            "vertex": self.label,
            "extremizes": self.objective,
            "dts_bp_years": self.dts,
            "duration": self.duration,
            "exp_return": self.exp_return,
            "cvar_eval": self.cvar,
            "l1_vs_baseline": self.l1_vs_baseline,
            "selected_using_eval_set": self.selected_using_eval_set,
            "status": self.status,
        }


@dataclass(frozen=True)
class MinRiskFace:
    """The min-DTS optimal face, sampled. A set, deliberately not a point."""

    dts_target: float
    dts_band: float
    floor_binds: bool
    alpha: float
    vertices: tuple[FaceVertex, ...]
    exp_return_lo: float
    exp_return_hi: float
    cvar_lo: float
    cvar_lo_is_exact: bool
    cvar_hi_observed: float
    max_vertex_l1: float
    looks_like_a_point: bool

    @property
    def exp_return_range(self) -> float:
        return self.exp_return_hi - self.exp_return_lo

    def vertex(self, label: str) -> FaceVertex:
        for v in self.vertices:
            if v.label == label:
                return v
        raise KeyError(label)

    def cvar_band(self, include_eval_selected: bool = True
                  ) -> tuple[float, float]:
        """(lowest, highest) evaluated CVaR over the recovered vertices.

        `include_eval_selected=False` drops the vertices whose search
        DIRECTION was computed on the evaluation set, leaving the band a
        scenario-free rule could actually have landed in. That is the band an
        attribution split should be quoted from; the wider one including them
        is a hindsight statement about the face's contents. Both are honest,
        they answer different questions, and mixing them would let a number
        chosen with knowledge of the evaluation draw be read as an achievable
        outcome.
        """
        vals = [v.cvar for v in self.vertices
                if np.isfinite(v.cvar)
                and (include_eval_selected or not v.selected_using_eval_set)]
        if not vals:
            return float("nan"), float("nan")
        return float(min(vals)), float(max(vals))

    def as_rows(self) -> list[dict[str, object]]:
        """One row per recovered vertex — the band's raw evidence."""
        return [v.as_row() for v in self.vertices]

    def as_band_row(self) -> dict[str, object]:
        """The band itself, as scalars, with its own honesty flags attached.

        `cvar_hi_observed` is deliberately named "observed": it is the worst
        tail among the vertices actually recovered, i.e. a LOWER bound on the
        worst tail available on the face, not the maximum. See the class
        docstring of `min_risk_face` for why the maximum is not computed.

        The `_exante` pair is the same band restricted to vertices whose
        direction never touched the evaluation set — the one to quote in an
        attribution split.
        """
        ex_lo, ex_hi = self.cvar_band(include_eval_selected=False)
        return {
            "face_dts_target": self.dts_target,
            "face_dts_band": self.dts_band,
            "face_floor_binds": self.floor_binds,
            "face_n_vertices": len(self.vertices),
            "face_exp_return_lo": self.exp_return_lo,
            "face_exp_return_hi": self.exp_return_hi,
            "face_exp_return_range": self.exp_return_range,
            "face_cvar_lo": self.cvar_lo,
            "face_cvar_lo_is_exact": self.cvar_lo_is_exact,
            "face_cvar_lo_uses_eval_set": self.cvar_lo_is_exact,
            "face_cvar_hi_observed": self.cvar_hi_observed,
            "face_cvar_hi_is_lower_bound": True,
            "face_cvar_lo_exante": ex_lo,
            "face_cvar_hi_exante": ex_hi,
            "face_max_vertex_l1": self.max_vertex_l1,
            "face_looks_like_a_point": self.looks_like_a_point,
        }


def min_risk_face(
    uni: BondUniverse,
    limits: CreditLimits,
    mu: np.ndarray,
    R: np.ndarray | None = None,
    R_eval: np.ndarray | None = None,
    alpha: float = 0.95,
    minimize_cvar_over_face: bool = False,
    face_rel_tol: float = 1e-9,
    point_atol: float = 1e-6,
    solver=cp.HIGHS,
) -> MinRiskFace:
    """Sample the min-DTS optimal face, so the baseline is quoted as a band.

    `min_risk_baseline` minimises `dts @ w` over a limit set that already
    contains `dts @ w >= dts_min`. The optimal VALUE is therefore forced by a
    constraint on the objective's own quantity, and the optimal FACE — every
    portfolio attaining it — is high-dimensional. The solver returns one
    vertex of it. Every attribution number computed from that vertex is a
    property of the vertex, not of the min-DTS portfolio as a class. This
    function recovers the extremes so the paper quotes a range.

    Four decisions, stated because each one could reasonably have gone the
    other way:

    **1. The face is pinned at the ACHIEVED DTS, not at the floor.** The
    baseline is solved first and `dts_target = dts @ w_base` is recomputed
    from the returned weights with numpy. Pinning at `limits.dts_min` instead
    would describe a different set whenever the floor does not bind (a limit
    sheet with `dts_min = 0` being the extreme case), and that set is not the
    argmin at all. `floor_binds` reports whether the two coincide, which is
    the fact the committed M2 sheet illustrates: 829.1918 = 0.9 x 921.3242.

    **2. The pin is a two-sided band, not an equality.** Width is
    `face_rel_tol * max(|dts_target|, 1)`, i.e. ~8e-7 bp-years on the M2
    universe — the same order as HiGHS's own primal feasibility tolerance, so
    the pin neither fights the solver nor materially enlarges the set. An
    exact equality would hand the effective width to whatever the solver's
    row-activity tolerance happens to be, which is the opposite of stating
    it. The band makes the reported range a slight OUTER approximation of the
    true face: it can only widen the band, never narrow it, which is the
    conservative direction here because a wider baseline band is harder for
    this paper's own dominance claim to survive. Each vertex carries its
    achieved DTS so a reader can check the pin held.

    **3. Return and CVaR are not the same kind of answer, and the object says
    so.**

    - Expected return: `max` and `min` of `mu @ w` over the face are two LPs,
      so `[exp_return_lo, exp_return_hi]` is the EXACT range of expected
      return over the face. That is a theorem, not a sample.
    - Out-of-sample CVaR, minimum: minimising CVaR over the face is also an
      LP (Rockafellar-Uryasev on `R_eval`), so `cvar_lo` is exact when
      `minimize_cvar_over_face=True`. It costs one more solve of full
      scenario size — on a 20,000-scenario evaluation set that is the same
      order as the headline solve, which is why it is opt-in rather than
      default. **That vertex is chosen using the evaluation set**, so its
      out-of-sample CVaR is an in-sample number for that set: a true
      statement about what the face contains in hindsight, and not an
      outcome any scenario-free rule could have reached in advance. The
      vertex carries `selected_using_eval_set=True` and `cvar_band()` will
      drop it on request; an attribution split must quote the band without
      it.
    - Out-of-sample CVaR, maximum: **not computed, because it is not an LP.**
      Maximising a convex function over a polytope is a different and much
      harder problem; its optimum sits at a vertex but no linear program
      finds it. What is reported is `cvar_hi_observed`, the worst tail among
      the vertices actually recovered — a lower bound on the worst tail
      available on the face, labelled as one. Quoting it as "the range of
      out-of-sample CVaR" would be exactly the conflation this function
      exists to end.

    **4. If the face is a point.** `looks_like_a_point` is set when every
    recovered vertex coincides in weight space to within `point_atol` in L1
    (weights sum to 1, so 1e-6 is one part in a million of the book). It is
    EVIDENCE, not proof: two objectives can be optimised at the same vertex
    of a face that is not a point. What is exact either way is the return
    band — a `[lo, hi]` of zero width (to solver tolerance; separate solves
    landing on one point still differ in the last few digits) says expected
    return is constant on the face and licenses quoting it as a point. It
    licenses nothing about CVaR, which was never extremised upward. On a
    genuine point face the band rows still come back, all identical, so a
    caller emits a degenerate band rather than special-casing.

    `R` is passed to the constraint builder and is only needed when
    `limits.te_bp` is finite. `R_eval` is the evaluation scenario set — pass
    the OUT-OF-SAMPLE draw, since that is the axis the paper quotes.
    """
    from .check import empirical_cvar   # local: only this path needs it

    dts_vec = uni.dts
    dur_vec = uni.duration
    mu = np.asarray(mu, dtype=np.float64).ravel()
    if mu.shape != (uni.n_bonds,):
        raise ValueError(f"mu has shape {mu.shape}, expected "
                         f"({uni.n_bonds},)")

    w_var, cons = _baseline_program(uni, limits, R=R)
    w_base, st_base = _solve_for_weights(cp.Minimize(dts_vec @ w_var), cons,
                                         w_var, solver, "min_dts")
    dts_target = float(dts_vec @ w_base)
    band = face_rel_tol * max(abs(dts_target), 1.0)
    floor_binds = abs(dts_target - limits.dts_min) <= band

    def _pinned() -> tuple[cp.Variable, list[cp.Constraint]]:
        w, c = _baseline_program(uni, limits, R=R)
        c = c + [dts_vec @ w <= dts_target + band,
                 dts_vec @ w >= dts_target - band]
        return w, c

    # (label, what it extremizes, weights, direction used R_eval?, status)
    found: list[tuple[str, str, np.ndarray, bool, str]] = [
        ("baseline_vertex", "min DTS (arbitrary vertex of the face)",
         w_base, False, st_base),
    ]

    w_var, cons = _pinned()
    w_hi, st_hi = _solve_for_weights(cp.Maximize(mu @ w_var), cons, w_var,
                                     solver, "max_return")
    found.append(("max_return", "max mu'w at fixed DTS", w_hi, False, st_hi))

    w_var, cons = _pinned()
    w_lo, st_lo = _solve_for_weights(cp.Minimize(mu @ w_var), cons, w_var,
                                     solver, "min_return")
    found.append(("min_return", "min mu'w at fixed DTS", w_lo, False, st_lo))

    if minimize_cvar_over_face:
        if R_eval is None:
            raise ValueError("minimize_cvar_over_face=True needs R_eval: "
                             "there is no CVaR to minimise without a "
                             "scenario set")
        # Local import: the optional branch is the only reason credit.py
        # would ever need the model builder, and the dependency runs one way
        # (model.py knows nothing about credit.py).
        from .model import build_mean_cvar

        def _builder(w: cp.Variable):
            c, names = credit_constraint_builder(uni, limits, R=R)(w)
            c = c + [dts_vec @ w <= dts_target + band,
                     dts_vec @ w >= dts_target - band]
            return c, names + ["face_dts_hi", "face_dts_lo"]

        # mu = 0 and risk_aversion = 1 turns the Mean-CVaR LP into pure CVaR
        # minimisation, on the SAME budget/box/mandate rows the baseline uses.
        model = build_mean_cvar(
            R_eval, np.zeros(uni.n_bonds), alpha=alpha, risk_aversion=1.0,
            w_max=limits.w_max, constraint_builder=_builder)
        model.problem.solve(solver=solver, verbose=False, warm_start=False)
        if model.w.value is None:
            raise RuntimeError(f"min-CVaR-over-face solve failed: "
                               f"{model.problem.status}")
        found.append(("min_cvar", f"min CVaR{alpha:.0%} at fixed DTS",
                      model.weights(), True, str(model.problem.status)))

    vertices: list[FaceVertex] = []
    for label, objective, w_v, eval_selected, status in found:
        cvar = np.nan
        if R_eval is not None:
            # Recomputed from the weights by the same exact discrete CVaR the
            # solution checker uses — never read off the solve that produced
            # the vertex.
            cvar, _ = empirical_cvar(-(R_eval @ w_v), alpha)
        vertices.append(FaceVertex(
            label=label, objective=objective, w=w_v,
            dts=float(dts_vec @ w_v), duration=float(dur_vec @ w_v),
            exp_return=float(mu @ w_v), cvar=float(cvar),
            l1_vs_baseline=float(np.abs(w_v - w_base).sum()),
            selected_using_eval_set=eval_selected, status=status))

    rets = [v.exp_return for v in vertices]
    cvars = [v.cvar for v in vertices if np.isfinite(v.cvar)]
    W = np.array([v.w for v in vertices])
    # Pairwise L1 over every recovered vertex, not just the two extremes: the
    # min-CVaR vertex can sit off the segment joining them.
    max_l1 = float(np.abs(W[:, None, :] - W[None, :, :]).sum(axis=2).max())

    return MinRiskFace(
        dts_target=dts_target, dts_band=band, floor_binds=bool(floor_binds),
        alpha=alpha, vertices=tuple(vertices),
        # min/max over ALL recovered vertices, every one of which lies on the
        # face: the two extremal solves supply the endpoints, and taking the
        # extremes over the whole set keeps lo <= hi under solver noise.
        exp_return_lo=float(min(rets)), exp_return_hi=float(max(rets)),
        cvar_lo=float(min(cvars)) if cvars else float("nan"),
        cvar_lo_is_exact=bool(minimize_cvar_over_face),
        cvar_hi_observed=float(max(cvars)) if cvars else float("nan"),
        max_vertex_l1=max_l1,
        looks_like_a_point=bool(max_l1 <= point_atol),
    )


def portfolio_stats(uni: BondUniverse, w: np.ndarray) -> dict:
    """Headline characteristics a credit PM would ask for first."""
    held = w > 1e-6
    return {
        "n_positions": int(held.sum()),
        "n_issuers": int(uni.bonds.loc[held, "issuer_id"].nunique()),
        "yield": float(w @ uni.ytm),
        "oas_bp": float(w @ uni.oas_bp),
        "duration": float(w @ uni.duration),
        "spread_duration": float(w @ uni.spread_duration),
        "dts_bp_years": float(w @ uni.dts),
        "max_weight": float(w.max()),
    }

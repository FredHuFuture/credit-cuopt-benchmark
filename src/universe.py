"""Synthetic corporate-bond universe with real fixed-income mechanics.

Bonds are not abstract "assets" here: every instrument is priced from its own
semiannual cash-flow schedule, and its modified duration comes out of that
same schedule rather than being drawn from a distribution. Spread duration,
DTS, benchmark weights and duration-bucket contributions then follow from the
arithmetic instead of from hand-waving — which matters, because the credit
constraint set in `src/credit.py` is exactly a set of statements about those
quantities.

Data policy: 100% synthetic and seeded, so the repo is reproducible by anyone
with no data licence at all. The *shape* of the distributions is governed by
`CalibrationConfig`, whose defaults are stylized long-run market levels. Those
defaults are the only place real data would ever enter: `CalibrationConfig`
documents, per field, the public series a user could calibrate from
(FRED ICE BofA OAS indices, Treasury par yields). Swapping the config swaps
the calibration — no other code changes.

Scope: fixed-rate bullet corporates. Callables, floaters, sinkers and
amortisers are out of scope (their spread duration is not their modified
duration, which is the assumption this module rests on).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd
import scipy.sparse as sp

SECTORS: tuple[str, ...] = (
    "Financials", "Industrials", "Utilities", "Energy",
    "Technology", "Healthcare", "Consumer", "Communications",
)

RATINGS: tuple[str, ...] = ("AAA", "AA", "A", "BBB", "BB", "B")

#: Maturity cells, in years. Bonds are bucketed by MATURITY (as index desks
#: define cells), and the quantity matched against the benchmark within each
#: cell is the portfolio's DURATION CONTRIBUTION. Naming both explicitly
#: because conflating the two is a classic source of confusion.
MATURITY_CELL_EDGES: tuple[float, ...] = (0.0, 3.0, 7.0, 12.0, 40.0)
MATURITY_CELL_NAMES: tuple[str, ...] = ("0-3y", "3-7y", "7-12y", "12y+")


@dataclass(frozen=True)
class CalibrationConfig:
    """Everything that would be calibrated to market data, in one place.

    Defaults are stylized long-run levels, NOT fitted values. To calibrate:

    - ``oas_by_rating_bp`` -> FRED ICE BofA option-adjusted spread series,
      e.g. BAMLC0A1CAAA / BAMLC0A2CAA / BAMLC0A3CA / BAMLC0A4CBBB (IG by
      rating) and BAMLH0A1HYBB / BAMLH0A2HYB (HY by rating).
    - ``treasury_curve_bp`` -> FRED constant-maturity Treasury series
      (DGS1, DGS2, DGS5, DGS7, DGS10, DGS20, DGS30).
    - ``rating_weights`` / ``sector_weights`` -> published index composition,
      or an ETF holdings file (e.g. an IG or HY corporate-bond ETF).

    Using any of those sources is the user's call and their licence question;
    nothing is downloaded here.
    """

    #: Long-run average OAS by rating, in basis points.
    oas_by_rating_bp: Mapping[str, float] = field(default_factory=lambda: {
        "AAA": 45.0, "AA": 60.0, "A": 90.0,
        "BBB": 140.0, "BB": 260.0, "B": 420.0,
    })
    #: Cross-sectional dispersion of OAS within a rating (lognormal sigma).
    oas_dispersion: float = 0.35
    #: Multiplicative sector adjustment to spread level.
    sector_spread_mult: Mapping[str, float] = field(default_factory=lambda: {
        "Financials": 1.05, "Industrials": 1.00, "Utilities": 0.85,
        "Energy": 1.25, "Technology": 0.90, "Healthcare": 0.90,
        "Consumer": 0.95, "Communications": 1.10,
    })
    #: Issuer-count distribution across ratings (IG-heavy, with an HY sleeve).
    rating_weights: Mapping[str, float] = field(default_factory=lambda: {
        "AAA": 0.01, "AA": 0.07, "A": 0.32,
        "BBB": 0.45, "BB": 0.10, "B": 0.05,
    })
    #: Issuer-count distribution across sectors.
    sector_weights: Mapping[str, float] | None = None  # None => uniform
    #: Par Treasury yields (decimal) at the knot maturities below.
    treasury_knots_y: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0, 30.0)
    treasury_curve: tuple[float, ...] = (0.0425, 0.0410, 0.0395, 0.0415, 0.0440)
    #: Issue size (USD mm): lognormal, median and sigma, floor for index
    #: eligibility.
    issue_size_median_mm: float = 600.0
    issue_size_sigma: float = 0.60
    issue_size_floor_mm: float = 250.0
    #: Coupons are set near the yield prevailing at issue; seasoning noise
    #: (decimal st.dev.) creates the usual premium/discount mix.
    coupon_seasoning_sd: float = 0.0075


@dataclass
class BondUniverse:
    """A priced bond universe plus everything the optimizer needs about it."""

    bonds: pd.DataFrame            # one row per bond
    w_benchmark: np.ndarray        # market-value index weights, sums to 1
    w0: np.ndarray                 # incumbent portfolio, sums to 1
    config: CalibrationConfig
    seed: int

    # ---- convenience accessors (arrays aligned with `bonds`) -------------
    @property
    def n_bonds(self) -> int:
        return len(self.bonds)

    def one_way_distance_to_benchmark(self) -> float:
        """One-way turnover needed to move the incumbent onto the index.

        A turnover budget below this cannot reach the benchmark, so the
        optimum is partly a property of `w0` rather than of the objective —
        worth knowing before reading any result.
        """
        return 0.5 * float(np.abs(self.w_benchmark - self.w0).sum())

    @property
    def duration(self) -> np.ndarray:
        return self.bonds["mod_duration"].to_numpy()

    @property
    def spread_duration(self) -> np.ndarray:
        return self.bonds["spread_duration"].to_numpy()

    @property
    def oas_bp(self) -> np.ndarray:
        return self.bonds["oas_bp"].to_numpy()

    @property
    def dts(self) -> np.ndarray:
        """Duration times spread, in bp-years (Ben Dor et al., 2007)."""
        return self.spread_duration * self.oas_bp

    @property
    def convexity(self) -> np.ndarray:
        return self.bonds["convexity"].to_numpy()

    @property
    def ytm(self) -> np.ndarray:
        return self.bonds["ytm"].to_numpy()

    @property
    def oas(self) -> np.ndarray:
        """OAS in decimal (not bp)."""
        return self.bonds["oas_bp"].to_numpy() / 10_000.0

    def group_matrix(self, column: str) -> tuple[sp.csr_array, list[str]]:
        """0/1 matrix G (n_groups x n_bonds): (G @ w)_g = weight in group g."""
        codes, labels = pd.factorize(self.bonds[column], sort=True)
        n_g = len(labels)
        G = sp.csr_array(
            (np.ones(self.n_bonds), (codes, np.arange(self.n_bonds))),
            shape=(n_g, self.n_bonds),
        )
        return G, [str(x) for x in labels]

    def duration_contribution_matrix(self) -> tuple[sp.csr_array, list[str]]:
        """Matrix D (n_cells x n_bonds) holding duration in the cell rows.

        (D @ w)_c is the portfolio's duration contribution from maturity cell
        c — the quantity an index-matched credit portfolio pins to the
        benchmark. Contribution, not cell-average duration: the average is
        undefined at zero cell weight and is gameable by shifting weight
        between cells.
        """
        codes = self.bonds["maturity_cell_code"].to_numpy()
        n_c = len(MATURITY_CELL_NAMES)
        D = sp.csr_array(
            (self.duration, (codes, np.arange(self.n_bonds))),
            shape=(n_c, self.n_bonds),
        )
        return D, list(MATURITY_CELL_NAMES)


# --------------------------------------------------------------------------
# Bond mathematics
# --------------------------------------------------------------------------

def price_and_duration(
    coupon: np.ndarray,
    ytm: np.ndarray,
    maturity: np.ndarray,
    freq: int = 2,
    face: float = 100.0,
    atol_grid: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Price, modified duration and convexity from explicit cash flows.

    `maturity` must lie on the 1/freq grid (checked, not silently rounded).
    Macaulay duration is the PV-weighted mean time to cash flow; modified
    duration divides by (1 + y/freq); convexity is
    ``sum(pv_k k(k+1)) / (P freq^2 (1+y/freq)^2)``, i.e. (1/P) d2P/dy2.

    Verified against closed forms: par bonds price to exactly 100, a zero's
    Macaulay duration equals its maturity, and modified duration matches a
    central finite difference of the price function to 8 decimals.
    """
    scaled = maturity * freq
    n_per = np.rint(scaled).astype(int)
    if np.any(np.abs(scaled - n_per) > atol_grid):
        raise ValueError("maturities must lie on the 1/freq coupon grid")
    if np.any(n_per < 1):
        raise ValueError("maturities must be at least one coupon period")

    k = np.arange(1, int(n_per.max()) + 1)          # period index
    times = k / freq                                 # in years

    alive = k[None, :] <= n_per[:, None]             # (N, n_max)
    cf = (coupon[:, None] * face / freq) * alive
    cf[np.arange(len(n_per)), n_per - 1] += face     # principal at maturity

    r = (ytm / freq)[:, None]
    disc = (1.0 + r) ** (-k[None, :])
    pv = cf * disc

    price = pv.sum(axis=1)
    macaulay = (pv * times[None, :]).sum(axis=1) / price
    modified = macaulay / (1.0 + ytm / freq)
    kk = (k * (k + 1.0))[None, :]
    convexity = (pv * kk).sum(axis=1) / (
        price * freq ** 2 * (1.0 + ytm / freq) ** 2)
    return price, modified, convexity


def _treasury_yield(maturity: np.ndarray, cfg: CalibrationConfig) -> np.ndarray:
    """Linear interpolation on the par curve, flat beyond the knots."""
    return np.interp(maturity, cfg.treasury_knots_y, cfg.treasury_curve)


# --------------------------------------------------------------------------
# Universe generation
# --------------------------------------------------------------------------

def make_universe(
    n_bonds: int = 500,
    seed: int = 7,
    cfg: CalibrationConfig | None = None,
    bonds_per_issuer_lambda: float = 2.5,
    portfolio_coverage: float = 0.65,
    portfolio_tilt_sd: float = 0.20,
) -> BondUniverse:
    """Generate a seeded corporate-bond universe.

    Issuers carry the credit identity (sector, rating, issuer spread level);
    each issues several bonds along the curve, so per-issuer concentration
    limits and duration-bucket constraints interact the way they do on a real
    desk.

    `portfolio_coverage` is the fraction of the universe the incumbent
    portfolio `w0` actually holds — real credit portfolios hold a subset of
    the index, which is what makes a turnover budget bind. Coverage and
    `portfolio_tilt_sd` are deliberately moderate so that `w0` is a plausible
    *managed* book sitting near the index: an incumbent far from the index
    would make a one-month turnover budget mechanically dominate the answer,
    and the optimizer's reported portfolio would mostly reflect `w0`'s random
    construction. `one_way_distance_to_benchmark()` reports the resulting
    distance so the reader can judge that for themselves.
    """
    cfg = cfg or CalibrationConfig()
    rng = np.random.default_rng(seed)

    # ---- issuers ----------------------------------------------------------
    n_issuers = max(1, int(np.ceil(n_bonds / bonds_per_issuer_lambda)))

    r_names = list(cfg.rating_weights)
    r_probs = np.array([cfg.rating_weights[r] for r in r_names], dtype=float)
    r_probs /= r_probs.sum()
    issuer_rating = rng.choice(r_names, size=n_issuers, p=r_probs)

    if cfg.sector_weights:
        s_names = list(cfg.sector_weights)
        s_probs = np.array([cfg.sector_weights[s] for s in s_names], float)
        s_probs /= s_probs.sum()
    else:
        s_names, s_probs = list(SECTORS), None
    issuer_sector = rng.choice(s_names, size=n_issuers, p=s_probs)

    # Issuer-level spread idiosyncrasy (persistent across that issuer's curve)
    issuer_shock = rng.lognormal(0.0, cfg.oas_dispersion * 0.8, n_issuers)

    # ---- bonds ------------------------------------------------------------
    n_per_issuer = 1 + rng.poisson(bonds_per_issuer_lambda - 1.0, n_issuers)
    issuer_of_bond = np.repeat(np.arange(n_issuers), n_per_issuer)
    if len(issuer_of_bond) > n_bonds:                 # trim to requested size
        issuer_of_bond = issuer_of_bond[:n_bonds]
    while len(issuer_of_bond) < n_bonds:              # top up if short
        extra = rng.integers(0, n_issuers, n_bonds - len(issuer_of_bond))
        issuer_of_bond = np.concatenate([issuer_of_bond, extra])
    issuer_of_bond = np.sort(issuer_of_bond)
    n = len(issuer_of_bond)

    rating = issuer_rating[issuer_of_bond]
    sector = issuer_sector[issuer_of_bond]

    # Maturities on the coupon grid, concentrated at the classic issue points.
    tenors = np.array([2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])
    tenor_p = np.array([0.14, 0.12, 0.22, 0.14, 0.24, 0.06, 0.08])
    maturity = rng.choice(tenors, size=n, p=tenor_p)
    maturity = maturity + rng.choice([-0.5, 0.0, 0.5], size=n,
                                     p=[0.25, 0.5, 0.25])
    maturity = np.clip(maturity, 1.0, 30.0)

    # OAS: rating level x sector multiplier x issuer shock x bond noise,
    # with a mild term structure (longer bonds trade a touch wider).
    base = np.array([cfg.oas_by_rating_bp[r] for r in rating])
    smult = np.array([cfg.sector_spread_mult[s] for s in sector])
    term = 1.0 + 0.012 * (maturity - 7.0)
    noise = rng.lognormal(0.0, cfg.oas_dispersion * 0.5, n)
    oas_bp = base * smult * issuer_shock[issuer_of_bond] * term * noise
    oas_bp = np.clip(oas_bp, 5.0, 2500.0)

    # Yield = Treasury + OAS; coupon set near the yield at issue, seasoned.
    treas = _treasury_yield(maturity, cfg)
    ytm = treas + oas_bp / 10_000.0
    coupon = np.maximum(
        0.0, ytm + rng.normal(0.0, cfg.coupon_seasoning_sd, n))
    coupon = np.round(coupon * 800.0) / 800.0        # 12.5bp coupon grid

    price, mod_dur, convexity = price_and_duration(coupon, ytm, maturity)

    # Fixed-rate bullets: spread duration == modified duration (see module
    # docstring for why this assumption is stated rather than assumed away).
    spread_duration = mod_dur

    issue_size = np.maximum(
        cfg.issue_size_floor_mm,
        rng.lognormal(np.log(cfg.issue_size_median_mm),
                      cfg.issue_size_sigma, n),
    )

    cell_code = np.clip(
        np.digitize(maturity, MATURITY_CELL_EDGES[1:-1]), 0,
        len(MATURITY_CELL_NAMES) - 1)

    bonds = pd.DataFrame({
        "bond_id": [f"B{i:05d}" for i in range(n)],
        "issuer_id": [f"ISS{j:04d}" for j in issuer_of_bond],
        "sector": sector,
        "rating": rating,
        "maturity": maturity,
        "coupon": coupon,
        "ytm": ytm,
        "oas_bp": oas_bp,
        "price": price,
        "mod_duration": mod_dur,
        "spread_duration": spread_duration,
        "convexity": convexity,
        "issue_size_mm": issue_size,
        "maturity_cell_code": cell_code,
        "maturity_cell": [MATURITY_CELL_NAMES[c] for c in cell_code],
    })

    # ---- benchmark and incumbent portfolio --------------------------------
    market_value = issue_size * price / 100.0
    w_benchmark = market_value / market_value.sum()

    # The incumbent holds a subset of the index (coverage), roughly index-like
    # within that subset, with active tilts.
    n_hold = max(10, int(round(portfolio_coverage * n)))
    held = rng.choice(n, size=n_hold, replace=False,
                      p=w_benchmark / w_benchmark.sum())
    w0 = np.zeros(n)
    tilt = rng.lognormal(0.0, portfolio_tilt_sd, n_hold)
    w0[held] = w_benchmark[held] * tilt
    w0 /= w0.sum()

    return BondUniverse(bonds=bonds, w_benchmark=w_benchmark, w0=w0,
                        config=cfg, seed=seed)


def summarize(uni: BondUniverse, w: np.ndarray | None = None) -> pd.DataFrame:
    """Rating-level exposure summary for a portfolio (default: benchmark)."""
    w = uni.w_benchmark if w is None else w
    df = uni.bonds.assign(weight=w)
    out = df.groupby("rating", observed=True).apply(
        lambda g: pd.Series({
            "weight": g["weight"].sum(),
            "oas_bp": np.average(g["oas_bp"], weights=g["weight"])
            if g["weight"].sum() > 0 else np.nan,
            "duration": np.average(g["mod_duration"], weights=g["weight"])
            if g["weight"].sum() > 0 else np.nan,
            "dts": float((g["weight"] * g["spread_duration"]
                          * g["oas_bp"]).sum()),
            "n_bonds": int((g["weight"] > 1e-6).sum()),
        }), include_groups=False)
    return out.reindex([r for r in RATINGS if r in out.index])

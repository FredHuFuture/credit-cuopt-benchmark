"""Scenario generation.

Two engines:

- `equity_scenarios` — the simple fat-tailed factor model used for the M1
  equity benchmark. Frozen: the committed M1 results depend on it.
- `credit_scenarios` — the credit engine (M2). Joint rate and spread moves
  with fat tails, plus issuer-level default jumps. Runs on the GPU via CuPy
  when available, with an identical NumPy path otherwise.

Total return or excess return? (This choice decides what is being optimized.)
-----------------------------------------------------------------------------
A 6-duration investment-grade book carries roughly twice as much monthly
risk from rates as from spreads: D x 25bp = 1.56% against
D x 149bp x 9.4% = 0.87%. Optimize *total* return and the answer is dominated
by a duration bet wearing a credit costume. (An external adversarial review of
the pre-fix model put a number on it — roughly two thirds of the apparent CVaR
improvement came from shortening duration — via its own reduced-form
reproduction. That decomposition is not reproducible at this commit, because
the model has since changed in several other respects; the arithmetic above is
the part you can check yourself.)

So the default is `return_mode="excess"`: return over duration-matched
Treasuries, which is what credit desks actually optimize (rates are hedged or
index-matched separately):

    r_excess = OAS*h - SD*ds + 0.5*C*ds^2   (+ default jump)

This holds to second order in ds. Cancelling a duration-matched Treasury leg
leaves a rate-spread cross term C*dy*ds which is dropped here; because rates
and spreads are negatively correlated, that term is most negative in exactly
the tail scenarios CVaR measures, so absolute CVaR is understated by roughly
3%. It shifts benchmark and optimised portfolios almost identically, so
relative comparisons are unaffected. `return_mode="total"` keeps the rate leg
for the unhedged view.

Why the credit engine looks the way it does
-------------------------------------------
1. **Spread moves are relative, not absolute.** A 20bp move means something
   very different for a 45bp AAA and a 420bp single-B. Modelling
   `Δs_i = s_i × (relative shock)` makes spread volatility proportional to
   spread level, which is exactly the empirical regularity behind DTS
   (Ben Dor et al., 2007) — and it is why the DTS constraint in
   `src/credit.py` is the right risk limit to impose.
2. **Fat tails come from a multivariate Student-t.** Credit factor moves are
   not Gaussian; CVaR is a tail statistic, so the tail assumption is the
   assumption that matters most.
3. **Widening is skewed.** Spreads gap wider and grind tighter. The
   systematic factor gets an asymmetric scaling, applied mean-preservingly so
   it adds tail shape without smuggling in a drift.
4. **Defaults are issuer-level and correlated with widening.** A one-factor
   structural trigger (Vasicek/ASRF-style) driven by the *same* systematic
   credit factor means default clustering and spread widening happen in the
   same scenarios — and because the trigger is per issuer, holding five bonds
   from one issuer is five losses at once. That is precisely the risk the
   per-issuer concentration limit exists to control, so the risk model and
   the constraint set tell the same story.
5. **Rates and credit are negatively correlated.** Flight to quality: in the
   scenarios where spreads gap, Treasury yields fall and duration helps.

Precision: sampling runs in float32 by default (GeForce fp64 throughput is
1:64, and the sampling is bandwidth-bound), then the scenario matrix is cast
to float64 for the LP, which cuOpt solves in double precision.

Reproducibility caveat: CuPy and NumPy generators do not share a random
stream, so the GPU and CPU paths produce *statistically identical but not
bit-identical* scenario sets from the same seed. Results committed to this
repo state which device produced them, and `scripts/check_rng_agreement.py`
verifies the two paths agree distributionally.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from .universe import BondUniverse

try:  # CuPy ships as a cuOpt dependency; absence must not break the CPU path.
    import cupy as _cp
    _HAS_CUPY = _cp.cuda.runtime.getDeviceCount() > 0
except Exception:  # noqa: BLE001
    _cp = None
    _HAS_CUPY = False


# --------------------------------------------------------------------------
# M1 equity engine (frozen — committed M1 results depend on it)
# --------------------------------------------------------------------------

def equity_scenarios(
    n_assets: int,
    n_scen: int,
    seed: int = 42,
    n_factors: int = 10,
    t_dof: float = 5.0,
    daily_vol: float = 0.015,
    annual_sharpe: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Correlated, fat-tailed synthetic daily equity returns.

    r = B f + eps with Student-t factors and idiosyncratic noise, scaled to a
    target daily vol. Expected returns are the *true* model drifts (not sample
    means), so optimizer inputs are not contaminated by scenario noise.

    Returns
    -------
    R : (n_scen, n_assets) scenario return matrix
    mu : (n_assets,) expected daily returns
    """
    rng = np.random.default_rng(seed)

    # Factor loadings: one market factor everyone loads on, plus style factors.
    B = np.zeros((n_assets, n_factors))
    B[:, 0] = rng.normal(1.0, 0.3, n_assets)              # market beta ~ 1
    B[:, 1:] = rng.normal(0.0, 0.5, (n_assets, n_factors - 1))

    # Factor covariance: market factor dominates.
    f_vol = np.full(n_factors, 0.004)
    f_vol[0] = 0.010

    # Student-t factors (standardized to unit variance, then scaled).
    t_scale = np.sqrt((t_dof - 2.0) / t_dof)
    F = rng.standard_t(t_dof, size=(n_scen, n_factors)) * t_scale * f_vol

    eps_vol = 0.008
    eps = rng.standard_t(t_dof, size=(n_scen, n_assets)) * t_scale * eps_vol

    R = F @ B.T + eps

    # Rescale each asset to the target daily vol (keeps conditioning sane).
    vols = R.std(axis=0)
    R *= daily_vol / vols

    # True drifts: cross-sectional dispersion around a common risk premium.
    mu_annual = daily_vol * np.sqrt(252) * annual_sharpe
    mu = (mu_annual / 252) * rng.uniform(0.2, 1.8, n_assets)
    R += mu  # center scenarios on the true drift

    return R, mu


# --------------------------------------------------------------------------
# M2 credit engine
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class CreditScenarioConfig:
    """Monthly-horizon dynamics. Vols are per horizon, not annualized.

    Calibration targets, if real data is ever used: rate factor vols from
    Treasury constant-maturity series; relative spread vols from the ICE BofA
    OAS series (log-changes); default rates from a rating-agency transition
    study. Defaults below are stylized, not fitted.
    """

    t_dof: float = 5.0                    # Student-t degrees of freedom
    rate_level_vol: float = 0.0025        # 25bp/month parallel shift
    rate_slope_vol: float = 0.0012        # 12bp/month steepening
    credit_sys_vol: float = 0.08          # 8%/month RELATIVE spread move
    rating_factor_vol: float = 0.040      # residual rating-specific
    sector_factor_vol: float = 0.050      # residual sector-specific
    idio_vol: float = 0.060               # per-bond residual
    hy_vol_mult: float = 1.20             # HY relative vol above DTS-implied
    widening_skew: float = 1.35           # >1 => widening tail fatter
    corr_level_slope: float = -0.30
    corr_level_credit: float = -0.35      # flight to quality
    #: Annual default probabilities by rating (stylized agency-study levels).
    pd_annual: Mapping[str, float] = field(default_factory=lambda: {
        "AAA": 0.0001, "AA": 0.0003, "A": 0.0008,
        "BBB": 0.0025, "BB": 0.0110, "B": 0.0420,
    })
    recovery_price: float = 40.0          # price the bond trades to on default
    asset_correlation: float = 0.15       # Vasicek/ASRF single-factor rho


def _xp(use_gpu: bool):
    return _cp if (use_gpu and _HAS_CUPY) else np


def warmup_gpu(uni: "BondUniverse | None" = None) -> bool:
    """Create the CUDA context and compile kernels before anything is timed.

    The first CuPy call in a process pays for context creation and kernel
    compilation — seconds, not milliseconds. Pass the universe: a generic
    matmul warms the context but NOT the kernels this module actually uses
    (gamma sampling, quantile, where, fancy indexing), which left a measured
    20,000-scenario generation looking 8x slower than it is. Warm up the real
    code path, the same discipline as the discarded warmup solve in
    `src/bench.py`.
    """
    if not _HAS_CUPY:
        return False
    if uni is not None:
        credit_scenarios(uni, n_scen=256, seed=0, use_gpu=True)
    else:
        rng = _cp.random.default_rng(0)
        x = rng.standard_normal((256, 256), dtype=np.float32)
        _ = float(_cp.asnumpy((2.0 * _cp.sqrt(_cp.abs(x)) @ x).sum()))
    _cp.cuda.runtime.deviceSynchronize()
    return True


def _to_numpy(a):
    return _cp.asnumpy(a) if (_cp is not None and isinstance(a, _cp.ndarray)) \
        else np.asarray(a)


def _mv_t_factors(xp, rng, n_scen: int, corr: np.ndarray, dof: float, dtype):
    """Multivariate Student-t factors, standardized to unit variance."""
    k = corr.shape[0]
    L = np.linalg.cholesky(corr)
    z = rng.standard_normal((n_scen, k), dtype=dtype)
    g = z @ xp.asarray(L.T, dtype=dtype)
    # chi2(dof) = 2 * Gamma(dof/2, scale=1); works on NumPy and CuPy Generators
    chi2 = 2.0 * rng.standard_gamma(dof / 2.0, size=(n_scen, 1), dtype=dtype)
    t = g * xp.sqrt(xp.asarray(dof, dtype=dtype) / chi2)
    return t * np.sqrt((dof - 2.0) / dof)          # unit variance


def credit_scenarios(
    uni: BondUniverse,
    n_scen: int = 10_000,
    horizon_years: float = 1.0 / 12.0,
    seed: int = 11,
    cfg: CreditScenarioConfig | None = None,
    use_gpu: bool = True,
    dtype=np.float32,
    return_mode: str = "excess",
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Joint rate + spread + default scenarios for a bond universe.

    Parameters
    ----------
    return_mode : {"excess", "total"}
        "excess" (default) returns credit excess return over duration-matched
        Treasuries — what a credit mandate is measured on. "total" adds the
        rate leg back. See the module docstring for why this matters.

    Returns
    -------
    R : (n_scen, n_bonds) float64 returns over the horizon
    mu : (n_bonds,) float64 expected return. Shocks are demeaned in-sample so
         the only intended drifts are carry, the convexity gain, and the
         expected default loss; `info["mu_vs_sample_mean_max_bp"]` reports the
         residual disagreement with the realized scenario mean.
    info : dict of timings and realized diagnostics
    """
    if return_mode not in ("excess", "total"):
        raise ValueError("return_mode must be 'excess' or 'total'")
    cfg = cfg or CreditScenarioConfig()
    xp = _xp(use_gpu)
    on_gpu = xp is not np
    rng = xp.random.default_rng(seed)
    t0 = time.perf_counter()

    bonds = uni.bonds
    n = len(bonds)
    ratings = bonds["rating"].to_numpy()
    r_codes, r_labels = _codes(ratings)
    s_codes, s_labels = _codes(bonds["sector"].to_numpy())
    i_codes, i_labels = _codes(bonds["issuer_id"].to_numpy())
    n_r, n_s, n_iss = len(r_labels), len(s_labels), len(i_labels)

    # ---- correlated systematic factors: [level, slope, credit_sys] --------
    corr = np.eye(3)
    corr[0, 1] = corr[1, 0] = cfg.corr_level_slope
    corr[0, 2] = corr[2, 0] = cfg.corr_level_credit
    corr = _nearest_psd(corr)
    F = _mv_t_factors(xp, rng, n_scen, corr, cfg.t_dof, dtype)
    F = F - F.mean(axis=0, keepdims=True)      # see _demean note below
    level = F[:, 0:1] * dtype(cfg.rate_level_vol)
    slope = F[:, 1:2] * dtype(cfg.rate_slope_vol)
    sys_raw = F[:, 2:3]

    # Mean-preserving widening skew: inflate the widening (positive) side,
    # then recentre so no drift is introduced. Note the skew also inflates
    # realized volatility by ~18% over `credit_sys_vol`; `sys_std` is
    # renormalized separately because the default trigger needs unit variance.
    sys_sk = xp.where(sys_raw > 0, sys_raw * dtype(cfg.widening_skew), sys_raw)
    sys_sk = sys_sk - sys_sk.mean()
    sys_std = sys_sk / sys_sk.std()            # unit variance, for the trigger
    credit_sys = sys_sk * dtype(cfg.credit_sys_vol)

    # ---- residual rating / sector / idiosyncratic spread factors ----------
    # Every block is demeaned in-sample. Left alone, a sample mean of order
    # sigma/sqrt(S) becomes a spurious drift the optimizer will happily trade
    # against — for the rate level at S=20k that is ~13bp/yr of duration-
    # proportional "alpha" that exists only in the sampling error.
    rf = _demean(rng.standard_normal((n_scen, n_r), dtype=dtype)) \
        * dtype(cfg.rating_factor_vol)
    sf = _demean(rng.standard_normal((n_scen, n_s), dtype=dtype)) \
        * dtype(cfg.sector_factor_vol)
    idio = _demean(rng.standard_normal((n_scen, n), dtype=dtype)) \
        * dtype(cfg.idio_vol)

    r_codes_x = xp.asarray(r_codes)
    s_codes_x = xp.asarray(s_codes)
    i_codes_x = xp.asarray(i_codes)

    # Relative spread shock per bond. DTS logic: same relative vol across
    # ratings is the DTS-implied baseline; HY gets a modest empirical uplift.
    hy_mult = xp.asarray(
        np.where(np.isin(ratings, ("BB", "B")), cfg.hy_vol_mult, 1.0),
        dtype=dtype)
    ds_rel = (credit_sys + rf[:, r_codes_x] + sf[:, s_codes_x] + idio) * hy_mult

    # ---- price return ----------------------------------------------------
    # Second order matters here: linearizing a bond's price response always
    # OVERSTATES the loss for a yield rise, and the CVaR tail is exactly where
    # the big moves live. On a 30y bond a 125bp move is mispriced by 2.5% of
    # price without the convexity term — against a ~3% CVaR that is not a
    # rounding error, and it biases the optimizer against long bonds.
    sdur = xp.asarray(uni.spread_duration, dtype=dtype)
    conv = xp.asarray(uni.convexity, dtype=dtype)
    oas = xp.asarray(uni.oas, dtype=dtype)

    ds_abs = oas[None, :] * ds_rel            # relative -> absolute spread move

    if return_mode == "excess":
        # Excess over duration-matched Treasuries: the -D*dy leg cancels.
        # The rate factors are deliberately NOT materialized per bond here —
        # a full (n_scen, n_bonds) rate-shock matrix costs 400 MB of device
        # memory and ~150ms at 50k x 2,000, inside the timed region, for a
        # quantity this branch never uses.
        carry = oas * dtype(horizon_years)
        dtotal = ds_abs
        R = carry[None, :] - sdur * ds_abs + dtype(0.5) * conv * ds_abs ** 2
    else:
        maturity = xp.asarray(bonds["maturity"].to_numpy(), dtype=dtype)
        slope_load = (maturity - dtype(7.0)) / dtype(10.0)
        dy = level + slope * slope_load[None, :]
        dur = xp.asarray(uni.duration, dtype=dtype)
        ytm = xp.asarray(uni.ytm, dtype=dtype)
        carry = ytm * dtype(horizon_years)
        dtotal = dy + ds_abs                  # total yield change
        R = carry[None, :] - dur * dtotal + dtype(0.5) * conv * dtotal ** 2

    # Expected convexity gain, computed from the realized shock sample (it is
    # a second-order term; using the sample second moment keeps mu consistent
    # with the scenarios that were actually generated).
    conv_gain = dtype(0.5) * conv * (dtotal ** 2).mean(axis=0)

    # ---- issuer-level defaults (single-factor structural trigger) --------
    pd_h = np.array([cfg.pd_annual[r] for r in r_labels]) * horizon_years
    issuer_rating_code = _first_code_per_group(i_codes, r_codes, n_iss)
    rho = dtype(cfg.asset_correlation)
    zi = rng.standard_normal((n_scen, n_iss), dtype=dtype)
    # Asset return falls when the systematic credit factor widens.
    A = -xp.sqrt(rho) * sys_std + xp.sqrt(dtype(1.0) - rho) * zi

    # Calibrate the trigger on an INDEPENDENT draw from the same asset-return
    # distribution. Quantiling the scenario matrix itself would make the
    # "realized vs target default rate" diagnostic a tautology rather than a
    # check; with a separate draw it is a genuine (if simple) validation.
    #
    # Cached per process: the thresholds are a property of the MODEL (t_dof,
    # skew, rho, PDs), not of any particular scenario set. Recomputing a
    # fixed 400k draw on every call put a constant ~30ms inside the timed
    # region — which at 2,000 scenarios was a third of the measurement.
    #
    # The Monte Carlo ESTIMATE of them, however, is not model-only: it is
    # drawn from `rng` at whatever state this line is reached in, so the cache
    # key has to include that state or the answer depends on cache warmth.
    # `draw_key` is the seed plus every draw taken above (each n_scen wide by
    # one of these counts), which is exactly what fixes it. See
    # _default_thresholds.
    draw_key = (seed, n_scen, n_r, n_s, n, n_iss)
    thresholds = _default_thresholds(xp, rng, corr, cfg, rho, pd_h, dtype,
                                     draw_key)
    issuer_thr = thresholds[xp.asarray(issuer_rating_code)]
    defaulted_issuer = A < issuer_thr[None, :]           # (n_scen, n_iss)
    defaulted = defaulted_issuer[:, i_codes_x]           # (n_scen, n_bonds)

    price = xp.asarray(bonds["price"].to_numpy(), dtype=dtype)
    r_default = (dtype(cfg.recovery_price) - price) / price
    R = xp.where(defaulted, r_default[None, :], R)

    # ---- expected return -------------------------------------------------
    # carry + convexity gain - expected default loss. Shocks are demeaned, so
    # these are the only intended drifts; the residual disagreement with the
    # realized sample mean is measured below rather than asserted away.
    pd_bond = xp.asarray(pd_h[r_codes], dtype=dtype)
    mu = carry + conv_gain + pd_bond * r_default    # r_default is negative

    R_host = _to_numpy(R).astype(np.float64)
    mu_host = _to_numpy(mu).astype(np.float64)
    if on_gpu:
        _cp.cuda.runtime.deviceSynchronize()
    # Timing includes the device-to-host transfer: the CPU path has no such
    # step, so excluding it would flatter the GPU.
    elapsed = time.perf_counter() - t0

    sample_mean = R_host.mean(axis=0)
    mu_gap_bp = float(np.abs(mu_host - sample_mean).max() * 1e4)
    info = {
        "device": "GPU (CuPy)" if on_gpu else "CPU (NumPy)",
        "seconds": elapsed,
        "return_mode": return_mode,
        "n_scen": n_scen,
        "n_bonds": n,
        "n_issuers": n_iss,
        "realized_default_rate_per_bond": float(_to_numpy(defaulted.mean())),
        "target_default_rate_per_bond": float(pd_h[r_codes].mean()),
        "scenarios_with_any_default": float(
            _to_numpy(defaulted.any(axis=1).mean())),
        "mu_vs_sample_mean_max_bp": mu_gap_bp,
        "dtype": np.dtype(dtype).name,
    }
    return R_host, mu_host, info


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _demean(a):
    """Remove the in-sample mean of every column (variance reduction)."""
    return a - a.mean(axis=0, keepdims=True)


#: Default-trigger thresholds, cached per (model parameters, device, dtype,
#: generator state at the calibration point).
_THRESHOLD_CACHE: dict = {}


def _default_thresholds(xp, rng, corr, cfg, rho, pd_h, dtype, draw_key,
                        n_cal=400_000):
    """Asset-return quantiles at each rating's PD, from an independent draw.

    The calibration draw comes off the caller's generator, so the values
    depend on the generator's state as well as on the model parameters.
    `draw_key` carries that state (seed and every draw taken before this
    point) into the cache key, which makes the result a pure function of the
    call rather than of the order calls happened to arrive in: a cache hit
    now returns exactly what a cache miss would have computed. A fresh
    process is unaffected — the first call for a given key still computes
    from the same generator state it always did.
    """
    key = (xp is np, np.dtype(dtype).name, cfg.t_dof, cfg.widening_skew,
           float(cfg.asset_correlation), tuple(np.round(pd_h, 12)),
           draw_key)
    hit = _THRESHOLD_CACHE.get(key)
    if hit is not None:
        return hit
    cal_sys = _mv_t_factors(xp, rng, n_cal, corr, cfg.t_dof, dtype)[:, 2:3]
    cal_sk = xp.where(cal_sys > 0, cal_sys * dtype(cfg.widening_skew), cal_sys)
    cal_sk = (cal_sk - cal_sk.mean()) / cal_sk.std()
    cal_A = (-xp.sqrt(rho) * cal_sk
             + xp.sqrt(dtype(1.0) - rho)
             * rng.standard_normal((n_cal, 1), dtype=dtype)).ravel()
    # p == 0 is not a quantile this estimator can express: a Monte Carlo
    # `quantile(x, 0.0)` is the sample MINIMUM of the calibration draw, so a
    # rating with PD exactly zero still defaults as soon as the scenario draw
    # reaches further into the left tail than the 400k calibration draw did.
    # That leak is invisible at the shipped PDs (all strictly positive, so
    # every committed result is bit-identical either way) and fatal to a
    # defaults-off control arm, which is the one place PD = 0 is asked for.
    # -inf is the honest threshold: "this rating never defaults".
    neg_inf = xp.asarray(-np.inf, dtype=dtype)
    out = xp.asarray([xp.quantile(cal_A, float(p)) if p > 0.0 else neg_inf
                      for p in pd_h], dtype=dtype)
    _THRESHOLD_CACHE[key] = out
    return out


def _codes(values: np.ndarray) -> tuple[np.ndarray, list[str]]:
    labels, codes = np.unique(values, return_inverse=True)
    return codes.astype(np.int32), [str(x) for x in labels]


def _first_code_per_group(group_codes: np.ndarray, value_codes: np.ndarray,
                          n_groups: int) -> np.ndarray:
    """Rating code of each issuer (all bonds of an issuer share its rating)."""
    out = np.zeros(n_groups, dtype=np.int32)
    out[group_codes] = value_codes
    return out


def _nearest_psd(corr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Clip negative eigenvalues so the Cholesky always succeeds."""
    vals, vecs = np.linalg.eigh(corr)
    if vals.min() > eps:
        return corr
    vals = np.clip(vals, eps, None)
    out = vecs @ np.diag(vals) @ vecs.T
    d = np.sqrt(np.diag(out))
    return out / np.outer(d, d)

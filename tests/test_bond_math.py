"""The bond math, checked against the closed forms the README claims.

`src/universe.py` prices every bond from an explicit semiannual cash-flow
schedule and derives modified duration and convexity from that same schedule.
Its docstring — and the README — make four specific claims about that
arithmetic. This file is those four claims, written as assertions:

1. A par bond (coupon == yield) prices to exactly 100.
2. A zero-coupon bond's Macaulay duration equals its maturity.
3. Modified duration equals a central finite difference of the price function,
   -(1/P) dP/dy, to ~1e-7 absolute.
4. Convexity equals a second-order finite difference, (1/P) d2P/dy2.

Claims 3 and 4 are the ones worth having: 1 and 2 are identities that a
cash-flow bug could still satisfy, whereas the finite differences compare the
analytic sensitivities against the pricing function itself, which is the only
thing that can catch a mismatch between the price loop and the duration loop.

The public API is used as-is (`price_and_duration`); nothing here reimplements
the math, otherwise the test would only prove the two copies agree.

CPU-only and dependency-light on purpose: no GPU, no solver, no cvxpy. This
must be runnable by anyone who clones the repo.

    pytest tests/test_bond_math.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.universe import price_and_duration  # noqa: E402

FREQ = 2

#: A spread of shapes: sub-year, on-the-run tenors, a half-year off the grid
#: point, deep discount, deep premium, and a 30y where convexity is large.
#: Maturities all lie on the 1/FREQ grid, which the function enforces.
MATURITY = np.array([0.5, 1.0, 2.0, 3.5, 5.0, 7.5, 10.0, 20.0, 30.0])
COUPON = np.array([0.00, 0.030, 0.050, 0.020, 0.070, 0.045, 0.060, 0.0125, 0.09])
YTM = np.array([0.050, 0.040, 0.050, 0.030, 0.060, 0.050, 0.045, 0.055, 0.040])


def _price(ytm: np.ndarray) -> np.ndarray:
    return price_and_duration(COUPON, ytm, MATURITY, freq=FREQ)[0]


def test_par_bond_prices_to_100() -> None:
    """Claim 1: coupon == yield => price == face, at every maturity."""
    y = YTM.copy()
    price, _, _ = price_and_duration(y, y, MATURITY, freq=FREQ)
    assert price == pytest.approx(np.full_like(price, 100.0), rel=0, abs=1e-11)


def test_zero_coupon_macaulay_duration_equals_maturity() -> None:
    """Claim 2: a zero's PV-weighted mean time to cash flow is its maturity.

    The function returns MODIFIED duration, so the identity is checked after
    undoing the 1/(1 + y/freq) factor — which also pins that factor down.
    """
    zeros = np.zeros_like(MATURITY)
    _, modified, _ = price_and_duration(zeros, YTM, MATURITY, freq=FREQ)
    macaulay = modified * (1.0 + YTM / FREQ)
    assert macaulay == pytest.approx(MATURITY, rel=0, abs=1e-12)


def test_modified_duration_matches_central_finite_difference() -> None:
    """Claim 3: D_mod == -(1/P) dP/dy, to ~1e-7 in years of duration.

    h = 1e-6 balances the O(h^2) truncation error against cancellation in the
    price difference; the observed error at this step is ~3.5e-9 years, so the
    1e-7 gate has room but would still catch a real sign or off-by-one-period
    error immediately.
    """
    h = 1e-6
    _, modified, _ = price_and_duration(COUPON, YTM, MATURITY, freq=FREQ)
    fd = -(_price(YTM + h) - _price(YTM - h)) / (2.0 * h) / _price(YTM)
    assert np.abs(fd - modified).max() < 1e-7


def test_convexity_matches_second_order_finite_difference() -> None:
    """Claim 4: convexity == (1/P) d2P/dy2.

    A larger step than for duration (1e-4): the second difference divides by
    h^2, so cancellation, not truncation, is the binding error term here —
    at h = 1e-6 the same check is only good to ~7e-4 relative. Compared in
    relative terms because convexity spans ~0.5 to ~330 across this set.
    """
    h = 1e-4
    _, _, convexity = price_and_duration(COUPON, YTM, MATURITY, freq=FREQ)
    p0 = _price(YTM)
    fd = (_price(YTM + h) - 2.0 * p0 + _price(YTM - h)) / (h * h * p0)
    assert fd == pytest.approx(convexity, rel=1e-5)

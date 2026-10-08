"""Two gates, named, and the boundary that separates them.

The repo's original mandate gate is `|rel_slack| <= 1e-6`: a numerical
statement about whether a point sits on a constraint. A credit PM asks a
different question — would anyone notice? — and answers it at about 1bp of
portfolio weight, because that is the precision positions are reported at and
because lot rounding moves a book orders of magnitude further than 1e-6. The
two verdicts disagree on real solves, so both are reported and neither is
quietly chosen.

Every report here is hand-built, with breaches placed exactly where they need
to be. A solver cannot be asked to land on a gate boundary on demand, and a
test that hopes it does is not testing the boundary.

Units are the other half of this. The report mixes weight fractions, years of
duration, bp-years of DTS and bp of annualized active return; a dollar figure
is emitted only where the breach is a fraction of capital, and the rows where
it is not carry the reason instead of a fabricated number.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.credit import (  # noqa: E402
    DESK_TOL_BY_UNIT, DESK_WEIGHT_GATE_BP, NOTIONAL_USD, PURITY_REL_GATE,
    UNIT_DTS, UNIT_RETURN_BP, UNIT_WEIGHT, UNIT_WEIGHT_L1, UNIT_YEARS,
    VERDICT_BREACH, VERDICT_DESK_ONLY, VERDICT_OK, default_limits,
    exposure_report, gate_verdicts, score_breaches, worst_desk_violation,
    worst_violation,
)
from src.universe import make_universe  # noqa: E402


def _report(*rows: tuple[str, str, float, float, str]) -> pd.DataFrame:
    """Build the frame `exposure_report` builds, from (item, unit, slack,
    scale, family) tuples, and score it the same way."""
    rep = pd.DataFrame(
        [{"family": fam, "item": item, "portfolio": np.nan,
          "benchmark": np.nan, "limit": "-", "slack": slack, "scale": scale,
          "unit": unit}
         for item, unit, slack, scale, fam in rows])
    rep["rel_slack"] = rep["slack"] / rep["scale"]
    rep["binding"] = rep["rel_slack"] < PURITY_REL_GATE
    return score_breaches(rep)


def test_the_documented_constants_are_the_constants():
    """These numbers are quoted in prose in three files. Pin them here so a
    silent retune fails a test instead of rewriting a published claim."""
    assert PURITY_REL_GATE == 1e-6
    assert DESK_WEIGHT_GATE_BP == 1.0
    assert NOTIONAL_USD == 1_000_000_000.0
    assert DESK_TOL_BY_UNIT[UNIT_WEIGHT] == 1e-4        # 1bp of weight
    assert DESK_TOL_BY_UNIT[UNIT_WEIGHT_L1] == 1e-4


def test_one_basis_point_is_the_boundary_and_the_boundary_is_inclusive():
    """The case that separates the gates, at 1bp and a hair over.

    A 1bp weight breach on a $1bn book is exactly $100,000, sits exactly on
    the desk tolerance, and is a pass; 1.01bp is $101,000 and is not. The
    purity gate fails on both, which is the point — it is measuring something
    else.
    """
    at = _report(("BBB band", UNIT_WEIGHT, -1e-4, 0.1, "rating"))
    over = _report(("BBB band", UNIT_WEIGHT, -1.01e-4, 0.1, "rating"))

    assert at["breach_bp_weight"].iloc[0] == pytest.approx(1.0, rel=1e-12)
    assert at["breach_usd"].iloc[0] == pytest.approx(100_000.0, rel=1e-12)
    assert at["breach_desk_multiple"].iloc[0] == pytest.approx(1.0, rel=1e-12)
    assert not at["breaches_desk_gate"].iloc[0]
    assert at["breaches_purity_gate"].iloc[0]

    assert over["breach_usd"].iloc[0] == pytest.approx(101_000.0, rel=1e-12)
    assert over["breaches_desk_gate"].iloc[0]

    assert gate_verdicts(at).verdict == VERDICT_DESK_ONLY
    assert gate_verdicts(over).verdict == VERDICT_BREACH


def test_the_23e_6_breach_that_changes_the_headline():
    """The case from the review, in dollars.

    A 2.3e-6 relative breach of a 25% one-way turnover budget is 5.75e-7 of
    the book: 0.0058bp, $575 on $1bn. It fails the 1e-6 gate — which is what
    currently marks the configuration as non-compliant — and it is 174x
    INSIDE the desk gate. Both verdicts are reported; the paper does not get
    to pick.
    """
    rep = _report(("one-way turnover", UNIT_WEIGHT, -2.3e-6 * 0.25, 0.25,
                   "turnover"))
    g = gate_verdicts(rep)

    assert rep["breach"].iloc[0] == pytest.approx(5.75e-7, rel=1e-12)
    assert rep["breach_bp_weight"].iloc[0] == pytest.approx(0.00575, rel=1e-9)
    assert rep["breach_usd"].iloc[0] == pytest.approx(575.0, rel=1e-9)

    assert not g.purity_pass
    assert g.purity_worst_rel == pytest.approx(2.3e-6, rel=1e-9)
    assert g.desk_pass
    assert g.desk_worst_multiple == pytest.approx(0.00575, rel=1e-9)
    assert g.verdict == VERDICT_DESK_ONLY
    assert g.n_purity_breaches == 1 and g.n_desk_breaches == 0


def test_dollars_only_where_dollars_mean_something():
    """A breach in dollars is a sentence about capital. Years of duration and
    bp-years of DTS are not capital, and no assumed market move is invented
    to make them look like it."""
    rep = _report(
        ("BBB band", UNIT_WEIGHT, -2e-4, 0.1, "rating"),
        ("active weight L1", UNIT_WEIGHT_L1, -3e-4, 0.5, "tracking_error"),
        ("total duration", UNIT_YEARS, -0.02, 0.1, "duration"),
        ("DTS vs floor (bp-y)", UNIT_DTS, -2.0, 900.0, "dts"),
        ("TE sigma (bp ann.)", UNIT_RETURN_BP, -0.4, 10.0, "tracking_error"),
    )
    usd = rep.set_index("item")["breach_usd"]
    bp = rep.set_index("item")["breach_bp_weight"]

    assert usd["BBB band"] == pytest.approx(200_000.0, rel=1e-12)
    assert usd["active weight L1"] == pytest.approx(300_000.0, rel=1e-12)
    assert bp["BBB band"] == pytest.approx(2.0, rel=1e-12)
    for item in ("total duration", "DTS vs floor (bp-y)",
                 "TE sigma (bp ann.)"):
        assert np.isnan(usd[item]), f"{item} must not carry a dollar figure"
        assert np.isnan(bp[item])
    basis = rep.set_index("item")["usd_basis"]
    assert "yield move" in basis["total duration"]
    assert "spread move" in basis["DTS vs floor (bp-y)"]
    assert "not a fraction of capital" in basis["TE sigma (bp ann.)"]

    # ...and the non-weight families are still SCORED. A 0.02y duration
    # breach is four ticks of the 0.01y a desk reads duration at, so the desk
    # gate fails on it even though it has no dollar value.
    assert rep.set_index("item")["breaches_desk_gate"]["total duration"]
    assert rep.set_index("item")["breach_desk_multiple"]["total duration"] \
        == pytest.approx(4.0, rel=1e-12)


def test_the_two_gates_rank_rows_differently():
    """Not a second opinion on the same ordering — a different ordering.

    The DTS row is the worse breach relative to its own limit; the issuer row
    is the worse breach in the units a desk reads. A single gate has to
    discard one of those facts.
    """
    rep = _report(
        ("DTS vs floor (bp-y)", UNIT_DTS, -0.4, 900.0, "dts"),
        ("max issuer weight", UNIT_WEIGHT, -1.5e-4, 1.0, "issuer"),
    )
    purity_val, purity_item = worst_violation(rep)
    desk_val, desk_item = worst_desk_violation(rep)

    assert purity_item == "DTS vs floor (bp-y)"
    assert purity_val == pytest.approx(0.4 / 900.0, rel=1e-12)
    assert desk_item == "max issuer weight"
    assert desk_val == pytest.approx(1.5, rel=1e-12)

    g = gate_verdicts(rep)
    assert g.verdict == VERDICT_BREACH
    assert g.purity_worst_limit == "DTS vs floor (bp-y)"
    assert g.desk_worst_limit == "max issuer weight"
    # The dollar figure belongs to the DESK-worst row, and it exists because
    # that row happens to be weight-denominated.
    assert g.desk_worst_bp_weight == pytest.approx(1.5, rel=1e-12)
    assert g.desk_worst_usd == pytest.approx(150_000.0, rel=1e-12)


def test_no_dollars_when_the_worst_desk_row_has_none():
    """A DTS-only breach fails the desk gate with an empty dollar column
    rather than a made-up one."""
    rep = _report(("DTS vs floor (bp-y)", UNIT_DTS, -2.0, 900.0, "dts"))
    g = gate_verdicts(rep)

    assert g.verdict == VERDICT_BREACH
    assert g.desk_worst_multiple == pytest.approx(4.0, rel=1e-12)
    assert np.isnan(g.desk_worst_bp_weight)
    assert np.isnan(g.desk_worst_usd)


def test_a_clean_report_passes_both_and_reports_no_breach():
    """Including the exactly-binding case: slack of zero is compliance, and
    it must not come back as a signed-zero breach."""
    rep = _report(
        ("BBB band", UNIT_WEIGHT, 0.0, 0.1, "rating"),
        ("total duration", UNIT_YEARS, 0.05, 0.1, "duration"),
        ("TE sigma (bp ann.)", UNIT_RETURN_BP, np.inf, 10.0,
         "tracking_error"),
    )
    g = gate_verdicts(rep)

    assert (rep["breach"] == 0.0).all()
    assert not np.signbit(rep["breach"]).any(), "breach came back as -0.0"
    assert not rep["breaches_purity_gate"].any()
    assert not rep["breaches_desk_gate"].any()
    assert g.purity_pass and g.desk_pass and g.verdict == VERDICT_OK
    assert worst_desk_violation(rep) == (0.0, "-")
    # An unconstrained row is not a pass the portfolio earned: infinite slack
    # stays infinite and cannot dominate the ranking.
    assert np.isinf(rep["rel_slack"].iloc[2])


def test_an_unmapped_unit_fails_loudly():
    """A new limit family with no desk tolerance must not silently pass the
    desk gate forever. NaN would have done exactly that."""
    rep = pd.DataFrame([{
        "family": "new", "item": "something", "portfolio": np.nan,
        "benchmark": np.nan, "limit": "-", "slack": -1.0, "scale": 1.0,
        "unit": "furlongs_per_fortnight", "rel_slack": -1.0,
        "binding": True}])
    with pytest.raises(ValueError, match="furlongs_per_fortnight"):
        score_breaches(rep)


def test_exposure_report_carries_the_columns_on_a_real_portfolio():
    """End to end on a real universe: the benchmark breaches the turnover
    budget by its own distance from the incumbent, and that breach now has a
    dollar value a PM can read."""
    uni = make_universe(n_bonds=60, seed=7)
    limits = default_limits(uni)
    rep = exposure_report(uni, uni.w_benchmark, limits)

    for col in ("unit", "breach", "breach_bp_weight", "breach_usd",
                "usd_basis", "desk_tol", "breach_desk_multiple",
                "breaches_purity_gate", "breaches_desk_gate"):
        assert col in rep.columns

    to = rep.set_index("item").loc["one-way turnover"]
    excess = uni.one_way_distance_to_benchmark() - limits.turnover
    assert to["breach"] == pytest.approx(excess, rel=1e-12)
    assert to["breach_usd"] == pytest.approx(excess * NOTIONAL_USD,
                                             rel=1e-12)

    g = gate_verdicts(rep)
    assert g.verdict == VERDICT_BREACH
    assert g.desk_worst_limit == "one-way turnover"
    # `worst_violation` is untouched: same 2-tuple, same meaning, and it
    # agrees with the purity half of the verdict.
    viol, which = worst_violation(rep)
    assert (viol, which) == (g.purity_worst_rel, g.purity_worst_limit)

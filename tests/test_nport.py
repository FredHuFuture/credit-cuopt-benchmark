"""The N-PORT universe builder, against a filing set written by hand.

`src/nport.py` turns a 463 MB quarterly SEC download into a `BondUniverse`.
Almost everything that can go wrong in it goes wrong SILENTLY -- a filter that
excludes the wrong category still returns a plausible-looking universe, a
deduplication that averages the wrong thing still returns one row per bond,
and a coupon read on the wrong scale still prices. So the tests here are built
the only way that catches those: a miniature N-PORT ZIP whose every row was
chosen to have a known answer, run through the real `build_universe` end to
end.

The fixture is a real ZIP on disk with the real member names and the real
column sets, so nothing is stubbed and no code path is skipped. Its bonds are
par bonds against a flat-ish hand-made Treasury curve, which makes the yields
and spreads exact integers rather than approximations: ACME11AA1 is a 5-year
5% bond priced at 100 against a 4% five-year Treasury, so its spread is
100.0000 bp and any arithmetic error shows up in the first decimal.

CPU-only, no network, no solver except the one feasibility LP, ~2 seconds.

    pytest tests/test_nport.py -q
"""

from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.credit import default_limits  # noqa: E402
from src.nport import (  # noqa: E402
    _DEBT_COLS, _HOLDING_COLS, FREQ, NportConfig, build_universe,
    classify_sector_by_issuer, classify_sector_keyword, frame_sha256,
    grid_maturity, quarter_url, recentre_levels, repair_rate_scale,
    solve_ytm, spread_implied_ratings, valid_cusip,
)
from src.universe import RATINGS, price_and_duration  # noqa: E402

REPORT_DATE = "31-DEC-2025"

#: Test config: three curve knots instead of eight, one Treasury per knot
#: instead of eight, because the fixture is hand-written. Everything else is
#: the production default, so the filters under test are the real ones.
TEST_CFG = NportConfig(
    treasury_knots_y=(1.0, 5.0, 10.0),
    treasury_min_per_knot=1,
)


# --------------------------------------------------------------------------
# Fixture construction
# --------------------------------------------------------------------------

class _Filing:
    """Accumulates holding + debt rows and writes them as an N-PORT ZIP."""

    def __init__(self) -> None:
        self.holdings: list[dict] = []
        self.debt: list[dict] = []
        self._hid = 100_000

    def add(self, *, cusip: str, name: str, price: float, balance: float,
            coupon: str, maturity: str, n_funds: int = 6,
            asset_cat: str = "DBT", issuer_type: str = "CORP",
            currency: str = "USD", payoff: str = "Long", unit: str = "PA",
            coupon_type: str = "Fixed", default: str = "N",
            arrears: str = "N", pik: str = "N", conv: str = "",
            country: str = "US", fund_offset: int = 0) -> None:
        """One bond, held identically by `n_funds` distinct funds."""
        for k in range(n_funds):
            self._hid += 1
            self.holdings.append({
                "ACCESSION_NUMBER": f"0001-26-{k + fund_offset:06d}",
                "HOLDING_ID": str(self._hid),
                "ISSUER_NAME": name,
                "ISSUER_CUSIP": cusip,
                "BALANCE": f"{balance:.6f}",
                "UNIT": unit,
                "CURRENCY_CODE": currency,
                "CURRENCY_VALUE": f"{balance * price / 100.0:.6f}",
                "PAYOFF_PROFILE": payoff,
                "ASSET_CAT": asset_cat,
                "ISSUER_TYPE": issuer_type,
                "INVESTMENT_COUNTRY": country,
            })
            self.debt.append({
                "HOLDING_ID": str(self._hid),
                "MATURITY_DATE": maturity,
                "COUPON_TYPE": coupon_type,
                "ANNUALIZED_RATE": coupon,
                "IS_DEFAULT": default,
                "ARE_ANY_INTEREST_PAYMENT": arrears,
                "IS_ANY_PORTION_INTEREST_PAID": pik,
                "IS_CONVTIBLE_MANDATORY": "",
                "IS_CONVTIBLE_CONTINGENT": conv,
            })

    def add_row(self, **over) -> None:
        """One extra holding row on an existing bond (an odd mark, say)."""
        self.add(n_funds=1, **over)

    def write(self, path: Path) -> Path:
        def tsv(rows: list[dict], cols) -> bytes:
            buf = io.StringIO()
            pd.DataFrame(rows, columns=list(cols)).to_csv(
                buf, sep="\t", index=False)
            return buf.getvalue().encode()

        sub = pd.DataFrame({
            "ACCESSION_NUMBER": sorted(
                {h["ACCESSION_NUMBER"] for h in self.holdings}),
        })
        sub["REPORT_DATE"] = REPORT_DATE
        sub["FILING_DATE"] = "15-FEB-2026"
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("SUBMISSION.tsv", tsv(
                sub.to_dict("records"),
                ["ACCESSION_NUMBER", "FILING_DATE", "REPORT_DATE"]))
            z.writestr("FUND_REPORTED_HOLDING.tsv",
                       tsv(self.holdings, _HOLDING_COLS))
            z.writestr("DEBT_SECURITY.tsv", tsv(self.debt, _DEBT_COLS))
        return path


def _make_filing() -> _Filing:
    """The fixture. Every row is here to prove one specific thing."""
    f = _Filing()

    # --- Treasuries: par bonds, so the fitted curve is exactly 3/4/4.5% ----
    f.add(cusip="912828AA1", name="UNITED STATES TREASURY", price=100.0,
          balance=1e6, coupon="3.0", maturity="31-DEC-2026",
          issuer_type="UST", n_funds=3)
    f.add(cusip="912828BB2", name="UNITED STATES TREASURY", price=100.0,
          balance=1e6, coupon="4.0", maturity="31-DEC-2030",
          issuer_type="UST", n_funds=3)
    f.add(cusip="912828CC3", name="UNITED STATES TREASURY", price=100.0,
          balance=1e6, coupon="4.5", maturity="31-DEC-2035",
          issuer_type="UST", n_funds=3)

    # --- the arithmetic bonds ---------------------------------------------
    # 5y par at 5%: ytm 5.00%, curve 4.00% -> spread exactly 100bp.
    f.add(cusip="ACME11AA1", name="ACME BANK CORP", price=100.0,
          balance=1e6, coupon="5.0", maturity="31-DEC-2030")
    # Same CUSIP6 prefix as the bond above -> must inherit its rating.
    f.add(cusip="ACME11AB2", name="ACME BANK CORP", price=100.0,
          balance=1e6, coupon="6.0", maturity="31-DEC-2035")
    # 2y par at 4%: curve interpolates to 3.25% -> spread exactly 75bp.
    # Half the market value of ACME11AA1, so w_benchmark must be exactly 2:1.
    f.add(cusip="BETA22AA1", name="BETA ENERGY PIPELINE LP", price=100.0,
          balance=5e5, coupon="4.0", maturity="31-DEC-2027")

    # --- consolidation: one filer's mark is 30 points off the others ------
    f.add(cusip="GAMA33AA1", name="GAMMA HEALTH SYSTEMS INC", price=100.0,
          balance=1e6, coupon="5.0", maturity="31-DEC-2030", n_funds=5)
    f.add_row(cusip="GAMA33AA1", name="GAMMA HEALTH SYSTEMS INC", price=130.0,
              balance=1e6, coupon="5.0", maturity="31-DEC-2030",
              fund_offset=50)

    # --- coupon reported on the decimal scale by two of six filers --------
    f.add(cusip="DELT44AA1", name="DELTA SOFTWARE HOLDINGS", price=100.0,
          balance=1e6, coupon="5.5", maturity="31-DEC-2030", n_funds=4)
    f.add(cusip="DELT44AA1", name="DELTA SOFTWARE HOLDINGS", price=100.0,
          balance=1e6, coupon="0.055", maturity="31-DEC-2030", n_funds=2,
          fund_offset=60)

    # --- one filer reports principal in thousands: price 100,000 ----------
    f.add(cusip="EPSI55AA1", name="EPSILON UTILITIES ELECTRIC", price=100.0,
          balance=1e6, coupon="5.0", maturity="31-DEC-2030", n_funds=6)
    f.add_row(cusip="EPSI55AA1", name="EPSILON UTILITIES ELECTRIC",
              price=100_000.0, balance=1e3, coupon="5.0",
              maturity="31-DEC-2030", fund_offset=70)

    # --- rows that must NOT reach the universe, one per filter ------------
    f.add(cusip="EQTY00019", name="ZETA COMMON STOCK", price=100.0,
          balance=1e6, coupon="", maturity="31-DEC-2030", asset_cat="EC")
    f.add(cusip="MUNI00017", name="ETA CITY WATER AUTHORITY", price=100.0,
          balance=1e6, coupon="5.0", maturity="31-DEC-2030",
          issuer_type="MUN")
    f.add(cusip="EURO00015", name="THETA EURO INDUSTRIALS", price=100.0,
          balance=1e6, coupon="5.0", maturity="31-DEC-2030", currency="EUR")
    f.add(cusip="SHRT00013", name="IOTA SHORT POSITION CORP", price=100.0,
          balance=1e6, coupon="5.0", maturity="31-DEC-2030", payoff="Short")
    f.add(cusip="SHRS00011", name="KAPPA SHARE UNIT CORP", price=100.0,
          balance=1e6, coupon="5.0", maturity="31-DEC-2030", unit="NS")
    f.add(cusip="FLOT00018", name="LAMBDA FLOATING RATE CORP", price=100.0,
          balance=1e6, coupon="5.0", maturity="31-DEC-2030",
          coupon_type="Floating")
    f.add(cusip="000000000", name="MU PLACEHOLDER CUSIP CORP", price=100.0,
          balance=1e6, coupon="5.0", maturity="31-DEC-2030")
    f.add(cusip="DFLT00016", name="NU DEFAULTED CORP", price=40.0,
          balance=1e6, coupon="5.0", maturity="31-DEC-2030", default="Y")
    f.add(cusip="ARRS00014", name="XI ARREARS CORP", price=70.0,
          balance=1e6, coupon="5.0", maturity="31-DEC-2030", arrears="Y")
    f.add(cusip="PIKK00012", name="OMICRON PIK CORP", price=90.0,
          balance=1e6, coupon="5.0", maturity="31-DEC-2030", pik="Y")
    f.add(cusip="CONV00010", name="PI CONVERTIBLE CORP", price=100.0,
          balance=1e6, coupon="5.0", maturity="31-DEC-2030", conv="Y")
    f.add(cusip="THIN00018", name="RHO THINLY HELD CORP", price=100.0,
          balance=1e6, coupon="5.0", maturity="31-DEC-2030", n_funds=2)
    f.add(cusip="SHMT00016", name="SIGMA SHORT MATURITY CORP", price=100.0,
          balance=1e6, coupon="5.0", maturity="30-JUN-2026")
    f.add(cusip="LGMT00014", name="TAU CENTURY BOND CORP", price=100.0,
          balance=1e6, coupon="5.0", maturity="31-DEC-2075")
    f.add(cusip="ZERO00012", name="UPSILON ZERO COUPON CORP", price=80.0,
          balance=1e6, coupon="0", maturity="31-DEC-2030")
    return f


#: The bonds the fixture is designed to admit, and nothing else.
EXPECTED_BONDS = {"ACME11AA1", "ACME11AB2", "BETA22AA1", "GAMA33AA1",
                  "DELT44AA1", "EPSI55AA1"}


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    path = tmp_path_factory.mktemp("nport") / "2026q1_nport.zip"
    _make_filing().write(path)
    return build_universe(str(path), "2026q1", downloaded_utc="2026-08-23",
                          cfg=TEST_CFG, seed=7), path


# --------------------------------------------------------------------------
# Filtering
# --------------------------------------------------------------------------

def test_only_the_intended_bonds_survive(built) -> None:
    """Every exclusion in the module docstring, exercised by one row each.

    The strong form of the assertion -- set equality, not a count -- because a
    filter that drops the right NUMBER of rows for the wrong reason is exactly
    the failure this fixture exists to catch.
    """
    nu, _ = built
    assert set(nu.universe.bonds["bond_id"]) == EXPECTED_BONDS


def test_filter_ledger_accounts_for_each_exclusion(built) -> None:
    """The provenance ledger is monotone and lands on the final count."""
    nu, _ = built
    stages = nu.provenance.stage_counts
    ordered = [stages[k] for k in sorted(stages)]
    assert ordered == sorted(ordered, reverse=True), stages
    assert stages["20_final"] == len(EXPECTED_BONDS)
    # The categorical filters bite in the order the docstring claims.
    assert stages["01_asset_cat_debt"] < stages["00_all_holdings"]
    assert stages["02_issuer_type"] < stages["01_asset_cat_debt"]
    assert stages["03_currency_usd"] < stages["02_issuer_type"]


def test_bond_level_screens_each_remove_their_row(built) -> None:
    nu, _ = built
    s = nu.provenance.stage_counts
    assert s["11_distinct_cusips"] - s["12_not_in_default"] == 1
    assert s["12_not_in_default"] - s["13_not_in_arrears"] == 1
    assert s["13_not_in_arrears"] - s["14_not_pik"] == 1
    assert s["14_not_pik"] - s["15_not_convertible"] == 1
    assert s["15_not_convertible"] - s["16_min_funds"] == 1
    assert s["16_min_funds"] - s["17_coupon_positive"] == 1
    # Two maturity exclusions: a 6-month bond and a 50-year bond.
    assert s["17_coupon_positive"] - s["18_maturity_window"] == 2


def test_valid_cusip_rejects_the_placeholders() -> None:
    s = pd.Series(["037833100", "000000000", "N/A", "", None, "12345678",
                   "abcdefghi", "0378331000"])
    assert list(valid_cusip(s)) == [True, False, False, False, False, False,
                                    True, False]


# --------------------------------------------------------------------------
# Consolidation across funds
# --------------------------------------------------------------------------

def test_outlier_mark_is_dropped_not_averaged(built) -> None:
    """GAMA33AA1: five filers at 100, one at 130.

    A mean would give 105 and a bond that does not exist. The median rule
    keeps 100 and the deviation screen removes the row entirely, so the
    aggregate market value must lose it too -- that second half is the part a
    price-only fix would miss, and it is what `w_benchmark` is built from.
    """
    nu, _ = built
    row = nu.universe.bonds.set_index("bond_id").loc["GAMA33AA1"]
    assert row["price"] == pytest.approx(100.0)
    assert row["n_funds_holding"] == 5
    assert row["market_value_usd"] == pytest.approx(5e6)
    assert nu.diagnostics.price_rows_dropped_dev == 1


def test_absurd_price_row_is_dropped_by_the_band(built) -> None:
    """EPSI55AA1: one filer reported principal in thousands, price 100,000."""
    nu, _ = built
    row = nu.universe.bonds.set_index("bond_id").loc["EPSI55AA1"]
    assert row["price"] == pytest.approx(100.0)
    assert row["n_funds_holding"] == 6
    assert nu.diagnostics.price_rows_dropped_band == 1


def test_coupon_decimal_scale_is_repaired(built) -> None:
    """DELT44AA1: two of six filers reported 0.055 instead of 5.5.

    Read naively the median coupon would be 2.7775% -- a bond that never
    existed, priced at a yield nothing supports. This is the single most
    dangerous field in the dataset because nothing else about the row looks
    wrong.
    """
    nu, _ = built
    row = nu.universe.bonds.set_index("bond_id").loc["DELT44AA1"]
    assert row["coupon"] == pytest.approx(0.055)
    assert nu.diagnostics.rate_scale_rows_repaired == 2
    assert nu.diagnostics.rate_scale_cusips_mixed == 1


def test_repair_rate_scale_leaves_percent_rows_alone() -> None:
    r = pd.Series([5.0, 0.05, 0.0, 12.5, 0.30, 0.29, np.nan])
    out, n = repair_rate_scale(r, 0.30)
    assert n == 2                       # 0.05 and 0.29
    assert list(out.dropna()) == pytest.approx(
        [5.0, 5.0, 0.0, 12.5, 0.30, 29.0])


# --------------------------------------------------------------------------
# Benchmark and incumbent weights
# --------------------------------------------------------------------------

def test_benchmark_weight_is_aggregate_market_value(built) -> None:
    """BETA22AA1 holds half the par of ACME11AA1 at the same price."""
    nu, _ = built
    uni = nu.universe
    w = pd.Series(uni.w_benchmark, index=uni.bonds["bond_id"])
    assert w.sum() == pytest.approx(1.0)
    assert w["ACME11AA1"] / w["BETA22AA1"] == pytest.approx(2.0)
    mv = uni.bonds["market_value_usd"].to_numpy()
    assert np.allclose(uni.w_benchmark, mv / mv.sum())


def test_incumbent_is_a_normalised_subset_near_the_benchmark(built) -> None:
    nu, _ = built
    uni = nu.universe
    assert uni.w0.sum() == pytest.approx(1.0)
    assert (uni.w0 >= 0).all()
    assert 0.0 < uni.one_way_distance_to_benchmark() < 0.5


# --------------------------------------------------------------------------
# Derived bond mathematics
# --------------------------------------------------------------------------

def test_treasury_curve_recovers_the_par_coupons(built) -> None:
    """Three par Treasuries at 3 / 4 / 4.5% must fit to exactly that."""
    nu, _ = built
    assert nu.treasury_knots == (1.0, 5.0, 10.0)
    assert np.allclose(nu.treasury_yields, [0.03, 0.04, 0.045], atol=1e-9)


def test_spreads_are_exact_against_the_hand_made_curve(built) -> None:
    """The whole derivation chain, checked on numbers chosen to be integers.

    ACME11AA1: 5y par 5% against a 4% five-year point -> 100.0000 bp.
    ACME11AB2: 10y par 6% against 4.5%                -> 150.0000 bp.
    BETA22AA1: 2y par 4% against 3.25% interpolated   ->  75.0000 bp.
    """
    nu, _ = built
    b = nu.universe.bonds.set_index("bond_id")
    assert b.loc["ACME11AA1", "ytm"] == pytest.approx(0.05, abs=1e-12)
    assert b.loc["ACME11AA1", "oas_bp"] == pytest.approx(100.0, abs=1e-6)
    assert b.loc["ACME11AB2", "oas_bp"] == pytest.approx(150.0, abs=1e-6)
    assert b.loc["BETA22AA1", "oas_bp"] == pytest.approx(75.0, abs=1e-6)


def test_solve_ytm_inverts_the_repo_pricer(built) -> None:
    """Round-trip: repricing at the solved yield must return the input price.

    The point of this one is that `solve_ytm` never gets its own pricing
    formula -- it bisects the function the optimizer's duration and convexity
    come from, so a disagreement here would be a disagreement inside a single
    implementation, which cannot happen. It is checked anyway because the
    bracket and the monotonicity direction can still be wrong.
    """
    nu, _ = built
    b = nu.universe.bonds
    price, _, _ = price_and_duration(
        b["coupon"].to_numpy(), b["ytm"].to_numpy(),
        b["maturity"].to_numpy(), freq=FREQ)
    assert np.abs(price - b["price"].to_numpy()).max() < 1e-9
    assert nu.diagnostics.ytm_solve_max_price_error < 1e-9


def test_solve_ytm_on_premium_and_discount_bonds() -> None:
    coupon = np.array([0.02, 0.05, 0.08, 0.0])
    maturity = np.array([2.0, 5.0, 10.0, 3.0])
    truth = np.array([0.045, 0.05, 0.035, 0.06])
    price, _, _ = price_and_duration(coupon, truth, maturity, freq=FREQ)
    assert solve_ytm(price, coupon, maturity) == pytest.approx(truth,
                                                               abs=1e-10)


def test_duration_and_convexity_come_from_the_shared_pricer(built) -> None:
    nu, _ = built
    b = nu.universe.bonds
    _, dur, cvx = price_and_duration(
        b["coupon"].to_numpy(), b["ytm"].to_numpy(),
        b["maturity"].to_numpy(), freq=FREQ)
    assert np.allclose(b["mod_duration"], dur)
    assert np.allclose(b["convexity"], cvx)
    # Fixed-rate bullets: the repo's stated scope assumption.
    assert np.allclose(b["spread_duration"], b["mod_duration"])


def test_grid_maturity_rounds_to_the_coupon_grid() -> None:
    cfg = NportConfig()
    got = grid_maturity(np.array([1.1, 1.3, 4.99, 7.26, 29.9, 40.0, 0.2]),
                        cfg)
    assert list(got) == [1.0, 1.5, 5.0, 7.5, 30.0, 30.0, 1.0]
    assert np.abs(got[:5] - np.array([1.1, 1.3, 4.99, 7.26, 29.9])).max() \
        <= 1.0 / (2 * FREQ) + 1e-12


# --------------------------------------------------------------------------
# The two proxies
# --------------------------------------------------------------------------

def test_rating_is_constant_within_an_issuer(built) -> None:
    """`src/scenarios.py` defaults ISSUERS, and takes one bond's label to
    stand for the whole name. A per-bond rating would make that arbitrary."""
    nu, _ = built
    b = nu.universe.bonds
    assert b.groupby("issuer_id")["rating"].nunique().max() == 1
    # ACME issues two bonds at different spreads (100bp and 150bp); the
    # issuer-level median must still give them one label between them.
    acme = b[b["issuer_id"] == "ACME11"]
    assert len(acme) == 2 and acme["rating"].nunique() == 1
    assert b["issuer_id"].nunique() == 5


def test_every_rating_label_is_one_the_pipeline_knows(built) -> None:
    """A label outside `RATINGS` raises a KeyError deep inside the scenario
    engine's `pd_annual` lookup, which is a terrible place to find out."""
    nu, _ = built
    assert set(nu.universe.bonds["rating"]) <= set(RATINGS)


def test_spread_implied_ratings_are_monotone_in_spread() -> None:
    levels = {"AAA": 45.0, "AA": 60.0, "A": 90.0, "BBB": 140.0,
              "BB": 260.0, "B": 420.0}
    oas = np.array([40.0, 62.0, 95.0, 150.0, 250.0, 500.0])
    iss = np.array(["I1", "I2", "I3", "I4", "I5", "I6"])
    got, _ = spread_implied_ratings(oas, iss, levels)
    assert list(got) == ["AAA", "AA", "A", "BBB", "BB", "B"]


def test_spread_quantile_scheme_hits_the_target_shares() -> None:
    rng = np.random.default_rng(0)
    n = 2000
    oas = np.exp(rng.normal(np.log(120.0), 0.6, n))
    iss = np.array([f"I{i:05d}" for i in range(n)])
    shares = {"AAA": 0.01, "AA": 0.07, "A": 0.32, "BBB": 0.45,
              "BB": 0.10, "B": 0.05}
    got, _ = spread_implied_ratings(
        oas, iss, {r: 1.0 for r in shares}, scheme="spread_quantile",
        shares=shares)
    counts = pd.Series(got).value_counts(normalize=True)
    for r, target in shares.items():
        assert counts.get(r, 0.0) == pytest.approx(target, abs=0.005)


def test_recentring_moves_the_level_and_keeps_the_log_spacing() -> None:
    levels = {"AAA": 45.0, "AA": 60.0, "A": 90.0, "BBB": 140.0,
              "BB": 260.0, "B": 420.0}
    shares = {"AAA": 0.01, "AA": 0.07, "A": 0.32, "BBB": 0.45,
              "BB": 0.10, "B": 0.05}
    oas = np.full(100, 60.0)
    out, shift = recentre_levels(levels, shares, oas, np.ones(100))
    assert shift < 1.0                       # 60bp is tighter than long-run
    ratios = [out[r] / levels[r] for r in levels]
    assert np.allclose(ratios, shift)        # shape preserved exactly


def test_sector_classifier_reports_what_it_could_not_decide() -> None:
    names = ["ACME BANK CORP", "BETA ENERGY PIPELINE LP",
             "GAMMA HEALTH SYSTEMS INC", "EPSILON UTILITIES ELECTRIC",
             "PANTHER ESCROW ISSUER", "IMOLA MERGER CORP"]
    labels, decided = classify_sector_keyword(names)
    assert labels[:4] == ["Financials", "Energy", "Healthcare", "Utilities"]
    assert list(decided) == [True, True, True, True, False, False]


def test_issuer_classifier_falls_back_to_another_spelling() -> None:
    """An opaque name on the big bond, an informative one on a small bond."""
    iss = np.array(["A1", "A1", "B1"])
    names = ["PANTHER ESCROW ISSUER", "DELTA AIR LINES INC",
             "IMOLA MERGER CORP"]
    weight = np.array([100.0, 1.0, 5.0])
    labels, decided = classify_sector_by_issuer(iss, names, weight)
    assert labels[0] == labels[1] == "Industrials"    # airline, decided
    assert decided[0] and decided[1]
    assert not decided[2]


def test_sector_is_constant_within_an_issuer(built) -> None:
    nu, _ = built
    b = nu.universe.bonds
    assert b.groupby("issuer_id")["sector"].nunique().max() == 1


# --------------------------------------------------------------------------
# Provenance and determinism
# --------------------------------------------------------------------------

def test_same_inputs_give_a_byte_identical_universe(built, tmp_path) -> None:
    """The determinism guarantee, checked by rebuilding from the same ZIP."""
    nu, path = built
    again = build_universe(str(path), "2026q1",
                           downloaded_utc="2026-08-23", cfg=TEST_CFG, seed=7)
    assert again.provenance.bonds_sha256 == nu.provenance.bonds_sha256
    assert np.allclose(again.universe.w_benchmark, nu.universe.w_benchmark)
    assert np.allclose(again.universe.w0, nu.universe.w0)


def test_a_different_seed_changes_only_the_incumbent(built, tmp_path) -> None:
    """`w0` is the one constructed quantity; nothing else may move with it."""
    nu, path = built
    other = build_universe(str(path), "2026q1",
                           downloaded_utc="2026-08-23", cfg=TEST_CFG, seed=11)
    assert other.provenance.bonds_sha256 == nu.provenance.bonds_sha256
    assert np.allclose(other.universe.w_benchmark, nu.universe.w_benchmark)
    assert not np.allclose(other.universe.w0, nu.universe.w0)


def test_provenance_records_the_source_and_serialises(built) -> None:
    nu, path = built
    p = nu.provenance
    assert p.quarter == "2026q1"
    assert p.source_url == quarter_url("2026q1")
    assert "publicdomain" in p.licence
    assert len(p.zip_sha256) == 64
    assert set(p.member_sha256) == {"FUND_REPORTED_HOLDING.tsv",
                                    "DEBT_SECURITY.tsv", "SUBMISSION.tsv"}
    assert p.report_date == REPORT_DATE
    assert p.n_bonds == len(EXPECTED_BONDS)
    round_trip = json.loads(p.to_json())
    assert round_trip["bonds_sha256"] == p.bonds_sha256
    assert round_trip["rating_anchor"]["recentred"] is True


def test_frame_hash_notices_a_one_ulp_change() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})
    other = df.copy()
    other.loc[0, "a"] = np.nextafter(1.0, 2.0)
    assert frame_sha256(df) == frame_sha256(df.copy())
    assert frame_sha256(df) != frame_sha256(other)


def test_quarter_url_rejects_nonsense() -> None:
    assert quarter_url("2026Q1").endswith("2026q1_nport.zip")
    for bad in ("2026", "2026q5", "26q1", "next quarter", ""):
        with pytest.raises(ValueError):
            quarter_url(bad)


def test_config_validation_rejects_unknown_proxies() -> None:
    with pytest.raises(ValueError):
        NportConfig(rating_scheme="agency").validate()
    with pytest.raises(ValueError):
        NportConfig(sector_scheme="gics").validate()


# --------------------------------------------------------------------------
# The universe is usable by the rest of the repo
# --------------------------------------------------------------------------

def test_every_column_the_pipeline_reads_is_present(built) -> None:
    """The contract with `src/universe.py`'s accessors and `src/credit.py`."""
    nu, _ = built
    uni = nu.universe
    for col in ("bond_id", "issuer_id", "sector", "rating", "maturity",
                "coupon", "ytm", "oas_bp", "price", "mod_duration",
                "spread_duration", "convexity", "issue_size_mm",
                "maturity_cell_code", "maturity_cell"):
        assert col in uni.bonds.columns, col
    assert uni.bonds.notna().all().all()
    for arr in (uni.duration, uni.spread_duration, uni.oas_bp, uni.dts,
                uni.convexity, uni.ytm, uni.oas):
        assert np.isfinite(arr).all()
    G, labels = uni.group_matrix("rating")
    assert G.shape == (len(labels), uni.n_bonds)
    D, cells = uni.duration_contribution_matrix()
    assert D.shape == (len(cells), uni.n_bonds)
    # Cell duration contributions must add up to total portfolio duration.
    assert (D @ uni.w_benchmark).sum() == pytest.approx(
        float(uni.w_benchmark @ uni.duration))


def test_default_limits_yields_a_feasible_mandate(built) -> None:
    """The mandate must admit at least one portfolio on this universe.

    Checked by a feasibility LP over the exact constraint set
    `credit_constraint_builder` hands the optimizer, not inferred from some
    other solve succeeding. If this fails, every benchmark run on the universe
    is measuring an empty polytope.
    """
    import cvxpy as cp

    from src.credit import credit_constraint_builder

    nu, _ = built
    uni = nu.universe
    limits = default_limits(uni)
    w = cp.Variable(uni.n_bonds, nonneg=True)
    cons = [cp.sum(w) == 1.0, w <= limits.w_max]
    extra, names = credit_constraint_builder(uni, limits)(w)
    prob = cp.Problem(cp.Minimize(0), cons + extra)
    prob.solve(solver=cp.HIGHS, verbose=False)
    assert prob.status == "optimal", (prob.status, names)
    assert w.value is not None
    assert np.asarray(w.value).sum() == pytest.approx(1.0, abs=1e-7)

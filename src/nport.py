"""A real corporate-bond universe, built from SEC Form N-PORT.

`src/universe.py` generates a synthetic universe. Everything downstream of it
-- the scenario engine, the credit mandate, the M6 tracking-error result -- is
real fixed-income machinery running on invented instruments, and the paper has
to say so. This module removes that sentence by rebuilding the same
`BondUniverse` object out of filings.

**The source.** SEC Form N-PORT Data Sets: quarterly structured extracts of
every registered fund's monthly portfolio holdings, published by the SEC's
Division of Economic and Risk Analysis. The data.gov catalog record for the
series gives the licence as http://www.usa.gov/publicdomain/label/1.0/ -- a US
Government Work, public domain, no restrictions (verified 2026-08-23). That
is the whole reason this module exists and every other security-level
corporate-bond source in the project's survey does not: TRACE, Mergent FISD,
ICE and Bloomberg are licence-blocked, and the open bond-pricing panels carry
no licence grant at all.

**What N-PORT gives you, and what it does not.** Per holding: issuer name and
LEI, CUSIP, ISIN, principal amount, USD fair value, percent of the filing
fund's net assets, asset and issuer category, domicile country, and -- via
DEBT_SECURITY -- coupon type, coupon rate, maturity date, a default flag, an
arrears flag and convertible flags. It does NOT give: credit ratings (those
are the agencies' property), any sector or industry classification, amount
outstanding, call schedules, or a Treasury curve.

So this module derives what is derivable and *labels* what is not:

===================  ========================================================
quantity             where it comes from
===================  ========================================================
price                100 x fair value / principal, median across filers. REAL
                     (derived from two reported fields).
coupon, maturity     REAL (DEBT_SECURITY), consolidated across filers.
yield to maturity    REAL, derived: the yield that reprices the bond under
                     `src.universe.price_and_duration`. No second pricer.
Treasury curve       REAL: fitted from the US Treasury holdings in the SAME
                     filing quarter, so the spread is a spread over a curve
                     the filings themselves imply.
OAS                  REAL, derived: ytm - interpolated Treasury yield. It is
                     a Z-spread-like nominal spread, NOT option-adjusted;
                     N-PORT carries no call schedule. See CAVEATS.
duration, convexity  REAL, derived from the same cash-flow schedule the repo
                     already uses.
issuer identity      REAL: the 6-character CUSIP issuer prefix.
benchmark weight     REAL aggregate, PROXY semantics: aggregate USD fair value
                     held across all filing funds. See `w_benchmark` note.
rating               **PROXY** -- spread-implied. N-PORT has no ratings.
sector               **PROXY** -- keyword classification of the issuer name,
                     or issuer domicile country. N-PORT has no sector.
incumbent w0         **CONSTRUCTED** -- seeded rule; no real incumbent exists.
===================  ========================================================

**CAVEATS, stated here rather than discovered later.**

1. *The bullet assumption is unverifiable.* The repo's bond math assumes
   fixed-rate bullets (spread duration == modified duration). N-PORT reports
   coupon type but no call schedule, and most USD corporates are callable.
   The derived yield is therefore a yield to MATURITY, not to worst; for a
   bond trading above par with a live call it overstates both yield and
   spread. `NportDiagnostics.share_above_par` reports the exposure.
2. *"OAS" is a misnomer kept for column compatibility.* Nothing here is
   option-adjusted. The column is named `oas_bp` because the whole downstream
   pipeline reads that name.
3. *Fair value may or may not include accrued interest.* Funds are not
   required to say. The implied-price distribution (median 100.2 on 2026q1)
   is consistent with clean pricing but does not prove it; a dirty mark would
   bias prices up and yields down by roughly a quarter of one coupon.
4. *Filers disagree about the coupon's units.* About 5% of holding rows
   report ANNUALIZED_RATE as a decimal fraction (0.045) rather than a percent
   (4.5). On 2026q1, 8,692 CUSIPs carry both conventions and the max/min ratio
   is exactly 100.0 at every quantile from the 1st to the 99th. `_repair_rate_
   scale` fixes this and counts what it touched. This is not a subtle
   modelling choice; read naively the field silently produces a universe of
   0.05%-coupon bonds.
5. *The benchmark is an ownership weight, not an index weight.* See below.
6. *Rating and sector proxies are correlated with quantities the optimizer
   trades.* See `spread_implied_ratings`.

**On `w_benchmark`.** N-PORT does not report amount outstanding, so there is
no way to build an issue-size-weighted index from it. What it does report is
every filing fund's position, so summing USD fair value across funds gives the
aggregate holding of the US registered-fund industry in each bond. That is a
real, fully-sourced aggregate -- and it is a *proxy* for index weight with a
known direction of error: it over-weights bonds that are index-eligible and
widely held, and under-weights bonds held mainly outside registered funds
(insurance general accounts, foreign official holders, buy-and-hold retail).
For a paper whose benchmark is meant to represent what a US credit index fund
tracks, that bias points the right way, but it is a bias and it is named.

Determinism: same quarter ZIP plus same `NportConfig` plus same seed gives a
byte-identical universe. `NportProvenance` records the source URL, the file
sha256, the download date, the per-stage row counts and a hash of the final
frame so a universe can be traced back to the filing data it came from.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import os
import re
import zipfile
from dataclasses import dataclass
from typing import Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

from .universe import (
    MATURITY_CELL_EDGES, MATURITY_CELL_NAMES, RATINGS, SECTORS,
    BondUniverse, CalibrationConfig, price_and_duration,
)

#: Where the quarterly ZIPs live. `{q}` is e.g. "2026q1".
NPORT_URL = ("https://www.sec.gov/files/dera/data/form-n-port-data-sets/"
             "{q}_nport.zip")

#: Landing page, for a citation that is not a raw file URL.
NPORT_LANDING = ("https://www.sec.gov/data-research/sec-markets-data/"
                 "form-n-port-data-sets")

#: The licence, verified against the data.gov catalog record on 2026-08-23.
NPORT_LICENCE = "http://www.usa.gov/publicdomain/label/1.0/"

#: SEC requires a descriptive User-Agent on automated requests and rate-limits
#: to ~10 req/s. One request per quarter is nowhere near that, but the header
#: is not optional. The fetch script fills in a contact address.
USER_AGENT_TEMPLATE = "bond-cvar-cuopt academic research ({contact})"

QUARTER_RE = re.compile(r"^(19|20)\d{2}q[1-4]$")

#: Coupon frequency assumed for every bond. USD corporates pay semiannually;
#: `price_and_duration` requires maturities on the 1/FREQ grid.
FREQ = 2

# --------------------------------------------------------------------------
# N-PORT code values used as filters. Spelled out so the filter set is
# readable as a sentence rather than as a set of magic strings.
# --------------------------------------------------------------------------

#: ASSET_CAT. "DBT" is plain debt. Everything else is excluded BY THIS CHOICE:
#: ABS-MBS / ABS-CBDO / ABS-O / ABS-APCP (securitized, amortising, no bullet
#: cash-flow schedule), LON (bank loans, floating), SN (structured notes),
#: EC / EP (equity), DFE / DIR / DCR / DCO / DO / DE (derivatives), RA (repo),
#: STIV (short-term investment vehicles), COMM, RE, OTHER.
ASSET_CAT_DEBT = "DBT"

#: ISSUER_TYPE. "CORP" excludes MUN (municipal), UST (Treasury), USGSE / USGA
#: (agency), NUSS (non-US sovereign), RF (registered fund), PF (private fund)
#: and OTHER. Note that supranationals (IBRD, EIB) self-report as CORP by some
#: filers; they survive this filter and are corporate-like in cash-flow terms.
ISSUER_TYPE_CORP = "CORP"
ISSUER_TYPE_TREASURY = "UST"

#: UNIT. "PA" is principal amount -- required, because the price is derived as
#: value / principal. "NS" (number of shares) holdings of debt cannot be
#: priced this way and are dropped rather than guessed at.
UNIT_PRINCIPAL = "PA"

PAYOFF_LONG = "Long"
COUPON_FIXED = "fixed"

_HOLDING_COLS = (
    "ACCESSION_NUMBER", "HOLDING_ID", "ISSUER_NAME", "ISSUER_CUSIP",
    "BALANCE", "UNIT", "CURRENCY_CODE", "CURRENCY_VALUE", "PAYOFF_PROFILE",
    "ASSET_CAT", "ISSUER_TYPE", "INVESTMENT_COUNTRY",
)
_DEBT_COLS = (
    "HOLDING_ID", "MATURITY_DATE", "COUPON_TYPE", "ANNUALIZED_RATE",
    "IS_DEFAULT", "ARE_ANY_INTEREST_PAYMENT", "IS_ANY_PORTION_INTEREST_PAID",
    "IS_CONVTIBLE_MANDATORY", "IS_CONVTIBLE_CONTINGENT",
)

_CUSIP_RE = re.compile(r"^[0-9A-Z]{9}$")


# --------------------------------------------------------------------------
# Sector proxy
#
# N-PORT has no industry classification and GICS/BICS/TRBC are all
# proprietary, so there is no licence-clean mapping to buy either. What is
# available is the issuer's legal name, which for large debt issuers is
# unusually informative ("... BANK", "... ENERGY TRANSFER LP", "... PHARMA").
#
# This classifier is a PROXY with an UNQUANTIFIED error rate: it cannot be
# validated, because validating it would require the proprietary
# classification it exists to replace. What CAN be reported honestly is how
# much of the universe it actually decided rather than defaulted, and
# `NportDiagnostics.sector_keyword_share` reports exactly that. Read the
# residual bucket as "unclassified", not as "Industrials".
#
# Order matters: the first pattern that matches wins, and the list is ordered
# so that the more specific token beats the more generic one (a "FINANCE" or
# "CAPITAL" suffix on an industrial captive lender is deliberately read as
# Financials, which is how index providers treat captives too).
# --------------------------------------------------------------------------

SECTOR_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("Financials", r"\b(BANK|BANC|BANCORP|BANKING)\b|BANK(?=CORP|SHARES)"),
    ("Financials", r"\b(INSUR\w*|REINSUR\w*|ASSURANCE|LIFE|ANNUIT\w*)\b"),
    ("Financials", r"\b(CAPITAL|FINANC\w*|FUNDING|CREDIT|LEASING|LEASE|"
                   r"MORTGAGE|LENDING|LOAN|INVEST\w*|SECURITIES|ASSET MGMT|"
                   r"HOLDINGS? FIN\w*|TRUST|REIT|PROPERT\w*|REALTY|"
                   r"REAL ESTATE)\b"),
    ("Financials", r"\b(GOLDMAN|MORGAN STANLEY|CITIGROUP|JPMORGAN|"
                   r"BARCLAYS|HSBC|UBS|DEUTSCHE|SANTANDER|BNP|"
                   r"AMERICAN EXPRESS|SCHWAB|BLACKSTONE|APOLLO|KKR|ARES|"
                   r"ATHENE|AERCAP|VISA|MASTERCARD|PAYPAL)\b"),
    ("Utilities", r"\b(UTILIT\w*|ELECTRIC|POWER|GRID|TRANSMISSION|"
                  r"WATER WORKS|EDISON|LIGHTING)\b"),
    ("Energy", r"\b(ENERGY|OIL|PETROL\w*|PETROBRAS|GAS|LNG|PIPELINE|"
               r"MIDSTREAM|REFIN\w*|DRILL\w*|EXPLORATION|OFFSHORE|"
               r"RESOURCES|COAL)\b"),
    # HEALTH without a leading word boundary: "ATHENAHEALTH", "UNITEDHEALTH"
    # and "WELLCAREHEALTH" are common issuer spellings and a boundary loses
    # all of them. The false-positive risk (a food brand called "...HEALTHY")
    # is real and much smaller.
    ("Healthcare", r"(HEALTH\w*)|\b(PHARMA\w*|BIO\w*|MEDIC\w*|HOSPITAL|"
                   r"THERAPEUT\w*|LABORATOR\w*|DENTAL|CLINIC\w*|SURG\w*|"
                   r"LILLY|MERCK|PFIZER|AMGEN|ABBVIE|CIGNA|ELEVANCE|"
                   r"CENTENE|MCKESSON|CVS)\b"),
    ("Technology", r"\b(TECHNOLOG\w*|SOFTWARE|SEMICONDUCTOR|MICRO\w*|"
                   r"DATA|DIGITAL|CYBER\w*|CLOUD|COMPUT\w*|SYSTEMS|"
                   r"ELECTRONICS?|ORACLE|BROADCOM|INTEL|NVIDIA|APPLE|"
                   r"MICROSOFT|DELL|HEWLETT|SALESFORCE|WORKDAY|ADOBE|"
                   r"ALPHABET|AMAZON|META PLATFORMS)\b"),
    ("Communications", r"\b(COMMUNICAT\w*|TELECOM\w*|WIRELESS|MOBILE|"
                       r"MEDIA|BROADCAST\w*|CABLE|NETWORKS?|"
                       r"ENTERTAINMENT|STUDIOS|PUBLISH\w*|"
                       r"T-MOBILE|VERIZON|COMCAST|CHARTER|WARNER|"
                       r"PARAMOUNT|NETFLIX|DISH|ECHOSTAR|AT&T)\b"),
    ("Consumer", r"\b(FOODS?|BEVERAGE\w*|BREW\w*|RESTAURANT\w*|RETAIL\w*|"
                 r"STORES?|MARKETS?|APPAREL|BRANDS?|TOBACCO|"
                 r"HOTEL\w*|RESORT\w*|CRUISE\w*|CASINO\w*|GAMING|"
                 r"AUTOMOTIVE|MOTORS?|AIRLINES?|TRAVEL|LEISURE|"
                 r"CONSUMER|HOME\w*|GROCER\w*|"
                 r"PHILIP MORRIS|ALTRIA|COCA-COLA|PEPSI|NIKE|"
                 r"MCDONALD|STARBUCKS|CARNIVAL|MARRIOTT|HILTON|"
                 r"LOWE'?S|TARGET|WALMART|COSTCO|KROGER|FORD|"
                 r"GENERAL MOTORS|TESLA|HYUNDAI|TOYOTA)\b"),
    ("Industrials", r"\b(INDUSTR\w*|AEROSP\w*|DEFENSE|AIRCRAFT|AVIATION|"
                    r"AIR ?LINES?|AIRWAYS|BOEING|"
                    r"MACHINER\w*|EQUIPMENT|MANUFACTUR\w*|STEEL|METALS?|"
                    r"MINING|CHEMICAL\w*|MATERIALS?|CEMENT|PAPER|"
                    r"PACKAGING|CONSTRUCT\w*|ENGINEER\w*|LOGISTICS|"
                    r"FREIGHT|RAIL\w*|SHIPPING|TRANSPORT\w*|WASTE|"
                    r"ELEVATOR|BUILDING PRODUCTS|"
                    r"SERVICES?|HOLDINGS?|CORP\w*|INC|PLC|LP|LLC|CO)\b"),
)

#: The bucket a name lands in when no pattern matched. NOT a claim about the
#: issuer's industry -- the label has to be one of `SECTORS` because the
#: downstream scenario engine indexes sector factors off it, but the count of
#: rows carrying it is reported separately as "unclassified".
SECTOR_RESIDUAL = "Industrials"

#: The last alternative in the Industrials pattern above matches almost any
#: corporate suffix, so a name that reaches it has NOT been classified on
#: anything industry-bearing. Matching it is recorded as a residual hit.
_SECTOR_GENERIC = re.compile(
    r"\b(SERVICES?|HOLDINGS?|CORP\w*|INC|PLC|LP|LLC|CO)\b")

_SECTOR_COMPILED = tuple(
    (s, re.compile(p)) for s, p in SECTOR_KEYWORDS)

# A label outside the repo's sector vocabulary would give `src/scenarios.py` a
# sector factor no other module knows about, and the failure would surface as
# a strange correlation rather than as an error. Checked at import.
_unknown_sectors = sorted({s for s, _ in SECTOR_KEYWORDS} - set(SECTORS))
if _unknown_sectors or SECTOR_RESIDUAL not in SECTORS:
    raise ImportError(
        f"SECTOR_KEYWORDS emits labels outside src.universe.SECTORS: "
        f"{_unknown_sectors or [SECTOR_RESIDUAL]}")


def classify_sector_keyword(names: Sequence[str]) -> tuple[list[str],
                                                           np.ndarray]:
    """Map issuer names onto `SECTORS`; also return a "was it decided" mask.

    The mask is False when the name matched nothing industry-bearing and fell
    through to `SECTOR_RESIDUAL`. Callers must report the False share: it is
    the only honest error statement available for a classifier that cannot be
    validated against the proprietary taxonomies it substitutes for.
    """
    out: list[str] = []
    decided = np.zeros(len(names), dtype=bool)
    for i, raw in enumerate(names):
        label, hit = _classify_one((raw or "").upper())
        out.append(label)
        decided[i] = hit
    return out, decided


def _classify_one(name: str) -> tuple[str, bool]:
    """One upper-cased name -> (sector, decided on industry evidence?)."""
    for sector, pat in _SECTOR_COMPILED:
        m = pat.search(name)
        if m is None:
            continue
        # A hit on the catch-all corporate-suffix alternative is not an
        # industry signal; record it as undecided.
        generic = (sector == SECTOR_RESIDUAL
                   and _SECTOR_GENERIC.fullmatch(m.group(0)) is not None)
        return sector, not generic
    return SECTOR_RESIDUAL, False


def classify_sector_by_issuer(
    issuer_id: np.ndarray, names: Sequence[str], weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Classify each ISSUER once, trying every spelling its bonds carry.

    Filers use roughly 30,000 distinct name spellings for roughly 6,000
    issuers, and the spellings are not equally informative: the same borrower
    appears as "DELTA AIR LINES/SKYMILES" on one bond and "Delta Air Lines
    Inc" on another, and CUSIP truncation to 24 characters mangles others. So
    the largest bond's spelling is tried first and, if it decides nothing, the
    issuer's remaining spellings are tried in descending market-value order.
    First decisive spelling wins.

    This extracts more signal from the same field without hand-tuning the
    keyword list to a particular quarter's names, which is the failure mode a
    classifier like this invites. It also guarantees one sector per issuer,
    which matters because an issuer whose curve straddled two sectors would
    put correlated bonds in two different scenario factors.

    Returns per-bond (sector, decided) arrays aligned with the inputs.
    """
    df = pd.DataFrame({"iss": np.asarray(issuer_id),
                       "name": [str(x) for x in names],
                       "w": np.asarray(weight, dtype=float)})
    # Sort by weight desc, then name, so ties resolve deterministically.
    df = df.sort_values(["iss", "w", "name"], ascending=[True, False, True])
    label_of: dict[str, str] = {}
    decided_of: dict[str, bool] = {}
    for iss, grp in df.groupby("iss", sort=True):
        label, hit = SECTOR_RESIDUAL, False
        seen: set[str] = set()
        for nm in grp["name"]:
            up = nm.upper()
            if up in seen:
                continue
            seen.add(up)
            cand, ok = _classify_one(up)
            if ok:
                label, hit = cand, True
                break
            if label == SECTOR_RESIDUAL:
                label = cand
        label_of[iss] = label
        decided_of[iss] = hit
    key = pd.Series(np.asarray(issuer_id))
    return (key.map(label_of).to_numpy(),
            key.map(decided_of).to_numpy().astype(bool))


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class NportConfig:
    """Every filter threshold and proxy choice, in one place.

    Defaults are the ones the committed run used. Each is a judgement call and
    each is here so it can be argued with rather than hunted for in the code.
    """

    # ---- row-level sanity -------------------------------------------------
    #: Implied price band, in points of par. A holding outside it is a
    #: reporting error (balance in thousands, value in local currency), not a
    #: bond. 99.67% of 2026q1 USD fixed-rate corporate rows fall inside.
    price_min: float = 20.0
    price_max: float = 150.0
    #: A filer's mark is dropped when it differs from the cross-filer median
    #: for the same CUSIP by more than this many points. Median cross-filer
    #: p90-p10 dispersion on 2026q1 is 0.55 points, so 5.0 is ~9x the typical
    #: spread of honest marks and removes 0.4% of rows.
    price_dev_max: float = 5.0
    #: A rate strictly between 0 and this is read as a decimal fraction and
    #: multiplied by 100. See caveat 4 in the module docstring. The threshold
    #: sits above any plausible decimal coupon (30%) and below the smallest
    #: coupon that would be misread (a genuine 0.30% coupon is left alone).
    rate_decimal_ceiling: float = 0.30

    # ---- bond-level eligibility ------------------------------------------
    #: Minimum number of distinct filing funds holding the bond. This is the
    #: only liquidity / index-eligibility screen available: N-PORT has no
    #: amount outstanding and no index membership flag. Five filers is enough
    #: for the median-of-marks consolidation to mean something.
    min_funds: int = 5
    #: Maturity window in years. The lower bound keeps every bond at least two
    #: coupon periods from maturity; the upper matches the synthetic
    #: universe's 30y cap and the last Treasury curve knot.
    maturity_min_y: float = 1.0
    maturity_max_y: float = 30.0
    #: Implied-yield plausibility band (decimal). The residual guard against
    #: a coupon or price the earlier screens did not catch: a 100x coupon
    #: error survives every price test but produces an absurd yield.
    ytm_min: float = 0.005
    ytm_max: float = 0.25
    #: Spread floor in bp. A handful of bonds price through the fitted
    #: Treasury curve; the scenario engine multiplies a RELATIVE spread shock
    #: by `oas`, so a negative spread would flip the sign of that bond's
    #: credit return. Clipped, and the count is reported.
    oas_floor_bp: float = 1.0
    oas_cap_bp: float = 3000.0

    # ---- Treasury curve ---------------------------------------------------
    #: Knot maturities (years) for the curve fitted from UST holdings.
    treasury_knots_y: tuple[float, ...] = (1.0, 2.0, 3.0, 5.0, 7.0, 10.0,
                                           20.0, 30.0)
    #: Half-width of the maturity window around each knot, as
    #: max(abs_y, rel * knot).
    treasury_window_abs_y: float = 0.75
    treasury_window_rel: float = 0.25
    #: Minimum distinct Treasury CUSIPs behind a knot before it is trusted.
    treasury_min_per_knot: int = 8

    # ---- proxies ----------------------------------------------------------
    #: "spread_implied" (nearest published OAS level, in log space) or
    #: "spread_quantile" (rank issuers by spread, cut at published index
    #: rating shares). Both are circular in the same way; see
    #: `spread_implied_ratings`.
    rating_scheme: str = "spread_implied"
    #: Re-anchor the published rating spread levels on THIS quarter's spread
    #: level before matching, preserving their log spacing. Without it the
    #: proxy compares a tight quarter against long-run average spreads and
    #: reads the whole universe a notch too high-grade: on 2026q1 the
    #: unanchored map puts 24% of benchmark weight in AAA/AA, against ~9% in
    #: the published index composition it is calibrated from. See
    #: `spread_implied_ratings`.
    rating_level_recentre: bool = True
    #: "keyword" (issuer-name classifier) or "country" (issuer domicile, a
    #: real N-PORT field but a country rather than an industry).
    sector_scheme: str = "keyword"

    # ---- universe assembly ------------------------------------------------
    #: Keep only the this-many largest bonds by aggregate fund-held market
    #: value; None keeps everything that passed the filters. The synthetic
    #: universe is 500 bonds and the benchmark runs are sized on that.
    n_bonds: int | None = None
    #: `w0` construction, identical in rule and in defaults to
    #: `src.universe.make_universe`, so the incumbent is not a second thing
    #: that changed between the synthetic and the real universe.
    portfolio_coverage: float = 0.65
    portfolio_tilt_sd: float = 0.20

    def validate(self) -> None:
        if self.rating_scheme not in ("spread_implied", "spread_quantile"):
            raise ValueError(f"unknown rating_scheme {self.rating_scheme!r}")
        if self.sector_scheme not in ("keyword", "country"):
            raise ValueError(f"unknown sector_scheme {self.sector_scheme!r}")
        if self.price_min <= 0 or self.price_max <= self.price_min:
            raise ValueError("price band must be positive and ordered")
        if self.maturity_min_y < 1.0 / FREQ:
            raise ValueError("maturity_min_y must be >= one coupon period")


@dataclass(frozen=True)
class NportProvenance:
    """Everything needed to trace a universe back to the filing data.

    Written next to the universe as JSON. The point is not bookkeeping for its
    own sake: a spread-implied rating and a keyword sector are only defensible
    if a reader can regenerate the exact frame they were computed on.
    """

    quarter: str
    source_url: str
    landing_page: str
    licence: str
    zip_sha256: str
    zip_bytes: int
    downloaded_utc: str
    member_sha256: Mapping[str, str]
    report_date: str
    n_filings: int
    stage_counts: Mapping[str, int]
    config: Mapping[str, object]
    seed: int
    treasury_curve: Mapping[str, object]
    rating_anchor: Mapping[str, object]
    bonds_sha256: str
    n_bonds: int
    n_issuers: int
    generated_utc: str

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True)


@dataclass(frozen=True)
class NportDiagnostics:
    """The honesty panel: what the proxies did, and what the data did not say.

    Every field here exists because reporting the universe without it would
    let a reader assume something the filings do not support.
    """

    share_above_par: float
    n_oas_clipped_low: int
    n_oas_clipped_high: int
    rate_scale_rows_repaired: int
    rate_scale_cusips_mixed: int
    price_rows_dropped_band: int
    price_rows_dropped_dev: int
    sector_keyword_share: float
    maturity_round_mae_y: float
    maturity_round_max_y: float
    cross_filer_price_p90_p10_median: float
    n_funds_median: float
    ytm_solve_max_price_error: float
    #: Median spread over every ELIGIBLE bond in the quarter versus over the
    #: bonds actually kept. Equal unless `NportConfig.n_bonds` trimmed the
    #: universe, and the gap is the selection bias that trim introduces: the
    #: most widely-held bonds are systematically WIDER than the market,
    #: because high-yield funds are numerous and crowd into a small number of
    #: large liquid issues. Quoting a trimmed universe as "the US corporate
    #: bond market" without this number would be wrong.
    oas_median_eligible_bp: float
    oas_median_kept_bp: float
    n_eligible_before_trim: int

    def as_row(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclass
class NportUniverse:
    """A `BondUniverse` plus the paperwork that makes it citable.

    Deliberately a wrapper rather than extra fields on `BondUniverse`: not one
    line of `src/universe.py`, `src/credit.py` or `src/scenarios.py` changes,
    so the real universe is a drop-in for the synthetic one and the M1-M6
    results stay comparable.
    """

    universe: BondUniverse
    provenance: NportProvenance
    diagnostics: NportDiagnostics
    #: The fitted Treasury curve, as (knots_y, yields_decimal, n_per_knot).
    treasury_knots: tuple[float, ...]
    treasury_yields: tuple[float, ...]
    treasury_counts: tuple[int, ...]


# --------------------------------------------------------------------------
# Source access
# --------------------------------------------------------------------------

def quarter_url(quarter: str) -> str:
    """URL of one quarterly ZIP. `quarter` looks like "2026q1"."""
    q = quarter.strip().lower()
    if not QUARTER_RE.match(q):
        raise ValueError(f"quarter must look like '2026q1', got {quarter!r}")
    return NPORT_URL.format(q=q)


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _member_sha256(zf: zipfile.ZipFile, names: Sequence[str]
                   ) -> dict[str, str]:
    """Hash the members actually read, not the whole archive.

    The archive hash pins the download; these pin the specific tables the
    universe was built from, which is what a reader would re-extract.
    """
    out: dict[str, str] = {}
    for name in names:
        h = hashlib.sha256()
        with zf.open(name) as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        out[name] = h.hexdigest()
    return out


def _read_chunks(zf: zipfile.ZipFile, member: str,
                 usecols: Sequence[str], chunksize: int
                 ) -> Iterator[pd.DataFrame]:
    """Stream one TSV member. Everything is read as str and cast later.

    Chunked because FUND_REPORTED_HOLDING is ~1 GB uncompressed and this has
    to run on a laptop-sized WSL allocation.
    """
    with zf.open(member) as f:
        yield from pd.read_csv(
            f, sep="\t", usecols=list(usecols), dtype=str,
            chunksize=chunksize, encoding="utf-8",
            encoding_errors="replace", low_memory=False,
        )


# --------------------------------------------------------------------------
# Cleaning primitives -- each one is separately testable without a download
# --------------------------------------------------------------------------

def valid_cusip(series: pd.Series) -> pd.Series:
    """True where the CUSIP is a usable 9-character identifier.

    Filers use "000000000", "N/A" and blanks as placeholders; those rows carry
    no way to tell one bond from another and are dropped rather than merged
    into a single phantom instrument.
    """
    s = series.fillna("").astype(str).str.strip().str.upper()
    return (s.str.len() == 9) & (s != "000000000") \
        & s.str.match(_CUSIP_RE.pattern, na=False)


def repair_rate_scale(rate: pd.Series, ceiling: float
                      ) -> tuple[pd.Series, int]:
    """Put ANNUALIZED_RATE on the percent scale, and count the repairs.

    A minority of filers report the coupon as a decimal fraction. On 2026q1
    the two conventions coexist within 8,692 CUSIPs and the ratio between them
    is exactly 100.0 across the whole interquartile range, which is what makes
    this a units bug rather than a disagreement about the bond.

    A genuine coupon strictly between 0 and `ceiling` percent would be
    wrongly inflated by this rule. USD corporates with sub-30bp coupons are
    vanishingly rare, and the yield plausibility screen catches any that slip
    through; the alternative -- believing the field -- turns a 4.5% bond into a
    0.045% bond, which is not survivable.
    """
    r = pd.to_numeric(rate, errors="coerce")
    mask = (r > 0.0) & (r < ceiling)
    return r.mask(mask, r * 100.0), int(mask.sum())


def solve_ytm(price: np.ndarray, coupon: np.ndarray, maturity: np.ndarray,
              lo: float = -0.20, hi: float = 1.50, iters: int = 100
              ) -> np.ndarray:
    """Invert `src.universe.price_and_duration` for the yield.

    Bisection, not Newton: price is strictly decreasing in yield over the
    bracket, so bisection cannot fail to converge and has no derivative to get
    wrong. 100 halvings of a 1.7-wide bracket is far below double precision,
    so the result is the exact root of the repo's own pricing function.

    This deliberately calls the existing pricer rather than writing a closed
    form. A second implementation of the cash-flow schedule would be a second
    thing to keep in agreement with the duration and convexity the optimizer
    actually uses.
    """
    lo_a = np.full(price.shape, float(lo))
    hi_a = np.full(price.shape, float(hi))
    for _ in range(iters):
        mid = 0.5 * (lo_a + hi_a)
        p, _, _ = price_and_duration(coupon, mid, maturity, freq=FREQ)
        # p > price means the bond is priced too richly at `mid`: raise yield.
        too_rich = p > price
        lo_a = np.where(too_rich, mid, lo_a)
        hi_a = np.where(too_rich, hi_a, mid)
    return 0.5 * (lo_a + hi_a)


def grid_maturity(ttm_years: np.ndarray, cfg: NportConfig) -> np.ndarray:
    """Round time to maturity onto the 1/FREQ coupon grid.

    `price_and_duration` requires it and refuses to round silently, which is
    the right default -- so the rounding happens here, once, visibly. The
    displacement is at most 1/(2*FREQ) = 0.25 years and moves modified
    duration by strictly less than that. `NportDiagnostics` reports the mean
    and worst displacement actually incurred.

    Doing better would need day counts, settlement conventions and accrued
    interest -- i.e. a real pricer, which this repo deliberately does not have
    and which would not change any conclusion the paper draws.
    """
    g = np.round(np.asarray(ttm_years, dtype=float) * FREQ) / FREQ
    return np.clip(g, max(1.0 / FREQ, cfg.maturity_min_y), cfg.maturity_max_y)


def recentre_levels(levels: Mapping[str, float], shares: Mapping[str, float],
                    oas_bp: np.ndarray, weight: np.ndarray,
                    ) -> tuple[dict[str, float], float]:
    """Slide the published rating spread levels onto this quarter's level.

    `CalibrationConfig.oas_by_rating_bp` holds LONG-RUN average spreads (the
    FRED ICE BofA series). Spreads are cyclical and a rating scale is not, so
    matching a tight quarter's bonds against long-run averages systematically
    reads them as higher-grade than they are: on 2026q1 the unshifted map puts
    24% of benchmark weight in AAA/AA against roughly 9% in the index
    composition the same config declares.

    The fix keeps the published SHAPE -- the log spacing between rating levels,
    which is the part of the calibration that carries the rating information --
    and moves the LEVEL so that the share-weighted geometric mean of the
    levels equals the market-value-weighted geometric mean of the universe's
    spreads. One scalar, in log space, applied to every bucket.

    Returns the shifted levels and the multiplicative shift, which the
    provenance records: a shift far from 1 is a statement about the quarter,
    and a reader should be able to see it.
    """
    order = [r for r in RATINGS if r in levels and r in shares]
    p = np.array([float(shares[r]) for r in order])
    if p.sum() <= 0:
        raise ValueError("`shares` must sum to something positive")
    p = p / p.sum()
    anchor = float(p @ np.log([float(levels[r]) for r in order]))
    w = np.asarray(weight, dtype=float)
    w = w / w.sum()
    observed = float(w @ np.log(np.maximum(np.asarray(oas_bp, float), 1e-6)))
    shift = float(np.exp(observed - anchor))
    return {r: float(v) * shift for r, v in levels.items()}, shift


def spread_implied_ratings(
    oas_bp: np.ndarray,
    issuer_id: np.ndarray,
    levels: Mapping[str, float],
    scheme: str = "spread_implied",
    shares: Mapping[str, float] | None = None,
) -> tuple[np.ndarray, pd.Series]:
    """Bucket bonds into `RATINGS` from their spread. A documented PROXY.

    **N-PORT contains no credit ratings and never will** -- they are the
    agencies' property. Three things were possible: drop the rating family,
    invent ratings from a seeded draw, or infer them from the one credit
    signal the filings do carry, which is the price. The first guts a
    constraint family the paper's mandate depends on (and `src/scenarios.py`
    keys its default probabilities off the rating label, so the label cannot
    simply be absent). The second is exactly the synthetic-data criticism this
    module exists to answer. So: infer.

    **The circularity, stated plainly.** A rating derived from spread is not
    independent of spread. Two consequences follow and both are real:

    1. The rating-band family and the DTS family in `src/credit.py` become
       correlated by construction. They are not the same constraint -- bands
       are on weight, DTS is on weight x spread x duration -- but a reader must
       not treat them as independent evidence.
    2. Any claim of the form "the optimizer found bonds whose spread is wide
       for their rating" is UNAVAILABLE on this universe. There is no such
       thing here by construction. The paper must not make that claim on
       N-PORT data.

    **What is NOT damaged.** The paper's lead result (M6) is that a
    tracking-error mandate written on the standard deviation of active returns
    is a second-order cone that no LP solver can express, while its MAD
    surrogate overshoots the true sigma budget. That is a statement about the
    *form* of a constraint. Rating labels enter it only as part of the shape
    of the feasible polytope; they cannot manufacture a cone, and three
    solvers from three algorithm families agreeing is not something a rating
    proxy can arrange. The proxy costs the paper the rating-arbitrage story it
    was never telling, and costs it nothing else.

    Assignment is at the ISSUER level: the issuer's median bond spread picks
    one bucket and all its bonds inherit it. Two reasons. Ratings are issued
    to issuers (and per seniority class, which N-PORT also does not carry), so
    bond-by-bond assignment would be wrong even with real data; and
    `src/scenarios.py` explicitly assumes a single rating per issuer when it
    builds the default trigger -- `_first_code_per_group` would otherwise pick
    one bond's label arbitrarily and use it for the whole name.

    Two schemes:

    - ``spread_implied`` -- the issuer takes the rating whose published
      long-run average OAS (`levels`, i.e. `CalibrationConfig.oas_by_rating_bp`
      sourced from the FRED ICE BofA series) is nearest in LOG spread. Log,
      because spread levels are roughly geometric across the rating scale, so
      a linear distance would drag everything toward the wide end.
    - ``spread_quantile`` -- issuers are ranked by spread and cut so the
      resulting bucket shares match `shares` (a published index rating
      composition). Imposes an index-like mix by construction, which is a
      different assumption, not a weaker one: it guarantees non-degenerate
      buckets at the cost of asserting the mix rather than measuring it.

    Returns the per-bond rating array and the per-issuer spread series used.
    """
    oas_bp = np.asarray(oas_bp, dtype=float)
    iss = pd.Series(np.asarray(issuer_id))
    issuer_oas = pd.Series(oas_bp).groupby(iss).median()

    order = [r for r in RATINGS if r in levels]
    if not order:
        raise ValueError("levels must contain at least one rating in RATINGS")

    if scheme == "spread_implied":
        lv = np.log(np.array([levels[r] for r in order], dtype=float))
        d = np.abs(np.log(np.maximum(issuer_oas.to_numpy(), 1e-6))[:, None]
                   - lv[None, :])
        pick = np.array(order, dtype=object)[d.argmin(axis=1)]
    elif scheme == "spread_quantile":
        if not shares:
            raise ValueError("spread_quantile needs `shares`")
        w = np.array([float(shares.get(r, 0.0)) for r in order])
        if w.sum() <= 0:
            raise ValueError("`shares` must sum to something positive")
        w = w / w.sum()
        # Rank issuers tightest-first; the tightest `shares['AAA']` fraction
        # becomes AAA, and so on down the scale. `method="first"` breaks ties
        # by position, which is deterministic because the groupby sorted.
        rank = issuer_oas.rank(method="first").to_numpy() - 1.0
        frac = rank / max(len(issuer_oas), 1)
        edges = np.cumsum(w)[:-1]
        pick = np.array(order, dtype=object)[np.searchsorted(edges, frac,
                                                             side="right")]
    else:
        raise ValueError(f"unknown rating scheme {scheme!r}")

    by_issuer = pd.Series(pick, index=issuer_oas.index)
    return by_issuer.reindex(iss.to_numpy()).to_numpy(), issuer_oas


# --------------------------------------------------------------------------
# Treasury curve, fitted from the same filing quarter
# --------------------------------------------------------------------------

def fit_treasury_curve(
    treasuries: pd.DataFrame, cfg: NportConfig,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[int, ...]]:
    """A par-ish Treasury curve out of the UST holdings in the same ZIP.

    `treasuries` needs columns `price`, `coupon` (percent), `maturity` (grid
    years) and `ttm` (exact years). Each bond's yield is solved with the same
    bisection the corporates use; the knot value is the MEDIAN yield of the
    bonds inside that knot's maturity window.

    Median, not a fit: the UST population under ISSUER_TYPE == "UST" contains
    TIPS (quoted on an inflation-adjusted principal, so their nominal yield
    reads far too low), STRIPS, and the occasional agency bond a filer
    misclassified. A least-squares fit would chase those; a median ignores
    them. Nothing here is bootstrapped into zero rates -- the downstream
    consumer is `numpy.interp` on a par curve, which is exactly what
    `src.universe._treasury_yield` does with the synthetic curve, so the real
    curve is a drop-in replacement for it.

    Raises if a knot is too thin, rather than interpolating over a hole.
    """
    y = solve_ytm(treasuries["price"].to_numpy(),
                  treasuries["coupon"].to_numpy() / 100.0,
                  treasuries["maturity"].to_numpy())
    ok = np.isfinite(y) & (y > 0.0) & (y < 0.15)
    t = treasuries.loc[ok].assign(ytm=y[ok])

    knots, yields, counts = [], [], []
    for k in cfg.treasury_knots_y:
        half = max(cfg.treasury_window_abs_y, cfg.treasury_window_rel * k)
        sel = t["ttm"].between(k - half, k + half)
        n = int(sel.sum())
        if n < cfg.treasury_min_per_knot:
            raise ValueError(
                f"Treasury knot {k}y has only {n} bonds "
                f"(< {cfg.treasury_min_per_knot}); widen the window or drop "
                f"the knot rather than interpolating over a hole")
        knots.append(float(k))
        yields.append(float(t.loc[sel, "ytm"].median()))
        counts.append(n)
    return tuple(knots), tuple(yields), tuple(counts)


# --------------------------------------------------------------------------
# Extraction and consolidation
# --------------------------------------------------------------------------

def load_debt_securities(zf: zipfile.ZipFile,
                         chunksize: int = 500_000) -> pd.DataFrame:
    """DEBT_SECURITY, restricted to fixed-coupon rows, indexed by HOLDING_ID.

    Floating and variable coupons are dropped here: their spread duration is
    not their modified duration, which is the assumption the whole repo's
    bond math rests on (see `src/universe.py`'s scope note).
    """
    parts = []
    for chunk in _read_chunks(zf, "DEBT_SECURITY.tsv", _DEBT_COLS, chunksize):
        c = chunk[chunk["COUPON_TYPE"].fillna("").str.strip().str.lower()
                  == COUPON_FIXED]
        parts.append(c)
    ds = pd.concat(parts, ignore_index=True)
    ds["HOLDING_ID"] = pd.to_numeric(ds["HOLDING_ID"], errors="coerce")
    ds = ds.dropna(subset=["HOLDING_ID"])
    ds["HOLDING_ID"] = ds["HOLDING_ID"].astype("int64")
    return ds.drop(columns=["COUPON_TYPE"]).set_index("HOLDING_ID")


def _holding_stream(zf: zipfile.ZipFile, debt: pd.DataFrame,
                    issuer_type: str, chunksize: int,
                    stage: dict[str, int]) -> pd.DataFrame:
    """Holdings of one issuer type, joined to the fixed-coupon debt table.

    The filters are applied inside the chunk loop so the ~1 GB holdings table
    never lands in memory whole. `stage` accumulates the survivor count after
    each filter -- that ledger is the audit trail for "what did you exclude,
    and how much of it was there".
    """
    parts = []

    def bump(key: str, n: int) -> None:
        stage[key] = stage.get(key, 0) + n

    for chunk in _read_chunks(zf, "FUND_REPORTED_HOLDING.tsv",
                              _HOLDING_COLS, chunksize):
        bump("00_all_holdings", len(chunk))
        c = chunk[chunk["ASSET_CAT"] == ASSET_CAT_DEBT]
        bump("01_asset_cat_debt", len(c))
        c = c[c["ISSUER_TYPE"] == issuer_type]
        bump("02_issuer_type", len(c))
        c = c[c["CURRENCY_CODE"] == "USD"]
        bump("03_currency_usd", len(c))
        c = c[c["PAYOFF_PROFILE"] == PAYOFF_LONG]
        bump("04_payoff_long", len(c))
        c = c[c["UNIT"] == UNIT_PRINCIPAL]
        bump("05_unit_principal", len(c))
        if c.empty:
            continue
        c = c.copy()
        c["HOLDING_ID"] = pd.to_numeric(c["HOLDING_ID"], errors="coerce")
        c = c.dropna(subset=["HOLDING_ID"])
        c["HOLDING_ID"] = c["HOLDING_ID"].astype("int64")
        c = c.join(debt, on="HOLDING_ID", how="inner")
        bump("06_fixed_coupon_debt", len(c))
        keep = valid_cusip(c["ISSUER_CUSIP"])
        c = c[keep]
        bump("07_valid_cusip", len(c))
        if c.empty:
            continue
        parts.append(c.drop(columns=["ASSET_CAT", "ISSUER_TYPE",
                                     "CURRENCY_CODE", "PAYOFF_PROFILE",
                                     "UNIT", "HOLDING_ID"]))
    if not parts:
        raise ValueError(f"no holdings survived the filters for "
                         f"ISSUER_TYPE={issuer_type!r}")
    return pd.concat(parts, ignore_index=True)


def _numeric_and_price(h: pd.DataFrame, cfg: NportConfig,
                       stage: dict[str, int]) -> tuple[pd.DataFrame, int, int]:
    """Cast, derive the implied price, and drop the two classes of bad mark.

    Two stages, because they catch different errors. The absolute band catches
    a filer who reported principal in thousands (price ~ 100,000) or value in
    a foreign currency. The deviation-from-median rule catches a single stale
    or fat-fingered mark on a bond whose other holders agree -- which the
    absolute band cannot see, and which would otherwise drag the consolidated
    price.
    """
    h = h.copy()
    h["cusip"] = h["ISSUER_CUSIP"].str.strip().str.upper()
    h["balance"] = pd.to_numeric(h["BALANCE"], errors="coerce")
    h["value_usd"] = pd.to_numeric(h["CURRENCY_VALUE"], errors="coerce")
    h["coupon_pct"], n_repaired = repair_rate_scale(
        h["ANNUALIZED_RATE"], cfg.rate_decimal_ceiling)
    h["maturity_date"] = pd.to_datetime(
        h["MATURITY_DATE"], format="%d-%b-%Y", errors="coerce")
    h = h.dropna(subset=["balance", "value_usd", "coupon_pct",
                         "maturity_date"])
    h = h[(h["balance"] > 0) & (h["value_usd"] > 0)]
    stage["08_positive_and_parsed"] = len(h)

    h["price"] = 100.0 * h["value_usd"] / h["balance"]
    before = len(h)
    h = h[h["price"].between(cfg.price_min, cfg.price_max)]
    dropped_band = before - len(h)
    stage["09_price_in_band"] = len(h)

    med = h.groupby("cusip")["price"].transform("median")
    before = len(h)
    h = h[(h["price"] - med).abs() <= cfg.price_dev_max]
    dropped_dev = before - len(h)
    stage["10_price_near_peers"] = len(h)
    h.attrs["rate_rows_repaired"] = n_repaired
    return h, dropped_band, dropped_dev


def _flag_any(series: pd.Series) -> bool:
    return bool((series.fillna("").str.strip().str.upper() == "Y").any())


def consolidate_holdings(h: pd.DataFrame) -> pd.DataFrame:
    """One row per CUSIP out of many rows per CUSIP-fund pair.

    **This is the step most likely to be silently wrong**, so the choices are
    spelled out.

    *Price: the cross-filer MEDIAN, not the value-weighted mean.* Filers
    mostly agree -- the median cross-filer p90-p10 spread on 2026q1 is 0.55
    points -- but the disagreement has a fat tail, and a value-weighted mean is
    a mean of ratios whose denominator a single mis-scaled filer can dominate.
    Measured on 2026q1 without the row-level screens, the 99.9th percentile of
    the value-weighted price is 17,699. The median is unmoved by it.

    *Coupon: the median of the scale-repaired rates.* Same reasoning, plus it
    is the tiebreaker for the percent-versus-decimal split when the row-level
    repair leaves a residue.

    *Maturity: the MODE, not the median.* A maturity date is a fact, not a
    measurement: when filers disagree (2% of CUSIPs on 2026q1) one of them is
    wrong, and the majority answer is the right estimator. A median of dates
    would invent a date no filer reported.

    *Market value and par: SUMS across filers.* These are the only genuinely
    additive quantities here, and the sum is the point -- it is the aggregate
    registered-fund position in the bond, which is what `w_benchmark` is
    built from.

    *Flags are ORs.* If any filer says the bond is in default, in arrears, PIK
    or contingently convertible, it leaves the universe. A disagreement about
    a credit event resolves toward exclusion, not toward the majority: the
    cost of dropping a clean bond is one row, the cost of keeping a defaulted
    one is a bond whose price and spread mean nothing.
    """
    g = h.groupby("cusip", sort=True)
    out = g.agg(
        n_funds=("ACCESSION_NUMBER", "nunique"),
        n_rows=("price", "size"),
        market_value_usd=("value_usd", "sum"),
        par_held_usd=("balance", "sum"),
        price=("price", "median"),
        coupon_pct=("coupon_pct", "median"),
        price_p10=("price", lambda s: s.quantile(0.10)),
        price_p90=("price", lambda s: s.quantile(0.90)),
    )
    out["maturity_date"] = g["maturity_date"].agg(
        lambda s: s.mode().iloc[0])
    out["in_default"] = g["IS_DEFAULT"].agg(_flag_any)
    out["in_arrears"] = g["ARE_ANY_INTEREST_PAYMENT"].agg(_flag_any)
    out["pays_in_kind"] = g["IS_ANY_PORTION_INTEREST_PAID"].agg(_flag_any)
    out["convertible"] = (g["IS_CONVTIBLE_MANDATORY"].agg(_flag_any)
                          | g["IS_CONVTIBLE_CONTINGENT"].agg(_flag_any))
    out["country"] = g["INVESTMENT_COUNTRY"].agg(
        lambda s: s.dropna().mode().iloc[0] if s.notna().any() else "XX")
    # Issuer name: the spelling attached to the most money. Filers use ~30k
    # distinct spellings for ~6k issuers, so any name-based logic has to pick
    # one deterministically, and the largest holder's spelling is the least
    # arbitrary rule available.
    name_mv = h.groupby(["cusip", "ISSUER_NAME"], sort=True)["value_usd"].sum()
    name_mv = name_mv.reset_index().sort_values(
        ["cusip", "value_usd", "ISSUER_NAME"], ascending=[True, False, True])
    out["issuer_name"] = name_mv.drop_duplicates("cusip").set_index(
        "cusip")["ISSUER_NAME"]
    return out.reset_index()


# --------------------------------------------------------------------------
# Universe assembly
# --------------------------------------------------------------------------

def build_universe(
    zip_path: str,
    quarter: str,
    downloaded_utc: str,
    cfg: NportConfig | None = None,
    calib: CalibrationConfig | None = None,
    seed: int = 7,
    chunksize: int = 400_000,
) -> NportUniverse:
    """Turn one quarterly N-PORT ZIP into a `BondUniverse`.

    Deterministic: the same ZIP, `cfg` and `seed` produce a byte-identical
    `bonds` frame, and `provenance.bonds_sha256` proves it.

    `calib` supplies the two things that are NOT in N-PORT and are not
    derivable from it: the published OAS-by-rating levels the spread-implied
    rating proxy is calibrated against, and the index rating shares the
    quantile scheme would target. Its Treasury curve is REPLACED by the one
    fitted from this quarter's UST holdings, and the returned universe's
    `config` carries the fitted curve so every downstream consumer of
    `_treasury_yield` sees the real one.
    """
    cfg = cfg or NportConfig()
    cfg.validate()
    calib = calib or CalibrationConfig()

    zf = zipfile.ZipFile(zip_path)
    members = ("FUND_REPORTED_HOLDING.tsv", "DEBT_SECURITY.tsv",
               "SUBMISSION.tsv")
    member_hashes = _member_sha256(zf, members)

    sub = pd.read_csv(zf.open("SUBMISSION.tsv"), sep="\t", dtype=str,
                      usecols=["ACCESSION_NUMBER", "REPORT_DATE"])
    report_date = str(sub["REPORT_DATE"].mode().iloc[0])
    n_filings = int(sub["ACCESSION_NUMBER"].nunique())

    debt = load_debt_securities(zf, chunksize=chunksize)

    # ---- Treasury curve, from the same quarter --------------------------
    t_stage: dict[str, int] = {}
    tre = _holding_stream(zf, debt, ISSUER_TYPE_TREASURY, chunksize, t_stage)
    tre, _, _ = _numeric_and_price(tre, cfg, t_stage)
    tre_b = consolidate_holdings(tre)
    asof = pd.Timestamp(report_date)
    tre_b["ttm"] = (tre_b["maturity_date"] - asof).dt.days / 365.25
    tre_b = tre_b[(~tre_b["in_default"]) & (tre_b["n_funds"] >= 3)
                  & (tre_b["ttm"] > 0.5)
                  & (tre_b["ttm"] <= cfg.maturity_max_y + 0.5)]
    tre_b["maturity"] = grid_maturity(tre_b["ttm"].to_numpy(), cfg)
    tre_b = tre_b.rename(columns={"coupon_pct": "coupon"})
    knots, curve, counts = fit_treasury_curve(tre_b, cfg)
    calib = dataclasses.replace(calib, treasury_knots_y=knots,
                                treasury_curve=curve)
    del tre, tre_b

    # ---- corporates ------------------------------------------------------
    stage: dict[str, int] = {}
    h = _holding_stream(zf, debt, ISSUER_TYPE_CORP, chunksize, stage)
    del debt
    h, drop_band, drop_dev = _numeric_and_price(h, cfg, stage)
    n_repaired = int(h.attrs.get("rate_rows_repaired", 0))

    # How many CUSIPs carry both coupon conventions - the size of caveat 4.
    raw = pd.to_numeric(h["ANNUALIZED_RATE"], errors="coerce")
    scale = pd.DataFrame({
        "cusip": h["cusip"].to_numpy(),
        "dec": ((raw > 0) & (raw < cfg.rate_decimal_ceiling)).to_numpy(),
        "pct": (raw >= cfg.rate_decimal_ceiling).to_numpy(),
    }).groupby("cusip").sum()
    n_mixed = int(((scale["dec"] > 0) & (scale["pct"] > 0)).sum())
    del raw, scale

    b = consolidate_holdings(h)
    stage["11_distinct_cusips"] = len(b)
    cross_disp = float((b["price_p90"] - b["price_p10"]).median())
    del h

    # ---- bond-level eligibility -----------------------------------------
    b["ttm"] = (b["maturity_date"] - asof).dt.days / 365.25
    for key, mask in (
        ("12_not_in_default", ~b["in_default"]),
        ("13_not_in_arrears", ~b["in_arrears"]),
        ("14_not_pik", ~b["pays_in_kind"]),
        ("15_not_convertible", ~b["convertible"]),
        ("16_min_funds", b["n_funds"] >= cfg.min_funds),
        ("17_coupon_positive", b["coupon_pct"] > 0),
        ("18_maturity_window", b["ttm"].between(cfg.maturity_min_y,
                                                cfg.maturity_max_y)),
    ):
        b = b[mask.reindex(b.index, fill_value=False)]
        stage[key] = len(b)
    b = b.reset_index(drop=True)
    if b.empty:
        raise ValueError("no bonds survived the eligibility filters")

    # ---- bond mathematics ------------------------------------------------
    b["maturity"] = grid_maturity(b["ttm"].to_numpy(), cfg)
    coupon = b["coupon_pct"].to_numpy() / 100.0
    ytm = solve_ytm(b["price"].to_numpy(), coupon, b["maturity"].to_numpy())
    b["ytm"] = ytm
    b = b[(b["ytm"] > cfg.ytm_min) & (b["ytm"] < cfg.ytm_max)]
    stage["19_ytm_plausible"] = len(b)
    b = b.reset_index(drop=True)
    if b.empty:
        raise ValueError("no bonds survived the yield plausibility screen")

    # Spread over the curve this quarter's own Treasury holdings imply,
    # computed on EVERY eligible bond before any trim: the rating anchor
    # below has to describe the quarter's spread level, not the spread level
    # of whichever subset was kept. Anchoring on the trimmed set instead made
    # the top-500-by-ownership subset (which is materially wider than the
    # whole market, because high-yield funds crowd into a handful of large
    # issues) reprice the entire rating scale, so the same bond came out AAA
    # at n=500 and A on the full universe. A rating that moves when you
    # change the sample size is not a rating.
    treas_all = np.interp(b["maturity"].to_numpy(),
                          np.asarray(calib.treasury_knots_y),
                          np.asarray(calib.treasury_curve))
    oas_all = np.clip((b["ytm"].to_numpy() - treas_all) * 1e4,
                      cfg.oas_floor_bp, cfg.oas_cap_bp)
    levels = dict(calib.oas_by_rating_bp)
    level_shift = 1.0
    n_anchor = len(b)
    if cfg.rating_level_recentre:
        levels, level_shift = recentre_levels(
            levels, calib.rating_weights, oas_all,
            b["market_value_usd"].to_numpy(dtype=float))

    if cfg.n_bonds is not None and len(b) > cfg.n_bonds:
        # Largest by aggregate fund-held market value: the same ordering the
        # benchmark weight uses, so trimming removes the tail of the index
        # rather than a random subset of it. Ties broken on CUSIP so the
        # trim is deterministic.
        b = b.sort_values(["market_value_usd", "cusip"],
                          ascending=[False, True]).head(cfg.n_bonds)
        b = b.sort_values("cusip").reset_index(drop=True)
    stage["20_final"] = len(b)
    round_err = np.abs(b["maturity"].to_numpy() - b["ttm"].to_numpy())

    coupon = b["coupon_pct"].to_numpy() / 100.0
    maturity = b["maturity"].to_numpy()
    price_model, mod_dur, convexity = price_and_duration(
        coupon, b["ytm"].to_numpy(), maturity, freq=FREQ)
    ytm_err = float(np.abs(price_model - b["price"].to_numpy()).max())

    treas = np.interp(maturity, np.asarray(calib.treasury_knots_y),
                      np.asarray(calib.treasury_curve))
    oas_raw = (b["ytm"].to_numpy() - treas) * 1e4
    n_low = int((oas_raw < cfg.oas_floor_bp).sum())
    n_high = int((oas_raw > cfg.oas_cap_bp).sum())
    oas_bp = np.clip(oas_raw, cfg.oas_floor_bp, cfg.oas_cap_bp)

    # ---- identity, proxies ----------------------------------------------
    # CUSIP6, the issue prefix, is the industry-standard issuer key and is
    # populated on every surviving row. ISSUER_LEI is present on only ~88% of
    # them and ISSUER_NAME has ~30k spellings for ~6k issuers, so neither is
    # usable as the key a concentration limit is enforced on.
    issuer_id = b["cusip"].str[:6].to_numpy()
    mv_raw = b["market_value_usd"].to_numpy(dtype=float)
    rating, _ = spread_implied_ratings(
        oas_bp, issuer_id, levels,
        scheme=cfg.rating_scheme, shares=calib.rating_weights)

    if cfg.sector_scheme == "keyword":
        sector, decided = classify_sector_by_issuer(
            issuer_id, b["issuer_name"].tolist(), mv_raw)
        kw_share = float(decided.mean())
    else:
        sector = b["country"].fillna("XX").to_numpy()
        kw_share = float("nan")

    cell_code = np.clip(
        np.digitize(maturity, MATURITY_CELL_EDGES[1:-1]), 0,
        len(MATURITY_CELL_NAMES) - 1)

    bonds = pd.DataFrame({
        "bond_id": b["cusip"].to_numpy(),
        "issuer_id": issuer_id,
        "sector": sector,
        "rating": rating,
        "maturity": maturity,
        "coupon": coupon,
        "ytm": b["ytm"].to_numpy(),
        "oas_bp": oas_bp,
        "price": b["price"].to_numpy(),
        "mod_duration": mod_dur,
        "spread_duration": mod_dur,      # fixed-rate bullets, per repo scope
        "convexity": convexity,
        # NOT amount outstanding: N-PORT does not report it. This is the par
        # amount held across all filing funds, in USD millions, and the name
        # is kept only because the column contract requires it.
        "issue_size_mm": b["par_held_usd"].to_numpy() / 1e6,
        "maturity_cell_code": cell_code,
        "maturity_cell": [MATURITY_CELL_NAMES[c] for c in cell_code],
        # ---- provenance columns, ignored by the pipeline -----------------
        "issuer_name": b["issuer_name"].to_numpy(),
        "investment_country": b["country"].to_numpy(),
        "maturity_date": b["maturity_date"].dt.strftime("%Y-%m-%d").to_numpy(),
        "maturity_exact_y": b["ttm"].to_numpy(),
        "maturity_round_err_y": round_err,
        "n_funds_holding": b["n_funds"].to_numpy(),
        "market_value_usd": b["market_value_usd"].to_numpy(),
        "oas_raw_bp": oas_raw,
        "rating_source": f"proxy:{cfg.rating_scheme}",
        "sector_source": f"proxy:{cfg.sector_scheme}",
    })

    mv = bonds["market_value_usd"].to_numpy(dtype=float)
    w_benchmark = mv / mv.sum()

    # ---- incumbent portfolio --------------------------------------------
    # Identical rule and defaults to `make_universe`, so the synthetic and
    # real universes differ in their DATA and not in how w0 was invented.
    rng = np.random.default_rng(seed)
    n = len(bonds)
    # `make_universe` writes `max(10, ...)` unclamped, which cannot draw
    # without replacement on a universe of fewer than 10 bonds. Clamped here
    # because a hand-built test fixture is exactly that small, and a helper
    # that only works above some unstated size is a helper that will be
    # trusted at the wrong moment.
    n_hold = min(n, max(10, int(round(cfg.portfolio_coverage * n))))
    held = rng.choice(n, size=n_hold, replace=False, p=w_benchmark)
    w0 = np.zeros(n)
    w0[held] = w_benchmark[held] * rng.lognormal(
        0.0, cfg.portfolio_tilt_sd, n_hold)
    w0 /= w0.sum()

    uni = BondUniverse(bonds=bonds, w_benchmark=w_benchmark, w0=w0,
                       config=calib, seed=seed)

    diag = NportDiagnostics(
        share_above_par=float((bonds["price"] > 100.0).mean()),
        n_oas_clipped_low=n_low,
        n_oas_clipped_high=n_high,
        rate_scale_rows_repaired=n_repaired,
        rate_scale_cusips_mixed=n_mixed,
        price_rows_dropped_band=drop_band,
        price_rows_dropped_dev=drop_dev,
        sector_keyword_share=kw_share,
        maturity_round_mae_y=float(round_err.mean()),
        maturity_round_max_y=float(round_err.max()),
        cross_filer_price_p90_p10_median=cross_disp,
        n_funds_median=float(np.median(bonds["n_funds_holding"])),
        ytm_solve_max_price_error=ytm_err,
        oas_median_eligible_bp=float(np.median(oas_all)),
        oas_median_kept_bp=float(np.median(oas_bp)),
        n_eligible_before_trim=int(n_anchor),
    )

    prov = NportProvenance(
        quarter=quarter,
        source_url=quarter_url(quarter),
        landing_page=NPORT_LANDING,
        licence=NPORT_LICENCE,
        zip_sha256=sha256_file(zip_path),
        zip_bytes=int(os.path.getsize(zip_path)),
        downloaded_utc=downloaded_utc,
        member_sha256=member_hashes,
        report_date=report_date,
        n_filings=n_filings,
        stage_counts=dict(sorted(stage.items())),
        config=dataclasses.asdict(cfg),
        seed=seed,
        treasury_curve={
            "knots_y": list(knots),
            "par_yield_decimal": list(curve),
            "n_bonds_per_knot": list(counts),
            "source": "UST holdings in the same N-PORT quarter",
        },
        rating_anchor={
            "scheme": cfg.rating_scheme,
            "recentred": bool(cfg.rating_level_recentre),
            "published_levels_bp": dict(calib.oas_by_rating_bp),
            "published_shares": dict(calib.rating_weights),
            "level_shift": level_shift,
            "levels_used_bp": {k: float(v) for k, v in levels.items()},
            "anchored_on_n_bonds": n_anchor,
            "anchor_note": ("the shift is computed on every eligible bond "
                            "in the quarter, BEFORE any n_bonds trim, so "
                            "the rating scale does not move with sample "
                            "size"),
            "note": ("levels are a PROXY anchor, not ratings; see "
                     "src.nport.spread_implied_ratings for the circularity "
                     "this buys and what it costs the paper"),
        },
        bonds_sha256=frame_sha256(bonds),
        n_bonds=int(len(bonds)),
        n_issuers=int(bonds["issuer_id"].nunique()),
        generated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
    )
    return NportUniverse(universe=uni, provenance=prov, diagnostics=diag,
                         treasury_knots=knots, treasury_yields=curve,
                         treasury_counts=counts)


def frame_sha256(df: pd.DataFrame) -> str:
    """Content hash of a DataFrame, stable across runs and platforms.

    Floats go through `repr` at 17 significant digits rather than through
    their raw bytes: the bytes are what determinism actually means here, but
    a CSV round-trip is what a reader would reproduce, and 17 digits is exact
    for IEEE754 doubles either way.
    """
    h = hashlib.sha256()
    h.update("\t".join(map(str, df.columns)).encode())
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_float_dtype(s):
            payload = "\n".join(f"{v:.17g}" for v in s.to_numpy())
        else:
            payload = "\n".join(map(str, s.to_numpy()))
        h.update(col.encode())
        h.update(payload.encode())
    return h.hexdigest()

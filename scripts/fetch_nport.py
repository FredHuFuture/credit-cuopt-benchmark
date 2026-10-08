#!/usr/bin/env python
"""Fetch one SEC Form N-PORT quarter, and build a real bond universe from it.

The repo's standing data policy is "fetch at runtime, commit nothing". N-PORT
is public domain (US Government Work,
http://www.usa.gov/publicdomain/label/1.0/, verified 2026-08-23) so the legal
half of that policy does not bind here -- but a 463 MB quarterly ZIP does not
belong in git under any licence, so the cache directory stays gitignored and
only the DERIVED universe and its provenance are committable.

Two things happen here and they are separable on purpose:

    fetch   download the quarter into data/cache/nport/, verify it, and write
            a sidecar recording the URL, the sha256 and the download date.
    build   turn the cached ZIP into a `BondUniverse`, write it with its
            provenance, and print the honest report: what the filters removed,
            what the proxies decided, whether `default_limits` is feasible on
            it, and how the result differs from `make_universe`.

`--build` implies a cached quarter; it never downloads implicitly, so a build
cannot silently depend on the network.

The SEC asks for a descriptive User-Agent on automated requests and rate-limits
to about 10 requests per second. This makes one request per quarter and
identifies itself; `--contact` is required for exactly that reason.

Examples:
    python scripts/fetch_nport.py --quarter 2026q1 --contact you@example.org
    python scripts/fetch_nport.py --quarter 2026q1 --build --n-bonds 500
    python scripts/fetch_nport.py --quarter 2026q1 --build   # full universe
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.credit import CreditLimits, default_limits  # noqa: E402
from src.nport import (  # noqa: E402
    NPORT_LANDING, NPORT_LICENCE, USER_AGENT_TEMPLATE, NportConfig,
    build_universe, quarter_url, sha256_file,
)
from src.universe import (  # noqa: E402
    MATURITY_CELL_NAMES, RATINGS, BondUniverse, make_universe,
)

DEFAULT_CACHE = REPO / "data" / "cache" / "nport"
DEFAULT_OUT = REPO / "results"

#: Read size for the streaming download. Large enough that the progress line
#: is not the bottleneck, small enough to keep memory flat.
_CHUNK = 1 << 20


# --------------------------------------------------------------------------
# Fetch
# --------------------------------------------------------------------------

def fetch_quarter(quarter: str, cache: Path, contact: str,
                  force: bool = False) -> tuple[Path, dict]:
    """Download one quarter into `cache`, or reuse what is already there.

    Downloads to a `.part` file and renames only on success, so an interrupted
    fetch can never be mistaken for a complete one -- which matters more than
    usual here, because a truncated ZIP fails in the middle of a table rather
    than at open time.
    """
    cache.mkdir(parents=True, exist_ok=True)
    url = quarter_url(quarter)
    dest = cache / f"{quarter}_nport.zip"
    side = cache / f"{quarter}_nport.json"

    if dest.exists() and side.exists() and not force:
        meta = json.loads(side.read_text())
        if meta.get("sha256") and meta.get("bytes") == dest.stat().st_size:
            print(f"[cache] {dest} ({dest.stat().st_size / 1e6:.0f} MB, "
                  f"downloaded {meta.get('downloaded_utc')})")
            return dest, meta

    tmp = dest.with_suffix(".zip.part")
    req = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT_TEMPLATE.format(
            contact=contact), "Accept-Encoding": "identity"})
    t0 = time.perf_counter()
    print(f"[fetch] {url}")
    try:
        with urllib.request.urlopen(req, timeout=120) as r, \
                open(tmp, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            got = 0
            while True:
                block = r.read(_CHUNK)
                if not block:
                    break
                f.write(block)
                got += len(block)
                if total and got % (32 * _CHUNK) < _CHUNK:
                    print(f"  {got / 1e6:7.0f} / {total / 1e6:.0f} MB",
                          flush=True)
    except urllib.error.HTTPError as e:            # noqa: PERF203
        tmp.unlink(missing_ok=True)
        raise SystemExit(
            f"SEC returned HTTP {e.code} for {url}. A quarter is published "
            f"a few weeks after its filing deadline; check the landing page "
            f"{NPORT_LANDING} for what exists.") from e
    tmp.replace(dest)
    meta = {
        "quarter": quarter,
        "url": url,
        "landing_page": NPORT_LANDING,
        "licence": NPORT_LICENCE,
        "bytes": dest.stat().st_size,
        "sha256": sha256_file(str(dest)),
        "downloaded_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "user_agent_contact": contact,
        "seconds": round(time.perf_counter() - t0, 1),
    }
    side.write_text(json.dumps(meta, indent=2, sort_keys=True))
    print(f"[ok] {dest} {meta['bytes'] / 1e6:.0f} MB in {meta['seconds']}s")
    print(f"     sha256 {meta['sha256']}")
    return dest, meta


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def _quantiles(x: np.ndarray, qs=(0.05, 0.25, 0.50, 0.75, 0.95)) -> str:
    v = np.quantile(np.asarray(x, dtype=float), qs)
    return "  ".join(f"{q:.0%}:{val:>8.2f}" for q, val in zip(qs, v))


def _weighted_share(uni: BondUniverse, column: str) -> pd.Series:
    df = uni.bonds.assign(w=uni.w_benchmark)
    return df.groupby(column, observed=True)["w"].sum().sort_values(
        ascending=False)


def feasibility(uni: BondUniverse, limits: CreditLimits
                ) -> tuple[bool, str, float]:
    """Is `default_limits(uni)` a feasible mandate on this universe?

    Solves a pure feasibility LP (zero objective) over exactly the constraint
    set `src/credit.py` hands the optimizer, with HiGHS on the CPU. Reported
    as a fact about the universe, not inferred from a successful optimisation
    somewhere else: a mandate that is infeasible makes every benchmark on this
    universe meaningless, so it is checked first and separately.

    Turnover is included. It measures distance from `w0`, which is
    constructed rather than real, so a turnover infeasibility would be a
    statement about the `w0` rule and the report says which limit bound.
    """
    import cvxpy as cp

    from src.credit import credit_constraint_builder

    w = cp.Variable(uni.n_bonds, nonneg=True)
    cons = [cp.sum(w) == 1.0, w <= limits.w_max]
    extra, _ = credit_constraint_builder(uni, limits)(w)
    prob = cp.Problem(cp.Minimize(0), cons + extra)
    prob.solve(solver=cp.HIGHS, verbose=False)
    ok = w.value is not None and prob.status in ("optimal",
                                                 "optimal_inaccurate")
    slack = float("nan")
    if ok:
        from src.credit import exposure_report
        rep = exposure_report(uni, np.asarray(w.value).ravel(), limits)
        slack = float(rep["rel_slack"].min())
    return bool(ok), str(prob.status), slack


def spread_sanity(uni: BondUniverse, n: int = 8) -> pd.DataFrame:
    """The tightest and widest large issuers, for an eyeball check.

    Every spread in this universe is DERIVED -- fair value over principal gives
    a price, the repo's pricer inverts it for a yield, and the quarter's own
    Treasury holdings give the curve to subtract. None of it is a reported
    spread, so the derivation needs a check that does not come from the same
    arithmetic. This is that check: the ordering of well-known borrowers by
    credit quality is public knowledge, and a derivation that gets it right
    across four hundred basis points is doing something real.

    Restricted to the 100 largest issuers so the extremes are recognisable
    names rather than one stale mark on a bond nobody holds.
    """
    df = uni.bonds.assign(w=uni.w_benchmark)
    g = df.groupby("issuer_id", observed=True).agg(
        issuer=("issuer_name", "first"),
        n_bonds=("bond_id", "size"),
        mv_bn=("market_value_usd", lambda s: s.sum() / 1e9),
        oas_bp=("oas_bp", "median"),
        duration=("mod_duration", "median"),
        rating=("rating", "first"),
        sector=("sector", "first"))
    g = g.nlargest(100, "mv_bn").sort_values("oas_bp")
    return pd.concat([g.head(n), g.tail(n)])


def compare(real: BondUniverse, synth: BondUniverse) -> pd.DataFrame:
    """Side-by-side on the characteristics the optimizer actually reads.

    This table is a paper result in its own right: it says how realistic the
    synthetic universe was. Everything is benchmark-weighted, because that is
    how the constraint set sees the universe.
    """
    rows = []

    def add(metric: str, r: float, s: float, unit: str = "") -> None:
        rows.append({"metric": metric, "real_nport": r, "synthetic": s,
                     "unit": unit,
                     "real_over_synth": (r / s if s not in (0.0,) and
                                         np.isfinite(s) else np.nan)})

    for label, fn, unit in (
        ("n_bonds", lambda u: float(u.n_bonds), "count"),
        ("n_issuers", lambda u: float(u.bonds["issuer_id"].nunique()),
         "count"),
        ("bonds_per_issuer",
         lambda u: u.n_bonds / u.bonds["issuer_id"].nunique(), "count"),
        ("bench_duration", lambda u: float(u.w_benchmark @ u.duration),
         "years"),
        ("bench_oas_bp", lambda u: float(u.w_benchmark @ u.oas_bp), "bp"),
        ("bench_dts", lambda u: float(u.w_benchmark @ u.dts), "bp-years"),
        ("bench_ytm_pct", lambda u: float(u.w_benchmark @ u.ytm) * 100.0,
         "percent"),
        ("bench_convexity", lambda u: float(u.w_benchmark @ u.convexity),
         "-"),
        ("bench_price", lambda u: float(u.w_benchmark
                                        @ u.bonds["price"].to_numpy()),
         "points"),
        ("bench_maturity", lambda u: float(u.w_benchmark
                                           @ u.bonds["maturity"].to_numpy()),
         "years"),
        ("max_bond_weight", lambda u: float(u.w_benchmark.max()), "fraction"),
        ("max_issuer_weight",
         lambda u: float(u.bonds.assign(w=u.w_benchmark).groupby(
             "issuer_id", observed=True)["w"].sum().max()), "fraction"),
        ("top10_bond_weight",
         lambda u: float(np.sort(u.w_benchmark)[-10:].sum()), "fraction"),
        ("effective_n_bonds",
         lambda u: float(1.0 / (u.w_benchmark ** 2).sum()), "count"),
        ("duration_p05", lambda u: float(np.quantile(u.duration, 0.05)),
         "years"),
        ("duration_p50", lambda u: float(np.quantile(u.duration, 0.50)),
         "years"),
        ("duration_p95", lambda u: float(np.quantile(u.duration, 0.95)),
         "years"),
        ("oas_p05", lambda u: float(np.quantile(u.oas_bp, 0.05)), "bp"),
        ("oas_p50", lambda u: float(np.quantile(u.oas_bp, 0.50)), "bp"),
        ("oas_p95", lambda u: float(np.quantile(u.oas_bp, 0.95)), "bp"),
        ("oas_log_sd", lambda u: float(np.std(np.log(u.oas_bp))), "log bp"),
        ("dts_p95", lambda u: float(np.quantile(u.dts, 0.95)), "bp-years"),
        ("w0_distance_to_bench",
         lambda u: u.one_way_distance_to_benchmark(), "one-way fraction"),
    ):
        add(label, fn(real), fn(synth), unit)

    for r in RATINGS:
        wr = _weighted_share(real, "rating")
        ws = _weighted_share(synth, "rating")
        add(f"rating_weight_{r}", float(wr.get(r, 0.0)),
            float(ws.get(r, 0.0)), "fraction")
    for c in MATURITY_CELL_NAMES:
        wr = _weighted_share(real, "maturity_cell")
        ws = _weighted_share(synth, "maturity_cell")
        add(f"cell_weight_{c}", float(wr.get(c, 0.0)),
            float(ws.get(c, 0.0)), "fraction")
    return pd.DataFrame(rows)


def report(nu, synth: BondUniverse, out_dir: Path, tag: str,
           write: bool, universe_csv: bool = True) -> None:
    uni, prov, diag = nu.universe, nu.provenance, nu.diagnostics
    b = uni.bonds
    line = "-" * 74

    print(f"\n{line}\nSOURCE\n{line}")
    print(f"quarter            {prov.quarter}   report date "
          f"{prov.report_date}")
    print(f"filings in ZIP     {prov.n_filings:,}")
    print(f"licence            {prov.licence}")
    print(f"zip sha256         {prov.zip_sha256}")
    print(f"bonds sha256       {prov.bonds_sha256}")

    print(f"\n{line}\nFILTER LEDGER (holding rows, then bonds)\n{line}")
    for k, v in prov.stage_counts.items():
        print(f"  {k:<26} {v:>12,}")

    print(f"\n{line}\nTREASURY CURVE, fitted from UST holdings in the same "
          f"quarter\n{line}")
    for k, y, n in zip(nu.treasury_knots, nu.treasury_yields,
                       nu.treasury_counts):
        print(f"  {k:>5.0f}y   {y * 100:>6.3f}%   (n={n})")

    print(f"\n{line}\nDATA-QUALITY / PROXY PANEL\n{line}")
    for k, v in diag.as_row().items():
        print(f"  {k:<38} {v:>14.6g}" if isinstance(v, float)
              else f"  {k:<38} {v:>14}")

    print(f"\n{line}\nUNIVERSE\n{line}")
    print(f"bonds {uni.n_bonds:,}   issuers "
          f"{b['issuer_id'].nunique():,}   "
          f"aggregate fund-held MV ${b['market_value_usd'].sum() / 1e9:.1f}bn")
    print(f"duration   {_quantiles(uni.duration)}")
    print(f"spread bp  {_quantiles(uni.oas_bp)}")
    print(f"ytm %      {_quantiles(uni.ytm * 100)}")
    print(f"maturity   {_quantiles(b['maturity'].to_numpy())}")
    print(f"price      {_quantiles(b['price'].to_numpy())}")

    print(f"\nrating ({b['rating_source'].iloc[0]}), benchmark-weighted:")
    wr = _weighted_share(uni, "rating")
    cr = b["rating"].value_counts()
    for r in RATINGS:
        print(f"  {r:<5} weight {float(wr.get(r, 0.0)):>7.2%}   "
              f"bonds {int(cr.get(r, 0)):>6,}")
    print(f"\nsector ({b['sector_source'].iloc[0]}), benchmark-weighted:")
    for s, v in _weighted_share(uni, "sector").head(12).items():
        print(f"  {str(s):<18} {v:>7.2%}   bonds "
              f"{int((b['sector'] == s).sum()):>6,}")

    print(f"\n{line}\nDERIVED-SPREAD SANITY: tightest and widest of the 100 "
          f"largest issuers\n{line}")
    with pd.option_context("display.width", 130,
                           "display.float_format", lambda v: f"{v:,.2f}"):
        print(spread_sanity(uni).to_string())
    ra = prov.rating_anchor
    print(f"\nrating anchor: recentred={ra['recentred']} "
          f"shift={float(ra['level_shift']):.3f}x  levels used (bp): "
          + ", ".join(f"{k}={v:.0f}"
                      for k, v in ra["levels_used_bp"].items()))

    print(f"\n{line}\nMANDATE FEASIBILITY: default_limits(uni)\n{line}")
    lim = default_limits(uni)
    print(f"  {lim.describe()}")
    ok, status, slack = feasibility(uni, lim)
    print(f"  feasible: {ok}   solver status: {status}   "
          f"min rel_slack at the feasible point: {slack:.3g}")
    print(f"  w0 one-way distance to benchmark: "
          f"{uni.one_way_distance_to_benchmark():.3f} "
          f"(turnover budget {lim.turnover:.0%})")

    print(f"\n{line}\nREAL vs SYNTHETIC (make_universe, same n_bonds)\n{line}")
    cmp_df = compare(uni, synth)
    with pd.option_context("display.width", 120,
                           "display.max_rows", 200,
                           "display.float_format", lambda v: f"{v:,.4f}"):
        print(cmp_df.to_string(index=False))

    if not write:
        print("\n[dry-run] nothing written; pass --write to emit files")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    if universe_csv:
        p_uni = out_dir / f"nport_universe_{tag}.csv"
        b.assign(w_benchmark=uni.w_benchmark, w0=uni.w0).to_csv(
            p_uni, index=False)
        written.append(p_uni)
    p_prov = out_dir / f"nport_provenance_{tag}.json"
    p_cmp = out_dir / f"nport_vs_synthetic_{tag}.csv"
    # Provenance and the honesty panel go in one file. Separating them would
    # let a universe be cited by its hash while the proxy diagnostics that
    # qualify it stayed on someone's terminal.
    payload = json.loads(prov.to_json())
    payload["diagnostics"] = diag.as_row()
    payload["mandate_feasibility"] = {
        "limits": lim.describe(),
        "feasible": bool(ok),
        "solver_status": status,
        "min_rel_slack_at_feasible_point": slack,
        "w0_one_way_distance_to_benchmark":
            uni.one_way_distance_to_benchmark(),
    }
    p_prov.write_text(json.dumps(payload, indent=2, sort_keys=True))
    cmp_df.to_csv(p_cmp, index=False)
    written += [p_prov, p_cmp]
    print("\nwrote " + "\n      ".join(str(p) for p in written))


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quarter", default="2026q1",
                    help="e.g. 2026q1 (default: %(default)s)")
    ap.add_argument("--contact", default=os.environ.get("SEC_CONTACT", ""),
                    help="contact address for the SEC User-Agent header; "
                         "required to download (env SEC_CONTACT)")
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--force", action="store_true",
                    help="re-download even if the quarter is cached")
    ap.add_argument("--no-fetch", action="store_true",
                    help="use the cached ZIP only; never touch the network")
    ap.add_argument("--build", action="store_true",
                    help="build the universe from the cached quarter")
    ap.add_argument("--write", action="store_true",
                    help="write the universe, provenance and comparison")
    ap.add_argument("--tag", default=None,
                    help="output filename stem (default: "
                         "<quarter>_n<n_bonds> or <quarter>_full)")
    ap.add_argument("--no-universe-csv", action="store_true",
                    help="write only the provenance and the comparison; the "
                         "full untrimmed universe is a few MB of CSV and the "
                         "provenance hash is what makes it reproducible")
    ap.add_argument("--n-bonds", type=int, default=None,
                    help="keep the N largest bonds by aggregate fund-held "
                         "market value (default: keep all)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--rating-scheme", default="spread_implied",
                    choices=("spread_implied", "spread_quantile"))
    ap.add_argument("--no-recentre", action="store_true",
                    help="match spreads against the published LONG-RUN "
                         "rating levels instead of re-anchoring them on this "
                         "quarter; reads a tight quarter as high-grade")
    ap.add_argument("--sector-scheme", default="keyword",
                    choices=("keyword", "country"))
    ap.add_argument("--min-funds", type=int, default=5)
    args = ap.parse_args()

    try:
        quarter_url(args.quarter)          # fail on the argument, not later
    except ValueError as e:
        raise SystemExit(str(e)) from e

    cache: Path = args.cache
    zip_path = cache / f"{args.quarter}_nport.zip"
    side = cache / f"{args.quarter}_nport.json"

    if args.no_fetch:
        if not zip_path.exists():
            raise SystemExit(f"--no-fetch but {zip_path} is not cached")
        meta = json.loads(side.read_text()) if side.exists() else {
            "downloaded_utc": "unknown"}
    else:
        if not args.contact:
            raise SystemExit(
                "--contact is required: the SEC asks automated clients to "
                "identify themselves in the User-Agent header. Pass an "
                "address you are willing to be emailed at, or set "
                "SEC_CONTACT.")
        zip_path, meta = fetch_quarter(args.quarter, cache, args.contact,
                                       force=args.force)

    if not args.build:
        print("\n(no --build: stopping after fetch)")
        return 0

    cfg = NportConfig(n_bonds=args.n_bonds,
                      rating_scheme=args.rating_scheme,
                      rating_level_recentre=not args.no_recentre,
                      sector_scheme=args.sector_scheme,
                      min_funds=args.min_funds)
    t0 = time.perf_counter()
    nu = build_universe(str(zip_path), args.quarter,
                        downloaded_utc=str(meta.get("downloaded_utc")),
                        cfg=cfg, seed=args.seed)
    print(f"[build] {nu.universe.n_bonds:,} bonds in "
          f"{time.perf_counter() - t0:.1f}s")

    synth = make_universe(n_bonds=nu.universe.n_bonds, seed=args.seed)
    tag = args.tag or (f"{args.quarter}_n{args.n_bonds}" if args.n_bonds
                       else f"{args.quarter}_full")
    report(nu, synth, args.out, tag, write=args.write,
           universe_csv=not args.no_universe_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

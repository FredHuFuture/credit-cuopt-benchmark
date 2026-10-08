"""Interactive credit Mean-CVaR re-optimization.

    streamlit run demo/app.py

The point of the whole project in one screen: move a desk limit, get a new
optimal portfolio before you have finished thinking about the last one. The
LP behind each slider move has ~10 million nonzeros.

What the numbers on screen mean, and why they are shown:

- **solve time** — the re-solve only, not the page. Cold by default: cuOpt's
  warm start requires presolve off, and presolve is worth far more here than
  warm start saves (see `src/interactive.py`). The warm path is a toggle so
  the trade can be watched rather than argued about.
- **limits respected** — a first-order solver returns a point inside its
  *residual tolerance*, which is not the same as a portfolio inside the
  mandate. At the default 1e-4 the turnover budget in particular can be
  breached by enough to matter, so the badge is live and honest: if the
  portfolio on screen is not compliant, it says so, and the tolerance control
  lets you buy compliance and watch what it costs in time.
- **out-of-sample CVaR** — measured on an independent scenario set the
  optimizer never saw. The in-sample figure is shown next to it so the
  overfit is visible.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.check import empirical_cvar  # noqa: E402
from src.credit import (  # noqa: E402
    default_limits, exposure_report, portfolio_stats, worst_violation,
)
from src.interactive import Knobs, build_interactive, solve  # noqa: E402
from src.scenarios import credit_scenarios, warmup_gpu  # noqa: E402
from src.universe import make_universe  # noqa: E402

st.set_page_config(page_title="Credit Mean-CVaR on a GPU", layout="wide")

N_BONDS = 500
N_SCEN = 20_000
ALPHA = 0.95


@st.cache_resource(show_spinner="Building the universe, scenarios and LP…")
def bootstrap():
    """Built once per session; the CUDA context lives here across reruns."""
    uni = make_universe(n_bonds=N_BONDS, seed=7)
    warmup_gpu(uni)
    R, mu, info = credit_scenarios(uni, n_scen=N_SCEN, seed=11, use_gpu=True)
    R_oos, _, _ = credit_scenarios(uni, n_scen=N_SCEN, seed=101, use_gpu=True)
    limits = default_limits(uni)
    bench_dts = float(uni.w_benchmark @ uni.dts)
    k0 = Knobs(alpha=ALPHA,
               sector_cap=float(max(limits.sector_caps.values())),
               issuer_cap=limits.issuer_cap,
               dts_min_mult=limits.dts_min / bench_dts,
               dts_max_mult=limits.dts_max / bench_dts,
               turnover=limits.turnover)
    model = build_interactive(uni, R, mu, limits, k0)
    return uni, R, R_oos, mu, limits, model, info, bench_dts


uni, R, R_oos, mu, limits, model, sinfo, bench_dts = bootstrap()

st.title("Credit Mean-CVaR, re-optimized while you move the slider")
st.caption(
    f"{uni.n_bonds} corporate bonds · {N_SCEN:,} monthly scenarios with "
    f"issuer-level defaults · LP with {model.A.nnz:,} nonzeros · "
    f"NVIDIA cuOpt PDLP on the GPU · excess return over duration-matched "
    f"Treasuries"
)

with st.sidebar:
    st.header("Desk limits")
    risk_aversion = st.slider("Risk aversion λ", 0.5, 10.0, 3.0, 0.5,
                              help="Weight on CVaR in the objective.")
    alpha = st.select_slider("CVaR confidence α", [0.90, 0.95, 0.975, 0.99],
                             value=ALPHA)
    st.divider()
    issuer_cap = st.slider("Max issuer weight", 0.005, 0.06,
                           float(limits.issuer_cap), 0.005, format="%.3f")
    sector_cap = st.slider("Max sector weight", 0.10, 0.60,
                           float(max(limits.sector_caps.values())), 0.01)
    dts_min_mult = st.slider("DTS floor (× benchmark)", 0.50, 1.00, 0.90, 0.05,
                             help="Stops the optimizer from simply "
                                  "de-risking to zero credit beta.")
    dts_max_mult = st.slider("DTS cap (× benchmark)", 1.00, 1.50, 1.10, 0.05)
    turnover = st.slider("Turnover budget (one-way)", 0.02, 0.60, 0.25, 0.01)
    st.divider()
    tol = st.select_slider(
        "Solver tolerance", [1e-4, 1e-5, 1e-6], value=1e-4,
        format_func=lambda v: f"{v:.0e}",
        help="Looser is faster. It is also how a limit breach gets in.")
    use_warm = st.toggle(
        "Warm start (disables presolve)", value=False,
        help="Reuse the previous iterate. cuOpt requires presolve off for "
             "this, and presolve is worth more than the warm start saves on "
             "this problem — try it and watch the clock.")

knobs = Knobs(risk_aversion=risk_aversion, alpha=alpha,
              sector_cap=sector_cap, issuer_cap=issuer_cap,
              dts_min_mult=dts_min_mult, dts_max_mult=dts_max_mult,
              turnover=turnover)

res = solve(model, knobs, warm=use_warm, tol=tol)
w = res["w"]

if w is None:
    st.error(f"Solver returned {res['status']} — try loosening a limit.")
    st.stop()

rep = exposure_report(uni, w, limits)
viol, which = worst_violation(rep)
stats = portfolio_stats(uni, w)
cvar_in, _ = empirical_cvar(-(R @ w), alpha)
cvar_oos, var_oos = empirical_cvar(-(R_oos @ w), alpha)
bench_oos, _ = empirical_cvar(-(R_oos @ uni.w_benchmark), alpha)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Re-solve", f"{res['seconds']:.2f}s",
          "warm, no presolve" if res["warm"] else "cold + presolve",
          delta_color="off")
c2.metric(f"CVaR{alpha:.0%} out-of-sample", f"{cvar_oos:.2%}",
          f"{cvar_oos / bench_oos - 1:+.1%} vs benchmark",
          delta_color="inverse")
c3.metric("Expected excess (monthly)", f"{float(mu @ w):.4%}")
c4.metric("Limits", "respected" if viol <= 1e-6 else "BREACH",
          "-" if viol <= 1e-6 else f"{which} by {viol:.1%}",
          delta_color="inverse" if viol > 1e-6 else "off")

if viol > 1e-6:
    st.warning(
        f"The solver stopped inside its {tol:.0e} residual tolerance but the "
        f"portfolio breaches **{which}** by {viol:.2%} of that limit. This is "
        f"the honest cost of a loose tolerance — tighten it in the sidebar "
        f"and watch the solve time change.")

left, right = st.columns([3, 2])

with left:
    st.subheader("Rating exposure vs the index")
    tilt = pd.DataFrame({
        "optimal": uni.bonds.assign(x=w).groupby(
            "rating", observed=True)["x"].sum(),
        "benchmark": uni.bonds.assign(x=uni.w_benchmark).groupby(
            "rating", observed=True)["x"].sum(),
    }).reindex(["AAA", "AA", "A", "BBB", "BB", "B"]).dropna(how="all")
    st.bar_chart(tilt, height=260)

    st.subheader("Where the limits bind")
    binding = rep[rep["binding"]][["family", "item", "portfolio",
                                   "benchmark", "limit"]]
    st.dataframe(binding, hide_index=True, use_container_width=True)
    if binding.empty:
        st.caption("Nothing binding — the mandate is looser than the "
                   "optimizer needs.")

with right:
    st.subheader("Portfolio")
    st.dataframe(pd.DataFrame({
        "": ["OAS (bp)", "Duration (y)", "DTS (bp-y)", "Positions",
             "Issuers", "Max weight", f"CVaR{alpha:.0%} in-sample",
             f"CVaR{alpha:.0%} out-of-sample", f"VaR{alpha:.0%} out-of-sample"],
        "portfolio": [f"{stats['oas_bp']:.0f}", f"{stats['duration']:.2f}",
                      f"{stats['dts_bp_years']:,.0f}",
                      f"{stats['n_positions']}", f"{stats['n_issuers']}",
                      f"{stats['max_weight']:.2%}", f"{cvar_in:.2%}",
                      f"{cvar_oos:.2%}", f"{var_oos:.2%}"],
        "benchmark": [f"{uni.w_benchmark @ uni.oas_bp:.0f}",
                      f"{uni.w_benchmark @ uni.duration:.2f}",
                      f"{bench_dts:,.0f}", f"{uni.n_bonds}",
                      f"{uni.bonds['issuer_id'].nunique()}",
                      f"{uni.w_benchmark.max():.2%}", "—",
                      f"{bench_oos:.2%}", "—"],
    }), hide_index=True, use_container_width=True)

    st.subheader("Largest active positions")
    active = (uni.bonds.assign(active=w - uni.w_benchmark)
              .reindex(columns=["bond_id", "issuer_id", "rating", "sector",
                                "maturity", "oas_bp", "active"])
              .sort_values("active", key=abs, ascending=False).head(10))
    st.dataframe(active.style.format({"active": "{:+.3%}",
                                      "oas_bp": "{:.0f}",
                                      "maturity": "{:.1f}"}),
                 hide_index=True, use_container_width=True)

with st.expander("Warm vs cold, measured on this problem right now"):
    if st.button("Run A/B (4 solves)"):
        warm_t, cold_t = [], []
        prog = st.progress(0.0)
        for i in range(2):
            kk = Knobs(**{**knobs.__dict__,
                          "risk_aversion": risk_aversion * (1 + 0.1 * (i + 1))})
            warm_t.append(solve(model, kk, warm=True, tol=tol)["seconds"])
            cold_t.append(solve(model, kk, warm=False, tol=tol)["seconds"])
            prog.progress((i + 1) / 2)
        wm, cm = float(np.median(warm_t)), float(np.median(cold_t))
        st.write(f"warm median **{wm:.2f}s** · cold median **{cm:.2f}s** · "
                 f"warm start saves **{100 * (1 - wm / cm):.0f}%**"
                 if cm > 0 else "")
        st.caption("Two knob values, each solved warm and cold, medians "
                   "reported. Single solves on this problem vary by tens of "
                   "percent, so a single pair would not mean much.")

st.caption(
    "Fully synthetic universe and scenarios (seeded, reproducible). Risk "
    "figures out-of-sample unless labelled otherwise. Not investment advice."
)

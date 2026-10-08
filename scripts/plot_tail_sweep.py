#!/usr/bin/env python
"""Plot the M9 tail dose-response from its committed CSV.

The referee objection this figure exists to kill is one sentence long:
*you assumed fat tails and then discovered fat tails.* The sweep answers it
by taking the two tail knobs apart, so the figure has to carry three things
and nothing else.

1. **A validation anchor.** With Gaussian factors and defaults switched off
   the surrogate must deliver exactly what it promises, and the residual
   above 1.0 has to be shown shrinking with the scenario count rather than
   argued away. That is in-sample selection by the optimizer over N weights
   against one scenario draw, not a mechanism. Panel C.
2. **The separation at MATCHED tail parameter.** At every single value of
   `t_dof` - including `t_dof = 1e6`, where the systematic factors are
   Gaussian to six decimal places - switching issuer defaults on moves the
   overshoot by a factor of roughly three. Sweeping `t_dof` with defaults
   off never gets past 1.3x even at `t_dof = 2.5`, which is far outside
   anything a credit desk would fit. Panel A. This is paired mechanism
   evidence that the default channel matters in this generator; it is not a
   structural causal attribution for credit portfolios in general.
3. **A dose-response indexed on something measurable.** `pd_mult` is a knob
   inside this repo's generator and nobody can read it off a real book. The
   REALIZED default rate can be read off any credit book's history, so that
   is the x axis of panel B, and the points are individual scenario draws
   rather than cell medians. Two of the three draws at `pd_mult = 0.01`
   realized no defaults at all and land exactly on top of the no-jump point;
   the third realized one and doubled the overshoot. A cell median would
   have hidden the cleanest evidence in the CSV.

Panel A is indexed on `t_dof` and not on a measurable, deliberately: it is a
controlled A/B in which `t_dof` is the CONTROLLED variable, and plotting a
controlled comparison against a downstream measurement would destroy the
pairing that makes it a controlled comparison. The measured consequence of
those same points is what panel B is for.

The convention trap this figure would otherwise fall into, stated once:
`src/credit.py` already multiplies MAD by `sqrt(pi/2)` both in the LP
constraint and in `tracking_error_bp`, so `te_mad_implied_bp` is a
sigma-scaled quantity. Under a Gaussian the ratio of `te_sigma_bp` to it is
therefore 1.0 and NOT 1.2533, and the y = 1 reference line is the Gaussian
prediction with no further conversion. Applying sqrt(pi/2) again here would
put the whole curve 25% too low.

Authored at the paper's text width so type reaches the page at the size it
was specified in; see PAPER_TEXTWIDTH_IN.

Reads committed results only; no solve, no GPU.

    python scripts/plot_tail_sweep.py --csv results/m9_tail_sweep.csv \
        --out results/m9_tail_sweep.png
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
# Type 42 (TrueType) rather than matplotlib's default Type 3: arXiv flags
# Type 3 fonts and this figure is headed for a preprint.
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402
from matplotlib.lines import Line2D             # noqa: E402
from matplotlib.ticker import (                 # noqa: E402
    FixedLocator, FuncFormatter, NullLocator,
)

#: Width of the text block in paper/main.tex, in inches: a4paper (210mm)
#: with geometry margin=1in, so 210mm - 2in = 6.2677in = 452.97 TeX pt. A
#: figure authored at this width and placed at \linewidth is scaled by
#: exactly 1.0, which is the only way a font size specified here is the font
#: size that reaches the page. Authoring wider and letting \linewidth shrink
#: it is how the 7.8pt notes on the previous version of this repo's lead
#: figure arrived at 3.6pt.
PAPER_TEXTWIDTH_IN = 6.2677

#: Total figure height. Chosen so figure + caption clears the a4 text block
#: (9.69in) as a single float rather than overflowing onto a float page.
DEFAULT_HEIGHT_IN = 7.9

CONVERGED = ("optimal", "optimal_inaccurate")

#: Below this many degrees of freedom the Student-t has no fourth moment, so
#: the sample excess kurtosis at those grid points is a finite random number
#: rather than an estimate of a population quantity that exists. Those points
#: are still measurements of the SAMPLE the optimizer actually saw - which is
#: the honest thing to plot - so they are drawn, and the region is shaded
#: instead of the markers being altered: the marker channel is already
#: carrying the arm, and overloading it here made the previous version of
#: this figure unreadable.
FINITE_KURTOSIS_DOF = 4.0

#: Panel A's series. Colour is the redundant channel, never the only one:
#: each series carries its own marker, dash pattern and line weight, so the
#: figure survives a monochrome printer. Palette follows
#: scripts/plot_te_mandate.py.
OFF_C, ON_C = "#1565c0", "#c62828"
JUMP_C = "#6a1b9a"

#: Budget -> (line weight, marker face, dash). The tight budget is the
#: heavier line because it is the M6 headline and the larger effect.
BUDGET_STYLE = {
    "tight": dict(lw=1.9, ms=5.6, mfc_off=OFF_C, mfc_on=ON_C,
                  ls_off="-", ls_on="--", alpha=1.0),
    "loose": dict(lw=1.2, ms=5.2, mfc_off="white", mfc_on="white",
                  ls_off=(0, (1, 1.6)), ls_on=(0, (4, 1.4, 1, 1.4)),
                  alpha=0.9),
}

CONTROL_ARM = "gaussian-control"
CROSS_CHECK_ARM = "cross-check"

# Type sizes. Every one of these is the size the reader gets on the printed
# page, because the figure is authored at PAPER_TEXTWIDTH_IN.
FS_SUP, FS_SUB = 10.5, 7.6
FS_TITLE, FS_LABEL, FS_TICK = 9.0, 7.8, 7.4
FS_LEGEND, FS_ANNOT, FS_NOTE = 7.2, 7.0, 6.9


def read_env(path: Path) -> dict[str, str]:
    """Parse the '# key: value' header `src.bench.write_csv` prepends."""
    env: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line.startswith("#"):
            break
        if ":" in line:
            k, v = line[1:].split(":", 1)
            env[k.strip()] = v.strip()
    return env


def load(path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    env = read_env(path)
    df = pd.read_csv(path, comment="#")
    for col in ("t_dof", "pd_mult", "te_budget_bp", "overshoot_x",
                "ref_active_exkurt", "opt_active_exkurt",
                "sigma_over_mad_ref", "ref_active_hill_alpha",
                "te_sigma_bp", "te_mad_implied_bp", "wall_s",
                "realized_default_rate", "n_scenarios"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["status"] = df["status"].astype(str)
    df["binding"] = df["binding"].astype(str).str.lower().isin(("true", "1"))
    df["converged"] = df["status"].isin(CONVERGED)
    # Only a converged solve on a BINDING budget measures the surrogate. A
    # slack budget measures whichever other limit bound instead, and drawing
    # it would put a point on the curve that the surrogate never produced.
    df["usable"] = df["converged"] & df["binding"]
    # The generator's default rate is per bond over one holding period; the
    # same linear scaling it uses internally (pd_h = pd_annual * horizon)
    # inverts it, so no approximation beyond the model's own is introduced.
    ppy = float(env.get("periods_per_year", 12.0))
    df["def_rate_pct"] = df["realized_default_rate"] * ppy * 100.0
    return df, env


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Median over replicates, with the observed min/max kept as the range.

    Median rather than mean: the default-jump tail is a rare-event statistic
    and one unlucky draw moves a mean much further than it moves the thing
    being estimated. The full range is carried alongside so the figure can
    draw it rather than imply a precision three draws do not have.
    """
    keys = ["arm", "point", "t_dof", "pd_mult", "n_scenarios", "te_budget_bp"]
    return (df[df["usable"]].groupby(keys, dropna=False)
            .agg(n=("overshoot_x", "size"),
                 y=("overshoot_x", "median"),
                 y_lo=("overshoot_x", "min"),
                 y_hi=("overshoot_x", "max"),
                 exkurt=("ref_active_exkurt", "median"),
                 exkurt_opt=("opt_active_exkurt", "median"),
                 hill=("ref_active_hill_alpha", "median"),
                 ratio_ref=("sigma_over_mad_ref", "median"),
                 defrate=("def_rate_pct", "median"),
                 defrate_lo=("def_rate_pct", "min"),
                 defrate_hi=("def_rate_pct", "max"))
            .reset_index())


def _log_yaxis(ax, lo: float, hi: float) -> None:
    """Shared y treatment: log scale, hand-picked ticks, 'x' suffix."""
    ax.set_yscale("log")
    ax.set_ylim(lo, hi)
    ticks = [t for t in (1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0)
             if lo <= t <= hi]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}x"))
    ax.yaxis.set_minor_locator(NullLocator())


def _dof_label(dof: float) -> str:
    return "1e6" if dof >= 1e5 else f"{dof:g}"


# --------------------------------------------------------------------------
# Panel A: the controlled comparison
# --------------------------------------------------------------------------

def draw_matched(ax, summary: pd.DataFrame, budgets: list[float],
                 repo_dof: float) -> list:
    """Overshoot at matched t_dof, defaults off vs on. Returns handles.

    The x axis is CATEGORICAL in t_dof rather than logarithmic in it. A log
    axis spanning 2.5 to 1e6 pushes seven of the eight grid points into the
    left tenth of the panel, which turns the one comparison the figure is
    for into a smudge. Evenly spaced columns, ordered Gaussian-first, put
    equal ink on every paired comparison - which is what a paired comparison
    is entitled to.
    """
    off = summary[summary["arm"].eq("t-sweep-nojump")]
    on = summary[summary["arm"].eq("t-sweep-repo")]
    dofs = sorted(set(off["t_dof"]) & set(on["t_dof"]), reverse=True)
    pos = {d: i for i, d in enumerate(dofs)}
    tight, loose = min(budgets), max(budgets)

    # Where the Student-t stops having a fourth moment. Shaded rather than
    # marked per point: see FINITE_KURTOSIS_DOF.
    no_moment = [pos[d] for d in dofs if d <= FINITE_KURTOSIS_DOF]
    if no_moment:
        ax.axvspan(min(no_moment) - 0.5, max(no_moment) + 0.5,
                   color="0.93", zorder=0)
        # Top of the shaded band: the only strip of this panel that stays
        # empty at every budget, so the caveat never lands on a curve.
        ax.text(0.5 * (min(no_moment) + max(no_moment)), 0.965,
                "Student-t has no fourth moment here",
                transform=ax.get_xaxis_transform(),
                fontsize=FS_ANNOT - 0.4, color="0.42", ha="center",
                va="top", zorder=1)
    if repo_dof in pos:
        ax.axvline(pos[repo_dof], color="0.55", lw=0.9, ls=(0, (2, 2)),
                   zorder=1)

    ax.axhline(1.0, color="0.20", lw=1.3, ls=(0, (7, 3)), zorder=2)

    def series(block: pd.DataFrame, bp: float) -> pd.DataFrame:
        b = block[np.isclose(block["te_budget_bp"], bp)].copy()
        b["xpos"] = b["t_dof"].map(pos)
        return b.dropna(subset=["xpos"]).sort_values("xpos")

    # The band between the two tight-budget curves IS the jump contribution;
    # filling it says so without a sentence.
    a, b = series(off, tight), series(on, tight)
    ax.fill_between(a["xpos"], a["y"], b["y"], color="#e8b4b4", alpha=0.32,
                    zorder=2, lw=0)

    handles = [Line2D([], [], color="0.20", lw=1.3, ls=(0, (7, 3)),
                      label="Gaussian prediction, 1.0x")]
    for bp, tag in ((tight, "tight"), (loose, "loose")):
        st = BUDGET_STYLE[tag]
        for block, colour, key, ls_key, name in (
                (off, OFF_C, "mfc_off", "ls_off", "defaults OFF"),
                (on, ON_C, "mfc_on", "ls_on", "defaults ON")):
            s = series(block, bp)
            if s.empty:
                continue
            marker = "o" if colour == OFF_C else "s"
            ax.plot(s["xpos"], s["y"], color=colour, lw=st["lw"],
                    ls=st[ls_key], marker=marker, ms=st["ms"],
                    mfc=st[key], mec=colour, mew=1.2, alpha=st["alpha"],
                    zorder=4)
            if tag == "tight":
                # Only the tight budget carries bars. At 50bp the whole
                # series lives inside a 0.9x band and its bars would cross
                # the tight-budget curves without adding a fact.
                ax.errorbar(s["xpos"], s["y"],
                            yerr=[s["y"] - s["y_lo"], s["y_hi"] - s["y"]],
                            fmt="none", ecolor=colour, elinewidth=0.8,
                            capsize=1.8, alpha=0.5, zorder=3)
            handles.append(Line2D([], [], color=colour, lw=st["lw"],
                                  ls=st[ls_key], marker=marker,
                                  ms=st["ms"], mfc=st[key], mec=colour,
                                  label=f"{name}, {bp:g}bp"))

    # One callout, not three. The two facts that answer the referee are the
    # size of the gap where the Student-t is switched off entirely and the
    # size of the whole Student-t effect; putting them in one box in the
    # only large empty region of the panel beats three boxes hunting for
    # gaps between four curves.
    g_off = a[a["xpos"].eq(0)]
    g_on = b[b["xpos"].eq(0)]
    fat = a[a["xpos"].eq(len(dofs) - 1)]
    if not (g_off.empty or g_on.empty or fat.empty):
        y0, y1 = float(g_off["y"].iloc[0]), float(g_on["y"].iloc[0])
        ax.annotate("", xy=(0.28, y1), xytext=(0.28, y0),
                    arrowprops=dict(arrowstyle="<->", color="0.15", lw=1.1,
                                    shrinkA=0, shrinkB=0), zorder=6)
        ax.text(0.020, 0.965,
                f"Gaussian factors (t_dof = 1e6), {tight:g}bp budget:\n"
                f"defaults off {y0:.2f}x, defaults on {y1:.2f}x.\n"
                f"Sweeping the Student-t instead, defaults off,\n"
                f"never passes {float(fat['y'].iloc[0]):.2f}x - and that is "
                f"at t_dof = {_dof_label(dofs[-1])}.",
                transform=ax.transAxes, fontsize=FS_ANNOT, color="0.10",
                ha="left", va="top", zorder=7, linespacing=1.35,
                bbox=dict(boxstyle="round,pad=0.30", fc="white", ec="0.72",
                          lw=0.7))

    ax.set_xlim(-0.55, len(dofs) - 0.45)
    ax.set_xticks(list(range(len(dofs))))
    ax.set_xticklabels([_dof_label(d) for d in dofs], fontsize=FS_TICK)
    if repo_dof in pos:
        ax.get_xticklabels()[pos[repo_dof]].set_color("0.10")
        ax.get_xticklabels()[pos[repo_dof]].set_fontweight("bold")
    _log_yaxis(ax, 0.96, float(summary["y_hi"].max()) * 1.55)
    ax.tick_params(labelsize=FS_TICK)
    ax.set_xlabel("Student-t degrees of freedom of the systematic factors\n"
                  "(fatter tails to the right; bold = the value this repo "
                  "ships)", fontsize=FS_LABEL)
    ax.set_ylabel("realized sigma-TE / requested budget", fontsize=FS_LABEL)
    ax.set_title("A.  Same tail parameter, defaults off vs on: paired "
                 "default-channel evidence", fontsize=FS_TITLE, loc="left")
    ax.grid(True, which="major", axis="y", alpha=0.22)
    return handles


# --------------------------------------------------------------------------
# Panel B: the dose-response, on a quantity somebody can measure
# --------------------------------------------------------------------------

def draw_dose(ax, df: pd.DataFrame, summary: pd.DataFrame, budget: float,
              repo_dof: float) -> None:
    """Overshoot against the REALIZED default rate, one marker per draw.

    Per draw and not per cell, because the collision that a cell median
    would hide is the finding: at `pd_mult = 0.01` two of three draws
    realized no default at all and reproduce the no-jump point to four
    decimals, and the one that realized a single default doubled the
    overshoot. Nothing about `pd_mult` predicts a draw's overshoot; the
    defaults it actually realized do.
    """
    at_repo_dof = (np.isclose(df["t_dof"], repo_dof)
                   & np.isclose(df["te_budget_bp"], budget) & df["usable"])
    raw = df[at_repo_dof & df["arm"].isin(
        ("t-sweep-nojump", "jump-sweep", "t-sweep-repo"))]
    cells = summary[np.isclose(summary["t_dof"], repo_dof)
                    & np.isclose(summary["te_budget_bp"], budget)
                    & summary["arm"].isin(
                        ("t-sweep-nojump", "jump-sweep", "t-sweep-repo"))]
    cells = cells.sort_values("pd_mult")
    if raw.empty or cells.empty:
        return

    ax.axhline(1.0, color="0.20", lw=1.3, ls=(0, (7, 3)), zorder=2)
    ax.plot(cells["defrate"], cells["y"], color=JUMP_C, lw=1.6, ls=":",
            zorder=3)

    def colour_of(arm: str) -> str:
        return {"t-sweep-nojump": OFF_C, "t-sweep-repo": ON_C}.get(arm,
                                                                   JUMP_C)

    def marker_of(arm: str) -> str:
        return {"t-sweep-nojump": "o", "t-sweep-repo": "s"}.get(arm, "^")

    for _, r in raw.iterrows():
        ax.plot([r["def_rate_pct"]], [r["overshoot_x"]],
                marker=marker_of(r["arm"]), ms=4.2, mfc="white",
                mec=colour_of(r["arm"]), mew=1.0, ls="none", zorder=4)
    for _, r in cells.iterrows():
        ax.plot([r["defrate"]], [r["y"]], marker=marker_of(r["arm"]),
                ms=6.4, color=colour_of(r["arm"]), zorder=5)

    # Measured excess kurtosis of the same series, so the OTHER measurable
    # the sweep records is on the figure and not only in the CSV, and the
    # empirical literature (which reports kurtosis, not pd multipliers) has
    # an axis to be placed against. Cell medians only: per draw it is a
    # rare-event statistic ranging over three orders of magnitude. Labels go
    # directly below their marker, the one direction that stays clear of a
    # curve rising left to right.
    # Below the marker for every point but the last, which is ringed and has
    # the callout arrow coming into it from below; that one goes up-left,
    # into the only gap its neighbours leave.
    offsets = {0.1: (0, -9, "center", "top"), 0.3: (0, -9, "center", "top"),
               1.0: (-11, 4, "right", "bottom")}
    for _, r in cells[cells["pd_mult"].isin(offsets)].iterrows():
        dx, dy, ha, va = offsets[float(r["pd_mult"])]
        ax.annotate(f"kurt {r['exkurt']:.2g}",
                    xy=(r["defrate"], r["y"]), xytext=(dx, dy),
                    textcoords="offset points", fontsize=FS_ANNOT - 0.7,
                    color="0.40", ha=ha, va=va, zorder=6,
                    bbox=dict(boxstyle="square,pad=0.10", fc="white",
                              ec="none", alpha=0.85))

    shipped = cells[np.isclose(cells["pd_mult"], 1.0)]
    if not shipped.empty:
        s = shipped.iloc[0]
        ax.plot([s["defrate"]], [s["y"]], marker="s", ms=12, mfc="none",
                mec="0.10", mew=1.2, zorder=6)
        # Anchored in the lower-right corner, the only quadrant of this
        # panel a rising dose-response leaves empty.
        ax.annotate("the generator\nas shipped",
                    xy=(s["defrate"], s["y"]), xytext=(0.975, 0.035),
                    textcoords="axes fraction", fontsize=FS_ANNOT,
                    color="0.10", ha="right", va="bottom", zorder=7,
                    linespacing=1.3,
                    arrowprops=dict(arrowstyle="->", color="0.40", lw=0.9,
                                    connectionstyle="arc3,rad=0.15"),
                    bbox=dict(boxstyle="round,pad=0.22", fc="#f2f2f2",
                              ec="0.55", lw=0.7))

    zero = raw[raw["def_rate_pct"].le(0.0)]
    if not zero.empty:
        # No arrow: the cluster is the leftmost thing in the panel and an
        # arrow to it would cross every marker on the way.
        ax.text(0.03, 0.965,
                f"{len(zero)} of {len(raw)} draws realized no default;\n"
                f"every one lands at "
                f"{float(zero['overshoot_x'].median()):.2f}x.",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=FS_ANNOT, color="0.10", linespacing=1.35, zorder=7,
                bbox=dict(boxstyle="round,pad=0.24", fc="white", ec="0.72",
                          lw=0.7))

    hi_x = float(raw["def_rate_pct"].max())
    ax.set_xscale("symlog", linthresh=0.02, linscale=0.45)
    ax.set_xlim(-0.0045, hi_x * 1.7)
    xt = [0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    ax.xaxis.set_major_locator(FixedLocator([t for t in xt
                                             if t <= hi_x * 1.7]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_locator(NullLocator())
    _log_yaxis(ax, 0.955, float(raw["overshoot_x"].max()) * 1.60)
    ax.tick_params(labelsize=FS_TICK)
    ax.set_xlabel("realized default rate of that draw\n"
                  "(% of names per year, annualized)", fontsize=FS_LABEL)
    ax.set_ylabel("realized sigma-TE / budget", fontsize=FS_LABEL)
    ax.set_title(f"B.  Dose-response at the {budget:g}bp budget\n"
                 f"t_dof fixed at {repo_dof:g}, one small marker per draw",
                 fontsize=FS_TITLE, loc="left")
    ax.grid(True, which="major", alpha=0.22)


# --------------------------------------------------------------------------
# Panel C: the harness check
# --------------------------------------------------------------------------

def draw_control(ax, ctrl: pd.DataFrame) -> None:
    """The validation panel: the Gaussian control must go to 1.0."""
    ctrl = ctrl.sort_values("n_scenarios")
    ax.axhline(1.0, color="0.20", lw=1.3, ls=(0, (7, 3)), zorder=2)
    ax.plot(ctrl["n_scenarios"], ctrl["y"], color=OFF_C, marker="o",
            ms=5.4, lw=1.7, ls="-", zorder=4)
    ax.plot(ctrl["n_scenarios"], ctrl["ratio_ref"], color="0.35", marker="D",
            ms=4.4, lw=1.2, ls="-.", zorder=4)
    for _, r in ctrl.iterrows():
        ax.annotate(f"{r['y']:.3f}", xy=(r["n_scenarios"], r["y"]),
                    xytext=(0, 6), textcoords="offset points",
                    fontsize=FS_ANNOT - 0.4, ha="center", color="#0d47a1")
    ax.set_xscale("log")
    ax.set_xticks(list(ctrl["n_scenarios"]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.xaxis.set_minor_locator(NullLocator())
    # Padding on both axes: the per-point value labels sit outside the data
    # range and would otherwise be clipped by the spine.
    xs = ctrl["n_scenarios"].to_numpy(dtype=float)
    ax.set_xlim(xs.min() * 0.74, xs.max() * 1.34)
    vals = np.concatenate([ctrl["y"].to_numpy(dtype=float),
                           ctrl["ratio_ref"].to_numpy(dtype=float), [1.0]])
    pad = 0.16 * (vals.max() - vals.min() + 1e-6)
    ax.set_ylim(vals.min() - pad, vals.max() + 2.2 * pad)
    ax.tick_params(labelsize=FS_TICK)
    ax.set_xlabel("scenarios in the draw (S)", fontsize=FS_LABEL)
    ax.set_ylabel("sigma / MAD-implied", fontsize=FS_LABEL)
    ax.set_title("C.  Harness check: Gaussian world\n"
                 "the residual is selection, and it shrinks",
                 fontsize=FS_TITLE, loc="left")
    ax.grid(True, which="major", alpha=0.22)
    ax.legend(handles=[
        Line2D([], [], color=OFF_C, marker="o", ms=5.4, lw=1.7,
               label="optimized portfolio"),
        Line2D([], [], color="0.35", marker="D", ms=4.4, lw=1.2, ls="-.",
               label="incumbent book (fixed)"),
        Line2D([], [], color="0.20", lw=1.3, ls=(0, (7, 3)),
               label="exact answer, 1.0")],
        fontsize=FS_LEGEND - 0.3, loc="upper right", framealpha=0.95,
        borderpad=0.4, labelspacing=0.32, handlelength=2.2)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Plot the M9 tail dose-response from its committed CSV.")
    ap.add_argument("--csv", default="results/m9_tail_sweep.csv")
    ap.add_argument("--out", default="results/m9_tail_sweep.png")
    ap.add_argument("--out-pdf", default="",
                    help="vector output (default: --out with .pdf)")
    ap.add_argument("--dpi", type=float, default=150.0)
    ap.add_argument("--width-in", type=float, default=PAPER_TEXTWIDTH_IN,
                    help="figure width in inches; the default is the "
                         "paper's text width, which is what makes the type "
                         "sizes in this script the type sizes on the page")
    ap.add_argument("--height-in", type=float, default=DEFAULT_HEIGHT_IN)
    args = ap.parse_args()

    path = Path(args.csv)
    if not path.exists():
        print(f"missing {path} - run scripts/run_tail_sweep.py first")
        return 1
    df, env = load(path)
    summary = summarize(df)
    if summary.empty:
        print("no converged, binding rows in the CSV - nothing to plot")
        return 1

    repo_dof = float(env.get("repo_operating_dof", 5.0))
    curve = summary[summary["arm"].isin(
        ("t-sweep-nojump", "t-sweep-repo", "jump-sweep"))]
    ctrl = summary[summary["arm"].eq(CONTROL_ARM)]
    budgets = sorted(curve["te_budget_bp"].unique())

    fig = plt.figure(figsize=(args.width_in, args.height_in))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.12, 1.0],
                          left=0.108, right=0.988, top=0.858, bottom=0.228,
                          hspace=0.46, wspace=0.34)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    handles = draw_matched(ax_a, curve, budgets, repo_dof)
    # The one series panel B introduces that panel A does not have. Named in
    # the figure legend rather than in panel B's title so the whole
    # marker vocabulary is defined in exactly one place.
    handles.append(Line2D([], [], color=JUMP_C, marker="^", ms=5.6, lw=1.6,
                          ls=":", label="pd multiplier swept (panel B)"))
    draw_dose(ax_b, df, summary, min(budgets), repo_dof)
    if not ctrl.empty:
        draw_control(ax_c, ctrl)
    else:
        ax_c.set_visible(False)

    size = (f"{env.get('n_bonds', '?')} bonds x "
            f"{int(float(env.get('n_scenarios', 0))):,} scenarios, "
            f"{env.get('solver', '?')} on CPU, "
            f"{len(env.get('scenario_seeds', '').split(','))} scenario "
            f"draws per grid point")
    fig.suptitle("MAD delivery under paired continuous-tail and default "
                 "controls", fontsize=FS_SUP, x=0.008, ha="left",
                 y=0.990)
    fig.text(0.008, 0.962, size + f"   ({path.name})", fontsize=FS_SUB,
             color="0.30", ha="left", va="top")
    if handles:
        fig.legend(handles=handles, loc="upper left",
                   bbox_to_anchor=(0.006, 0.950), ncol=3,
                   fontsize=FS_LEGEND, frameon=False, handlelength=2.4,
                   columnspacing=1.4, labelspacing=0.30)

    # Two figure-level notes. Stacked rather than side by side: at one text
    # column there is no second column to put the right-hand one in, and
    # wrapping is derived from the actual figure width so the same code
    # produces a legible page at any --width-in.
    wrap = max(60, int(args.width_in * 20.5))
    note_a = textwrap.fill(
        "Reading rule, applied rather than assumed: a point enters the "
        "figure only if the solve converged AND the MAD budget actually "
        f"bound (within {float(env.get('binding_rel_slack', 0.05)):.0%} of "
        f"it). All {len(df)} rows in this CSV qualify. Bars in panel A are "
        "the full min-max range over three scenario draws, not a standard "
        "error: with jumps on, the overshoot is a rare-event statistic and "
        "three draws do not estimate its spread. Shaded columns are "
        f"t_dof <= {FINITE_KURTOSIS_DOF:g}, where the Student-t has no "
        "fourth moment.", width=wrap)
    note_b = textwrap.fill(
        "The Gaussian reference is a population identity: the MAD constraint "
        "already includes sqrt(pi/2), with the same annualization as sigma. "
        "Finite samples and portfolio selection can move the ratio above "
        "1 even under a Gaussian generator, as panel C shows. Excess above "
        "1 is therefore not by itself a normality test or a unit conversion.",
        width=wrap)
    fig.text(0.008, 0.012, note_a + "\n\n" + note_b, fontsize=FS_NOTE,
             color="0.30", va="bottom", ha="left", linespacing=1.45)

    out_png = Path(args.out)
    out_pdf = (Path(args.out_pdf) if args.out_pdf
               else out_png.with_suffix(".pdf"))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=args.dpi)
    fig.savefig(out_pdf)
    print(f"wrote {out_png}  ({args.width_in:.4g} x {args.height_in:.4g} in)")
    print(f"wrote {out_pdf}")

    print(f"\n{path.name}: {size}")
    print("\nPanel A - matched t_dof, median over draws:")
    piv = (curve[curve["arm"].isin(("t-sweep-nojump", "t-sweep-repo"))]
           .pivot_table(index=["te_budget_bp", "t_dof"], columns="arm",
                        values="y")
           .reset_index())
    print(piv.to_string(index=False, float_format=lambda v: f"{v:,.4g}"))
    print("\nPanel B - dose-response at t_dof = "
          f"{repo_dof:g}, {min(budgets):g}bp:")
    dose = (curve[np.isclose(curve["t_dof"], repo_dof)
                  & np.isclose(curve["te_budget_bp"], min(budgets))]
            .sort_values("pd_mult"))
    print(dose[["arm", "pd_mult", "defrate", "exkurt", "y", "y_lo", "y_hi"]]
          .to_string(index=False, float_format=lambda v: f"{v:,.4g}"))
    if not ctrl.empty:
        print("\nPanel C - Gaussian control (t_dof=1e6, no jumps):")
        print(ctrl[["n_scenarios", "exkurt", "ratio_ref", "y"]]
              .sort_values("n_scenarios")
              .to_string(index=False, float_format=lambda v: f"{v:,.4g}"))
    xchk = summarize(df[df["arm"].eq(CROSS_CHECK_ARM)])
    if not xchk.empty:
        print("\nCross-check solver "
              f"({env.get('cross_check_solver', '?')}) at the repo "
              f"operating point:")
        print(xchk[["te_budget_bp", "exkurt", "y"]]
              .to_string(index=False, float_format=lambda v: f"{v:,.4g}"))
    skipped = df[df["converged"] & ~df["binding"]]
    if len(skipped):
        print(f"\nexcluded, budget did not bind: {len(skipped)} row(s)")
    bad = df[~df["converged"]]
    if len(bad):
        print(f"\nexcluded, not converged: {len(bad)} row(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

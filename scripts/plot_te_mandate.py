#!/usr/bin/env python
"""Plot the M6 tracking-error mandate result from its committed CSVs.

The finding this figure has to carry, without a caption:

1. Bloomberg's optimized fixed income index methodology 34100 writes the
   ex-ante tracking-error budget on **sigma** of active return. That is a
   second-order cone constraint. An LP solver cannot express it - not slowly,
   not approximately, at all - and `run_te_mandate.py` records those cells as
   `NOT-IN-CLASS` rather than as a failed solve.
2. The LP-representable surrogate constrains MAD instead. It hits its own
   budget to ~1e-5bp while the **realized sigma** overshoots by multiples,
   and the overshoot grows monotonically as the budget tightens.

Two design decisions worth stating, because both could be got wrong quietly.

**How NOT-IN-CLASS is drawn.** An absent point reads as "not run"; a point at
zero or at a time limit reads as "measured, badly". Neither is true here, so
NOT-IN-CLASS gets its own visual grammar and appears twice: in the main panel
as a circled cross sitting exactly on the mandate point (10bp, 10bp) - the
place the mandate asks the optimizer to reach and no LP solver can address -
and in the cross-solver panel as a hatched band filling the whole row where a
value would go, labelled in words. Three states, each distinguishable in
greyscale: filled marker = a converged measurement, open marker = converged
but the budget was slack, hatched band = outside the solver's problem class.

**The panels are stacked, and the figure is authored at the paper's text
width.** This figure leads the paper, and at \\linewidth in a one-column
article a figure authored 979pt wide is scaled to 0.46: the 7.8pt notes
below reached the page at 3.6pt and the two of them might as well not have
been drawn. Authoring at PAPER_TEXTWIDTH_IN makes the scale factor exactly
1.0, so every size specified in this file is the size the reader gets. Side
by side at that width neither panel has room, hence one column: panel A is
a log-log scatter that tolerates being wide, panel B is a row chart whose
natural direction is horizontal, and both want the full measure.

**The Gaussian reference is the identity line, not sqrt(pi/2) times it.**
`src/credit.py` already multiplies the MAD by `MAD_TO_SIGMA = sqrt(pi/2)`
both inside the LP constraint and inside `tracking_error_bp`, so the CSV's
`te_mad_implied_bp` is a sigma-scaled quantity and the LP enforces
`te_mad_implied_bp <= budget`. Under Gaussian active returns sigma equals
that number exactly, so **y = x is simultaneously the mandate as written and
the Gaussian prediction**, and a second line at sqrt(pi/2) * x would apply
the conversion twice and overstate what the surrogate is entitled to miss by.
Everything above the identity line is measured non-normality, not a unit
conversion. The figure says so on its face.

Reads committed results only; no solve, no GPU.

    python scripts/plot_te_mandate.py \
        --csv results/m6_te_mandate.csv \
        --ref-csv results/m6_te_sigma_ref.csv \
        --out results/m6_te_mandate.png
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
# Type 42 (TrueType) rather than matplotlib's default Type 3: arXiv flags
# Type 3 fonts, and this figure is headed for a preprint.
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.ticker import (  # noqa: E402
    FixedLocator, FuncFormatter, NullLocator,
)

#: Width of the text block in paper/main.tex, in inches: a4paper (210mm)
#: with geometry margin=1in, so 210mm - 2in = 6.2677in = 452.97 TeX pt. A
#: figure authored at this width and placed at \linewidth is scaled by
#: exactly 1.0, which is the only thing that makes a font size specified in
#: this file the font size that reaches the page.
PAPER_TEXTWIDTH_IN = 6.2677

#: Total figure height. The a4 text block is 9.69in and this figure's
#: caption sets to about 1.5in, so anything past ~7.9in makes LaTeX report
#: "Float too large for page" and defer the figure to the end of the document.
DEFAULT_HEIGHT_IN = 7.85

#: Solver statuses whose weight vector is a converged answer. Copied from
#: `CONVERGED` in scripts/run_te_mandate.py deliberately: anything else is an
#: arbitrary iterate, and the tracking error recomputed from it is a number
#: with a unit and no meaning. The committed CSV still carries those values
#: (that run predates the runner's own NaN-ing of them), so this script has
#: to apply the filter itself rather than trust the column.
CONVERGED = ("optimal", "optimal_inaccurate")

#: A budget counts as *binding* only if the solve landed within this relative
#: distance of it. Any threshold near a solver tolerance would misclassify: a
#: converged first-order solve sits O(tol) inside its own constraint, and in
#: this CSV the genuinely-binding rows sit at most 1.2e-3 below budget while
#: the genuinely-slack rows sit 0.41-0.44 below it. 5% is two and a half
#: orders of magnitude above the former and an order below the latter, so the
#: classification does not turn on the exact value.
BINDING_REL_SLACK = 0.05

#: Colour is the redundant channel here, never the only one: every series
#: also carries its own marker and dash pattern, so the figure survives a
#: black-and-white printer. Greens follow scripts/plot_sweep.py.
STYLE = {
    "cuopt@1e-4": dict(color="#76b900", marker="o", linestyle="-",
                       label="cuOpt PDLP @1e-4 (GPU)",
                       short="cuOpt PDLP @1e-4"),
    "cuopt@1e-6": dict(color="#2e7d32", marker="^", linestyle="--",
                       label="cuOpt PDLP @1e-6 (GPU)",
                       short="cuOpt PDLP @1e-6"),
    "cuopt@1e-4+xover": dict(color="#1b5e20", marker="s", linestyle="-.",
                             label="cuOpt PDLP +crossover (GPU)",
                             short="cuOpt +crossover"),
    "highs-simplex": dict(color="#c62828", marker="D", linestyle=":",
                          label="HiGHS simplex (CPU)",
                          short="HiGHS simplex"),
    "clarabel": dict(color="#6a1b9a", marker="v", linestyle="-",
                     label="Clarabel interior point (CPU)",
                     short="Clarabel (conic)"),
    "ortools-pdlp": dict(color="#1565c0", marker="P", linestyle="--",
                         label="OR-Tools PDLP (CPU)",
                         short="OR-Tools PDLP"),
}

# Type sizes. Every one of these is the size the reader gets on the printed
# page, because the figure is authored at PAPER_TEXTWIDTH_IN and placed at
# \linewidth. Nothing here is below 7.5pt, which is the smallest size this
# document's 11pt body text can sit next to without looking like a defect.
FS_SUP = 11.0
FS_TITLE, FS_AXIS, FS_TICK = 9.6, 8.6, 8.2
FS_LEGEND, FS_ANNOT, FS_RATIO, FS_NOTE = 7.8, 7.8, 9.0, 7.5

#: Unknown solver labels still have to plot: a regenerated CSV may add a
#: config, and a figure script that crashes on a new row is worse than one
#: that draws it in grey.
_FALLBACK_MARKERS = ("X", "*", "h", "8", "p")
_FALLBACK_LINES = ("-", "--", "-.", ":")


def style_for(name: str, index: int = 0) -> dict:
    """Style for a solver label, generated if the label is not known."""
    if name in STYLE:
        return STYLE[name]
    return dict(color="0.35",
                marker=_FALLBACK_MARKERS[index % len(_FALLBACK_MARKERS)],
                linestyle=_FALLBACK_LINES[index % len(_FALLBACK_LINES)],
                label=str(name), short=str(name))


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
    """Read one M6 CSV and attach the derived columns the figure needs."""
    env = read_env(path)
    df = pd.read_csv(path, comment="#")
    for col in ("te_budget_bp", "wall_s", "te_sigma_bp",
                "te_mad_implied_bp", "te_breach_bp"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["status"] = df["status"].astype(str)
    df["converged"] = df["status"].isin(CONVERGED)
    df["not_in_class"] = df["status"].eq("NOT-IN-CLASS")

    # Which quantity the budget was applied to. Same rule as the runner: the
    # mode that was CONSTRAINED is the one scored, so a sigma-mode row is
    # judged on sigma and a mad-mode row on the (already sigma-scaled)
    # MAD-implied value.
    df["te_scored_bp"] = np.where(df["te_mode"].eq("mad"),
                                  df["te_mad_implied_bp"], df["te_sigma_bp"])
    df["budget_slack_rel"] = 1.0 - df["te_scored_bp"] / df["te_budget_bp"]
    df["binding"] = df["converged"] & (df["budget_slack_rel"]
                                       <= BINDING_REL_SLACK)
    # The headline ratio: realized sigma against the budget that was asked
    # for. A Gaussian world would return 1.0.
    df["sigma_over_budget"] = df["te_sigma_bp"] / df["te_budget_bp"]
    df["sigma_over_mad"] = df["te_sigma_bp"] / df["te_mad_implied_bp"]
    return df, env


def size_label(env: dict[str, str]) -> str:
    """'500 bonds x 10,000 scenarios' from the CSV header, or a shrug."""
    try:
        return (f"{int(env['n_bonds'])} bonds"
                f" x {int(env['n_scenarios']):,} scenarios")
    except (KeyError, ValueError):
        return "problem size not recorded in the CSV header"


# --------------------------------------------------------------------------
# Panel A: the budget sweep
# --------------------------------------------------------------------------

def draw_sweep(ax, df: pd.DataFrame, env: dict[str, str], csv_name: str,
               mandate_bp: float
               ) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """Realized sigma-TE against the requested MAD budget.

    Returns (per-budget summary, dropped rows, legend handles). The handles
    come back rather than being drawn here because at one text column an
    in-panel legend costs a third of the panel: six entries at FS_LEGEND is
    0.95in, and this panel is under three inches tall. The caller puts them
    in a figure-level legend under the title, where the height is spent
    once instead of inside the data area.
    """
    mad = df[df["te_mode"].eq("mad")]
    curves = mad[mad["converged"]]
    dropped = mad[~mad["converged"]]
    if curves.empty:
        raise SystemExit("no converged MAD rows in the CSV - nothing to plot")

    budgets = np.sort(curves["te_budget_bp"].unique())
    # Equal decade span on both axes so y = x reads as a diagonal, with
    # headroom above the data and below the diagonal for the two callouts.
    lo = float(min(budgets.min(), curves["te_sigma_bp"].min())) * 0.62
    hi = float(max(budgets.max(), curves["te_sigma_bp"].max())) * 1.85
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    # More headroom on y than on x. On log-log, y = x is a straight line for
    # any pair of ranges - only its slope changes - and the alternative here
    # is a callout box sitting on top of the data point it describes.
    ax.set_ylim(lo, hi * 1.80)

    # Highlight the mandate budget before anything is drawn on top of it.
    if np.isfinite(mandate_bp):
        ax.axvspan(mandate_bp * 0.93, mandate_bp * 1.075, color="0.925",
                   zorder=0)

    # y = x. Under a Gaussian this is also exactly what the MAD surrogate
    # delivers, because the budget already carries the sqrt(pi/2) factor.
    diag = np.array([lo, hi * 1.80])
    ax.plot(diag, diag, color="0.20", lw=1.5, ls=(0, (7, 3)), zorder=2)

    handles: list = [Line2D([], [], color="0.20", lw=1.5, ls=(0, (7, 3)),
                            label="y = x: the mandate as written")]
    for i, (sname, sub) in enumerate(curves.groupby("solver", sort=True)):
        st = style_for(sname, i)
        sub = sub.sort_values("te_budget_bp")
        ax.plot(sub["te_budget_bp"], sub["te_sigma_bp"], lw=1.9,
                color=st["color"], linestyle=st["linestyle"],
                marker=st["marker"], ms=7.5, zorder=3)
        slack = sub[~sub["binding"]]
        if not slack.empty:
            # Open marker: converged, but the TE budget was not the binding
            # limit, so the point measures some other constraint.
            ax.plot(slack["te_budget_bp"], slack["te_sigma_bp"],
                    linestyle="none", marker=st["marker"], ms=9.5,
                    mfc="white", mec=st["color"], mew=1.9, zorder=4)
        handles.append(Line2D([], [], color=st["color"],
                              linestyle=st["linestyle"], marker=st["marker"],
                              ms=7, lw=1.9, label=st["label"]))

    # Scale references straight from the CSV header.
    for key, label, dash in (
            ("incumbent_te_sigma_bp", "incumbent book", (0, (1, 2))),
            ("unconstrained_te_sigma_bp", "no TE constraint",
             (0, (5, 2, 1, 2)))):
        try:
            val = float(env[key])
        except (KeyError, ValueError):
            continue
        ax.axhline(val, color="0.45", lw=1.1, ls=dash, zorder=1)
        handles.append(Line2D([], [], color="0.45", lw=1.1, ls=dash,
                              label=f"{label}: {val:.1f}bp"))

    # Per-budget summary. The ratio annotations quote the TIGHTEST converged
    # arm (cuopt@1e-6) rather than the median across arms. The two cuOpt
    # tolerances agree to <0.1% at every binding budget, so the choice is
    # cosmetic -- but the paper's table prints the @1e-6 arm, and at 50bp
    # the across-arm median rounds to 1.46x while @1e-6 rounds to 1.45x.
    # Two adjacent renderings of the same quantity must not round apart, so
    # both now quote the same arm. Median remains the fallback for any
    # budget the tight arm did not converge on.
    tight = curves[curves["solver"].str.contains("1e-6", regex=False)]
    fallback = curves[~curves["te_budget_bp"].isin(tight["te_budget_bp"])]
    src = pd.concat([tight, fallback])
    summary = (src.groupby("te_budget_bp")
               .agg(sigma_bp=("te_sigma_bp", "median"),
                    mad_bp=("te_mad_implied_bp", "median"))
               .reset_index())
    counts = (curves.groupby("te_budget_bp")
              .agg(n_solvers=("solver", "size"),
                   n_binding=("binding", "sum"))
              .reset_index())
    summary = summary.merge(counts, on="te_budget_bp")
    summary["ratio"] = summary["sigma_bp"] / summary["te_budget_bp"]
    summary["binding"] = summary["n_binding"] > 0

    for _, r in summary.iterrows():
        if not r["binding"]:
            continue
        b, s = float(r["te_budget_bp"]), float(r["sigma_bp"])
        ax.annotate("", xy=(b, s), xytext=(b, b),
                    arrowprops=dict(arrowstyle="<->", color="0.30", lw=1.0,
                                    shrinkA=0, shrinkB=0), zorder=5)
        ax.text(b * 1.07, np.sqrt(b * s), f"{r['ratio']:.2f}x",
                fontsize=FS_RATIO, color="0.10", ha="left", va="center",
                zorder=6,
                bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none",
                          alpha=0.88))

    # The slack budget gets words, not a ratio: past the point where the
    # budget binds, the figure is no longer measuring the surrogate.
    slack_rows = summary[~summary["binding"]]
    if not slack_rows.empty:
        r = slack_rows.iloc[0]
        b, s = float(r["te_budget_bp"]), float(r["sigma_bp"])
        unc = env.get("unconstrained_te_sigma_bp", "")
        tail = (f" sigma-TE has settled on the unconstrained optimum, "
                f"{unc}bp." if unc else "")
        # Top-right, above the curve. The other callout has the empty
        # wedge below the identity line and there is not room for two
        # multi-line boxes down there; this one has a slot of its own
        # between the top of the data and the top of the panel.
        ax.annotate(
            textwrap.fill(
                f"open marker: at {b:g}bp the budget is slack. Realized "
                f"MAD-implied TE is only {r['mad_bp']:.0f}bp, so other "
                f"limits bind, not this one;{tail}", width=46),
            xy=(b, s * 1.05), xytext=(0.985, 0.985),
            textcoords="axes fraction", ha="right", va="top",
            fontsize=FS_ANNOT, color="0.15", linespacing=1.35,
            arrowprops=dict(arrowstyle="->", color="0.45", lw=1.0,
                            connectionstyle="arc3,rad=-0.2"),
            bbox=dict(boxstyle="round,pad=0.30", fc="white", ec="0.75",
                      lw=0.8))

    # The mandate point itself: a circled cross, because the honest mark for
    # "no LP solver can address this" is not a data point.
    sig = df[df["te_mode"].eq("sigma")]
    nic = sig[sig["not_in_class"]]
    if np.isfinite(mandate_bp):
        ax.plot([mandate_bp], [mandate_bp], marker="o", ms=15, mfc="white",
                mec="0.10", mew=1.8, zorder=7)
        ax.plot([mandate_bp], [mandate_bp], marker="x", ms=9, color="0.10",
                mew=1.8, zorder=8)
        # Four lines, not six: the box has to sit entirely below the
        # identity line, and the geometry only allows that at this width
        # if it is four lines tall. The list of which solvers returned
        # NOT-IN-CLASS moves to the figure note, where provenance belongs
        # anyway.
        msg = textwrap.fill(
            f"The mandate point. Bloomberg 34100 as written is sigma-TE "
            f"<= {mandate_bp:g}bp, a second-order cone: {len(nic)} of "
            f"{len(sig)} solvers run here return NOT-IN-CLASS. Not a slow "
            f"solve - the constraint cannot be expressed, so there is no "
            f"point to plot.", width=52)
        # Bottom-right in AXES fraction, not data coordinates: everything
        # below the identity line is empty by construction, and a data-space
        # anchor drifts off the panel the moment the axis limits change.
        ax.annotate(msg, xy=(mandate_bp * 1.16, mandate_bp * 0.90),
                    xytext=(0.985, 0.025), textcoords="axes fraction",
                    ha="right", va="bottom", fontsize=FS_ANNOT,
                    color="0.10", linespacing=1.35,
                    arrowprops=dict(arrowstyle="->", color="0.35", lw=1.0),
                    bbox=dict(boxstyle="round,pad=0.30", fc="#f2f2f2",
                              ec="0.55", lw=0.8))
        handles.append(Line2D([], [], color="0.10", marker="x", ls="none",
                              ms=8, mew=1.8,
                              label="NOT-IN-CLASS - see box"))

    ticks = [t for t in (5, 10, 20, 30, 50, 75, 100, 150, 200)
             if lo <= t <= hi]
    fmt = FuncFormatter(lambda v, _: f"{v:g}")
    ax.xaxis.set_major_locator(FixedLocator(list(budgets)))
    ax.xaxis.set_major_formatter(fmt)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(fmt)
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("requested MAD-surrogate budget (bp, annualized)",
                  fontsize=FS_AXIS)
    ax.set_ylabel("realized tracking error: sigma of active return\n"
                  "(bp annualized, recomputed from the scenario matrix)",
                  fontsize=FS_AXIS)
    ax.set_title("A. The LP surrogate hits its own budget and misses the "
                 "mandate\n" + size_label(env) + f"   ({csv_name})",
                 fontsize=FS_TITLE, loc="left")
    ax.grid(True, which="major", alpha=0.25)
    # One column, not two: two columns of these labels are 4.3in wide and
    # would lie on top of the curve. One column is 2.1in and sits in the
    # upper-left wedge above it, which is empty at every budget in the CSV.
    ax.legend(handles=handles, fontsize=FS_LEGEND, loc="upper left",
              framealpha=0.95, borderpad=0.4, labelspacing=0.32,
              handlelength=2.0)
    return summary, dropped, handles


# --------------------------------------------------------------------------
# Panel B: the cross-solver check at the mandate budget
# --------------------------------------------------------------------------

#: Display order for the two formulations: the surrogate first, because that
#: is what panel A measured, then the constraint the mandate actually writes.
_MODE_ORDER = ("mad", "sigma")
_MODE_TITLE = {
    "mad": "LP surrogate:  sqrt(pi/2) * mean|active|  <=  budget",
    "sigma": "as written:  std(active)  <=  budget   (second-order cone)",
}


def draw_reference(ax, ref: pd.DataFrame, env: dict[str, str],
                   csv_name: str, mandate_bp: float,
                   ratio_a: float = float("nan")) -> pd.DataFrame:
    """One row per (formulation, solver) at the mandate budget.

    `ratio_a` is panel A's overshoot at the same budget, quoted here only so
    the two panels cannot be misread as one series: they are different
    instances and they do not give the same number.
    """
    at_mandate = ref[np.isclose(ref["te_budget_bp"], mandate_bp)]
    if at_mandate.empty:
        raise SystemExit(f"reference CSV has no rows at {mandate_bp}bp")
    xmax = float(np.nanmax(at_mandate["te_sigma_bp"])) * 1.40

    y = 0.0
    ticks: list[tuple[float, str]] = []
    mad_vals: list[float] = []
    for mode in _MODE_ORDER:
        block = at_mandate[at_mandate["te_mode"].eq(mode)]
        if block.empty:
            continue
        ax.text(0.0, y - 0.85, _MODE_TITLE[mode], fontsize=FS_ANNOT + 0.4,
                va="center",
                ha="left", color="0.15", style="italic", zorder=4,
                bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="none"))
        vals: list[float] = []
        for i, (_, r) in enumerate(block.iterrows()):
            st = style_for(r["solver"], i)
            ticks.append((y, st["short"]))
            if r["not_in_class"]:
                # A hatched band across the whole row: the value is not
                # missing and not bad, it does not exist.
                ax.add_patch(Rectangle((0, y - 0.33), xmax, 0.66,
                                       facecolor="0.94", edgecolor="0.55",
                                       hatch="///", lw=0.8, zorder=1))
                ax.text(xmax * 0.03, y,
                        "NOT-IN-CLASS: cannot express an SOC constraint",
                        fontsize=FS_ANNOT, va="center", ha="left",
                        color="0.15", zorder=3,
                        bbox=dict(boxstyle="square,pad=0.25",
                                  fc="white", ec="none", alpha=0.88))
            elif r["converged"]:
                val = float(r["te_sigma_bp"])
                ax.plot([0, val], [y, y], color=st["color"], lw=1.4,
                        alpha=0.5, zorder=2)
                ax.plot([val], [y], marker=st["marker"], ms=9,
                        color=st["color"], zorder=3)
                ax.text(val + xmax * 0.025, y, f"{val:.2f}bp",
                        fontsize=FS_ANNOT + 0.4,
                        va="center", ha="left", color="0.10", zorder=3)
                vals.append(val)
            else:
                # Same filtering rule as panel A, stated in place.
                ax.plot([float(r["te_sigma_bp"])], [y], marker=st["marker"],
                        ms=9, mfc="white", mec=st["color"], mew=1.7,
                        zorder=3)
                ax.text(xmax * 0.03, y,
                        f"unconverged ({r['status']}) - excluded",
                        fontsize=FS_ANNOT, va="center", ha="left",
                        color="0.35")
            y += 1.0
        if mode == "mad" and len(vals) > 1:
            mad_vals = vals
            spread = (max(vals) - min(vals)) / float(np.median(vals))
            # The cross-panel caveat rides along here rather than in its own
            # text object next to the ratio: at this panel height there is
            # not a clear line of type between the ratio and the first row.
            also = (f" Panel A's larger instance: {ratio_a:.2f}x."
                    if np.isfinite(ratio_a) else "")
            # Two lines, hand-broken. A third line does not fit between this
            # block and the next formulation heading at this panel height,
            # and a wrapped third line is exactly what a reader would see
            # sitting on top of the heading.
            ax.text(0.0, y - 0.34,
                    f"{min(vals):.3f}-{max(vals):.3f}bp across "
                    f"{len(vals)} solvers and {len(vals)} algorithm "
                    f"families (spread {spread * 100:.2f}%):\nthe overshoot "
                    f"is the formulation, not the solver.{also}",
                    fontsize=FS_NOTE, va="top", ha="left", color="0.25",
                    zorder=4, bbox=dict(boxstyle="square,pad=0.25",
                                        fc="white", ec="none", alpha=0.9))
            y += 1.75
        y += 0.90

    # Three lines of header hang above the first row - the mandate rule,
    # the ratio arrow and the ratio - and each needs a line of type, not a
    # fraction of one. -3.1 is 0.8 units more than the row pitch this panel
    # would otherwise have; anything less and the ratio lands on the first
    # formulation heading.
    top = -3.10
    ax.set_ylim(y - 0.55, top)
    ax.set_xlim(0, xmax)
    ax.set_yticks([p for p, _ in ticks])
    ax.set_yticklabels([t for _, t in ticks], fontsize=FS_TICK)

    ax.axvline(mandate_bp, color="0.20", lw=1.3, ls=(0, (7, 3)), zorder=1)
    ax.text(mandate_bp, top + 0.62, f"mandate: {mandate_bp:g}bp",
            fontsize=FS_ANNOT, ha="center", va="bottom", color="0.10")

    if mad_vals:
        med = float(np.median(mad_vals))
        ax.axvline(med, color="0.45", lw=1.0, ls=(0, (2, 2)), zorder=1)
        ax.annotate("", xy=(med, top + 1.60),
                    xytext=(mandate_bp, top + 1.60),
                    arrowprops=dict(arrowstyle="<->", color="0.25", lw=1.1))
        ax.text(0.5 * (mandate_bp + med), top + 1.60,
                f"{med / mandate_bp:.2f}x", fontsize=FS_RATIO, ha="center",
                va="bottom", color="0.10")

    ax.set_xlabel("realized tracking error: sigma of active return "
                  "(bp annualized)", fontsize=FS_AXIS)
    ax.set_title("B. Not the solver: three algorithm families, one answer\n"
                 + size_label(env) + f"   ({csv_name})",
                 fontsize=FS_TITLE, loc="left")
    ax.grid(True, axis="x", alpha=0.25)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    return at_mandate


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Plot the M6 tracking-error mandate figure from the "
                    "committed CSVs (no solve, no GPU).")
    ap.add_argument("--csv", default="results/m6_te_mandate.csv",
                    help="the MAD budget sweep plus the sigma-form rows")
    ap.add_argument("--ref-csv", default="results/m6_te_sigma_ref.csv",
                    help="smaller instance with a conic solver present; "
                         "pass '' to draw panel A alone")
    ap.add_argument("--out", default="results/m6_te_mandate.png",
                    help="raster output for the README")
    ap.add_argument("--out-pdf", default="",
                    help="vector output for the paper "
                         "(default: --out with a .pdf suffix)")
    ap.add_argument("--dpi", type=float, default=150.0)
    ap.add_argument("--mandate-bp", type=float, default=float("nan"),
                    help="mandate level in bp; default: read it from the "
                         "sigma-mode rows of --csv")
    ap.add_argument("--width-in", type=float, default=PAPER_TEXTWIDTH_IN,
                    help="figure width in inches. The default is the "
                         "paper's text width, which is what makes the type "
                         "sizes in this script the type sizes on the page; "
                         "the layout below is proportional, so a larger "
                         "value renders the same figure with relatively "
                         "smaller type")
    ap.add_argument("--height-in", type=float, default=DEFAULT_HEIGHT_IN)
    args = ap.parse_args()

    path = Path(args.csv)
    if not path.exists():
        print(f"missing {path} - run scripts/run_te_mandate.py first")
        return 1
    df, env = load(path)

    mandate_bp = args.mandate_bp
    if not np.isfinite(mandate_bp):
        sig = df[df["te_mode"].eq("sigma")]
        mandate_bp = (float(sig["te_budget_bp"].iloc[0]) if len(sig)
                      else float("nan"))

    ref_path = Path(args.ref_csv) if args.ref_csv else None
    have_ref = ref_path is not None and ref_path.exists()
    if ref_path is not None and not have_ref:
        print(f"note: {ref_path} not found - drawing panel A only")

    # One column, panels stacked, explicit rectangles rather than a shared
    # GridSpec: the two panels need DIFFERENT left margins (panel B's y tick
    # labels are solver names and eat 1.1in, panel A's are three digits) and
    # a GridSpec can only give them one. Rectangles are in figure fractions,
    # so --width-in / --height-in scale the whole thing without a second
    # code path.
    fig = plt.figure(figsize=(args.width_in, args.height_in))
    h = args.height_in
    # Vertical budget, in inches, top to bottom. Written out rather than
    # eyeballed because the defect this layout fixes was a figure whose
    # geometry nobody had added up: suptitle 0.30, panel title 0.34,
    # panel A (whatever is left, 2.42 at the default height), its x label
    # 0.34, gap 0.12, panel title 0.34, panel B 2.25, its x label 0.34,
    # gap 0.08, notes 1.32. Every one of those except the two panels hangs
    # OUTSIDE an axes rectangle, which is what the previous version forgot.
    notes_in, xlab_in, title_in, sup_in = 1.32, 0.34, 0.34, 0.30
    b_h = 2.25 if have_ref else 0.0
    b_bottom = (notes_in + 0.08 + xlab_in) / h
    a_bottom = (b_bottom + (b_h + title_in + xlab_in + 0.12) / h
                if have_ref else b_bottom)
    a_h = (1.0 - a_bottom) * h - title_in - sup_in
    ax_a = fig.add_axes([0.115, a_bottom, 0.870, a_h / h])
    # Panel B needs a wider left margin than panel A: its y tick labels are
    # solver names, panel A's are three digits. One shared GridSpec cannot
    # give them different margins, hence explicit rectangles.
    ax_b = (fig.add_axes([0.185, b_bottom, 0.800, b_h / h])
            if have_ref else None)
    wrap_w = max(60, int(args.width_in * 17.5))

    summary, dropped, handles = draw_sweep(ax_a, df, env, path.name,
                                           mandate_bp)

    ref_rows = None
    if have_ref:
        at = summary[np.isclose(summary["te_budget_bp"], mandate_bp)]
        ratio_a = float(at["ratio"].iloc[0]) if len(at) else float("nan")
        ref, ref_env = load(ref_path)
        ref_rows = draw_reference(ax_b, ref, ref_env, ref_path.name,
                                  mandate_bp, ratio_a=ratio_a)

    fig.suptitle("A mandate no LP solver can take, and an LP surrogate that "
                 "does not deliver it",
                 fontsize=FS_SUP, x=0.007, ha="left", y=0.992)

    # The two things a referee would otherwise have to take on trust.
    n_by_solver = (dropped.groupby("solver").size().sort_index()
                   if not dropped.empty else pd.Series(dtype=int))
    n_budgets = int(df[df["te_mode"].eq("mad")]["te_budget_bp"].nunique())
    who = "; ".join(f"{style_for(s)['short']} {n} of {n_budgets}"
                    for s, n in n_by_solver.items())
    # What the dropped rows were, rather than what they contained. Until the
    # runner was fixed, a non-converged row still carried a tracking error
    # recomputed from whatever iterate the solver stopped on -- numbers in the
    # thousands of bp, noise with a unit attached -- and this note quoted the
    # largest as evidence that dropping them mattered. Those rows now write
    # NaN, which is correct and left this branch reporting "none dropped"
    # directly after saying how many were dropped. The status breakdown is
    # the useful form: a time cap and a solver error are different failures
    # and the paper distinguishes them.
    if dropped.empty:
        worst_txt = "none dropped"
    else:
        by_status = dropped["status"].value_counts().sort_index()
        parts = []
        for st, n in by_status.items():
            label = ("hit the time cap" if "limit" in str(st).lower()
                     else "returned a solver error"
                     if "error" in str(st).lower() else str(st))
            parts.append(f"{n} {label}")
        recomputed = dropped["te_sigma_bp"].notna().sum()
        tail = ("; none carries a recomputed tracking error"
                if recomputed == 0 else
                f"; {recomputed} still carries a TE recomputed from a "
                f"non-converged iterate")
        worst_txt = ", ".join(parts) + tail
    nic_names = ", ".join(
        style_for(s)["short"]
        for s in df[df["te_mode"].eq("sigma") & df["not_in_class"]]["solver"])
    nic_txt = (f" NOT-IN-CLASS in panel A: {nic_names}." if nic_names else "")
    filter_note = textwrap.fill(
        "Filtering rule, applied rather than assumed: a row enters the "
        "curves only if its solver status is converged, which drops "
        f"{who} ({worst_txt.strip()}). Filled marker = the budget binds; "
        f"open = converged but slack by over {BINDING_REL_SLACK:.0%}."
        f"{nic_txt}", width=wrap_w)
    gaussian_note = textwrap.fill(
        "Why the identity line is also the Gaussian line: src/credit.py "
        "constrains sqrt(pi/2) * mean|A_c| <= budget and reports that same "
        "sigma-scaled quantity, so on Gaussian active returns the surrogate "
        "would land exactly on y = x. Every bp above the line is measured "
        "non-normality, not a unit conversion.",
        width=wrap_w)
    # Stacked, not side by side: at one text column there is no second
    # column to put the right-hand note in.
    fig.text(0.007, 0.012, filter_note + "\n\n" + gaussian_note,
             fontsize=FS_NOTE, color="0.30", va="bottom", ha="left",
             linespacing=1.45)

    out_png = Path(args.out)
    out_pdf = (Path(args.out_pdf) if args.out_pdf
               else out_png.with_suffix(".pdf"))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=args.dpi)
    fig.savefig(out_pdf)
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")

    print(f"\n{path.name}: {size_label(env)}")
    print("MAD-surrogate sweep, converged rows only "
          f"({', '.join(CONVERGED)}):")
    print("  budget    realized       realized   sigma/   sigma/  budget")
    print("    (bp)  sigma (bp)  MAD-impl (bp)   budget      MAD  binds")
    mad = df[df["te_mode"].eq("mad") & df["converged"]].sort_values(
        ["te_budget_bp", "solver"])
    for _, r in mad.iterrows():
        print(f"  {r['te_budget_bp']:6.1f}  {r['te_sigma_bp']:10.4f}  "
              f"{r['te_mad_implied_bp']:13.4f}  "
              f"{r['sigma_over_budget']:7.3f}  {r['sigma_over_mad']:7.3f}  "
              f"{str(bool(r['binding'])):>5}   {r['solver']}")
    print("\nper-budget median across converged solvers:")
    for _, r in summary.iterrows():
        tag = "binding" if r["binding"] else "BUDGET SLACK - not a delivery"
        print(f"  {r['te_budget_bp']:6.1f}bp budget -> "
              f"{r['sigma_bp']:8.3f}bp sigma  = {r['ratio']:6.3f}x  ({tag})")
    print("\nexcluded (status not converged):")
    if dropped.empty:
        print("  (none)")
    for _, r in dropped.iterrows():
        print(f"  {r['te_budget_bp']:6.1f}  {r['solver']:<18} "
              f"{r['status']:<18} te_sigma_bp in CSV = "
              f"{r['te_sigma_bp']:,.1f} (discarded)")

    if ref_rows is not None:
        print(f"\n{ref_path.name}: cross-solver check at {mandate_bp:g}bp")
        for _, r in ref_rows.iterrows():
            val = ("NOT-IN-CLASS" if r["not_in_class"]
                   else f"{r['te_sigma_bp']:.4f}bp sigma")
            print(f"  {r['te_mode']:<6} {r['solver']:<16} "
                  f"{r['status']:<14} {val}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

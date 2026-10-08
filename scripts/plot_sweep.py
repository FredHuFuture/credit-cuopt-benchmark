#!/usr/bin/env python
"""Plot the M3 scaling study from its committed CSV.

Reads results only — the chart reproduces in a second without re-running the
sweep, which takes hours. DNFs are drawn as open markers at the time limit
rather than dropped, because "the CPU stopped finishing here" is the finding.

Authored at the paper's text width. This figure was previously drawn 9.6in
wide and placed at \\linewidth in a 6.27in text block, so it arrived on the
page at scale 0.655 and its 8pt annotations reached the reader at 5.2pt.
Nothing about the figure was wrong except its size in inches. See
PAPER_TEXTWIDTH_IN.

Emits BOTH a raster (for the README, where nothing is scaled) and a vector
PDF (for the paper, where a 150dpi raster of 8pt type is visibly soft next
to Type 1 body text).

    python scripts/plot_sweep.py --csv results/m3_sweep.csv \
        --out results/m3_scaling.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
# Type 42 (TrueType) rather than matplotlib's default Type 3: arXiv flags
# Type 3 fonts, and this figure now ships a PDF into the preprint.
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

#: Width of the text block in paper/main.tex, in inches: a4paper (210mm)
#: with geometry margin=1in, so 210mm - 2in = 6.2677in = 452.97 TeX pt. A
#: figure authored at this width and placed at \linewidth is scaled by
#: exactly 1.0, which is the only thing that makes a font size specified in
#: this file the font size that reaches the page.
PAPER_TEXTWIDTH_IN = 6.2677

#: Height. Single panel, so the only constraint is that it plus its caption
#: fit the 9.69in a4 text block; 4.1in leaves room for a five-line caption
#: and body text on the same page.
DEFAULT_HEIGHT_IN = 4.1

# Type sizes, which are the sizes on the page because of PAPER_TEXTWIDTH_IN.
FS_TITLE, FS_AXIS, FS_TICK, FS_LEGEND, FS_ANNOT = 9.6, 8.6, 8.2, 8.0, 7.8

#: Keys are canonical solver ids. The CSV mixes two spellings — worker rows
#: carry the benchmark harness label ("cuopt-pdlp+xover@0.0001") while rows
#: the orchestrator writes itself (DNFs, memory refusals) carry the factory
#: name ("cuopt_pdlp+xover") — so match on a normalised form rather than on
#: either spelling.
STYLE = {
    "cuopt-pdlp+xover": dict(color="#2e7d32", marker="s",
                             label="cuOpt PDLP + crossover (GPU)"),
    "cuopt-pdlp":       dict(color="#76b900", marker="o",
                             label="cuOpt PDLP (GPU)"),
    "ortools-pdlp":     dict(color="#1565c0", marker="^",
                             label="OR-Tools PDLP (CPU)"),
    "highs":            dict(color="#c62828", marker="D",
                             label="HiGHS simplex (CPU)"),
}


def canonical(name: str) -> str:
    """Map either spelling of a solver label onto its canonical id."""
    s = str(name).replace("_", "-").split("@")[0].strip()
    for key in STYLE:                      # '+xover' first: it is a prefix clash
        if s.startswith(key):
            return key
    return s


def read_env(path: Path) -> dict[str, str]:
    env = {}
    for line in path.read_text().splitlines():
        if not line.startswith("#"):
            break
        if ":" in line:
            k, v = line[1:].split(":", 1)
            env[k.strip()] = v.strip()
    return env


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/m3_sweep.csv")
    ap.add_argument("--out", default="results/m3_scaling.png",
                    help="raster output for the README")
    ap.add_argument("--out-pdf", default="",
                    help="vector output for the paper "
                         "(default: --out with a .pdf suffix)")
    ap.add_argument("--dpi", type=float, default=200.0,
                    help="raster resolution. 200 rather than 150 because "
                         "main.tex currently includes the PNG, and at "
                         "\\linewidth a 150dpi raster of 8pt type is soft "
                         "next to Type 1 body text")
    ap.add_argument("--width-in", type=float, default=PAPER_TEXTWIDTH_IN,
                    help="figure width in inches; the default is the "
                         "paper's text width, which is what makes the type "
                         "sizes in this script the type sizes on the page")
    ap.add_argument("--height-in", type=float, default=DEFAULT_HEIGHT_IN)
    args = ap.parse_args()

    path = Path(args.csv)
    if not path.exists():
        print(f"missing {path} — run scripts/run_sweep.py first")
        return 1
    env = read_env(path)
    df = pd.read_csv(path, comment="#")
    df["wall_s_median"] = pd.to_numeric(df["wall_s_median"], errors="coerce")
    df["dnf"] = df["dnf"].astype(str).str.lower().isin(("true", "1"))
    df["canon"] = df["solver"].map(canonical)
    limit = float(env.get("time_limit_s", 300.0))

    # Two different failures, drawn differently: a solver that ran out of time
    # is a fact about the solver; a cell the host could not build is a fact
    # about the machine. Collapsing them into one marker would misattribute
    # a 16 GB memory ceiling to whichever solver happened to be listed.
    mem = df["status"].astype(str).str.startswith("skipped_memory")
    fig, ax = plt.subplots(figsize=(args.width_in, args.height_in))

    if mem.any():
        ceiling = float(df.loc[mem, "nnz"].min())
        ax.axvspan(ceiling * 0.86, df["nnz"].max() * 1.3,
                   color="0.92", zorder=0)
        ax.text(ceiling * 0.95, limit * 0.55,
                "beyond this host's\nmemory ceiling\n(not attempted)",
                fontsize=FS_ANNOT, color="0.35", ha="left", va="center",
                linespacing=1.35)

    for solver, style in STYLE.items():
        sub = df[(df["canon"] == solver) & (~mem)]
        if sub.empty:
            continue
        ok = sub[~sub["dnf"]].sort_values("nnz")
        if not ok.empty:
            ax.plot(ok["nnz"], ok["wall_s_median"], lw=2, ms=7, **style)
        bad = sub[sub["dnf"]]
        if not bad.empty:
            ax.plot(bad["nnz"], np.full(len(bad), limit),
                    linestyle="none", marker=style["marker"], ms=10,
                    mfc="none", mec=style["color"], mew=2,
                    label="_nolegend_")

    ax.axhline(limit, color="0.6", lw=1, ls=":")
    # Below the line, not above it. Above it the label lands on the top
    # spine at this figure height, and on the open DNF marker it describes.
    ax.text(df["nnz"].min() * 0.98, limit * 0.80,
            f"{limit:.0f}s time limit; open markers did not finish",
            fontsize=FS_ANNOT, color="0.4", va="top")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("nonzeros in the scenario block  (scenarios × bonds)",
                  fontsize=FS_AXIS)
    ax.set_ylabel("solve time, median of repeats (s)", fontsize=FS_AXIS)
    ax.set_title("Credit Mean-CVaR: solve time vs problem size\n"
                 + env.get("gpu", "GPU").split(",")[0]
                 + " vs " + env.get("cpu_model", "CPU"), fontsize=FS_TITLE,
                 loc="left")
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=FS_LEGEND, loc="center left", framealpha=0.95,
              borderpad=0.4, labelspacing=0.34, handlelength=2.0)

    # Where do the two hardware paths cross, among configurations whose
    # limits actually hold? That is the only comparison worth annotating.
    gpu = df[(df["canon"] == "cuopt-pdlp+xover") & (~df["dnf"])]
    cpu = df[(df["canon"] == "highs") & (~df["dnf"])]
    merged = gpu.merge(cpu, on="nnz", suffixes=("_gpu", "_cpu"))
    if not merged.empty:
        merged = merged.sort_values("nnz")
        ratio = (merged["wall_s_median_cpu"] / merged["wall_s_median_gpu"])
        note = "  ".join(f"{n / 1e6:.0f}M: {r:.2f}x"
                         for n, r in zip(merged["nnz"], ratio))
        ax.text(0.98, 0.025,
                "CPU/GPU time ratio (limits-respecting configs)\n" + note,
                transform=ax.transAxes, fontsize=FS_ANNOT, ha="right",
                va="bottom", linespacing=1.35,
                bbox=dict(boxstyle="round,pad=0.30", fc="white", ec="0.8",
                          lw=0.8))

    fig.tight_layout(pad=0.6)
    out_png = Path(args.out)
    out_pdf = (Path(args.out_pdf) if args.out_pdf
               else out_png.with_suffix(".pdf"))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=args.dpi)
    fig.savefig(out_pdf)
    print(f"wrote {out_png}  "
          f"({args.width_in:.4g} x {args.height_in:.4g} in)")
    print(f"wrote {out_pdf}")

    print("\nCPU/GPU ratio among limits-respecting configurations:")
    if merged.empty:
        print("  (no overlapping finished cells yet)")
    else:
        for n, g, c in zip(merged["nnz"], merged["wall_s_median_gpu"],
                           merged["wall_s_median_cpu"]):
            print(f"  {n / 1e6:8.1f}M nnz   GPU {g:8.2f}s   CPU {c:8.2f}s   "
                  f"{c / g:5.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())

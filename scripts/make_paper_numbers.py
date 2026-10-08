#!/usr/bin/env python
"""Generate paper/numbers.tex from the committed CSVs in results/.

The paper is not allowed to contain a typed number. Every quantity it prints
comes through a `\\newcommand` defined here, and every one of those macros is
read out of a CSV that is committed to the repository. Nothing is transcribed
by hand and nothing is computed by hand: speedups, ratios and percentage
reductions are arithmetic performed in this file on values read from disk, so
they cannot drift away from the measurements they summarise.

WHY THIS EXISTS
---------------
An adversarial review of the README found three result tables with no
committed CSV behind them, and a scenario draw that had changed underneath
the numbers quoting it. Both failures are the same failure: a number in prose
has no link back to the artifact that produced it, so nothing notices when the
artifact changes or was never written. This script closes that loop
structurally. Regenerate the data, re-run this, rebuild the PDF, and the paper
is correct by construction. Fail to regenerate the data and the build stops.

THE STRICT DEFAULT IS THE POINT
-------------------------------
If a source CSV is absent, or the row a macro selects is not in it, or the
cell it reads is empty, this script fails and names what is missing and the
command that produces it. It never emits a plausible-looking placeholder in
strict mode, because a placeholder that reaches a PDF is indistinguishable
from a measurement. `--allow-missing` emits a loud `\\textbf{??}` instead, so
the skeleton still compiles while drafting; the `??` is meant to be visible
from across the room.

Run:
    python scripts/make_paper_numbers.py                    # strict
    python scripts/make_paper_numbers.py --allow-missing    # drafting
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------
# Sources
#
# One entry per CSV the paper depends on. `producer` is printed verbatim in
# the failure message, so it has to be a command someone can paste -- naming
# the script without its arguments is how a reproducibility appendix rots.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Source:
    """A committed CSV the paper reads, and how to (re)produce it."""

    filename: str
    producer: str
    what: str


SOURCES: dict[str, Source] = {
    "m1": Source(
        "m1_equity_400x10k.csv",
        "python scripts/run_bench.py --n-assets 400 --n-scenarios 10000 "
        "--runs 3 --out results/m1_equity_400x10k.csv",
        "equity Mean-CVaR benchmark, 400 assets x 10,000 scenarios",
    ),
    "m2econ": Source(
        "m2_credit_500x20k.csv",
        "python scripts/run_credit.py --n-bonds 500 --n-scenarios 20000 "
        "--runs 3 --out results/m2_credit_500x20k.csv",
        "credit portfolio economics, in- and out-of-sample",
    ),
    "m2limits": Source(
        "m2_credit_500x20k_limits.csv",
        "python scripts/run_credit.py --n-bonds 500 --n-scenarios 20000 "
        "--runs 3 --out results/m2_credit_500x20k.csv",
        "per-limit exposure report for the chosen portfolio",
    ),
    "m2solves": Source(
        "m2_credit_500x20k_solves.csv",
        "python scripts/run_credit.py --n-bonds 500 --n-scenarios 20000 "
        "--runs 3 --out results/m2_credit_500x20k.csv",
        "per-configuration solve timings and mandate verdicts",
    ),
    "m3": Source(
        "m3_sweep.csv",
        "python scripts/run_sweep.py --grid full --resume "
        "--out results/m3_sweep.csv",
        "scaling sweep across problem cells",
    ),
    "m6": Source(
        "m6_te_mandate.csv",
        "python scripts/run_te_mandate.py --out results/m6_te_mandate.csv",
        "tracking-error mandate sweep, MAD surrogate and the SOC row",
    ),
    "m6ref": Source(
        "m6_te_sigma_ref.csv",
        "python scripts/run_te_mandate.py --n-bonds 300 --n-scenarios 3000 "
        "--te-grid 10 --solvers clarabel highs cuopt "
        "--out results/m6_te_sigma_ref.csv",
        "conic reference: what the sigma-form mandate actually costs",
    ),
    "scen": Source(
        "rng_agreement.csv",
        "python scripts/check_rng_agreement.py --out results/"
        "rng_agreement.csv",
        "scenario-generation timings and GPU/CPU distributional agreement",
    ),
    "warm": Source(
        "warmstart_tradeoff.csv",
        "python scripts/check_warmstart_tradeoff.py 500 20000 "
        "--out results/warmstart_tradeoff.csv",
        "warm start without presolve vs cold solve with presolve",
    ),
    "abl": Source(
        "ablation.csv",
        "python scripts/run_ablation.py --out results/ablation.csv",
        "constraint-family ablation: what crossover costs per limit family",
    ),
    "qpbranch": Source(
        "qp_branch.csv",
        "python scripts/run_qp_branch.py --out results/qp_branch.csv "
        "--resume",
        "M7 QP branch: dense vs factored Hessian, GPU barrier vs CPU "
        "interior-point, plus the crossover-behaviour probe row",
    ),
    "qpprobe": Source(
        "qp_probe.csv",
        "python scripts/run_qp_branch.py --grid 2000,5000 --runs 1 "
        "--out results/qp_probe.csv",
        "M7 pre-flight: the same four arms re-run at two sizes in a "
        "separate process, kept as a cross-process replication check",
    ),
    "m9": Source(
        "m9_tail_sweep.csv",
        "python scripts/run_tail_sweep.py --out results/m9_tail_sweep.csv",
        "M9 tail dose-response: surrogate overshoot vs Student-t dof and "
        "jump-to-default intensity, with a Gaussian control",
    ),
    "demo": Source(
        "demo_latency.csv",
        "python scripts/check_demo_latency.py --out results/demo_latency.csv",
        "interactive demo latency: hand-rolled CSR path vs CVXPY",
    ),
    "p0socpref": Source(
        "p0_socp_reference.csv",
        "python scripts/run_te_mandate.py --n-bonds 300 --n-scenarios 3000 "
        "--runs 3 --te-grid --solvers cuopt-barrier-socp clarabel-socp "
        "--time-limit 600 --out results/p0_socp_reference.csv",
        "P0 native-SOCP reference-size validation",
    ),
    "p0socwork": Source(
        "p0_socp_working.csv",
        "python scripts/run_te_mandate.py --n-bonds 500 --n-scenarios 20000 "
        "--runs 3 --te-grid --solvers cuopt-barrier-socp clarabel-socp "
        "--time-limit 1800 --out results/p0_socp_working.csv",
        "P0 native-SOCP working-size go/no-go",
    ),
    "p0recal": Source(
        "p0_te_recalibration.csv",
        "python scripts/run_te_mandate.py --p0-candidate-grid --p0-bisect "
        "--validation-scen-seeds 101 211 307 --skip-sigma "
        "--solvers cuopt-pdlp-1e-4 cuopt-pdlp-1e-6 "
        "--out results/p0_te_recalibration.csv",
        "P0 MAD recalibration and independent-draw validation",
    ),
    "p0endpoint": Source(
        "p0_endpoint_ladder.csv",
        "python scripts/run_feasibility_repair.py --loose-source cuopt-pdlp "
        "--loose-tol 1e-4 1e-6 --te-grid 10 --time-limit 1800 "
        "--out results/p0_endpoint_ladder.csv",
        "P0 raw, repaired, basic, and exact endpoint ladder",
    ),
    "p0loo": Source(
        "p0_ablation_loo.csv",
        "python scripts/run_ablation.py --levels L0_budget_box,L6_full_mandate,LOO_no_turnover "
        "--runs 5 --resume --out results/p0_ablation_loo.csv",
        "P0 paired full-mandate versus no-turnover ablation",
    ),
    "p0qp": Source(
        "p0_qp_matched.csv",
        "python scripts/run_qp_branch.py "
        "--grid 500,1000,2000,3000,5000,7500,10000,15000,29222 --runs 3 "
        "--arms clarabel_strict_dense clarabel_strict_factored "
        "cuopt_barrier_dense cuopt_barrier_factored "
        "--repair-strict-clarabel --resume --out results/p0_qp_matched.csv",
        "P0 matched-quality solver-stack QP benchmark",
    ),
    "p1lots": Source(
        "p1_bond_lots_cpu.csv",
        "python scripts/run_bond_lots.py --out results/p1_bond_lots_cpu.csv",
        "P1 CPU continuous, rounded, and lot-repaired quantity pilot",
    ),
    "p1cpu": Source(
        "p1_request_timing_cpu.csv",
        "python scripts/run_request_bench.py --out results/p1_request_timing_cpu.csv",
        "P1 CPU single-request timing and independent quality pilot",
    ),
    "p1gpu": Source(
        "p1_request_timing_gpu_smoke.csv",
        "python scripts/run_request_bench.py --allow-gpu "
        "--solvers cuopt-pdlp-1e-4 cuopt-pdlp-1e-6 cuopt-barrier-socp "
        "--time-limit 30 --out results/p1_request_timing_gpu_smoke.csv",
        "P1 GPU single-request smoke; failed gates remain diagnostic observations",
    ),
}

#: Header fields that describe the machine and the software stack rather than
#: one run. If two committed CSVs disagree on any of these, the paper cannot
#: have a single environment paragraph, which is exactly the "assembled from
#: different days" failure the review flagged. Checked, not assumed.
ENV_MUST_AGREE = (
    "platform", "python", "cvxpy", "cuopt", "cudf", "highspy", "ortools",
    "numpy", "cpu_model", "gpu",
)

#: Preference order for the hardware/version block. The first source that
#: exists AND carries an environment header wins; \EnvSourceFile records
#: which one it was, so the choice is visible in the paper's own output
#: rather than buried here.
ENV_PREFERENCE = ("m2solves", "m3", "m1", "m6")


# --------------------------------------------------------------------------
# Formatting -- decided once, here, and nowhere else
#
# Every macro names one of these. The rule matters more than the specific
# digits: solve times on this problem class vary by tens of percent run to
# run (one crossover configuration measured 22.0s and 38.8s on consecutive
# runs), so printing five significant figures would be reporting noise as
# precision, and letting each table pick its own rounding is how two numbers
# that came from the same cell end up looking like different measurements.
#
# Units are NOT baked into the macros. `\MThreeWideHighsSeconds` expands to
# the bare number so the surrounding prose can write "s", "\,s" or put it in
# a table column header as it likes. The one exception is percentages, where
# the "%" is a LaTeX escape rather than a unit choice.
# --------------------------------------------------------------------------

#: `<`, `>` and `|` are not typos in this table. They are legal LaTeX in
#: text mode and they silently render as something else entirely in the
#: default font encoding -- `<=` in a limit description would come out as
#: an inverted exclamation mark. A CSV cell like
#: "ex-ante TE <= 10bp" is exactly the sort of string that lands here.
LATEX_SPECIALS = {
    "\\": r"\textbackslash{}",
    "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
    "_": r"\_", "{": r"\{", "}": r"\}",
    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    "<": r"\textless{}", ">": r"\textgreater{}", "|": r"\textbar{}",
}


def latex_escape(text: str) -> str:
    """Make an arbitrary CSV string safe to paste into running LaTeX text."""
    return "".join(LATEX_SPECIALS.get(ch, ch) for ch in str(text))


def _sig3(x: float) -> str:
    """Three significant figures, capped at two decimals, comma-grouped.

    A single rule for every dimensionless magnitude -- speedups, ratios,
    basis-point levels -- so a 6.93x and a 3,050x are rounded by the same
    policy and the reader can compare them without checking a footnote.
    """
    ax = abs(x)
    if ax >= 1000.0:
        return f"{x:,.0f}"
    if ax >= 100.0:
        return f"{x:.0f}"
    if ax >= 10.0:
        return f"{x:.1f}"
    return f"{x:.2f}"


def fmt_seconds(x: float) -> str:
    """Wall-clock seconds.

    Sub-second timings keep three decimals because scenario generation lives
    there; everything else gets two decimals below 100s and one above, which
    is about four significant figures at the top of the range and never more
    resolution than a median of three repeats can support.
    """
    ax = abs(x)
    if ax < 1.0:
        return f"{x:.3f}"
    if ax < 100.0:
        return f"{x:.2f}"
    return f"{x:.1f}"


def fmt_ratio(x: float) -> str:
    """A multiple ("18.5" meaning 18.5x). No trailing 'x' -- prose adds it."""
    return _sig3(x)


def fmt_bp(x: float) -> str:
    """A basis-point level. Same rounding policy as any other magnitude."""
    return _sig3(x)


def fmt_bp_four(x: float) -> str:
    """Four fixed decimals so a slight lot-repair sigma breach stays visible."""
    return f"{x:.4f}"


def fmt_bp_five(x: float) -> str:
    """Five fixed decimals for strict sigma checks and small sigma excesses."""
    return f"{x:.5f}"


def fmt_pct(x: float) -> str:
    """A fraction printed as a percentage, one decimal, escaped for LaTeX."""
    return f"{100.0 * x:.1f}\\%"


def fmt_pct_two(x: float) -> str:
    """A percentage where the second decimal carries information.

    Risk levels and limit utilisations, not changes. Adjacent rows of the
    economics table differ by around a tenth of a percentage point, so one
    decimal would round two genuinely different portfolios to the same
    number -- and a 2.65\\% issuer cap would print as 2.6\\%, which is a
    different limit. Named separately rather than tightening `pct`, so the
    choice is visible at every call site.
    """
    return f"{100.0 * x:.2f}\\%"


def fmt_pct_three(x: float) -> str:
    """A fractional MIP gap as a percentage with three fixed decimals."""
    return f"{100.0 * x:.3f}\\%"


def fmt_signed_pct(x: float) -> str:
    """A percentage change that carries its sign, because -15.3% is a claim.

    Used where the direction is the finding. A bare "15.3%" next to the word
    "reduction" says the same thing twice and stops saying it correctly the
    moment the sign flips under a re-run.
    """
    return f"{100.0 * x:+.1f}\\%"


def fmt_pp(x: float) -> str:
    """A difference between two percentages, in percentage points."""
    return f"{100.0 * x:.1f}"


def fmt_years(x: float) -> str:
    """Duration in years. Two decimals: a desk quotes duration to 0.01y."""
    return f"{x:.2f}"


def fmt_int(x: float) -> str:
    """A count. Comma-grouped, because 20,040,000 is otherwise unreadable."""
    return f"{int(round(x)):,}"


def fmt_weight(x: float) -> str:
    """A portfolio weight sum. Four decimals: 1.0054 is the whole finding.

    Two decimals would print the over-invested interior points as 1.01 and
    1.00, which is the rounding that lets an infeasible portfolio look fine.
    """
    return f"{x:.4f}"


def fmt_millions(x: float) -> str:
    """A count expressed in millions, one decimal (nonzeros, mostly)."""
    return f"{x / 1e6:.1f}"


def fmt_sci(x: float) -> str:
    """A residual, as LaTeX maths. Wrapped so it works in text mode too.

    Residuals span 1e-16 to 1e-2 in these tables and a fixed decimal rule
    would print most of them as 0.00, which is the one rendering that would
    let a feasibility failure look clean.
    """
    if x == 0.0:
        return r"\ensuremath{0}"
    exp = int(np.floor(np.log10(abs(x))))
    mant = x / (10.0 ** exp)
    return rf"\ensuremath{{{mant:.1f}\times 10^{{{exp}}}}}"


def fmt_sci_four(x: float) -> str:
    """Four significant figures for objective gaps near a strict threshold."""
    if x == 0.0:
        return r"\ensuremath{0}"
    exp = int(np.floor(np.log10(abs(x))))
    mant = x / (10.0 ** exp)
    return rf"\ensuremath{{{mant:.3f}\times 10^{{{exp}}}}}"


def fmt_text(x: Any) -> str:
    """A verbatim string from a CSV, escaped."""
    return latex_escape(x)


FORMATTERS: dict[str, Callable[[Any], str]] = {
    "seconds": fmt_seconds,
    "ratio": fmt_ratio,
    "bp": fmt_bp,
    "bp4": fmt_bp_four,
    "bp5": fmt_bp_five,
    "pct": fmt_pct,
    "pct2": fmt_pct_two,
    "pct3": fmt_pct_three,
    "signed_pct": fmt_signed_pct,
    "pp": fmt_pp,
    "years": fmt_years,
    "int": fmt_int,
    "weight": fmt_weight,
    "millions": fmt_millions,
    "sci": fmt_sci,
    "sci4": fmt_sci_four,
    "text": fmt_text,
}


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------


class MissingData(Exception):
    """A macro's input is absent. Carries a message someone can act on."""


class RecordedNonResult(Exception):
    """A recorded outcome, or a number excluded from publication with a reason.

    A solver that did not finish is a measurement -- "this configuration
    does not complete here" is the finding on that cell -- and it must not
    be confused with a quantity nobody ever ran. Treating a DNF as missing
    data would make the strict build unsatisfiable forever on any cell that
    legitimately does not finish, which would turn the build gate into
    something to be switched off rather than something to be believed.

    A contaminated timing is also a non-result, even when its raw cell is
    numeric. Raised, rather than returned, so derived ratios inherit the
    exclusion instead of accidentally publishing an invalid comparison.
    """

    def __init__(self, label: str) -> None:
        super().__init__(label)
        self.label = label


#: Statuses that mean the solver reached an answer. Anything else on a row
#: with an empty timing is a recorded non-result rather than a gap.
CONVERGED_STATUSES = frozenset({"optimal", "Optimal", "OPTIMAL", "solved"})


@dataclass
class Table:
    """One CSV: its `# key: value` preamble and its rows."""

    key: str
    path: Path
    env: dict[str, str]
    df: pd.DataFrame


@dataclass
class Access:
    """One read, recorded so the emitted comment cannot lie about sources."""

    path: str
    selector: str
    column: str
    raw: str


def _read_table(key: str, results_dir: Path) -> Table:
    """Load a results CSV, splitting the commented environment preamble off.

    The preamble is counted and skipped by line rather than handed to
    pandas as `comment="#"`: that option strips from a '#' anywhere on a
    line, including inside a quoted field, and several of these CSVs carry
    free-text `notes` and JSON `config` columns where that would silently
    truncate a row.
    """
    src = SOURCES[key]
    path = results_dir / src.filename
    if not path.exists():
        raise MissingData(
            f"{path} does not exist ({src.what}). Produce it with:\n"
            f"      {src.producer}"
        )
    env: dict[str, str] = {}
    n_comment = 0
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            n_comment += 1
            if ":" in line:
                k, v = line[1:].split(":", 1)
                env[k.strip()] = v.strip()
    df = pd.read_csv(path, skiprows=n_comment)
    return Table(key=key, path=path, env=env, df=df)


def _selector_text(sel: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v}" for k, v in sel.items()) or "(all rows)"


class Ctx:
    """Loads tables on demand and records every value a macro touches."""

    def __init__(self, results_dir: Path) -> None:
        self.results_dir = results_dir
        self._tables: dict[str, Table] = {}
        self._failed: dict[str, MissingData] = {}
        self.accesses: list[Access] = []

    # -- loading ---------------------------------------------------------

    def table(self, key: str) -> Table:
        if key in self._failed:
            raise self._failed[key]
        if key not in self._tables:
            try:
                self._tables[key] = _read_table(key, self.results_dir)
            except MissingData as exc:
                self._failed[key] = exc
                raise
        return self._tables[key]

    @property
    def loaded(self) -> dict[str, Table]:
        return dict(self._tables)

    # -- tracing ---------------------------------------------------------

    def begin(self) -> None:
        self.accesses = []

    def _record(self, tab: Table, sel: dict[str, Any], column: str,
                raw: str) -> None:
        rel = tab.path.relative_to(REPO) if tab.path.is_relative_to(REPO) \
            else tab.path
        self.accesses.append(
            Access(str(rel).replace("\\", "/"), _selector_text(sel),
                   column, raw))

    # -- row access ------------------------------------------------------

    def rows(self, key: str, sel: dict[str, Any]) -> pd.DataFrame:
        """Every row matching `sel`, or a MissingData naming the miss."""
        tab = self.table(key)
        mask = pd.Series(True, index=tab.df.index)
        for col, want in sel.items():
            if col not in tab.df.columns:
                raise MissingData(
                    f"{tab.path.name} has no column '{col}' (selector "
                    f"{_selector_text(sel)}). The schema changed, or this "
                    f"macro is pointed at the wrong file."
                )
            series = tab.df[col]
            if isinstance(want, bool) or not isinstance(want, (int, float)):
                mask &= series.astype(str).str.strip() == str(want)
            else:
                mask &= np.isclose(
                    pd.to_numeric(series, errors="coerce").to_numpy(
                        dtype=float),
                    float(want), rtol=0.0, atol=1e-9, equal_nan=False)
        return tab.df[mask]

    def row(self, key: str, sel: dict[str, Any]) -> pd.Series:
        """Exactly one row matching `sel`."""
        tab = self.table(key)
        got = self.rows(key, sel)
        if len(got) == 0:
            raise MissingData(
                f"{tab.path.name} has no row where {_selector_text(sel)} "
                f"({SOURCES[key].what}). Regenerate it with:\n"
                f"      {SOURCES[key].producer}"
            )
        if len(got) > 1:
            raise MissingData(
                f"{tab.path.name} has {len(got)} rows where "
                f"{_selector_text(sel)}; the selector must identify one row "
                f"or the macro is averaging things it should not."
            )
        return got.iloc[0]

    def get(self, key: str, sel: dict[str, Any], column: str) -> float:
        """One numeric cell. Empty and NaN are treated as absent, loudly."""
        tab = self.table(key)
        row = self.row(key, sel)
        if column not in tab.df.columns:
            raise MissingData(
                f"{tab.path.name} has no column '{column}'. Regenerate with:"
                f"\n      {SOURCES[key].producer}"
            )
        raw = row[column]
        text = "" if pd.isna(raw) else str(raw).strip()
        if text == "" or text.lower() == "nan":
            raise MissingData(
                f"{tab.path.name}, row [{_selector_text(sel)}], column "
                f"'{column}' is empty. The run wrote the row but not this "
                f"measurement. Regenerate with:\n"
                f"      {SOURCES[key].producer}"
            )
        try:
            value = float(text)
        except ValueError as exc:
            raise MissingData(
                f"{tab.path.name}, row [{_selector_text(sel)}], column "
                f"'{column}' is '{text}', which is not a number."
            ) from exc
        self._record(tab, sel, column, text)
        return value

    def timing(self, key: str, sel: dict[str, Any], column: str) -> float:
        """A timing cell, tolerant of a row that records a non-result.

        Use this and not `get` for wall-clock columns. If the row exists,
        carries a non-converged status, and has no number, the cell is not
        a hole in the data --- the run is telling us the configuration did
        not produce a time. That is reported as such and does not fail the
        build. A converged row with an empty timing still fails, because
        that one really is a hole.
        """
        tab = self.table(key)
        row = self.row(key, sel)
        if column in tab.df.columns:
            raw = row[column]
            text = "" if pd.isna(raw) else str(raw).strip()
            if text not in ("", "nan", "NaN"):
                return self.get(key, sel, column)
        status = ""
        if "status" in tab.df.columns and not pd.isna(row["status"]):
            status = str(row["status"]).strip()
        if status and status not in CONVERGED_STATUSES:
            self._record(tab, sel, column,
                         f"(no timing; status={status})")
            raise RecordedNonResult(status)
        return self.get(key, sel, column)

    def text(self, key: str, sel: dict[str, Any], column: str,
             default: str | None = None) -> str:
        """One string cell, verbatim.

        Empty raises, because a blank where the paper expects prose usually
        means the run did not produce it. Pass `default` only where an empty
        cell is itself a legitimate outcome -- a converged row has no DNF note
        to quote -- so that intent is visible here rather than inferred.
        """
        tab = self.table(key)
        row = self.row(key, sel)
        if column not in tab.df.columns:
            raise MissingData(
                f"{tab.path.name} has no column '{column}'. Regenerate with:"
                f"\n      {SOURCES[key].producer}"
            )
        raw = row[column]
        text = "" if pd.isna(raw) else str(raw).strip()
        if text == "":
            if default is not None:
                self._record(tab, sel, column, f"(empty) -> {default}")
                return default
            raise MissingData(
                f"{tab.path.name}, row [{_selector_text(sel)}], column "
                f"'{column}' is empty."
            )
        self._record(tab, sel, column, text)
        return text

    def agg(self, key: str, sel: dict[str, Any], column: str,
            how: str) -> float:
        """An aggregate: row count, numeric count, max, min, sum, or median.

        `median` exists for replicate cells -- the M9 sweep repeats every
        (dof, pd_mult) point over three scenario seeds, and the paper quotes
        the median draw rather than letting one seed speak for the point.
        """
        tab = self.table(key)
        got = self.rows(key, sel)
        if how == "count":
            self._record(tab, sel, column or "(rows)",
                         f"count={len(got)}")
            return float(len(got))
        if column not in tab.df.columns:
            raise MissingData(
                f"{tab.path.name} has no column '{column}'."
            )
        vals = pd.to_numeric(got[column], errors="coerce").dropna()
        if how == "count_numeric":
            self._record(tab, sel, column,
                         f"numeric count={len(vals)} over {len(got)} row(s)")
            return float(len(vals))
        if vals.empty:
            raise MissingData(
                f"{tab.path.name} has no numeric '{column}' among rows where "
                f"{_selector_text(sel)}. Regenerate with:\n"
                f"      {SOURCES[key].producer}"
            )
        value = {"max": vals.max(), "min": vals.min(),
                 "sum": vals.sum(), "median": vals.median()}[how]
        self._record(tab, sel, column,
                     f"{how} over {len(vals)} row(s) = {value:g}")
        return float(value)

    # -- environment access ----------------------------------------------

    def env(self, key: str, field_name: str) -> str:
        """One `# key: value` header field."""
        tab = self.table(key)
        if field_name not in tab.env:
            raise MissingData(
                f"{tab.path.name} carries no '# {field_name}:' header field. "
                f"The committed file predates the environment-header writer, "
                f"or the run parameter was never recorded. Regenerate with:\n"
                f"      {SOURCES[key].producer}"
            )
        value = tab.env[field_name]
        self._record(tab, {"header": field_name}, field_name, value)
        return value

    def env_source(self, record: bool = False) -> str:
        """The first preference-ordered source that carries a header block."""
        tried = []
        for key in ENV_PREFERENCE:
            tried.append(SOURCES[key].filename)
            try:
                tab = self.table(key)
            except MissingData:
                continue
            if tab.env:
                if record:
                    self._record(tab, {"preference": "->".join(tried)},
                                 "(header block)",
                                 f"{len(tab.env)} header fields")
                return key
        raise MissingData(
            "no committed CSV carries an environment header, so the paper's "
            "hardware and version paragraph cannot be generated. Tried: "
            + ", ".join(tried)
        )

    def env_any(self, field_name: str) -> str:
        """A hardware/version field, from the preferred environment source."""
        return self.env(self.env_source(), field_name)


def env_disagreements(ctx: Ctx) -> list[str]:
    """Fields in ENV_MUST_AGREE that differ across the loaded CSVs.

    A disagreement here is not a formatting problem. It means the tables the
    paper prints side by side were measured on different software stacks, and
    the single environment paragraph the paper wants to write would be false
    for at least one of them.
    """
    seen: dict[str, dict[str, list[str]]] = {}
    for key, tab in ctx.loaded.items():
        for f in ENV_MUST_AGREE:
            if f in tab.env:
                seen.setdefault(f, {}).setdefault(
                    tab.env[f], []).append(tab.path.name)
    out = []
    for f, values in sorted(seen.items()):
        if len(values) > 1:
            detail = "; ".join(
                f"'{v}' in {', '.join(files)}" for v, files in values.items())
            out.append(f"{f}: {detail}")
    return out


# --------------------------------------------------------------------------
# The macro registry
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Macro:
    """One `\\newcommand` the paper may use."""

    name: str
    section: str
    fmt: str
    arith: str
    compute: Callable[[Ctx], Any]


# Row selectors, named once so a typo is a NameError rather than a wrong row.
M1_CUOPT = {"solver": "cuopt-pdlp@0.0001"}
M1_XOVER = {"solver": "cuopt-pdlp+xover@0.0001"}
M1_ORTOOLS = {"solver": "ortools-pdlp@0.0001"}
M1_HIGHS = {"solver": "highs-simplex"}

CELL_BASE, CELL_TALL, CELL_WIDE = "1000x5000", "1000x20000", "2500x8000"
SOLVER_CUOPT = "cuopt-pdlp@0.0001"
SOLVER_XOVER = "cuopt-pdlp+xover@0.0001"
SOLVER_ORTOOLS = "ortools-pdlp@0.0001"
SOLVER_HIGHS = "highs-simplex"


def _m3(cell: str, solver: str) -> dict[str, Any]:
    return {"cell": cell, "solver": solver}


P_BENCH = {"portfolio": "benchmark"}
P_INCUMBENT = {"portfolio": "incumbent w0"}
P_MINDTS = {"portfolio": "min-DTS (no scenarios)"}
P_OPT = {"portfolio": "CVaR-optimal"}
P_FACE_MAXR = {"portfolio": "min-DTS face: max return"}
P_FACE_MINR = {"portfolio": "min-DTS face: min return"}
P_FACE_MINC = {"portfolio": "min-DTS face: min CVaR_oos"}

C_LOOSE = {"config": "cuopt@1e-4"}
C_TIGHT = {"config": "cuopt@1e-6"}
C_XOVER = {"config": "cuopt@1e-4+xover"}
C_HIGHS = {"config": "highs-simplex"}


def _m6(case: str, solver: str) -> dict[str, Any]:
    return {"case": case, "solver": solver}


def _macros_env() -> list[Macro]:
    """Hardware and software stack. Global, and checked to be global."""
    s = "environment"

    def gpu_part(idx: int) -> Callable[[Ctx], str]:
        # "NVIDIA GeForce RTX 4080, 16376 MiB, 595.95" -> name / MiB / driver
        return lambda c: c.env_any("gpu").split(",")[idx].strip()

    return [
        Macro("EnvSourceFile", s, "text",
              "filename of the CSV whose header supplied this block; the "
              "preference order is tried in turn and the first file with a "
              "header wins",
              lambda c: SOURCES[c.env_source(record=True)].filename),
        Macro("EnvSourceTimestamp", s, "text",
              "timestamp_utc of that CSV, so the environment paragraph is "
              "dated by the run rather than by the build",
              lambda c: c.env(c.env_source(), "timestamp_utc")),
        Macro("EnvGpu", s, "text", "field 1 of the '# gpu:' header",
              gpu_part(0)),
        Macro("EnvGpuMemory", s, "text", "field 2 of the '# gpu:' header",
              gpu_part(1)),
        Macro("EnvGpuDriver", s, "text", "field 3 of the '# gpu:' header",
              gpu_part(2)),
        Macro("EnvCpu", s, "text", "'# cpu_model:'",
              lambda c: c.env_any("cpu_model")),
        Macro("EnvCpuThreads", s, "int", "'# cpu_count_logical:'",
              lambda c: float(c.env_any("cpu_count_logical"))),
        Macro("EnvPlatform", s, "text", "'# platform:'",
              lambda c: c.env_any("platform")),
        Macro("EnvPython", s, "text", "'# python:'",
              lambda c: c.env_any("python")),
        Macro("EnvCuopt", s, "text", "'# cuopt:'",
              lambda c: c.env_any("cuopt")),
        Macro("EnvCudf", s, "text", "'# cudf:'",
              lambda c: c.env_any("cudf")),
        Macro("EnvHighs", s, "text", "'# highspy:'",
              lambda c: c.env_any("highspy")),
        Macro("EnvOrtools", s, "text", "'# ortools:'",
              lambda c: c.env_any("ortools")),
        Macro("EnvCvxpy", s, "text", "'# cvxpy:'",
              lambda c: c.env_any("cvxpy")),
        Macro("EnvNumpy", s, "text", "'# numpy:'",
              lambda c: c.env_any("numpy")),
    ]


def _macros_m1() -> list[Macro]:
    """M1, the equity benchmark. This is the result the credit run inverts."""
    s = "M1 equity benchmark"
    return [
        Macro("MOneAssets", s, "int", "n_assets on the cuOpt PDLP row",
              lambda c: c.get("m1", M1_CUOPT, "n_assets")),
        Macro("MOneScenarios", s, "int", "n_scenarios on the same row",
              lambda c: c.get("m1", M1_CUOPT, "n_scenarios")),
        Macro("MOneNonzerosM", s, "millions", "nnz_est / 1e6",
              lambda c: c.get("m1", M1_CUOPT, "nnz_est")),
        Macro("MOneCuoptPdlpSeconds", s, "seconds", "wall_s_median",
              lambda c: c.timing("m1", M1_CUOPT, "wall_s_median")),
        Macro("MOneCuoptXoverSeconds", s, "seconds", "wall_s_median",
              lambda c: c.timing("m1", M1_XOVER, "wall_s_median")),
        Macro("MOneOrtoolsPdlpSeconds", s, "seconds", "wall_s_median",
              lambda c: c.timing("m1", M1_ORTOOLS, "wall_s_median")),
        Macro("MOneHighsSeconds", s, "seconds", "wall_s_median",
              lambda c: c.timing("m1", M1_HIGHS, "wall_s_median")),
        Macro("MOneXoverVsHighs", s, "ratio",
              "HiGHS wall / (cuOpt PDLP + crossover) wall",
              lambda c: (c.timing("m1", M1_HIGHS, "wall_s_median")
                         / c.timing("m1", M1_XOVER, "wall_s_median"))),
        Macro("MOneCuoptVsOrtools", s, "ratio",
              "OR-Tools PDLP wall / cuOpt PDLP wall, both at 1e-4",
              lambda c: (c.timing("m1", M1_ORTOOLS, "wall_s_median")
                         / c.timing("m1", M1_CUOPT, "wall_s_median"))),
        Macro("MOneCrossoverMultiple", s, "ratio",
              "(cuOpt PDLP + crossover) wall / cuOpt PDLP wall -- what the "
              "vertex costs on the equity constraint set",
              lambda c: (c.timing("m1", M1_XOVER, "wall_s_median")
                         / c.timing("m1", M1_CUOPT, "wall_s_median"))),
        Macro("MOneCuoptGapVsRefBp", s, "bp",
              "gap_vs_ref_bp against the HiGHS feasibility-1e-9 reference",
              lambda c: c.get("m1", M1_CUOPT, "gap_vs_ref_bp")),
        Macro("MOnePositions", s, "int", "positions on the crossover row",
              lambda c: c.get("m1", M1_XOVER, "positions")),
        Macro("MOneReference", s, "text", "'# reference:'",
              lambda c: c.env("m1", "reference")),
        Macro("MOneRunsPerSolver", s, "int", "'# runs_per_solver:'",
              lambda c: float(c.env("m1", "runs_per_solver"))),
        Macro("MOneTimeLimitSeconds", s, "seconds", "'# time_limit_s:'",
              lambda c: float(c.env("m1", "time_limit_s"))),
    ]


def _macros_m3() -> list[Macro]:
    """M3, the scaling sweep. Three cells; the paper must not imply more."""
    s = "M3 scaling sweep"
    out: list[Macro] = [
        Macro("MThreeCellCount", s, "int",
              "distinct cells with at least one timed row",
              lambda c: float(c.rows("m3", {"dnf": "False"})["cell"]
                              .nunique())),
        Macro("MThreeTimeLimitSeconds", s, "seconds", "'# time_limit_s:'",
              lambda c: float(c.env("m3", "time_limit_s"))),
        Macro("MThreeRunsPerConfig", s, "int", "'# runs_per_config:'",
              lambda c: float(c.env("m3", "runs_per_config"))),
        Macro("MThreeReplicates", s, "int",
              "'# replicates:' -- clean processes per vertex-producing "
              "configuration; wall_s_median on those rows is the median of "
              "the per-process medians",
              lambda c: float(c.env("m3", "replicates"))),
        Macro("MThreeThreads", s, "int", "'# threads_cpu_solvers:'",
              lambda c: float(c.env("m3", "threads_cpu_solvers"))),
        Macro("MThreeAlpha", s, "text", "'# alpha:'",
              lambda c: c.env("m3", "alpha")),
        Macro("MThreeRiskAversion", s, "text", "'# risk_aversion:'",
              lambda c: c.env("m3", "risk_aversion")),
        Macro("MThreePdlpTolerance", s, "text", "'# pdlp_tolerance:'",
              lambda c: c.env("m3", "pdlp_tolerance")),
        Macro("MThreeGpuContended", s, "text", "'# gpu_contended:'",
              lambda c: c.env("m3", "gpu_contended")),
        Macro("MThreeRamVisibleGiB", s, "text", "'# ram_visible_gib:'",
              lambda c: c.env("m3", "ram_visible_gib")),
    ]

    cells = (("Base", CELL_BASE), ("Tall", CELL_TALL), ("Wide", CELL_WIDE))
    solvers = (("Cuopt", SOLVER_CUOPT), ("Xover", SOLVER_XOVER),
               ("Ortools", SOLVER_ORTOOLS), ("Highs", SOLVER_HIGHS))
    for cname, cell in cells:
        out.append(Macro(
            f"MThree{cname}Bonds", s, "int", "n_bonds",
            (lambda cl: lambda c: c.get("m3", _m3(cl, SOLVER_CUOPT),
                                        "n_bonds"))(cell)))
        out.append(Macro(
            f"MThree{cname}Scenarios", s, "int", "n_scenarios",
            (lambda cl: lambda c: c.get("m3", _m3(cl, SOLVER_CUOPT),
                                        "n_scenarios"))(cell)))
        out.append(Macro(
            f"MThree{cname}NonzerosM", s, "millions", "nnz / 1e6",
            (lambda cl: lambda c: c.get("m3", _m3(cl, SOLVER_CUOPT),
                                        "nnz"))(cell)))
        for sname, solver in solvers:
            out.append(Macro(
                f"MThree{cname}{sname}Seconds", s, "seconds",
                "wall_s_median",
                (lambda cl, sv: lambda c: c.timing("m3", _m3(cl, sv),
                                                "wall_s_median"))(
                    cell, solver)))

    # Feasibility, reported in the same view as the speed it bought.
    for cname, cell in cells:
        out.append(Macro(
            f"MThree{cname}CuoptBudgetResid", s, "sci",
            "check_budget_resid = |sum(w) - 1| from the independent checker",
            (lambda cl: lambda c: c.get("m3", _m3(cl, SOLVER_CUOPT),
                                        "check_budget_resid"))(cell)))
        out.append(Macro(
            f"MThree{cname}OrtoolsBudgetResid", s, "sci",
            "check_budget_resid for the tolerance-matched CPU PDLP",
            (lambda cl: lambda c: c.get("m3", _m3(cl, SOLVER_ORTOOLS),
                                        "check_budget_resid"))(cell)))
        out.append(Macro(
            f"MThree{cname}ResidRatio", s, "ratio",
            "cuOpt budget residual / OR-Tools budget residual at the same "
            "requested 1e-4: how far apart two PDLPs stop",
            (lambda cl: lambda c: (
                c.get("m3", _m3(cl, SOLVER_CUOPT), "check_budget_resid")
                / c.get("m3", _m3(cl, SOLVER_ORTOOLS),
                        "check_budget_resid")))(cell)))
        out.append(Macro(
            f"MThree{cname}CuoptCheckOk", s, "text", "check_ok",
            (lambda cl: lambda c: c.text("m3", _m3(cl, SOLVER_CUOPT),
                                         "check_ok"))(cell)))

    # The checker now keeps the SIGN: check_budget_signed is sum(w) - 1
    # as measured, not its magnitude. The earlier draft asserted a
    # direction ("over-invested", "$5.4mm not funded") from the
    # magnitude-only column, and the signed measurement says the opposite;
    # these macros exist so the paper prints direction from a column
    # rather than inferring it.
    for cname, cell in cells:
        out.append(Macro(
            f"MThree{cname}CuoptBudgetSigned", s, "sci",
            "check_budget_signed = sum(w) - 1 with the sign kept; "
            "negative means the portfolio is under-invested",
            (lambda cl: lambda c: c.get("m3", _m3(cl, SOLVER_CUOPT),
                                        "check_budget_signed"))(cell)))

    out += [
        Macro("MThreeWideUnfundedPerBillionMM", s, "ratio",
              "check_budget_resid on the wide cell x $1,000mm: the dollars "
              "of portfolio that do not balance on a $1bn book. The checker "
              "stores |sum(w) - 1|, so this is a magnitude and the paper "
              "must not call it over- or under-invested.",
              lambda c: 1000.0 * c.get("m3", _m3(CELL_WIDE, SOLVER_CUOPT),
                                       "check_budget_resid")),
        Macro("MThreeWideCuoptVsHighs", s, "ratio",
              "HiGHS wall / cuOpt PDLP wall on the wide (2500x8000) cell",
              lambda c: (c.timing("m3", _m3(CELL_WIDE, SOLVER_HIGHS),
                               "wall_s_median")
                         / c.timing("m3", _m3(CELL_WIDE, SOLVER_CUOPT),
                                 "wall_s_median"))),
        Macro("MThreeTallCuoptVsOrtools", s, "ratio",
              "OR-Tools PDLP wall / cuOpt PDLP wall on the tall cell",
              lambda c: (c.timing("m3", _m3(CELL_TALL, SOLVER_ORTOOLS),
                               "wall_s_median")
                         / c.timing("m3", _m3(CELL_TALL, SOLVER_CUOPT),
                                 "wall_s_median"))),
        Macro("MThreeBaseCuoptVsHighs", s, "ratio",
              "HiGHS wall / cuOpt PDLP wall on the 5M-nonzero cell",
              lambda c: (c.timing("m3", _m3(CELL_BASE, SOLVER_HIGHS),
                               "wall_s_median")
                         / c.timing("m3", _m3(CELL_BASE, SOLVER_CUOPT),
                                 "wall_s_median"))),
        Macro("MThreeBaseXoverVsHighs", s, "ratio",
              "HiGHS wall / (cuOpt + crossover) wall at 5M -- below 1 means "
              "the CPU wins among configurations whose limits hold",
              lambda c: (c.timing("m3", _m3(CELL_BASE, SOLVER_HIGHS),
                               "wall_s_median")
                         / c.timing("m3", _m3(CELL_BASE, SOLVER_XOVER),
                                 "wall_s_median"))),
        Macro("MThreeWideXoverVsHighs", s, "ratio",
              "HiGHS wall / (cuOpt + crossover) wall on the wide cell",
              lambda c: (c.timing("m3", _m3(CELL_WIDE, SOLVER_HIGHS),
                               "wall_s_median")
                         / c.timing("m3", _m3(CELL_WIDE, SOLVER_XOVER),
                                 "wall_s_median"))),
        Macro("MThreeTallXoverVsHighs", s, "ratio",
              "HiGHS wall / (cuOpt + crossover) wall on the tall cell -- "
              "the cell where the two vertex paths are nearest parity",
              lambda c: (c.timing("m3", _m3(CELL_TALL, SOLVER_HIGHS),
                               "wall_s_median")
                         / c.timing("m3", _m3(CELL_TALL, SOLVER_XOVER),
                                 "wall_s_median"))),
        Macro("MThreeBaseCrossoverMultiple", s, "ratio",
              "(cuOpt + crossover) / cuOpt PDLP at 5M -- compare against "
              "MOneCrossoverMultiple, which is the same ratio on the equity "
              "constraint set",
              lambda c: (c.timing("m3", _m3(CELL_BASE, SOLVER_XOVER),
                               "wall_s_median")
                         / c.timing("m3", _m3(CELL_BASE, SOLVER_CUOPT),
                                 "wall_s_median"))),
        Macro("MThreeWideCrossoverMultiple", s, "ratio",
              "(cuOpt + crossover) / cuOpt PDLP on the wide cell",
              lambda c: (c.timing("m3", _m3(CELL_WIDE, SOLVER_XOVER),
                               "wall_s_median")
                         / c.timing("m3", _m3(CELL_WIDE, SOLVER_CUOPT),
                                 "wall_s_median"))),
        Macro("MThreeTallXoverNote", s, "text",
              "the notes cell verbatim WHEN THERE IS ONE. This row was a DNF "
              "in the previous sweep and the paper quoted the solver's own "
              "words, because a crossover DNF is a status on a discarded "
              "warmup rather than a wall-clock timeout and paraphrasing it "
              "had already gone wrong once. On the regenerated sweep the "
              "cell converges, so there is nothing to quote; an empty notes "
              "cell on a converged row is an outcome, not missing data.",
              lambda c: c.text("m3", _m3(CELL_TALL, SOLVER_XOVER), "notes",
                               default="converged; no solver note")),
        Macro("MThreeTallXoverStatus", s, "text", "status",
              lambda c: c.text("m3", _m3(CELL_TALL, SOLVER_XOVER), "status")),
        Macro("MThreeMemCeilingNonzerosM", s, "millions",
              "smallest nnz among rows the harness refused on memory",
              lambda c: c.agg("m3", {"status": "skipped_memory"}, "nnz",
                              "min")),
        Macro("MThreeMemRefusalNote", s, "text",
              "the notes cell on the first memory refusal, verbatim",
              lambda c: c.text("m3", {"status": "skipped_memory",
                                      "cell": "1000x50000",
                                      "solver": "cuopt_pdlp"}, "notes")),
    ]

    # Growth from the 5M cell to each 20M cell, per solver. Three cells
    # cannot fit an exponent, so these are quoted as the ratios they are.
    for sname, solver in solvers:
        for cname, cell in (("Tall", CELL_TALL), ("Wide", CELL_WIDE)):
            out.append(Macro(
                f"MThree{sname}Growth{cname}", s, "ratio",
                f"wall on the {cname.lower()} 20M cell / wall on the 5M cell",
                (lambda sv, cl: lambda c: (
                    c.timing("m3", _m3(cl, sv), "wall_s_median")
                    / c.timing("m3", _m3(CELL_BASE, sv),
                            "wall_s_median")))(solver, cell)))

    # The per-cell feasibility-1e-9 reference ran on the regenerated
    # sweep, so the reference-gap column is populated for every timed row.
    for cname, cell in cells:
        out.append(Macro(
            f"MThree{cname}CuoptGapVsRefBp", s, "bp",
            "gap_vs_ref_bp on the raw cuOpt row against that cell's own "
            "HiGHS feasibility-1e-9 reference; positive is worse than the "
            "reference",
            (lambda cl: lambda c: c.get("m3", _m3(cl, SOLVER_CUOPT),
                                        "gap_vs_ref_bp"))(cell)))
        out.append(Macro(
            f"MThree{cname}XoverIqrRatio", s, "ratio",
            "iqr_across_over_within on the crossover row: the IQR of the "
            "per-process medians over the median within-process IQR. "
            "Below 1 means process identity contributes less spread than "
            "run-to-run noise inside one process.",
            (lambda cl: lambda c: c.get("m3", _m3(cl, SOLVER_XOVER),
                                        "iqr_across_over_within"))(cell)))
    out.append(Macro(
        "MThreeBaseHighsIqrRatio", s, "ratio",
        "iqr_across_over_within on the HiGHS row of the 5M cell: the one "
        "replicated configuration where across-process spread exceeds "
        "within-process spread",
        lambda c: c.get("m3", _m3(CELL_BASE, SOLVER_HIGHS),
                        "iqr_across_over_within")))
    return out


def _macros_m2_solves() -> list[Macro]:
    """M2 timings: the crossover inversion, at the credit working size."""
    s = "M2 credit solve timings"
    return [
        Macro("MTwoCuoptLooseSeconds", s, "seconds", "wall_s_median",
              lambda c: c.timing("m2solves", C_LOOSE, "wall_s_median")),
        Macro("MTwoCuoptLooseSecondsMin", s, "seconds", "wall_s_min",
              lambda c: c.timing("m2solves", C_LOOSE, "wall_s_min")),
        Macro("MTwoCuoptLooseSecondsMax", s, "seconds", "wall_s_max",
              lambda c: c.timing("m2solves", C_LOOSE, "wall_s_max")),
        Macro("MTwoCuoptTightSeconds", s, "seconds", "wall_s_median",
              lambda c: c.timing("m2solves", C_TIGHT, "wall_s_median")),
        Macro("MTwoCuoptXoverSeconds", s, "seconds", "wall_s_median",
              lambda c: c.timing("m2solves", C_XOVER, "wall_s_median")),
        Macro("MTwoHighsSeconds", s, "seconds", "wall_s_median",
              lambda c: c.timing("m2solves", C_HIGHS, "wall_s_median")),
        Macro("MTwoHighsSecondsMin", s, "seconds", "wall_s_min",
              lambda c: c.timing("m2solves", C_HIGHS, "wall_s_min")),
        Macro("MTwoHighsSecondsMax", s, "seconds", "wall_s_max",
              lambda c: c.timing("m2solves", C_HIGHS, "wall_s_max")),
        Macro("MTwoCuoptLooseVsHighs", s, "ratio",
              "HiGHS wall / cuOpt@1e-4 wall -- the headline that a mandate "
              "verdict then takes away",
              lambda c: (c.timing("m2solves", C_HIGHS, "wall_s_median")
                         / c.timing("m2solves", C_LOOSE, "wall_s_median"))),
        Macro("MTwoXoverVsHighs", s, "ratio",
              "(cuOpt + crossover) wall / HiGHS wall -- above 1 means the GPU "
              "path loses once the answer has to be a vertex",
              lambda c: (c.timing("m2solves", C_XOVER, "wall_s_median")
                         / c.timing("m2solves", C_HIGHS, "wall_s_median"))),
        Macro("MTwoXoverSlowerThanHighsPct", s, "pct",
              "(cuOpt + crossover) / HiGHS - 1",
              lambda c: (c.timing("m2solves", C_XOVER, "wall_s_median")
                         / c.timing("m2solves", C_HIGHS, "wall_s_median")
                         - 1.0)),
        Macro("MTwoCrossoverMultiple", s, "ratio",
              "(cuOpt + crossover) / cuOpt@1e-4 on the full credit mandate; "
              "the equity counterpart is MOneCrossoverMultiple",
              lambda c: (c.timing("m2solves", C_XOVER, "wall_s_median")
                         / c.timing("m2solves", C_LOOSE, "wall_s_median"))),
        Macro("MTwoCuoptLooseWorstBreachPct", s, "pct",
              "worst_rel_violation on the cuOpt@1e-4 row: the worst mandate "
              "breach as a fraction of that limit's own scale",
              lambda c: c.get("m2solves", C_LOOSE, "worst_rel_violation")),
        Macro("MTwoCuoptLooseWorstLimit", s, "text", "worst_limit",
              lambda c: c.text("m2solves", C_LOOSE, "worst_limit")),
        Macro("MTwoCuoptTightWorstBreach", s, "sci", "worst_rel_violation",
              lambda c: c.get("m2solves", C_TIGHT, "worst_rel_violation")),
        Macro("MTwoXoverWorstBreach", s, "sci", "worst_rel_violation",
              lambda c: c.get("m2solves", C_XOVER, "worst_rel_violation")),
        Macro("MTwoHighsWorstBreach", s, "sci", "worst_rel_violation",
              lambda c: c.get("m2solves", C_HIGHS, "worst_rel_violation")),
        Macro("MTwoCuoptLooseBudgetSigned", s, "sci",
              "budget_signed = sum(w) - 1 on the cuopt@1e-4 row, sign "
              "kept; negative means under-invested",
              lambda c: c.get("m2solves", C_LOOSE, "budget_signed")),
        Macro("MTwoEconomicsReportedFrom", s, "text",
              "'# economics_reported_from:' -- which configuration's weights "
              "the economics table is actually built on",
              lambda c: c.env("m2solves", "economics_reported_from")),
        Macro("MTwoScenarios", s, "int", "'# n_scenarios:'",
              lambda c: float(c.env("m2solves", "n_scenarios"))),
        Macro("MTwoTurnoverLimitPct", s, "pct", "'# turnover:'",
              lambda c: float(c.env("m2solves", "turnover"))),
        Macro("MTwoRunsPerConfig", s, "int", "'# runs_per_config:'",
              lambda c: float(c.env("m2solves", "runs_per_config"))),
    ]


def _macros_m2_econ() -> list[Macro]:
    """M2 economics: what the optimizer earned, out of sample."""
    s = "M2 credit economics"
    out: list[Macro] = []
    rows = (("Bench", P_BENCH), ("Incumbent", P_INCUMBENT),
            ("MinDts", P_MINDTS), ("Optimal", P_OPT))
    cols = (("Excess", "exp.excess", "text"),
            ("CvarIn", "CVaR_in", "pct2"),
            ("CvarOos", "CVaR_oos", "pct2"),
            ("VarOos", "VaR_oos", "pct2"),
            ("Oas", "OAS(bp)", "bp"),
            ("Duration", "duration", "years"),
            ("Dts", "DTS", "bp"),
            ("Positions", "positions", "int"),
            ("Issuers", "issuers", "int"))
    for rname, sel in rows:
        for cname, col, fmt in cols:
            if cname == "Excess":
                # Monthly expected excess return is ~0.0012; a percentage
                # with one decimal would print 0.1% for every row and hide
                # the differences the attribution section is about. Quoted
                # in basis points per month instead, via its own formatter.
                out.append(Macro(
                    f"MTwo{rname}ExcessBpMonthly", s, "bp",
                    "exp.excess x 1e4, monthly basis points",
                    (lambda sl: lambda c: 1e4 * c.get("m2econ", sl,
                                                      "exp.excess"))(sel)))
                continue
            out.append(Macro(
                f"MTwo{rname}{cname}", s, fmt, col,
                (lambda sl, cl: lambda c: c.get("m2econ", sl, cl))(sel, col)))

    out += [
        Macro("MTwoBonds", s, "int",
              "positions on the benchmark row -- the benchmark holds the "
              "whole universe, so this is the bond count. The committed "
              "economics CSV records the universe size nowhere else.",
              lambda c: c.get("m2econ", P_BENCH, "positions")),
        Macro("MTwoCvarOosVsBench", s, "signed_pct",
              "CVaR_oos(optimal) / CVaR_oos(benchmark) - 1",
              lambda c: (c.get("m2econ", P_OPT, "CVaR_oos")
                         / c.get("m2econ", P_BENCH, "CVaR_oos") - 1.0)),
        Macro("MTwoCvarOosVsIncumbent", s, "signed_pct",
              "CVaR_oos(optimal) / CVaR_oos(incumbent) - 1 -- where the trade "
              "actually starts",
              lambda c: (c.get("m2econ", P_OPT, "CVaR_oos")
                         / c.get("m2econ", P_INCUMBENT, "CVaR_oos") - 1.0)),
        Macro("MTwoCvarInVsBench", s, "signed_pct",
              "CVaR_in(optimal) / CVaR_in(benchmark) - 1 -- the fit statistic",
              lambda c: (c.get("m2econ", P_OPT, "CVaR_in")
                         / c.get("m2econ", P_BENCH, "CVaR_in") - 1.0)),
        Macro("MTwoOverfitGapPp", s, "pp",
              "in-sample reduction minus out-of-sample reduction, in "
              "percentage points. Positive means the in-sample number "
              "flatters. An upper bound on the overfit, not a measurement "
              "of it: the two scenario sets are independent draws, so this "
              "mixes tail-overfit with sampling error in the evaluation set.",
              lambda c: ((c.get("m2econ", P_OPT, "CVaR_oos")
                          / c.get("m2econ", P_BENCH, "CVaR_oos"))
                         - (c.get("m2econ", P_OPT, "CVaR_in")
                            / c.get("m2econ", P_BENCH, "CVaR_in")))),
        Macro("MTwoMinDtsCvarOosVsBench", s, "signed_pct",
              "CVaR_oos(min-DTS) / CVaR_oos(benchmark) - 1: what a "
              "scenario-free de-risking rule gets on its own",
              lambda c: (c.get("m2econ", P_MINDTS, "CVaR_oos")
                         / c.get("m2econ", P_BENCH, "CVaR_oos") - 1.0)),
        Macro("MTwoAttributionCornerShare", s, "pct",
              "min-DTS reduction / optimal reduction, both vs benchmark "
              "out-of-sample: the share of the headline that is "
              "mandate-corner de-risking",
              lambda c: ((1.0 - c.get("m2econ", P_MINDTS, "CVaR_oos")
                          / c.get("m2econ", P_BENCH, "CVaR_oos"))
                         / (1.0 - c.get("m2econ", P_OPT, "CVaR_oos")
                            / c.get("m2econ", P_BENCH, "CVaR_oos")))),
        Macro("MTwoAttributionSelectionShare", s, "pct",
              "1 - MTwoAttributionCornerShare: the incremental, "
              "scenario-driven share",
              lambda c: (1.0 - (1.0 - c.get("m2econ", P_MINDTS, "CVaR_oos")
                                / c.get("m2econ", P_BENCH, "CVaR_oos"))
                         / (1.0 - c.get("m2econ", P_OPT, "CVaR_oos")
                            / c.get("m2econ", P_BENCH, "CVaR_oos")))),
        Macro("MTwoMinDtsGiveUpPct", s, "pct",
              "1 - exp.excess(min-DTS) / exp.excess(benchmark)",
              lambda c: (1.0 - c.get("m2econ", P_MINDTS, "exp.excess")
                         / c.get("m2econ", P_BENCH, "exp.excess"))),
        Macro("MTwoOptimalGiveUpPct", s, "pct",
              "1 - exp.excess(CVaR-optimal) / exp.excess(benchmark)",
              lambda c: (1.0 - c.get("m2econ", P_OPT, "exp.excess")
                         / c.get("m2econ", P_BENCH, "exp.excess"))),
        Macro("MTwoCarryEdgeOverMinDtsPct", s, "pct",
              "exp.excess(CVaR-optimal) / exp.excess(min-DTS) - 1, at the "
              "same DTS and the same duration",
              lambda c: (c.get("m2econ", P_OPT, "exp.excess")
                         / c.get("m2econ", P_MINDTS, "exp.excess") - 1.0)),
        Macro("MTwoTailEdgeOverMinDtsPct", s, "pct",
              "1 - CVaR_oos(CVaR-optimal) / CVaR_oos(min-DTS)",
              lambda c: (1.0 - c.get("m2econ", P_OPT, "CVaR_oos")
                         / c.get("m2econ", P_MINDTS, "CVaR_oos"))),
        Macro("MTwoSharedDtsGap", s, "sci",
              "|DTS(CVaR-optimal) - DTS(min-DTS)| -- printed rather than "
              "asserted, so 'identical DTS' is a number the reader checks "
              "instead of a word the author chose",
              lambda c: abs(c.get("m2econ", P_OPT, "DTS")
                            - c.get("m2econ", P_MINDTS, "DTS"))),
        Macro("MTwoSharedDurationGap", s, "sci",
              "|duration(CVaR-optimal) - duration(min-DTS)|",
              lambda c: abs(c.get("m2econ", P_OPT, "duration")
                            - c.get("m2econ", P_MINDTS, "duration"))),
        # The min-DTS optimal face, sampled at its extreme vertices. The
        # committed run solves the max-return and min-return vertices at
        # fixed DTS (ex-ante selectable: no evaluation data used) and the
        # min-CVaR_oos vertex (selected USING the evaluation set:
        # hindsight, quoted only as such). These macros are why the paper
        # reports a band instead of the 65/35 split, which a comparison of
        # them shows is not identified.
        Macro("MTwoFaceMaxRetExcessBpMonthly", s, "bp",
              "exp.excess x 1e4 on the max-return vertex of the min-DTS "
              "optimal face, monthly basis points",
              lambda c: 1e4 * c.get("m2econ", P_FACE_MAXR, "exp.excess")),
        Macro("MTwoFaceMinRetExcessBpMonthly", s, "bp",
              "exp.excess x 1e4 on the min-return vertex of the same face",
              lambda c: 1e4 * c.get("m2econ", P_FACE_MINR, "exp.excess")),
        Macro("MTwoFaceBestExanteCvarOosVsBench", s, "signed_pct",
              "CVaR_oos(max-return vertex) / CVaR_oos(benchmark) - 1: the "
              "best out-of-sample tail among face vertices chosen without "
              "the evaluation set",
              lambda c: (c.get("m2econ", P_FACE_MAXR, "CVaR_oos")
                         / c.get("m2econ", P_BENCH, "CVaR_oos") - 1.0)),
        Macro("MTwoFaceWorstExanteCvarOosVsBench", s, "signed_pct",
              "CVaR_oos(min-return vertex) / CVaR_oos(benchmark) - 1: the "
              "worst such vertex, which is WORSE than the benchmark",
              lambda c: (c.get("m2econ", P_FACE_MINR, "CVaR_oos")
                         / c.get("m2econ", P_BENCH, "CVaR_oos") - 1.0)),
        Macro("MTwoFaceCornerShareLoPct", s, "signed_pct",
              "reduction(min-return vertex) / reduction(CVaR-optimal), "
              "both vs benchmark out-of-sample: the lower end of the "
              "interval the 'mandate-corner share' actually lives in",
              lambda c: ((1.0 - c.get("m2econ", P_FACE_MINR, "CVaR_oos")
                          / c.get("m2econ", P_BENCH, "CVaR_oos"))
                         / (1.0 - c.get("m2econ", P_OPT, "CVaR_oos")
                            / c.get("m2econ", P_BENCH, "CVaR_oos")))),
        Macro("MTwoFaceCornerShareHiPct", s, "signed_pct",
              "reduction(max-return vertex) / reduction(CVaR-optimal): "
              "the upper end of the same interval",
              lambda c: ((1.0 - c.get("m2econ", P_FACE_MAXR, "CVaR_oos")
                          / c.get("m2econ", P_BENCH, "CVaR_oos"))
                         / (1.0 - c.get("m2econ", P_OPT, "CVaR_oos")
                            / c.get("m2econ", P_BENCH, "CVaR_oos")))),
        Macro("MTwoFaceHindsightCvarOosVsBench", s, "signed_pct",
              "CVaR_oos(min-CVaR_oos vertex) / CVaR_oos(benchmark) - 1. "
              "Selected USING the evaluation set: hindsight, excluded "
              "from the ex-ante band -- and it beats the optimizer.",
              lambda c: (c.get("m2econ", P_FACE_MINC, "CVaR_oos")
                         / c.get("m2econ", P_BENCH, "CVaR_oos") - 1.0)),
    ]
    return out


def _macros_m2_limits() -> list[Macro]:
    """The limit sheet as the chosen portfolio actually sits against it."""
    s = "M2 limit sheet"
    turnover = {"family": "turnover", "item": "one-way turnover"}
    issuer = {"family": "issuer", "item": "max issuer weight"}
    dts_floor = {"family": "dts", "item": "DTS vs floor (bp-y)"}
    dur_total = {"family": "duration", "item": "total duration"}
    rating_b = {"family": "rating", "item": "B"}
    rating_bb = {"family": "rating", "item": "BB"}
    return [
        Macro("MTwoLimitRowCount", s, "int",
              "rows in the exposure report -- every limit, scored",
              lambda c: c.agg("m2limits", {}, "", "count")),
        Macro("MTwoBindingCount", s, "int",
              "rows with binding = True",
              lambda c: c.agg("m2limits", {"binding": "True"}, "", "count")),
        Macro("MTwoFamilyCount", s, "int",
              "distinct values of the family column: the mandate as a PM "
              "counts it",
              lambda c: float(c.table("m2limits").df["family"].nunique())),
        Macro("MTwoTurnoverRealizedPct", s, "pct2",
              "portfolio column of the turnover row",
              lambda c: c.get("m2limits", turnover, "portfolio")),
        Macro("MTwoIssuerCapRealizedPct", s, "pct2",
              "portfolio column of the max-issuer-weight row",
              lambda c: c.get("m2limits", issuer, "portfolio")),
        Macro("MTwoDtsFloorMultiple", s, "ratio",
              "portfolio DTS / benchmark DTS on the DTS-floor row: the "
              "optimizer sits exactly on the floor",
              lambda c: (c.get("m2limits", dts_floor, "portfolio")
                         / c.get("m2limits", dts_floor, "benchmark"))),
        Macro("MTwoDurationVsBenchYears", s, "sci",
              "|portfolio - benchmark| on the total-duration row",
              lambda c: abs(c.get("m2limits", dur_total, "portfolio")
                            - c.get("m2limits", dur_total, "benchmark"))),
        Macro("MTwoRatingBVsBench", s, "ratio",
              "portfolio / benchmark weight in single-B: the band's floor is "
              "multiplicative, and the optimizer is standing on it",
              lambda c: (c.get("m2limits", rating_b, "portfolio")
                         / c.get("m2limits", rating_b, "benchmark"))),
        Macro("MTwoRatingBBVsBench", s, "ratio",
              "portfolio / benchmark weight in BB",
              lambda c: (c.get("m2limits", rating_bb, "portfolio")
                         / c.get("m2limits", rating_bb, "benchmark"))),
    ]


def _macros_m6() -> list[Macro]:
    """M6: the mandate as written is not a linear program."""
    s = "M6 tracking-error mandate"
    grid = (("Five", "mad@5bp", 5.0), ("Ten", "mad@10bp", 10.0),
            ("Twenty", "mad@20bp", 20.0), ("Fifty", "mad@50bp", 50.0),
            ("Hundred", "mad@100bp", 100.0))
    out: list[Macro] = [
        Macro("MSixBonds", s, "int", "'# n_bonds:'",
              lambda c: float(c.env("m6", "n_bonds"))),
        Macro("MSixScenarios", s, "int", "'# n_scenarios:'",
              lambda c: float(c.env("m6", "n_scenarios"))),
        Macro("MSixUniverseSeed", s, "text", "'# universe_seed:'",
              lambda c: c.env("m6", "universe_seed")),
        Macro("MSixScenarioSeed", s, "text", "'# scenario_seed:'",
              lambda c: c.env("m6", "scenario_seed")),
        Macro("MSixPeriodsPerYear", s, "text", "'# periods_per_year:'",
              lambda c: c.env("m6", "periods_per_year")),
        Macro("MSixBudgetSource", s, "text", "'# te_budget_source:'",
              lambda c: c.env("m6", "te_budget_source")),
        Macro("MSixMandateBudgetBp", s, "bp",
              "te_budget_bp on the sigma row: the published 10bp mandate",
              lambda c: c.get("m6", _m6("sigma@10bp", "cuopt@1e-4"),
                              "te_budget_bp")),
        Macro("MSixIncumbentTeSigmaBp", s, "bp",
              "'# incumbent_te_sigma_bp:' -- where the book already sits",
              lambda c: float(c.env("m6", "incumbent_te_sigma_bp"))),
        Macro("MSixUnconstrainedTeSigmaBp", s, "bp",
              "'# unconstrained_te_sigma_bp:'",
              lambda c: float(c.env("m6", "unconstrained_te_sigma_bp"))),
        Macro("MSixUnconstrainedVsBudget", s, "ratio",
              "unconstrained TE sigma / the 10bp mandate budget",
              lambda c: (float(c.env("m6", "unconstrained_te_sigma_bp"))
                         / c.get("m6", _m6("sigma@10bp", "cuopt@1e-4"),
                                 "te_budget_bp"))),
        Macro("MSixSigmaNotInClassCount", s, "int",
              "LP solvers that returned NOT-IN-CLASS on the sigma row -- the "
              "result that no timing caveat can touch",
              lambda c: c.agg("m6", {"te_mode": "sigma",
                                     "status": "NOT-IN-CLASS"}, "",
                              "count")),
        Macro("MSixSigmaNotInClassNote", s, "text",
              "the note column on the sigma row, verbatim",
              lambda c: c.text("m6", _m6("sigma@10bp", "cuopt@1e-4"),
                               "note")),
    ]

    # The MAD surrogate, scored at the tightest LP tolerance that converged.
    # 1e-6 rather than 1e-4 on purpose: at 1e-4 the reader can dismiss the
    # miss as solver slop, and the whole point is that it is not.
    for gname, case, budget in grid:
        out.append(Macro(
            f"MSixMad{gname}BudgetBp", s, "bp", "te_budget_bp",
            (lambda cs: lambda c: c.get("m6", _m6(cs, "cuopt@1e-6"),
                                        "te_budget_bp"))(case)))
        out.append(Macro(
            f"MSixMad{gname}SigmaBp", s, "bp",
            "te_sigma_bp achieved by the MAD-constrained solution at 1e-6, "
            "recomputed from the scenario matrix independently of the LP",
            (lambda cs: lambda c: c.get("m6", _m6(cs, "cuopt@1e-6"),
                                        "te_sigma_bp"))(case)))
        out.append(Macro(
            f"MSixMad{gname}MadImpliedBp", s, "bp",
            "te_mad_implied_bp: the quantity the LP constraint actually "
            "bounds, recomputed from the weights. Where this equals the "
            "budget the surrogate did its job and the mandate still missed; "
            "where it falls short of the budget the constraint was not "
            "binding and the overshoot ratio below is not a mandate miss.",
            (lambda cs: lambda c: c.get("m6", _m6(cs, "cuopt@1e-6"),
                                        "te_mad_implied_bp"))(case)))
        out.append(Macro(
            f"MSixMad{gname}Binding", s, "ratio",
            "te_mad_implied_bp / te_budget_bp: 1.00 means the LP sat on its "
            "own constraint, below 1.00 means the budget was slack",
            (lambda cs: lambda c: (
                c.get("m6", _m6(cs, "cuopt@1e-6"), "te_mad_implied_bp")
                / c.get("m6", _m6(cs, "cuopt@1e-6"),
                        "te_budget_bp")))(case)))
        out.append(Macro(
            f"MSixMad{gname}Overshoot", s, "ratio",
            "te_sigma_bp / te_budget_bp: how many times the mandate the "
            "LP-representable surrogate actually delivers",
            (lambda cs: lambda c: (
                c.get("m6", _m6(cs, "cuopt@1e-6"), "te_sigma_bp")
                / c.get("m6", _m6(cs, "cuopt@1e-6"),
                        "te_budget_bp")))(case)))
        out.append(Macro(
            f"MSixMad{gname}Seconds", s, "seconds",
            "wall_s at 1e-6",
            (lambda cs: lambda c: c.timing("m6", _m6(cs, "cuopt@1e-6"),
                                        "wall_s"))(case)))

    out += [
        Macro("MSixMadTenLooseSeconds", s, "seconds",
              "wall_s for cuopt@1e-4 at the 10bp MAD budget",
              lambda c: c.timing("m6", _m6("mad@10bp", "cuopt@1e-4"),
                              "wall_s")),
        Macro("MSixMadTenTightVsLoose", s, "ratio",
              "wall_s(1e-6) / wall_s(1e-4) at the 10bp MAD budget",
              lambda c: (c.timing("m6", _m6("mad@10bp", "cuopt@1e-6"),
                                  "wall_s")
                         / c.timing("m6", _m6("mad@10bp", "cuopt@1e-4"),
                                 "wall_s"))),
        Macro("MSixMadTenXoverStatus", s, "text",
              "status of cuopt@1e-4+xover at the 10bp MAD budget",
              lambda c: c.text("m6", _m6("mad@10bp", "cuopt@1e-4+xover"),
                               "status")),
        Macro("MSixMadTenXoverSeconds", s, "seconds", "wall_s",
              lambda c: c.timing("m6", _m6("mad@10bp", "cuopt@1e-4+xover"),
                              "wall_s")),
        Macro("MSixMadTenHighsStatus", s, "text",
              "status of HiGHS dual simplex at the 10bp MAD budget",
              lambda c: c.text("m6", _m6("mad@10bp", "highs-simplex"),
                               "status")),
        Macro("MSixMadTenHighsSeconds", s, "seconds", "wall_s",
              lambda c: c.timing("m6", _m6("mad@10bp", "highs-simplex"),
                              "wall_s")),
        Macro("MSixRespectingCount", s, "int",
              "rows with respects_mandate = True across the whole sweep",
              lambda c: c.agg("m6", {"respects_mandate": "True"}, "",
                              "count")),
        Macro("MSixRowCount", s, "int", "rows in the sweep",
              lambda c: c.agg("m6", {}, "", "count")),
    ]

    # ---------------------------------------------------------------
    # The sweep as a benchmark, not only as a formulation argument.
    #
    # The working-size run carries a clean environment header
    # (gpu_contended: no), so unlike the conic reference it is entitled
    # to report seconds. Everything below is a timing or a
    # did-not-finish, and the did-not-finish rows are results: a vertex
    # method that does not reach a vertex inside the cap is the finding
    # on that cell, not a gap in the data.
    # ---------------------------------------------------------------
    MAD_ROWS = {"te_mode": "mad"}
    S_LOOSE, S_TIGHT = "cuopt@1e-4", "cuopt@1e-6"
    S_XOVER, S_HIGHS = "cuopt@1e-4+xover", "highs-simplex"

    def _mad(solver: str, **extra: Any) -> dict[str, Any]:
        return {**MAD_ROWS, "solver": solver, **extra}

    out += [
        Macro("MSixTimeLimitS", s, "int",
              "'# time_limit_s:' -- the per-solve cap every DNF below hit",
              lambda c: float(c.env("m6", "time_limit_s"))),
        Macro("MSixBudgetCount", s, "int",
              "MAD budgets in the sweep (rows on the tightest cuOpt arm)",
              lambda c: c.agg("m6", _mad(S_TIGHT), "", "count")),

        # The two converging cuOpt arms, as a range over all five budgets.
        Macro("MSixLooseSecondsMin", s, "seconds",
              "fastest wall_s on the cuopt@1e-4 arm across the budget grid",
              lambda c: c.agg("m6", _mad(S_LOOSE), "wall_s", "min")),
        Macro("MSixLooseSecondsMax", s, "seconds",
              "slowest wall_s on the cuopt@1e-4 arm across the budget grid",
              lambda c: c.agg("m6", _mad(S_LOOSE), "wall_s", "max")),
        Macro("MSixTightSecondsMin", s, "seconds",
              "fastest wall_s on the cuopt@1e-6 arm across the budget grid",
              lambda c: c.agg("m6", _mad(S_TIGHT), "wall_s", "min")),
        Macro("MSixTightSecondsMax", s, "seconds",
              "slowest wall_s on the cuopt@1e-6 arm across the budget grid",
              lambda c: c.agg("m6", _mad(S_TIGHT), "wall_s", "max")),

        # Is the fast arm admissible? Scored on the desk gate, not on the
        # raw tolerance, and reported both ways round so the fast number
        # cannot be quoted without its verdict.
        Macro("MSixLooseDeskOkCount", s, "int",
              "budgets where cuopt@1e-4 clears the desk gate",
              lambda c: c.agg("m6", _mad(S_LOOSE, gate_verdict="DESK-OK"),
                              "", "count")),
        Macro("MSixLooseBreachCount", s, "int",
              "budgets where cuopt@1e-4 fails the desk gate",
              lambda c: c.agg("m6", _mad(S_LOOSE, gate_verdict="BREACH"),
                              "", "count")),
        Macro("MSixLooseBreachWorstLimit", s, "text",
              "desk_worst_limit on the cuopt@1e-4 breach rows -- which "
              "limit family actually fails there",
              lambda c: c.text("m6", _m6("mad@50bp", S_LOOSE),
                               "desk_worst_limit")),
        Macro("MSixLooseBreachSameLimitCount", s, "int",
              "how many of the breach rows name that same limit; equal to "
              "MSixLooseBreachCount means the failure is one family, not a "
              "scatter",
              lambda c: c.agg("m6", _mad(S_LOOSE, gate_verdict="BREACH",
                                         desk_worst_limit="one-way "
                                                          "turnover"),
                              "", "count")),
        Macro("MSixMadWorstBindingBreachBp", s, "sci",
              "worst te_breach_bp on the cuopt@1e-6 arm: how far the LP "
              "sits outside the constraint it was actually given. The "
              "slack budget's row is negative, so the max is the worst "
              "binding miss.",
              lambda c: c.agg("m6", _mad(S_TIGHT), "te_breach_bp", "max")),

        # The vertex-seeking arms: what they did instead of finishing.
        Macro("MSixHighsDnfCount", s, "int",
              "budgets where HiGHS dual simplex hit the cap",
              lambda c: c.agg("m6", _mad(S_HIGHS, status="user_limit"),
                              "", "count")),
        Macro("MSixHighsDnfSecondsMin", s, "seconds",
              "shortest wall_s among those DNFs",
              lambda c: c.agg("m6", _mad(S_HIGHS, status="user_limit"),
                              "wall_s", "min")),
        Macro("MSixHighsDnfSecondsMax", s, "seconds",
              "longest wall_s among those DNFs",
              lambda c: c.agg("m6", _mad(S_HIGHS, status="user_limit"),
                              "wall_s", "max")),
        Macro("MSixXoverDnfCount", s, "int",
              "budgets where cuOpt PDLP + crossover hit the cap",
              lambda c: c.agg("m6", _mad(S_XOVER, status="user_limit"),
                              "", "count")),
        Macro("MSixXoverConvergedCount", s, "int",
              "budgets where cuOpt PDLP + crossover converged at all",
              lambda c: c.agg("m6", _mad(S_XOVER, status="optimal"),
                              "", "count")),
        Macro("MSixXoverErrorStatus", s, "text",
              "the status cuOpt PDLP + crossover returned at the one "
              "budget where it neither converged nor hit the cap",
              lambda c: c.text("m6", _m6("mad@20bp", S_XOVER), "status")),
        Macro("MSixXoverErrorBudgetBp", s, "bp",
              "the budget that error occurs at",
              lambda c: c.get("m6", _m6("mad@20bp", S_XOVER),
                              "te_budget_bp")),

        # The one crossover that finished, and what it cost.
        Macro("MSixMadHundredLooseSeconds", s, "seconds",
              "wall_s for cuopt@1e-4 at the 100bp MAD budget -- the raw "
              "first-order point the crossover starts from",
              lambda c: c.timing("m6", _m6("mad@100bp", S_LOOSE),
                                 "wall_s")),
        Macro("MSixMadHundredXoverSeconds", s, "seconds",
              "wall_s for cuopt@1e-4+xover at the 100bp MAD budget: the "
              "only budget in the sweep where crossover converged",
              lambda c: c.timing("m6", _m6("mad@100bp", S_XOVER),
                                 "wall_s")),
        Macro("MSixMadHundredXoverVsLoose", s, "ratio",
              "crossover wall / raw wall at the 100bp MAD budget -- the "
              "TE family's crossover bill, a second instance of the "
              "ablation's mechanism on a different constraint family",
              lambda c: (c.timing("m6", _m6("mad@100bp", S_XOVER),
                                  "wall_s")
                         / c.timing("m6", _m6("mad@100bp", S_LOOSE),
                                    "wall_s"))),
        Macro("MSixMadHundredXoverSigmaBp", s, "bp",
              "te_sigma_bp of the crossed-over vertex: it lands on the "
              "unconstrained optimum, which is why the budget was slack",
              lambda c: c.get("m6", _m6("mad@100bp", S_XOVER),
                              "te_sigma_bp")),
    ]

    # The conic reference: a smaller instance, solved by a solver that CAN
    # express the constraint, so the MAD gap is measured against a mandate
    # that was actually met rather than inferred.
    #
    # THIS FILE IS A FORMULATION CHECK, NOT A BENCHMARK. Its environment
    # header reads `gpu_contended: yes -- timings unreliable`, because
    # capture_environment() samples the card at WRITE time, after every
    # solve in the run has finished; the sample therefore describes the
    # run's own tail rather than the conditions the solves ran under.
    # The *_Seconds macros below stay defined so the file's contents
    # remain fully documented here, but the paper must not quote them or
    # any ratio built from them -- MSixRefCuoptVsClarabelMad included.
    # Wall-clock claims about the TE family belong to the working-size
    # file above, which carries `gpu_contended: no`.
    r = "M6 conic reference"
    ref_clar_sigma = {"case": "sigma@10bp", "solver": "clarabel"}
    ref_clar_mad = {"case": "mad@10bp", "solver": "clarabel"}
    ref_cuopt_mad = {"case": "mad@10bp", "solver": "cuopt@1e-4"}
    ref_highs_sigma = {"case": "sigma@10bp", "solver": "highs-simplex"}
    ref_highs_mad = {"case": "mad@10bp", "solver": "highs-simplex"}
    out += [
        Macro("MSixRefBonds", r, "int", "'# n_bonds:'",
              lambda c: float(c.env("m6ref", "n_bonds"))),
        Macro("MSixRefScenarios", r, "int", "'# n_scenarios:'",
              lambda c: float(c.env("m6ref", "n_scenarios"))),
        Macro("MSixRefSigmaAchievedBp", r, "bp",
              "te_sigma_bp from the SOC solve: the mandate, met exactly",
              lambda c: c.get("m6ref", ref_clar_sigma, "te_sigma_bp")),
        Macro("MSixRefSigmaSeconds", r, "seconds", "wall_s",
              lambda c: c.timing("m6ref", ref_clar_sigma, "wall_s")),
        Macro("MSixRefMadSigmaBp", r, "bp",
              "te_sigma_bp from the MAD solve by the SAME solver on the SAME "
              "instance -- so the gap is the constraint form, not the solver",
              lambda c: c.get("m6ref", ref_clar_mad, "te_sigma_bp")),
        Macro("MSixRefMadSeconds", r, "seconds", "wall_s",
              lambda c: c.timing("m6ref", ref_clar_mad, "wall_s")),
        Macro("MSixRefMadOvershoot", r, "ratio",
              "te_sigma_bp(MAD solve) / te_sigma_bp(SOC solve), one solver, "
              "one instance, one budget",
              lambda c: (c.get("m6ref", ref_clar_mad, "te_sigma_bp")
                         / c.get("m6ref", ref_clar_sigma, "te_sigma_bp"))),
        Macro("MSixRefSigmaOverMadImplied", r, "ratio",
              "te_sigma_bp / te_mad_implied_bp on the SOC solution: above 1 "
              "means active returns are fatter-tailed than Gaussian, which "
              "is why a MAD budget calibrated on a Gaussian is loose",
              lambda c: (c.get("m6ref", ref_clar_sigma, "te_sigma_bp")
                         / c.get("m6ref", ref_clar_sigma,
                                 "te_mad_implied_bp"))),
        Macro("MSixRefCuoptMadSeconds", r, "seconds",
              "wall_s for cuopt@1e-4 on the MAD form at reference scale",
              lambda c: c.timing("m6ref", ref_cuopt_mad, "wall_s")),
        Macro("MSixRefCuoptMadSigmaBp", r, "bp", "te_sigma_bp",
              lambda c: c.get("m6ref", ref_cuopt_mad, "te_sigma_bp")),
        Macro("MSixRefCuoptVsClarabelMad", r, "ratio",
              "clarabel wall / cuopt wall on the identical MAD problem",
              lambda c: (c.timing("m6ref", ref_clar_mad, "wall_s")
                         / c.timing("m6ref", ref_cuopt_mad, "wall_s"))),
        Macro("MSixRefHighsSigmaStatus", r, "text",
              "status of HiGHS on the sigma form at reference scale",
              lambda c: c.text("m6ref", ref_highs_sigma, "status")),
        Macro("MSixRefHighsMadSigmaBp", r, "bp",
              "te_sigma_bp from the MAD solve by HiGHS dual simplex: a "
              "third algorithm family on the identical instance",
              lambda c: c.get("m6ref", ref_highs_mad, "te_sigma_bp")),
        Macro("MSixRefMadClarabelHighsAgreementBp", r, "sci",
              "|te_sigma_bp(Clarabel MAD) - te_sigma_bp(HiGHS MAD)|: how "
              "closely an interior-point method and a dual simplex agree "
              "on the surrogate solution's true tracking error",
              lambda c: abs(c.get("m6ref", ref_clar_mad, "te_sigma_bp")
                            - c.get("m6ref", ref_highs_mad,
                                    "te_sigma_bp"))),
        Macro("MSixRefRunsPerConfig", r, "int", "'# runs_per_config:'",
              lambda c: float(c.env("m6ref", "runs_per_config"))),
        Macro("MSixRefMadSpreadPct", r, "pct2",
              "(max - min) / median of te_sigma_bp across the three "
              "algorithm families on the identical MAD instance: the "
              "quantity this file exists to report, and the one no "
              "timing caveat can touch",
              lambda c: ((c.agg("m6ref", {"case": "mad@10bp"},
                                "te_sigma_bp", "max")
                          - c.agg("m6ref", {"case": "mad@10bp"},
                                  "te_sigma_bp", "min"))
                         / c.agg("m6ref", {"case": "mad@10bp"},
                                 "te_sigma_bp", "median"))),
    ]
    return out


def _macros_scen() -> list[Macro]:
    """Scenario generation: the one unambiguous GPU win in the pipeline."""
    s = "scenario generation"
    sizes = (("Small", 2000, 500), ("Mid", 20000, 500),
             ("Large", 100000, 500), ("Wide", 50000, 2000))
    out: list[Macro] = []
    for name, n_scen, n_bonds in sizes:
        sel = {"n_scen": n_scen, "n_bonds": n_bonds}
        out.append(Macro(f"Scen{name}Scenarios", s, "int", "n_scen",
                         (lambda sl: lambda c: c.get("scen", sl,
                                                     "n_scen"))(sel)))
        out.append(Macro(f"Scen{name}Bonds", s, "int", "n_bonds",
                         (lambda sl: lambda c: c.get("scen", sl,
                                                     "n_bonds"))(sel)))
        out.append(Macro(f"Scen{name}GpuSeconds", s, "seconds", "gpu_s",
                         (lambda sl: lambda c: c.get("scen", sl,
                                                     "gpu_s"))(sel)))
        out.append(Macro(f"Scen{name}CpuSeconds", s, "seconds", "cpu_s",
                         (lambda sl: lambda c: c.get("scen", sl,
                                                     "cpu_s"))(sel)))
        out.append(Macro(
            f"Scen{name}Speedup", s, "ratio",
            "cpu_s / gpu_s, recomputed here rather than read from the CSV's "
            "own speedup column, so the paper's arithmetic is the paper's",
            (lambda sl: lambda c: (c.get("scen", sl, "cpu_s")
                                   / c.get("scen", sl, "gpu_s")))(sel)))
    out += [
        Macro("ScenMaxCvarDiffInSe", s, "ratio",
              "worst |CVaR_gpu - CVaR_cpu| in standard errors across sizes",
              lambda c: c.agg("scen", {}, "cvar_diff_in_se", "max")),
        Macro("ScenAgreementGate", s, "text", "'# gate:'",
              lambda c: c.env("scen", "gate")),
        Macro("ScenAgreementVerdict", s, "text", "'# verdict:'",
              lambda c: c.env("scen", "verdict")),
    ]
    return out


def _macros_warm() -> list[Macro]:
    """The warm-start negative result."""
    s = "warm start vs presolve"
    cold_p = {"mode": "cold + presolve"}
    cold_np = {"mode": "cold, no presolve"}
    warm_np = {"mode": "warm, no presolve"}
    return [
        Macro("WarmColdPresolveSeconds", s, "seconds", "wall_s_median",
              lambda c: c.timing("warm", cold_p, "wall_s_median")),
        Macro("WarmColdNoPresolveSeconds", s, "seconds", "wall_s_median",
              lambda c: c.timing("warm", cold_np, "wall_s_median")),
        Macro("WarmWarmNoPresolveSeconds", s, "seconds", "wall_s_median",
              lambda c: c.timing("warm", warm_np, "wall_s_median")),
        Macro("WarmStartSavingPct", s, "pct",
              "1 - warm-no-presolve / cold-no-presolve: what warm start is "
              "worth on equal terms",
              lambda c: (1.0 - c.timing("warm", warm_np, "wall_s_median")
                         / c.timing("warm", cold_np, "wall_s_median"))),
        Macro("WarmPresolveMultiple", s, "ratio",
              "cold-no-presolve / cold-with-presolve: what presolve is worth",
              lambda c: (c.timing("warm", cold_np, "wall_s_median")
                         / c.timing("warm", cold_p, "wall_s_median"))),
        Macro("WarmNetPenaltyMultiple", s, "ratio",
              "warm-no-presolve / cold-with-presolve: the choice an "
              "interactive app actually faces, since warm start requires "
              "presolve off",
              lambda c: (c.timing("warm", warm_np, "wall_s_median")
                         / c.timing("warm", cold_p, "wall_s_median"))),
        Macro("WarmColdPresolveViolation", s, "sci",
              "worst_rel_violation of the cold+presolve solution",
              lambda c: c.get("warm", cold_p, "worst_rel_violation")),
        Macro("WarmColdNoPresolveViolation", s, "sci",
              "worst_rel_violation of the cold, no-presolve solution",
              lambda c: c.get("warm", cold_np, "worst_rel_violation")),
        Macro("WarmWarmNoPresolveViolation", s, "sci",
              "worst_rel_violation of the warm, no-presolve solution -- "
              "warm start is not only fastest, it lands closest to "
              "feasible",
              lambda c: c.get("warm", warm_np, "worst_rel_violation")),
        Macro("WarmVerdict", s, "text", "'# verdict:'",
              lambda c: c.env("warm", "verdict")),
        Macro("WarmBonds", s, "int", "n_bonds",
              lambda c: c.get("warm", cold_p, "n_bonds")),
        Macro("WarmScenarios", s, "int", "n_scenarios",
              lambda c: c.get("warm", cold_p, "n_scenarios")),
    ]


def _macros_ablation() -> list[Macro]:
    """The ablation: the experiment that earns the word 'because'."""
    s = "constraint ablation"
    levels = (("Zero", "L0_budget_box"), ("One", "L1_turnover"),
              ("Two", "L2_rating"), ("Three", "L3_sector"),
              ("Four", "L4_issuer"), ("Five", "L5_duration"),
              ("Six", "L6_full_mandate"))
    out: list[Macro] = []
    for name, level in levels:
        base = {"level": level, "solver": "cuopt_pdlp"}
        xo = {"level": level, "solver": "cuopt_pdlp+xover"}
        highs = {"level": level, "solver": "highs"}
        out.append(Macro(
            f"AblLevel{name}Families", s, "int", "n_families",
            (lambda sl: lambda c: c.get("abl", sl, "n_families"))(base)))
        out.append(Macro(
            f"AblLevel{name}MandateRows", s, "int",
            "mandate_rows: limit statements, as a PM counts them",
            (lambda sl: lambda c: c.get("abl", sl, "mandate_rows"))(base)))
        out.append(Macro(
            f"AblLevel{name}LpRowsAdded", s, "int",
            "lp_eq_rows at this level minus lp_eq_rows at L0: canonical rows "
            "this mandate contributes. NOT lp_ineq_rows_added, which is "
            "structurally zero -- CVXPY canonicalises to equality rows plus "
            "bounds, so there are no explicit inequality rows to count.",
            (lambda sl: lambda c: (
                c.get("abl", sl, "lp_eq_rows")
                - c.get("abl", {"level": "L0_budget_box",
                                "solver": sl["solver"]}, "lp_eq_rows")))(base)))
        out.append(Macro(
            f"AblLevel{name}LpVarsAdded", s, "int",
            "lp_vars at this level minus lp_vars at L0: auxiliary variables "
            "this mandate contributes. This is the quantity crossover cost "
            "actually tracks, because these are the variables whose basis "
            "status crossover has to resolve.",
            (lambda sl: lambda c: (
                c.get("abl", sl, "lp_vars")
                - c.get("abl", {"level": "L0_budget_box",
                                "solver": sl["solver"]}, "lp_vars")))(base)))
        out.append(Macro(
            f"AblLevel{name}PdlpSeconds", s, "seconds", "wall_s_median",
            (lambda sl: lambda c: c.timing("abl", sl, "wall_s_median"))(base)))
        out.append(Macro(
            f"AblLevel{name}XoverSeconds", s, "seconds", "wall_s_median",
            (lambda sl: lambda c: c.timing("abl", sl, "wall_s_median"))(xo)))
        out.append(Macro(
            f"AblLevel{name}HighsSeconds", s, "seconds", "wall_s_median",
            (lambda sl: lambda c: c.timing("abl", sl,
                                           "wall_s_median"))(highs)))
        out.append(Macro(
            f"AblLevel{name}CrossoverCostSeconds", s, "seconds",
            "crossover_cost_s = (cuopt+xover wall) - (cuopt pdlp wall) at "
            "this level. Read down this column: it is the experiment.",
            (lambda sl: lambda c: c.get("abl", sl,
                                        "crossover_cost_s"))(xo)))
        out.append(Macro(
            f"AblLevel{name}CrossoverMultiple", s, "ratio",
            "crossover_multiple = (cuopt+xover wall) / (cuopt pdlp wall)",
            (lambda sl: lambda c: c.get("abl", sl,
                                        "crossover_multiple"))(xo)))
        out.append(Macro(
            f"AblLevel{name}IpmCrossoverMultiple", s, "ratio",
            "(HiGHS IPM + crossover wall) / (HiGHS IPM wall): the same "
            "vertex conversion, starting from an interior-point iterate "
            "instead of a first-order one",
            (lambda lv: lambda c: (
                c.timing("abl", {"level": lv, "solver": "highs_ipm+xover"},
                         "wall_s_median")
                / c.timing("abl", {"level": lv, "solver": "highs_ipm"},
                           "wall_s_median")))(level)))

    def _ipm_multiple(c: Ctx, lv: str) -> float:
        return (c.timing("abl", {"level": lv, "solver": "highs_ipm+xover"},
                         "wall_s_median")
                / c.timing("abl", {"level": lv, "solver": "highs_ipm"},
                           "wall_s_median"))

    out += [
        Macro("AblIpmCrossoverMultipleMin", s, "ratio",
              "min over the seven levels of the HiGHS IPM crossover "
              "multiple",
              lambda c: min(_ipm_multiple(c, lv) for _, lv in levels)),
        Macro("AblIpmCrossoverMultipleMax", s, "ratio",
              "max over the seven levels of the HiGHS IPM crossover "
              "multiple -- with the min, the whole range an interior-point "
              "method pays for the conversion PDLP pays 5-9x for",
              lambda c: max(_ipm_multiple(c, lv) for _, lv in levels)),
    ]
    out += [
        Macro("AblCrossoverCostGrowth", s, "ratio",
              "crossover_cost_s at the full mandate / crossover_cost_s at "
              "the equity-equivalent L0: the causal claim, as one number",
              lambda c: (c.get("abl", {"level": "L6_full_mandate",
                                       "solver": "cuopt_pdlp+xover"},
                               "crossover_cost_s")
                         / c.get("abl", {"level": "L0_budget_box",
                                         "solver": "cuopt_pdlp+xover"},
                                 "crossover_cost_s"))),
        Macro("AblTurnoverRowShare", s, "pct",
              "(L1 - L0) / (L6 - L0) on lp_eq_rows: the share of the "
              "mandate's canonical rows that one limit statement contributes. "
              "Crossover cost tracking this rather than the family count is "
              "what makes the story one enormous family, not six families.",
              lambda c: ((c.get("abl", {"level": "L1_turnover",
                                        "solver": "cuopt_pdlp"}, "lp_eq_rows")
                          - c.get("abl", {"level": "L0_budget_box",
                                          "solver": "cuopt_pdlp"},
                                  "lp_eq_rows"))
                         / (c.get("abl", {"level": "L6_full_mandate",
                                          "solver": "cuopt_pdlp"},
                                  "lp_eq_rows")
                            - c.get("abl", {"level": "L0_budget_box",
                                            "solver": "cuopt_pdlp"},
                                    "lp_eq_rows")))),
        Macro("AblTurnoverVarShare", s, "pct",
              "(L1 - L0) / (L6 - L0) on lp_vars: the same share in auxiliary "
              "variables, which is the quantity crossover cost tracks.",
              lambda c: ((c.get("abl", {"level": "L1_turnover",
                                        "solver": "cuopt_pdlp"}, "lp_vars")
                          - c.get("abl", {"level": "L0_budget_box",
                                          "solver": "cuopt_pdlp"}, "lp_vars"))
                         / (c.get("abl", {"level": "L6_full_mandate",
                                          "solver": "cuopt_pdlp"}, "lp_vars")
                            - c.get("abl", {"level": "L0_budget_box",
                                            "solver": "cuopt_pdlp"},
                                    "lp_vars")))),
        Macro("AblTurnoverCrossoverShare", s, "pct",
              "L1 crossover_cost_s / L6 crossover_cost_s: the same share, "
              "measured in time instead of rows",
              lambda c: (c.get("abl", {"level": "L1_turnover",
                                       "solver": "cuopt_pdlp+xover"},
                               "crossover_cost_s")
                         / c.get("abl", {"level": "L6_full_mandate",
                                         "solver": "cuopt_pdlp+xover"},
                                 "crossover_cost_s"))),
    ]
    return out


# M7 arm names, spelled once. The CSV's arm strings encode formulation and
# hardware together; the macro names split them back out so the paper's
# 2x2 table reads (dense|factored) x (GPU|CPU) without a decoder ring.
QP_DENSE_GPU = "cuopt_barrier_dense"
QP_DENSE_CPU = "clarabel_dense"
QP_FACT_CPU = "clarabel_factored"
QP_FACT_GPU = "cuopt_barrier_factored"


def _qp(arm: str, n_bonds: float) -> dict[str, Any]:
    return {"arm": arm, "n_bonds": n_bonds}


def _m7_largest_factored_seconds(c: Ctx) -> float:
    """Wall of the faster factored arm at the largest factored N solved.

    The N is found by aggregation rather than typed, so if a rerun pushes
    the factored arms past 15,000 this macro follows the data.
    """
    n = c.agg("qpbranch", {"form": "factored", "status": "optimal"},
              "n_bonds", "max")
    return min(
        c.timing("qpbranch", _qp(QP_FACT_GPU, n), "wall_s_median"),
        c.timing("qpbranch", _qp(QP_FACT_CPU, n), "wall_s_median"))


def _m7_probe_worst_gap(c: Ctx) -> float:
    """Worst |probe wall / branch wall - 1| across the probe's own grid.

    qp_probe.csv re-ran the identical four arms at two sizes in a separate
    process before the full branch; the worst relative disagreement between
    the two files is the cross-process reproducibility of these timings,
    measured rather than asserted.
    """
    worst = 0.0
    for _, row in c.table("qpprobe").df.iterrows():
        arm = str(row["arm"]).strip()
        n = int(row["n_bonds"])
        probe = c.timing("qpprobe", _qp(arm, n), "wall_s_median")
        branch = c.timing("qpbranch", _qp(arm, n), "wall_s_median")
        worst = max(worst, abs(probe / branch - 1.0))
    return worst


def _macros_m7() -> list[Macro]:
    """M7 QP branch: on the covariance objective, formulation beats hardware.

    Dense rows at N >= 10,000 carry status skipped_host_memory and no wall
    time. Those are recorded non-results -- "this representation does not fit
    here" is the finding on the cell -- so their Seconds macros render the
    status text via the RecordedNonResult path instead of failing strict.
    """
    s = "M7 QP branch"
    arms = (("DenseGpu", QP_DENSE_GPU), ("DenseCpu", QP_DENSE_CPU),
            ("FactoredCpu", QP_FACT_CPU), ("FactoredGpu", QP_FACT_GPU))
    sizes = (("FiveHundred", 500), ("TwoThousand", 2000),
             ("FiveThousand", 5000), ("SeventyFiveHundred", 7500),
             ("TenThousand", 10000), ("FifteenThousand", 15000))
    out: list[Macro] = [
        Macro("MSevenGrid", s, "text", "'# grid_n_bonds:'",
              lambda c: c.env("qpbranch", "grid_n_bonds")),
        Macro("MSevenAmundiIndexScaleBonds", s, "int",
              "'# amundi_index_scale_n:' -- the index size the branch was "
              "aimed at and died short of (CUDA OOM in the factored arm's "
              "scenario generation; no factored row at this N exists)",
              lambda c: float(c.env("qpbranch", "amundi_index_scale_n"))),
        Macro("MSevenScenariosForSigma", s, "int",
              "'# n_scenarios_for_sigma:'",
              lambda c: float(c.env("qpbranch", "n_scenarios_for_sigma"))),
    ]
    for aname, arm in arms:
        for nname, n in sizes:
            out.append(Macro(
                f"MSeven{aname}{nname}Seconds", s, "seconds",
                "wall_s_median; a dense cell the harness refused on host "
                "memory renders its recorded status instead of a number",
                (lambda a, nn: lambda c: c.timing(
                    "qpbranch", _qp(a, nn), "wall_s_median"))(arm, n)))
    out += [
        Macro("MSevenFactoredAdvantageCpuSmallest", s, "ratio",
              "Clarabel dense wall / Clarabel factored wall at N=500: what "
              "the factored Hessian buys on the CPU at the small end",
              lambda c: (c.timing("qpbranch", _qp(QP_DENSE_CPU, 500),
                                  "wall_s_median")
                         / c.timing("qpbranch", _qp(QP_FACT_CPU, 500),
                                    "wall_s_median"))),
        Macro("MSevenFactoredAdvantageCpuLargest", s, "ratio",
              "Clarabel dense wall / Clarabel factored wall at N=7,500, the "
              "largest N where the dense representation still fits",
              lambda c: (c.timing("qpbranch", _qp(QP_DENSE_CPU, 7500),
                                  "wall_s_median")
                         / c.timing("qpbranch", _qp(QP_FACT_CPU, 7500),
                                    "wall_s_median"))),
        Macro("MSevenFactoredAdvantageGpuSmallest", s, "ratio",
              "cuOpt dense wall / cuOpt factored wall at N=500",
              lambda c: (c.timing("qpbranch", _qp(QP_DENSE_GPU, 500),
                                  "wall_s_median")
                         / c.timing("qpbranch", _qp(QP_FACT_GPU, 500),
                                    "wall_s_median"))),
        Macro("MSevenFactoredAdvantageGpuLargest", s, "ratio",
              "cuOpt dense wall / cuOpt factored wall at N=7,500",
              lambda c: (c.timing("qpbranch", _qp(QP_DENSE_GPU, 7500),
                                  "wall_s_median")
                         / c.timing("qpbranch", _qp(QP_FACT_GPU, 7500),
                                    "wall_s_median"))),
        Macro("MSevenDenseGpuOverFactoredCpu", s, "ratio",
              "cuOpt dense wall / Clarabel factored wall at N=7,500: the "
              "headline -- the slower hardware on the better formulation "
              "wins by this multiple",
              lambda c: (c.timing("qpbranch", _qp(QP_DENSE_GPU, 7500),
                                  "wall_s_median")
                         / c.timing("qpbranch", _qp(QP_FACT_CPU, 7500),
                                    "wall_s_median"))),
    ]
    for nname, n in (("FiveHundred", 500), ("FiveThousand", 5000),
                     ("SeventyFiveHundred", 7500),
                     ("FifteenThousand", 15000)):
        out.append(Macro(
            f"MSevenFactoredGpuOverCpu{nname}", s, "ratio",
            "cuOpt factored wall / Clarabel factored wall: above 1 the CPU "
            "leads, below 1 the GPU leads. The crossing sits BETWEEN the "
            "5,000 and 7,500 rows; no single interpolated N may be quoted "
            "as if it were measured.",
            (lambda nn: lambda c: (
                c.timing("qpbranch", _qp(QP_FACT_GPU, nn), "wall_s_median")
                / c.timing("qpbranch", _qp(QP_FACT_CPU, nn),
                           "wall_s_median")))(n)))
    out += [
        Macro("MSevenLargestFactoredBonds", s, "int",
              "max n_bonds over factored rows with status=optimal: the "
              "largest instance any arm actually solved",
              lambda c: c.agg("qpbranch", {"form": "factored",
                                           "status": "optimal"},
                              "n_bonds", "max")),
        Macro("MSevenLargestFactoredSeconds", s, "seconds",
              "wall_s_median of the faster factored arm at that largest N",
              _m7_largest_factored_seconds),
        Macro("MSevenFactoredAmundiRowCount", s, "int",
              "committed factored rows at the Amundi index scale N: the run "
              "died in the factored arm's scenario generation (CUDA OOM on "
              "the float64-promoted scenario matrix) before writing one, so "
              "this counts the disclosed defect rather than hiding it",
              lambda c: c.agg("qpbranch", {"form": "factored",
                                           "n_bonds": 29222}, "", "count")),
        Macro("MSevenDenseAmundiStatus", s, "text",
              "status of the dense GPU arm at the Amundi index scale N, "
              "verbatim",
              lambda c: c.text("qpbranch", _qp(QP_DENSE_GPU, 29222),
                               "status")),
        Macro("MSevenProbeCrossoverVerdict", s, "text",
              "crossover_verdict on the probe_crossover row, verbatim -- "
              "cuOpt's own answer to being handed a crossover flag on a QP. "
              "Nothing in the paper may lean on QP crossover; this macro "
              "exists so the paper quotes the probe instead of "
              "paraphrasing it.",
              lambda c: c.text("qpbranch", {"arm": "probe_crossover"},
                               "crossover_verdict")),
        Macro("MSevenProbeBranchWorstGapPct", s, "pct",
              "max over the probe grid (four arms x two sizes) of "
              "|probe wall / branch wall - 1|: cross-process timing "
              "reproducibility, from two separately committed files",
              _m7_probe_worst_gap),
    ]
    return out


# M9 helpers. Every aggregation point below is the MEDIAN over the three
# scenario draws at the TIGHTEST tracking-error budget in the sweep, with
# both the budget and the dof grid read from the CSV's own header rather
# than typed here. One seed must not speak for a (dof, pd_mult) point, and
# the tightest budget is where the surrogate's miss is least excusable.


def _m9_tight_budget(c: Ctx) -> float:
    return min(float(x) for x in c.env("m9", "te_grid_bp").split(","))


def _m9_point_median(c: Ctx, arm: str, dof: float) -> float:
    return c.agg("m9", {"arm": arm, "t_dof": dof,
                        "te_budget_bp": _m9_tight_budget(c)},
                 "overshoot_x", "median")


def _m9_grid_extreme(arm: str, how: str) -> Callable[[Ctx], float]:
    def f(c: Ctx) -> float:
        dofs = [float(x) for x in c.env("m9", "dof_grid").split(",")]
        meds = [_m9_point_median(c, arm, d) for d in dofs]
        return min(meds) if how == "min" else max(meds)
    return f


def _m9_corner(c: Ctx, jumps: bool, heavy: bool) -> float:
    """One corner of the 2x2 decomposition at the shipped operating point.

    heavy=True is the repo's operating Student-t dof (from the header);
    heavy=False is the largest dof in the grid, i.e. Gaussian factors.
    jumps toggles the default process (the pd=1 repo arm vs the pd=0 arm).
    """
    dofs = [float(x) for x in c.env("m9", "dof_grid").split(",")]
    dof = float(c.env("m9", "repo_operating_dof")) if heavy else max(dofs)
    arm = "t-sweep-repo" if jumps else "t-sweep-nojump"
    return _m9_point_median(c, arm, dof)


def _m9_decomp_share(which: str) -> Callable[[Ctx], float]:
    """t / jumps / interaction shares of the overshoot above the base corner.

    Computed from the four corner cells, never transcribed: with overshoot
    O at each corner, total = O(both) - O(base), the main effects are
    O(t-only) - O(base) and O(jumps-only) - O(base), and interaction is the
    remainder that makes the three shares sum to one.
    """
    def f(c: Ctx) -> float:
        base = _m9_corner(c, jumps=False, heavy=False)
        t_only = _m9_corner(c, jumps=False, heavy=True)
        j_only = _m9_corner(c, jumps=True, heavy=False)
        both = _m9_corner(c, jumps=True, heavy=True)
        total = both - base
        if which == "t":
            return (t_only - base) / total
        if which == "jumps":
            return (j_only - base) / total
        return (both - t_only - j_only + base) / total
    return f


def _m9_control_overshoot(extreme: str) -> Callable[[Ctx], float]:
    def f(c: Ctx) -> float:
        s_val = c.agg("m9", {"arm": "gaussian-control"}, "n_scenarios",
                      extreme)
        return c.get("m9", {"arm": "gaussian-control",
                            "n_scenarios": s_val}, "overshoot_x")
    return f


def _macros_m9() -> list[Macro]:
    """M9: the surrogate's failure is jump-to-default, not the t parameter."""
    s = "M9 tail sweep"
    out: list[Macro] = [
        Macro("MNineBonds", s, "int", "'# n_bonds:'",
              lambda c: float(c.env("m9", "n_bonds"))),
        Macro("MNineScenarios", s, "int", "'# n_scenarios:'",
              lambda c: float(c.env("m9", "n_scenarios"))),
        Macro("MNineScenarioSeeds", s, "text", "'# scenario_seeds:'",
              lambda c: c.env("m9", "scenario_seeds")),
        Macro("MNineDofGrid", s, "text", "'# dof_grid:'",
              lambda c: c.env("m9", "dof_grid")),
        Macro("MNinePdMultGrid", s, "text", "'# pd_mult_grid:'",
              lambda c: c.env("m9", "pd_mult_grid")),
        Macro("MNineTeGrid", s, "text", "'# te_grid_bp:'",
              lambda c: c.env("m9", "te_grid_bp")),
        Macro("MNineOperatingDof", s, "text", "'# repo_operating_dof:' -- "
              "the Student-t dof every other committed result was run at",
              lambda c: c.env("m9", "repo_operating_dof")),
        Macro("MNineControlBudgetBp", s, "bp", "'# control_budget_bp:'",
              lambda c: float(c.env("m9", "control_budget_bp"))),
        # The Gaussian control: with thin tails and no defaults the MAD
        # surrogate converges on the mandate as scenarios grow, which is
        # what licenses reading the overshoot elsewhere as a tail effect
        # rather than sampling noise.
        Macro("MNineControlSmallestS", s, "int",
              "min n_scenarios among gaussian-control rows",
              lambda c: c.agg("m9", {"arm": "gaussian-control"},
                              "n_scenarios", "min")),
        Macro("MNineControlLargestS", s, "int",
              "max n_scenarios among gaussian-control rows",
              lambda c: c.agg("m9", {"arm": "gaussian-control"},
                              "n_scenarios", "max")),
        Macro("MNineControlOvershootAtSmallestS", s, "ratio",
              "overshoot_x on the gaussian-control row with the fewest "
              "scenarios",
              _m9_control_overshoot("min")),
        Macro("MNineControlOvershootAtLargestS", s, "ratio",
              "overshoot_x on the gaussian-control row with the most "
              "scenarios: the number that should be, and is, approaching 1",
              _m9_control_overshoot("max")),
        # Defaults off: the whole dof grid, heaviest tail to Gaussian.
        Macro("MNineNoJumpOvershootMin", s, "ratio",
              "min over the dof grid of the per-dof median overshoot_x "
              "(defaults off, tightest budget)",
              _m9_grid_extreme("t-sweep-nojump", "min")),
        Macro("MNineNoJumpOvershootMax", s, "ratio",
              "max over the dof grid of the per-dof median overshoot_x "
              "(defaults off, tightest budget): the worst the Student-t "
              "parameter can do on its own",
              _m9_grid_extreme("t-sweep-nojump", "max")),
        # Defaults on: the same grid never comes close to the mandate.
        Macro("MNineDefaultsOnOvershootMin", s, "ratio",
              "min over the dof grid of the per-dof median overshoot_x "
              "with the default process on: the BEST any tail parameter "
              "achieves once jump-to-default exists",
              _m9_grid_extreme("t-sweep-repo", "min")),
    ]
    # Dose-response: dof pinned at the repo operating point, the default
    # intensity dial swept. Names are the pd_mult in hundredths of the
    # calibrated intensity: One = 0.01, Three = 0.03, Ten = 0.10,
    # Thirty = 0.30.
    doses = (("One", 0.01), ("Three", 0.03), ("Ten", 0.1), ("Thirty", 0.3))
    for dname, pd_mult in doses:
        out.append(Macro(
            f"MNineDose{dname}Overshoot", s, "ratio",
            f"median overshoot_x over the scenario draws at "
            f"pd_mult={pd_mult} (jump-sweep arm, tightest budget)",
            (lambda pm: lambda c: c.agg(
                "m9", {"arm": "jump-sweep", "pd_mult": pm,
                       "te_budget_bp": _m9_tight_budget(c)},
                "overshoot_x", "median"))(pd_mult)))
    out += [
        # The 2x2 corners behind the decomposition, printed so the reader
        # can redo the arithmetic in the margin.
        Macro("MNineCornerGaussianNoJump", s, "ratio",
              "median overshoot_x, Gaussian factors (largest dof in the "
              "grid), defaults off: the base corner",
              lambda c: _m9_corner(c, jumps=False, heavy=False)),
        Macro("MNineCornerTOnly", s, "ratio",
              "median overshoot_x at the operating dof, defaults off",
              lambda c: _m9_corner(c, jumps=False, heavy=True)),
        Macro("MNineCornerJumpsOnly", s, "ratio",
              "median overshoot_x, Gaussian factors, defaults on -- also "
              "the 'value at dof=1e6 with defaults on': kill the heavy "
              "tail entirely and the overshoot barely moves",
              lambda c: _m9_corner(c, jumps=True, heavy=False)),
        Macro("MNineCornerBoth", s, "ratio",
              "median overshoot_x at the operating dof with defaults on: "
              "the shipped configuration",
              lambda c: _m9_corner(c, jumps=True, heavy=True)),
        Macro("MNineDecompTSharePct", s, "pct",
              "share of the shipped point's overshoot-above-base explained "
              "by the Student-t main effect, from the four corner cells",
              _m9_decomp_share("t")),
        Macro("MNineDecompJumpSharePct", s, "pct",
              "share explained by the jump-to-default main effect",
              _m9_decomp_share("jumps")),
        Macro("MNineDecompInteractionSharePct", s, "pct",
              "the interaction remainder; the three shares sum to one by "
              "construction",
              _m9_decomp_share("interaction")),
    ]
    return out


def _macros_demo() -> list[Macro]:
    """Interactive demo latency: both paths, and the cross-file reconcile."""
    s = "demo latency"
    sizes = (("FourK", 4000), ("TenK", 10000), ("TwentyK", 20000))
    out: list[Macro] = [
        Macro("DemoBonds", s, "int", "n_bonds on the largest row",
              lambda c: c.get("demo", {"n_scenarios": 20000}, "n_bonds")),
        Macro("DemoRepsPerSize", s, "int", "'# reps_per_size:'",
              lambda c: float(c.env("demo", "reps_per_size"))),
        Macro("DemoTolerance", s, "text", "'# tolerance:'",
              lambda c: c.env("demo", "tolerance")),
    ]
    for sname, n_scen in sizes:
        sel = {"n_scenarios": n_scen}
        out.append(Macro(
            f"Demo{sname}Scenarios", s, "int", "n_scenarios",
            (lambda sl: lambda c: c.get("demo", sl, "n_scenarios"))(sel)))
        out.append(Macro(
            f"Demo{sname}HandSeconds", s, "seconds", "handrolled_s_median",
            (lambda sl: lambda c: c.timing("demo", sl,
                                           "handrolled_s_median"))(sel)))
        out.append(Macro(
            f"Demo{sname}CvxpySeconds", s, "seconds", "cvxpy_s_median",
            (lambda sl: lambda c: c.timing("demo", sl,
                                           "cvxpy_s_median"))(sel)))
        out.append(Macro(
            f"Demo{sname}CvxpyOverHand", s, "ratio",
            "cvxpy_s_median / handrolled_s_median: what the hand-rolled "
            "CSR path saves at this size",
            (lambda sl: lambda c: (
                c.timing("demo", sl, "cvxpy_s_median")
                / c.timing("demo", sl, "handrolled_s_median")))(sel)))
    out += [
        Macro("DemoFourKHandViolationPct", s, "pct",
              "handrolled_worst_rel_violation at 4,000 scenarios: the one "
              "surviving caveat -- at this size the fast path's answer "
              "breaches the mandate by a third of a limit's scale, so "
              "speed quoted at 4,000 scenarios is speed to a wrong answer",
              lambda c: c.get("demo", {"n_scenarios": 4000},
                              "handrolled_worst_rel_violation")),
        Macro("DemoTenKHandViolation", s, "sci",
              "handrolled_worst_rel_violation at 10,000 scenarios",
              lambda c: c.get("demo", {"n_scenarios": 10000},
                              "handrolled_worst_rel_violation")),
        Macro("DemoTwentyKHandViolation", s, "sci",
              "handrolled_worst_rel_violation at 20,000 scenarios",
              lambda c: c.get("demo", {"n_scenarios": 20000},
                              "handrolled_worst_rel_violation")),
        Macro("DemoWarmstartReconcileRatio", s, "ratio",
              "warmstart cold+presolve wall / demo 20k hand-rolled wall, "
              "read from the two separately committed files that once "
              "disagreed by 3.3x. Near 1 is the finding: the two scripts "
              "now time the same solve to within run-to-run noise.",
              lambda c: (c.timing("warm", {"mode": "cold + presolve"},
                                  "wall_s_median")
                         / c.timing("demo", {"n_scenarios": 20000},
                                    "handrolled_s_median"))),
    ]
    return out


# Match the P0 QP objective gate and src.credit's purity/TE desk gates.
# Keep the paper generator independent of solver imports and GPU runtimes.
P0_OBJECTIVE_REL_TOL = 1e-6
P0_PURITY_REL_TOL = 1e-6
P0_TE_DESK_TOL_BP = 0.05
P0_GPU_IDLE_MAX_PCT = 25.0  # src.bench.assert_gpu_idle


def _p0_socp_reference_timing(c: Ctx, sel: dict[str, Any]) -> float:
    """Publish a reference latency only with affirmative quality and telemetry.

    An existing row with unsafe or incomplete evidence is an explicit recorded
    exclusion, accepted by a strict build. Absent files/rows still fail strict
    generation. Feasibility macros use their original numeric cells directly.
    """
    tab = c.table("p0socpref")
    row = c.row("p0socpref", sel)

    def exclude(reason: str) -> None:
        raise RecordedNonResult(f"timing excluded: {reason}")

    def field(name: str, *, header: bool = False, numeric: bool = False):
        source = tab.env if header else row
        raw = source.get(name)
        value = "" if pd.isna(raw) else str(raw).strip()
        c._record(tab, {"header": name} if header else sel, name,
                  value or "(missing)")
        if not value:
            exclude(f"missing {name}")
        if numeric:
            try:
                number = float(value)
            except ValueError:
                exclude(f"invalid {name}={value}")
            if not np.isfinite(number):
                exclude(f"non-finite {name}={value}")
            return number
        return value

    contention = field("gpu_contended", header=True)
    if contention.lower() != "no":
        exclude(f"gpu_contended={contention}")
    for name, upper in (
        ("gpu_util_pct_at_capture", P0_GPU_IDLE_MAX_PCT),
        ("gpu_mem_used_mib_at_capture", np.inf),
    ):
        value = field(name, header=True, numeric=True)
        if not 0 <= value <= upper:
            exclude(f"{name}={value:g} outside timing gate")

    # After-solve utilization can be this run's own tail. Require valid
    # telemetry there, but apply the idle threshold only BEFORE each solve.
    for phase in ("before", "after"):
        for metric, upper in (
            ("gpu_util_pct", P0_GPU_IDLE_MAX_PCT if phase == "before" else 100),
            ("gpu_mem_used_mib", np.inf),
        ):
            name = f"resource_{phase}_{metric}"
            value = field(name, numeric=True)
            if not 0 <= value <= upper:
                exclude(f"{name}={value:g} outside timing gate")
    note = row.get("resource_note", "")
    if not pd.isna(note) and str(note).strip():
        exclude(f"resource_note={field('resource_note')}")

    status = field("status")
    if status not in CONVERGED_STATUSES:
        exclude(f"status={status}")
    for name in ("check_ok", "purity_pass", "respects_mandate", "desk_pass"):
        if field(name).lower() != "true":
            exclude(f"{name} did not pass")
    purity = field("purity_worst_rel", numeric=True)
    if not 0 <= purity <= P0_PURITY_REL_TOL:
        exclude("purity_worst_rel outside purity gate")

    # The TE runner labels the unmodified cuOpt/Clarabel interior solutions
    # 'barrier'/'interior-point'. They are equivalent raw SOCP endpoints only
    # under this explicit mapping, solver-returned status, and common gates.
    # Never generalize that equivalence to repaired, basic, or unknown rows.
    expected_endpoint = {
        "cuopt-barrier-socp": "barrier", "clarabel-socp": "interior-point",
    }
    solver = field("solver")
    for name, expected in (
        ("method_class", "SOCP"), ("status_source", "solver-returned"),
        ("endpoint", expected_endpoint.get(solver)), ("te_mode", "sigma"),
    ):
        value = field(name)
        if expected is None or value != expected:
            exclude(f"{name}={value} is not the reference SOCP endpoint")
    budget = field("te_budget_bp", numeric=True)
    if budget != 10.0:
        exclude("te_budget_bp is not the reference 10bp mandate")
    te = field("te_sigma_bp", numeric=True)
    if not 0 <= te <= budget + P0_TE_DESK_TOL_BP:
        exclude("te_sigma_bp outside mandate gate")

    solver_objective = field("solver_objective", numeric=True)
    objective = field("objective_recomputed", numeric=True)
    discrepancy = field("objective_discrepancy", numeric=True)
    tolerance = P0_OBJECTIVE_REL_TOL * max(abs(objective), 1e-12)
    # Recompute the comparison: an optimistic cached discrepancy cannot
    # override disagreeing solver and independently recomputed objectives.
    if abs(solver_objective - objective) > tolerance:
        exclude("solver_objective disagrees with objective_recomputed")
    if not 0 <= discrepancy <= tolerance:
        exclude("objective_discrepancy outside objective gate")
    seconds = field("steady_state_median_s", numeric=True)
    if seconds <= 0:
        exclude("steady_state_median_s must be positive")
    return seconds


def _p0_socp_reference_ratio(c: Ctx) -> float:
    """Clarabel/cuOpt only after both endpoints pass the same publication gate."""
    cu = {"case": "sigma@10bp", "solver": "cuopt-barrier-socp"}
    cl = {"case": "sigma@10bp", "solver": "clarabel-socp"}
    cu_seconds = _p0_socp_reference_timing(c, cu)
    cl_seconds = _p0_socp_reference_timing(c, cl)
    cu_objective = c.get("p0socpref", cu, "objective_recomputed")
    cl_objective = c.get("p0socpref", cl, "objective_recomputed")
    scale = max(abs(cu_objective), abs(cl_objective), 1e-12)
    if abs(cu_objective - cl_objective) > P0_OBJECTIVE_REL_TOL * scale:
        raise RecordedNonResult("timing ratio excluded: achieved objectives disagree")
    return cl_seconds / cu_seconds


def _macros_p0() -> list[Macro]:
    """P0 evidence used by the re-framed paper.

    These macros deliberately keep feasibility-only rows separate from timing
    rows and only expose QP ratios that the runner pre-qualified as matched.
    """
    s = "P0 constraint delivery"
    sigma = {"case": "sigma@10bp"}
    cu_soc = {**sigma, "solver": "cuopt-barrier-socp"}
    cl_soc = {**sigma, "solver": "clarabel-socp"}
    naive = {"case": "mad@10bp", "solver": "cuopt-pdlp-1e-6"}
    selected = {"case": "mad@2.25bp", "solver": "cuopt-pdlp-1e-6"}
    exact = cl_soc

    out = [
        Macro("PZeroSocpRefCuoptSeconds", s, "seconds",
              "reference-size cuOpt steady-state median wall time, subject to telemetry and independent quality gates",
              lambda c: _p0_socp_reference_timing(c, cu_soc)),
        Macro("PZeroSocpRefClarabelSeconds", s, "seconds",
              "reference-size Clarabel steady-state median wall time, subject to telemetry and independent quality gates",
              lambda c: _p0_socp_reference_timing(c, cl_soc)),
        Macro("PZeroSocpRefStackRatio", s, "ratio",
              "Clarabel / cuOpt reference-size steady-state wall time, only after telemetry, independent quality, and equivalent SOCP endpoint/objective gates",
              _p0_socp_reference_ratio),
        Macro("PZeroSocpRefCuoptTeBp", s, "bp", "independently recomputed sigma TE",
              lambda c: c.get("p0socpref", cu_soc, "te_sigma_bp")),
        Macro("PZeroSocpRefClarabelTeBp", s, "bp", "independently recomputed sigma TE",
              lambda c: c.get("p0socpref", cl_soc, "te_sigma_bp")),
        Macro("PZeroSocpWorkingCuoptStatus", s, "text",
              "working-size native cuOpt SOCP recorded status",
              lambda c: c.text("p0socwork", cu_soc, "status")),
        Macro("PZeroSocpWorkingClarabelTeBp", s, "bp",
              "working-size Clarabel achieved sigma TE; feasibility evidence only",
              lambda c: c.get("p0socwork", cl_soc, "te_sigma_bp")),
        Macro("PZeroSocpWorkingClarabelObjective", s, "bp",
              "working-size exact objective, multiplied by 10,000",
              lambda c: 1e4 * c.get("p0socwork", cl_soc, "objective_recomputed")),
        Macro("PZeroSocpWorkingClarabelExpectedExcessBp", s, "bp",
              "working-size exact expected excess return in bp/month",
              lambda c: c.get("p0socwork", cl_soc, "expected_excess_bp_monthly")),
        Macro("PZeroNaiveMadSigmaBp", s, "bp",
              "sigma TE delivered by the tight 10bp naive MAD row",
              lambda c: c.get("p0recal", naive, "te_sigma_bp")),
        Macro("PZeroNaiveMadOvershoot", s, "ratio",
              "delivered sigma TE / stated 10bp target",
              lambda c: c.get("p0recal", naive, "te_sigma_bp") /
                        c.get("p0recal", naive, "te_budget_bp")),
        Macro("PZeroSelectedMadBudgetBp", s, "bp",
              "highest-objective recalibrated MAD row that passes the true sigma and mandate gates",
              lambda c: c.get("p0recal", selected, "te_budget_bp")),
        Macro("PZeroSelectedMadSigmaBp", s, "bp",
              "in-model sigma TE of the selected recalibrated MAD row",
              lambda c: c.get("p0recal", selected, "te_sigma_bp")),
        Macro("PZeroSelectedMadHaircut", s, "ratio",
              "true sigma target / selected MAD budget",
              lambda c: c.get("p0recal", naive, "te_budget_bp") /
                        c.get("p0recal", selected, "te_budget_bp")),
        Macro("PZeroValidationSigmaMinBp", s, "bp",
              "minimum sigma TE across the three fixed validation draws",
              lambda c: c.agg("p0recal", {"status": "VALIDATION_ONLY"}, "te_sigma_bp", "min")),
        Macro("PZeroValidationSigmaMaxBp", s, "bp",
              "maximum sigma TE across the three fixed validation draws",
              lambda c: c.agg("p0recal", {"status": "VALIDATION_ONLY"}, "te_sigma_bp", "max")),
        Macro("PZeroValidationDrawCount", s, "int", "fixed validation draws",
              lambda c: c.agg("p0recal", {"status": "VALIDATION_ONLY"}, "", "count")),
        Macro("PZeroSelectedMadObjectiveLossBp", s, "bp",
              "10,000 times exact SOCP objective minus selected MAD objective",
              lambda c: 1e4 * (c.get("p0socwork", exact, "objective_recomputed") -
                               c.get("p0recal", selected, "objective_recomputed"))),
        Macro("PZeroSelectedMadExpectedExcessLossBp", s, "bp",
              "exact SOCP expected excess minus selected MAD, bp/month",
              lambda c: c.get("p0socwork", exact, "expected_excess_bp_monthly") -
                        c.get("p0recal", selected, "expected_excess_bp_monthly")),
        Macro("PZeroSelectedMadCvarCostBp", s, "bp",
              "10,000 times selected MAD CVaR minus exact SOCP CVaR",
              lambda c: 1e4 * (c.get("p0recal", selected, "cvar_recomputed") -
                               c.get("p0socwork", exact, "cvar_recomputed"))),
    ]

    tight_raw = {"case": "mad@10bp", "portfolio": "cuopt-pdlp@1e-06", "endpoint": "raw"}
    loose_raw = {"case": "mad@10bp", "portfolio": "cuopt-pdlp@0.0001", "endpoint": "raw"}
    loose_rep = {"case": "mad@10bp", "portfolio": "repair:l1", "endpoint": "repaired-feasible",
                 "repaired_from": "cuopt-pdlp@0.0001"}
    basic = {"case": "mad@10bp", "portfolio": "cuopt-pdlp+xover@0.0001", "endpoint": "basic"}
    out += [
        Macro("PZeroEndpointLooseRawSeconds", s, "seconds", "1e-4 raw total endpoint time",
              lambda c: c.get("p0endpoint", loose_raw, "total_endpoint_s")),
        Macro("PZeroEndpointLooseRawVerdict", s, "text", "1e-4 raw strict gate verdict",
              lambda c: c.text("p0endpoint", loose_raw, "gate_verdict")),
        Macro("PZeroEndpointLooseRepairSeconds", s, "seconds", "1e-4 raw plus L1 repair total time",
              lambda c: c.get("p0endpoint", loose_rep, "total_endpoint_s")),
        Macro("PZeroEndpointLooseRepairVerdict", s, "text", "1e-4 L1 repair strict gate verdict",
              lambda c: c.text("p0endpoint", loose_rep, "gate_verdict")),
        Macro("PZeroEndpointTightRawSeconds", s, "seconds", "1e-6 raw total endpoint time",
              lambda c: c.get("p0endpoint", tight_raw, "total_endpoint_s")),
        Macro("PZeroEndpointTightRawVerdict", s, "text", "1e-6 raw strict gate verdict",
              lambda c: c.text("p0endpoint", tight_raw, "gate_verdict")),
        Macro("PZeroEndpointBasicStatus", s, "text", "crossover/basic endpoint status",
              lambda c: c.text("p0endpoint", basic, "status")),
        Macro("PZeroEndpointBasicLowerBoundSeconds", s, "seconds", "crossover time-cap lower bound",
              lambda c: c.get("p0endpoint", basic, "total_endpoint_s")),
    ]

    pdlp_x = {"solver": "cuopt_pdlp+xover"}
    full = {**pdlp_x, "level": "L6_full_mandate"}
    loo = {**pdlp_x, "level": "LOO_no_turnover"}
    lzero = {**pdlp_x, "level": "L0_budget_box"}
    out += [
        Macro("PZeroLZeroCrossoverSeconds", s, "seconds", "L0 crossover overhead",
              lambda c: c.get("p0loo", lzero, "crossover_cost_s")),
        Macro("PZeroFullCrossoverSeconds", s, "seconds", "full-mandate crossover overhead",
              lambda c: c.get("p0loo", full, "crossover_cost_s")),
        Macro("PZeroNoTurnoverCrossoverSeconds", s, "seconds", "no-turnover crossover overhead",
              lambda c: c.get("p0loo", loo, "crossover_cost_s")),
        Macro("PZeroTurnoverMarginalSeconds", s, "seconds", "paired full minus no-turnover overhead",
              lambda c: c.get("p0loo", full, "turnover_marginal_cost_s")),
        Macro("PZeroTurnoverMarginalSharePct", s, "pct", "paired marginal / full overhead",
              lambda c: c.get("p0loo", full, "turnover_marginal_share")),
        Macro("PZeroTurnoverVerdict", s, "text", "pre-registered dominance classification",
              lambda c: c.text("p0loo", full, "turnover_dominance_verdict")),
        Macro("PZeroFullAuxVariables", s, "int", "full-mandate auxiliary variables added",
              lambda c: c.get("p0loo", full, "lp_vars_added")),
        Macro("PZeroNoTurnoverAuxVariables", s, "int", "no-turnover auxiliary variables added",
              lambda c: c.get("p0loo", loo, "lp_vars_added")),
    ]

    out += [
        Macro("PZeroQpFiveHundredBonds", s, "int", "smallest matched QP grid size",
              lambda c: c.get("p0qp", {"n_bonds": 500, "arm": "cuopt_barrier_dense"}, "n_bonds")),
        Macro("PZeroQpTwoThousandBonds", s, "int", "matched QP grid size",
              lambda c: c.get("p0qp", {"n_bonds": 2000, "arm": "cuopt_barrier_dense"}, "n_bonds")),
        Macro("PZeroQpThreeThousandBonds", s, "int", "matched QP grid size",
              lambda c: c.get("p0qp", {"n_bonds": 3000, "arm": "cuopt_barrier_dense"}, "n_bonds")),
        Macro("PZeroQpAmundiBonds", s, "int", "largest requested QP grid size",
              lambda c: c.get("p0qp", {"n_bonds": 29222, "arm": "cuopt_barrier_dense"}, "n_bonds")),
        Macro("PZeroQpEligibleRows", s, "int", "rows passing the independent quality gate",
              lambda c: c.agg("p0qp", {"ratio_eligible": "True"}, "", "count")),
        Macro("PZeroQpRatioRows", s, "int", "rows carrying a matched endpoint solver-stack ratio",
              lambda c: c.agg("p0qp", {}, "solver_stack_ratio", "count_numeric")),
        Macro("PZeroQpMaxObjectiveGap", s, "sci", "maximum relative best-feasible objective gap among eligible rows",
              lambda c: c.agg("p0qp", {"ratio_eligible": "True"}, "relative_best_objective_gap", "max")),
        Macro("PZeroQpAmundiFactoredStatus", s, "text", "29,222-name factored scenario-generation outcome",
              lambda c: c.text("p0qp", {"n_bonds": 29222, "arm": "cuopt_barrier_factored"}, "status")),
        Macro("PZeroQpLargestSolvedBonds", s, "int", "largest n with an optimal factored GPU row",
              lambda c: c.agg("p0qp", {"arm": "cuopt_barrier_factored", "status": "optimal"}, "n_bonds", "max")),
    ]
    for n, word in ((500, "FiveHundred"), (2000, "TwoThousand"), (3000, "ThreeThousand")):
        for form, title in (("dense", "Dense"), ("factored", "Factored")):
            out.append(Macro(
                f"PZeroQpRawRatio{title}{word}", s, "ratio",
                "precomputed matched raw-endpoint GPU/CPU solver-stack ratio",
                lambda c, n=n, form=form: c.get(
                    "p0qp", {"n_bonds": n, "arm": f"cuopt_barrier_{form}",
                             "endpoint": "raw", "ratio_eligible": "True"},
                    "solver_stack_ratio")))
    return out


P1_CPU_SOCP_COLD = {
    "solver": "clarabel-socp", "request_kind": "fresh_model_cold",
    "formulation": "mean_cvar_sigma_SOCP",
}
P1_GPU_SOCP_COLD = {
    "solver": "cuopt-barrier-socp", "request_kind": "fresh_model_cold",
    "formulation": "mean_cvar_sigma_SOCP",
}


def _p1_socp_objective_gap(c: Ctx) -> float:
    """Compare cold independently scored objectives only on the same model.

    This is diagnostic objective agreement, not an acceptance or speed claim.
    The model hash includes the instance and the full objective/constraint spec.
    """
    for field in ("model_hash", "instance_hash"):
        cpu = c.text("p1cpu", P1_CPU_SOCP_COLD, field)
        gpu = c.text("p1gpu", P1_GPU_SOCP_COLD, field)
        if cpu != gpu:
            raise MissingData(
                f"{SOURCES['p1cpu'].filename} and {SOURCES['p1gpu'].filename} "
                f"have mismatched {field}; cannot compare cold SOCP objectives.")
    return abs(c.get("p1gpu", P1_GPU_SOCP_COLD, "objective_recomputed")
               - c.get("p1cpu", P1_CPU_SOCP_COLD, "objective_recomputed"))


def _p1_accepted(c: Ctx, key: str, sel: dict[str, Any]) -> str:
    """Report the recorded acceptance flag, never infer it from solver status."""
    accepted = c.text(key, sel, "accepted").lower()
    if accepted not in {"true", "false"}:
        raise MissingData(
            f"{SOURCES[key].filename}, row [{_selector_text(sel)}], "
            f"accepted must be True or False, got '{accepted}'.")
    return "Yes" if accepted == "true" else "No"


def _p1_request_seconds(c: Ctx, key: str, sel: dict[str, Any]) -> float:
    """A diagnostic single observation, with status and acceptance provenance."""
    c.text(key, sel, "status")
    _p1_accepted(c, key, sel)
    return c.timing(key, sel, "total_request_s")


def _macros_p1() -> list[Macro]:
    """Measured quantity and request pilots; no comparative timing ratios.

    Seconds are single recorded observations, including failed-gate latency.
    Prose using them must call them diagnostic, report acceptance alongside
    them, and avoid CPU/GPU or cold/cached performance claims.
    """
    s = "P1 measured pilots"
    out = []
    for nav, word in ((1e6, "One"), (1e7, "Ten"), (1e8, "Hundred")):
        for endpoint, title in (("naive-rounding", "Rounded"),
                                ("lot-milp-repair", "Repair")):
            sel = {"nav_usd": nav, "endpoint": endpoint}
            out.append(Macro(
                f"POneLot{word}{title}SigmaBp", s, "bp4",
                "independently recomputed lot-pilot sigma TE; strict breaches remain visible",
                lambda c, sel=sel: c.get("p1lots", sel, "te_sigma_bp")))
    lot_hundred = {"nav_usd": 1e8, "endpoint": "lot-milp-repair"}
    out += [
        Macro("POneLotTenRepairGapPct", s, "pct3",
              "recorded relative MIP gap on L1 distance, not portfolio return or CVaR",
              lambda c: c.get("p1lots", {"nav_usd": 1e7, "endpoint": "lot-milp-repair"},
                              "mip_gap")),
        Macro("POneLotHundredRepairSigmaExcessBp", s, "bp5",
              "repaired sigma TE minus the same row's strict sigma target, in bp",
              lambda c: c.get("p1lots", lot_hundred, "te_sigma_bp")
                        - c.get("p1lots", lot_hundred, "sigma_target_bp")),
        Macro("POneLotRowCount", s, "int", "all recorded lot-pilot endpoints",
              lambda c: c.agg("p1lots", {}, "", "count")),
        Macro("POneGpuSocpObjectiveGap", s, "sci4",
              "recorded absolute solver-versus-recomputed objective discrepancy; failed gate",
              lambda c: c.get("p1gpu", P1_GPU_SOCP_COLD, "objective_discrepancy")),
        Macro("POneGpuSocpVsCpuObjectiveGap", s, "sci4",
              "absolute recomputed objective difference of same-model cold SOCPs; diagnostic only",
              _p1_socp_objective_gap),
        Macro("POneCpuHighsSigmaBp", s, "bp5", "true sigma TE of the cold CPU MAD request",
              lambda c: c.get("p1cpu", {"solver": "highs-simplex",
                                       "request_kind": "fresh_model_cold"}, "te_sigma_bp")),
        Macro("POneCpuRowCount", s, "int", "all recorded CPU requests",
              lambda c: c.agg("p1cpu", {}, "", "count")),
        Macro("POneGpuRowCount", s, "int", "all recorded GPU requests, including rejected rows",
              lambda c: c.agg("p1gpu", {}, "", "count")),
        Macro("POneGpuAcceptedCount", s, "int", "requests with recorded acceptance True",
              lambda c: c.agg("p1gpu", {"accepted": True}, "", "count")),
        Macro("POneGpuRatioCount", s, "int",
              "recorded numeric request-ratio count among accepted rows only; no ratio is computed",
              lambda c: c.agg("p1gpu", {"accepted": True}, "request_ratio", "count_numeric")),
    ]
    diagnostic = ("single recorded total-request seconds; diagnostic observation, including "
                  "failed-gate latency; report acceptance in prose; no performance or ratio claim")
    out.append(Macro(
        "POneCpuHighsSeconds", s, "seconds", diagnostic,
        lambda c: _p1_request_seconds(c, "p1cpu", {
            "solver": "highs-simplex", "request_kind": "fresh_model_cold"})))
    for word, kind in (("Cold", "fresh_model_cold"), ("Cached", "same_problem_cached")):
        sel = {**P1_CPU_SOCP_COLD, "request_kind": kind}
        out += [
            Macro(f"POneCpuSocp{word}Seconds", s, "seconds", diagnostic,
                  lambda c, sel=sel: _p1_request_seconds(c, "p1cpu", sel)),
            Macro(f"POneCpuSocp{word}SigmaBp", s, "bp5", "true sigma TE of the CPU SOCP request",
                  lambda c, sel=sel: c.get("p1cpu", sel, "te_sigma_bp")),
        ]
    for word, solver, formulation in (
        ("Loose", "cuopt-pdlp-1e-4", "mean_cvar_mad_LP"),
        ("Tight", "cuopt-pdlp-1e-6", "mean_cvar_mad_LP"),
        ("Socp", "cuopt-barrier-socp", "mean_cvar_sigma_SOCP"),
    ):
        sel = {"solver": solver, "request_kind": "fresh_model_cold",
               "formulation": formulation}
        out += [
            Macro(f"POneGpu{word}SigmaBp", s, "bp5", "true sigma TE of the cold GPU request",
                  lambda c, sel=sel: c.get("p1gpu", sel, "te_sigma_bp")),
            Macro(f"POneGpu{word}Seconds", s, "seconds", diagnostic,
                  lambda c, sel=sel: _p1_request_seconds(c, "p1gpu", sel)),
            Macro(f"POneGpu{word}Accepted", s, "text", "recorded independent acceptance, Yes or No",
                  lambda c, sel=sel: _p1_accepted(c, "p1gpu", sel)),
            Macro(f"POneGpu{word}Status", s, "text", "recorded solver status, not acceptance",
                  lambda c, sel=sel: c.text("p1gpu", sel, "status")),
        ]
    return out


def build_macros() -> list[Macro]:
    macros = (_macros_env() + _macros_m1() + _macros_m3()
              + _macros_m2_solves() + _macros_m2_econ() + _macros_m2_limits()
              + _macros_m6() + _macros_scen() + _macros_warm()
              + _macros_ablation() + _macros_m7() + _macros_m9()
              + _macros_demo() + _macros_p0() + _macros_p1())
    seen: set[str] = set()
    for m in macros:
        if m.name in seen:
            raise SystemExit(
                f"internal error: macro \\{m.name} is defined twice; "
                f"\\newcommand would fail at build time"
            )
        if not m.name.isalpha():
            raise SystemExit(
                f"internal error: macro name '{m.name}' contains a "
                f"non-letter; \\newcommand only accepts letters"
            )
        if m.fmt not in FORMATTERS:
            raise SystemExit(
                f"internal error: macro \\{m.name} names an unknown "
                f"formatter '{m.fmt}'"
            )
        seen.add(m.name)
    return macros


# --------------------------------------------------------------------------
# Emission
# --------------------------------------------------------------------------

MISSING_VALUE = r"\textbf{??}"


def _wrap(prefix: str, text: str, width: int = 76) -> list[str]:
    """Wrap a provenance comment without breaking a pasteable command."""
    words, lines, cur = str(text).split(), [], prefix
    for w in words:
        if len(cur) + 1 + len(w) > width and cur.strip() != prefix.strip():
            lines.append(cur)
            cur = "%     " + w
        else:
            cur = f"{cur} {w}" if cur.strip() != prefix.strip() else \
                f"{prefix}{w}"
    lines.append(cur)
    return lines


@dataclass
class Emitted:
    """One macro's rendered block plus whether it is a real measurement."""

    name: str
    section: str
    lines: list[str]
    missing_reason: str = ""
    non_result: str = ""


def report_problems(problems: list[tuple[str, str]]) -> list[str]:
    """Group the failure list by cause, so it reads as a work order.

    Reporting 59 macros blocked on one absent file as 59 paragraphs of
    identical text buries the four things someone actually has to run. The
    grouping key is the message, because every message already names its
    own producing command.
    """
    grouped: dict[str, list[str]] = {}
    for name, message in problems:
        grouped.setdefault(message, []).append(name)
    lines: list[str] = []
    for message, names in sorted(grouped.items(),
                                 key=lambda kv: -len(kv[1])):
        head = message.split("\n")
        lines.append(f"[{len(names)} macro(s)] {head[0]}")
        for extra in head[1:]:
            lines.append("    " + extra.strip())
        wrapped, cur = [], "    "
        for n in names:
            if len(cur) + len(n) + 2 > 74 and cur.strip():
                wrapped.append(cur)
                cur = "    "
            cur += f"\\{n}  "
        wrapped.append(cur)
        lines += wrapped
        lines.append("")
    return lines


def render(macros: list[Macro],
           ctx: Ctx) -> tuple[list[Emitted], list[tuple[str, str]]]:
    """Compute every macro, recording what could not be computed and why."""
    out: list[Emitted] = []
    problems: list[tuple[str, str]] = []
    for m in macros:
        ctx.begin()
        lines = [f"% \\{m.name}"]
        for chunk in _wrap("%   arith: ", m.arith):
            lines.append(chunk)
        try:
            value = m.compute(ctx)
        except RecordedNonResult as exc:
            # Includes measured failures and explicit publication exclusions.
            # An excluded numeric timing must not masquerade as a solver DNF.
            for chunk in _wrap(
                    "%   NON-RESULT: ",
                    f"{exc.label}; no numeric result is published for this "
                    f"recorded outcome or exclusion; the strict build accepts it"):
                lines.append(chunk)
            for acc in ctx.accesses:
                for chunk in _wrap(
                        "%   src: ",
                        f"{acc.path} [{acc.selector}] {acc.column} = "
                        f"{acc.raw}"):
                    lines.append(chunk)
            lines.append(f"\\newcommand{{\\{m.name}}}"
                         f"{{\\textit{{{latex_escape(exc.label)}}}}}")
            out.append(Emitted(m.name, m.section, lines,
                               non_result=exc.label))
            continue
        except MissingData as exc:
            for chunk in _wrap("%   UNAVAILABLE: ", str(exc).split("\n")[0]):
                lines.append(chunk)
            for extra in str(exc).split("\n")[1:]:
                lines.append("%   " + extra.strip())
            lines.append(f"\\newcommand{{\\{m.name}}}{{{MISSING_VALUE}}}")
            out.append(Emitted(m.name, m.section, lines, str(exc)))
            problems.append((m.name, str(exc)))
            continue
        for acc in ctx.accesses:
            for chunk in _wrap(
                    "%   src: ",
                    f"{acc.path} [{acc.selector}] {acc.column} = {acc.raw}"):
                lines.append(chunk)
        if not ctx.accesses:
            lines.append("%   src: (no CSV read -- check this macro)")
        text = FORMATTERS[m.fmt](value)
        lines.append(f"\\newcommand{{\\{m.name}}}{{{text}}}")
        out.append(Emitted(m.name, m.section, lines))
    return out, problems


def header_lines(ctx: Ctx, allow_missing: bool, n_missing: int,
                 disagreements: list[str]) -> list[str]:
    """The preamble of numbers.tex: what this file is and where it came from.

    Deliberately carries no generation timestamp. A timestamp would make the
    file churn on every build and turn a meaningful diff ("the wide cell got
    faster") into noise. The dates that matter are the source CSVs' own
    `timestamp_utc` headers, listed below.
    """
    lines = [
        "% ------------------------------------------------------------",
        "% numbers.tex - GENERATED. Do not edit by hand.",
        "%",
        "%   python scripts/make_paper_numbers.py",
        "%",
        "% Every quantity the paper prints is defined here and read out of "
        "a",
        "% committed CSV in results/. Derived quantities (speedups, ratios,",
        "% percentage reductions) are computed by that script from the CSV",
        "% values, never transcribed. Each macro carries the file, the row",
        "% selector and the raw value it came from.",
        "%",
        "% Source files and the runs behind them:",
    ]
    for key in SOURCES:
        src = SOURCES[key]
        tab = ctx.loaded.get(key)
        if tab is None:
            lines.append(f"%   {src.filename:<34} ABSENT")
        else:
            stamp = tab.env.get("timestamp_utc", "(no timestamp header)")
            lines.append(f"%   {src.filename:<34} {stamp}")
    lines.append("%")
    if disagreements:
        lines.append("% WARNING - the committed CSVs disagree about the "
                     "environment:")
        for d in disagreements:
            lines += _wrap("%   ", d)
        lines.append("%   The paper's environment paragraph cannot be true "
                     "of all of them.")
        lines.append("%")
    if allow_missing:
        lines += [
            f"% DRAFT BUILD: {n_missing} macro(s) had no committed "
            f"measurement and",
            "% expand to \\textbf{??}. Any ?? in the PDF is an unmeasured "
            "quantity,",
            "% not a rounding artifact. Rebuild without --allow-missing "
            "before",
            "% submitting.",
            "%",
        ]
    lines.append("% ------------------------------------------------------"
                 "------")
    lines.append("")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Emit paper/numbers.tex from the committed results CSVs.")
    ap.add_argument("--results-dir", default=str(REPO / "results"),
                    help="directory holding the committed result CSVs")
    ap.add_argument("--out", default=str(REPO / "paper" / "numbers.tex"),
                    help="LaTeX file to write")
    ap.add_argument("--allow-missing", action="store_true",
                    help="emit \\textbf{??} for quantities with no committed "
                         "measurement instead of failing. For drafting only: "
                         "a ?? that reaches a PDF is a visible hole, which "
                         "is the point, but a strict build is what a "
                         "submission gets")
    ap.add_argument("--quiet", action="store_true",
                    help="suppress the per-section summary")
    args = ap.parse_args()

    macros = build_macros()
    ctx = Ctx(Path(args.results_dir))
    emitted, problems = render(macros, ctx)
    disagreements = env_disagreements(ctx)

    body: list[str] = []
    last_section = ""
    for e in emitted:
        if e.section != last_section:
            body += ["", f"% ==== {e.section} "
                         + "=" * max(0, 66 - len(e.section)), ""]
            last_section = e.section
        body += e.lines
        body.append("")

    text = "\n".join(header_lines(ctx, args.allow_missing, len(problems),
                                  disagreements) + body) + "\n"

    if not args.quiet:
        by_section: dict[str, list[int]] = {}
        for e in emitted:
            slot = by_section.setdefault(e.section, [0, 0, 0])
            slot[0] += 1
            if e.missing_reason:
                slot[1] += 1
            if e.non_result:
                slot[2] += 1
        print(f"macros defined: {len(emitted)}")
        for sec, (total, miss, nonres) in by_section.items():
            flag = f"  ({miss} UNAVAILABLE)" if miss else ""
            if nonres:
                flag += f"  ({nonres} recorded non-result)"
            print(f"  {sec:<28} {total:>4}{flag}")

    if disagreements and not args.allow_missing:
        print("\nENVIRONMENT DISAGREEMENT -- the committed CSVs were not "
              "measured\non one software stack, so the paper cannot state "
              "one environment:", file=sys.stderr)
        for d in disagreements:
            print(f"  {d}", file=sys.stderr)
        print("\nRegenerate the affected results in a single session, or "
              "split the\nenvironment paragraph per experiment.",
              file=sys.stderr)
        return 2

    if problems and not args.allow_missing:
        print(f"\n{len(problems)} macro(s) have no committed measurement. "
              f"paper/numbers.tex\nwas NOT written. Grouped by what has to "
              f"be run:\n", file=sys.stderr)
        for line in report_problems(problems):
            print(line, file=sys.stderr)
        print("Re-run with --allow-missing to emit \\textbf{??} placeholders "
              "and keep\ndrafting.", file=sys.stderr)
        return 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(f"\nwrote {out_path}")
    if problems:
        print(f"  {len(problems)} macro(s) expand to {MISSING_VALUE} -- "
              f"see the UNAVAILABLE comments in the file")
    return 0


if __name__ == "__main__":
    sys.exit(main())

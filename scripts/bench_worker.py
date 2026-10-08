#!/usr/bin/env python
"""Benchmark worker: one solver configuration in one clean process.

Why a subprocess: highspy and OR-Tools bundle different HiGHS builds whose
C++ symbols collide when loaded together (undefined-symbol crash at best,
silent ODR corruption at worst). Each solver gets its own process; timings
happen entirely inside the worker so process startup never contaminates them.

One invocation is also one REPLICATE. `run_spec` repeats its timed solves
inside this process, so median-of-runs measures within-process spread only;
an orchestrator that wants the across-process spread (the larger of the two
on this project's hardware -- see the 22.0s/38.8s note in the README) invokes
this script again. Every record therefore carries `pid`, so a CSV row can
show its replicates came from distinct processes rather than from one process
counted twice.

Protocol: reads the problem from an .npz, runs the spec, prints one line
starting with RESULT_JSON: followed by the record dict.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RESULT_MARKER = "RESULT_JSON:"


def _num(x: float) -> Any:
    """Round for the CSV; NaN becomes an empty cell rather than "nan".

    An empty cell is read as absent by every consumer in the repo, which is
    what a NaN here means: the quantity has no value on this row (a duration
    breach has no dollar figure), not that it was measured as nothing.
    """
    v = float(x)
    if math.isnan(v):
        return ""
    # Ten SIGNIFICANT digits, not ten decimal places: a 1.1e-12 breach
    # multiple must not print as 0.0 in one column while its dollar value
    # survives in the next one.
    return float(f"{v:.10g}")


def mandate_columns(model, uni, limits, dnf: bool) -> dict[str, Any]:
    """Score the solved portfolio against the limit sheet it was built with.

    Scored here rather than in the orchestrator because this process is the
    only one holding the weight vector; shipping thousands of doubles back
    through the result line to score them elsewhere would buy nothing.

    What the sheet IS matters more than where the call lives. The sweep does
    not build a generic Mean-CVaR instance: `run_sweep.py` derives
    `default_limits(uni)` for each cell and pickles it next to the universe,
    and that same object is what `credit_constraint_builder` above turned
    into the LP's constraint rows. The mandate scored here is therefore the
    mandate the solver was actually given -- nothing is invented and no limit
    sheet is borrowed from another run. Two consequences the reader has to be
    told, so they are recorded in the CSV rather than left implicit:

    - The sheet is DERIVED PER CELL from that cell's own benchmark, so a
      breach at 300 bonds and one at 2,500 bonds are breaches of different
      numbers. The verdicts compare solvers within a cell; across cells they
      are not a like-for-like series.
    - `default_limits` leaves family 7 (tracking error) at te_bp = inf, so
      the LP never carried it and it is not scored. Families 1-6 are the
      whole mandate here. `exposure_report` is consequently called with no
      scenario matrix: at an infinite budget its TE rows would report an
      unconstrained number wearing a pass.

    Returns `mandate_scored: False` with a reason whenever there is nothing
    to score -- never a silently absent column.
    """
    if uni is None or limits is None:
        return {"mandate_scored": False,
                "mandate_note": "no limit sheet passed (--universe absent): "
                                "this run has no mandate to score against"}
    if dnf:
        return {"mandate_scored": False,
                "mandate_note": "no portfolio to score (DNF)"}

    from src.credit import exposure_report, gate_verdicts

    try:
        w = model.weights()
        rep = exposure_report(uni, w, limits)
        gates = gate_verdicts(rep)
    except Exception as e:  # noqa: BLE001
        # A scoring failure must not throw away timings that were already
        # paid for, and it must not masquerade as a pass either.
        return {"mandate_scored": False,
                "mandate_note": f"scoring failed ({type(e).__name__}: {e})"}

    row = gates.as_row()
    for k in ("purity_worst_rel", "desk_worst_multiple",
              "desk_worst_bp_weight", "desk_worst_usd"):
        row[k] = _num(row[k])
    row["mandate_scored"] = True
    row["mandate_note"] = ""
    row["n_limit_rows"] = int(len(rep))
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help=".npz with R, mu, params")
    ap.add_argument("--factory", required=True)
    ap.add_argument("--kwargs", default="{}", help="JSON kwargs for the factory")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--mode", choices=["bench", "reference"], default="bench")
    ap.add_argument("--ref", default="", help=".npz with ref_w, ref_obj")
    ap.add_argument("--out-ref", default="", help="where to save reference npz")
    ap.add_argument("--replicate", type=int, default=1,
                    help="1-based index of this replicate, echoed back in the "
                         "record so a row can be traced to the process that "
                         "produced it")
    ap.add_argument("--universe", default="",
                    help="pickled (BondUniverse, CreditLimits); when given, "
                         "the credit constraint set is attached to the model "
                         "AND the solved portfolio is scored against it")
    args = ap.parse_args()

    if args.factory == "ortools_pdlp":
        # MUST precede any cvxpy import: importing cvxpy loads highspy, whose
        # HiGHS build exports C++ symbols that shadow the ones OR-Tools'
        # bundled HiGHS provides — libortools then fails with an undefined
        # symbol. Loading libortools first binds its own symbols correctly.
        from ortools.pdlp.python import pdlp  # noqa: F401

    from src.bench import SPEC_FACTORIES, run_spec, solve_reference
    from src.model import build_mean_cvar

    d = np.load(args.data)
    R, mu = d["R"], d["mu"]
    alpha = float(d["alpha"])
    risk_aversion = float(d["risk_aversion"])
    w_max = float(d["w_max"])

    # The universe and the scenario matrix are generated ONCE by the
    # orchestrator and passed in as files. Regenerating them per worker would
    # be cheaper but wrong: the GPU and CPU scenario paths do not share a
    # random stream, so solvers would be racing on different problems.
    builder = None
    uni = limits = None
    if args.universe:
        import pickle

        from src.credit import credit_constraint_builder
        with open(args.universe, "rb") as f:
            uni, limits = pickle.load(f)
        builder = credit_constraint_builder(uni, limits)

    def build():
        return build_mean_cvar(
            R, mu, alpha=alpha, risk_aversion=risk_aversion, w_max=w_max,
            constraint_builder=builder,
        )

    spec = SPEC_FACTORIES[args.factory](**json.loads(args.kwargs))

    if args.mode == "reference":
        # Timed so the orchestrator can report what ground truth cost without
        # inferring it from the subprocess wall. It is build + solve in a
        # fresh process, and it is never part of a benchmarked timing.
        t0 = time.perf_counter()
        ref_w, ref_obj = solve_reference(build, spec)
        elapsed = time.perf_counter() - t0
        np.savez(args.out_ref, ref_w=ref_w, ref_obj=ref_obj)
        print(RESULT_MARKER + json.dumps(
            {"ref_obj": ref_obj, "label": spec.label,
             "ref_solve_s": round(elapsed, 4), "pid": os.getpid()}))
        return 0

    reference = None
    if args.ref:
        r = np.load(args.ref)
        reference = (r["ref_w"], float(r["ref_obj"]))

    # The model is bound to a name rather than passed inline: the mandate
    # scoring below needs the weight vector the solve produced, and nothing
    # else in this process holds it.
    model = build()
    rec = run_spec(model, spec, n_runs=args.runs, reference=reference)
    out = rec.row()
    out["wall_s_all"] = [round(w, 4) for w in rec.wall_s_all]
    out["pid"] = os.getpid()
    out["replicate"] = int(args.replicate)
    out.update(mandate_columns(model, uni, limits, bool(rec.dnf)))
    print(RESULT_MARKER + json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())

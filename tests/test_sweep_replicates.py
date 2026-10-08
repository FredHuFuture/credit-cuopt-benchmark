"""Across-process replication in the M3 sweep, and its resume key.

`run_spec` runs all of its timed solves inside ONE process, so median-of-runs
measures within-process spread. This repo's documented failure mode is across
process and session — 22.0s against 38.8s on consecutive runs of one
crossover configuration — so the sweep replicates at the PROCESS level and
reports the two spreads separately. The arithmetic that collapses those
replicates into one CSV row is what decides the error bar on the paper's
load-bearing rows, and it is pure: it can be tested without a solver, which
is the only way to exercise a DNF on the third replicate deliberately rather
than waiting for one.

The records below are hand-built worker records, not solves. Asking a solver
to produce a 26x spread on demand is not a test.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from run_sweep import (  # noqa: E402
    aggregate_replicates, job_of, row_is_complete,
)


def _rec(walls: list[float], dnf: bool = False, check_ok: bool = True,
         verdict: str = "OK", pid: int = 1) -> dict:
    """One worker record, carrying only what the aggregator reads."""
    med = sorted(walls)[len(walls) // 2] if walls else float("nan")
    return {
        "solver": "highs-simplex", "hardware": "CPU", "status": "optimal",
        "dnf": dnf, "n_runs": len(walls), "wall_s_median": med,
        "wall_s_min": min(walls) if walls else "",
        "wall_s_max": max(walls) if walls else "",
        "solver_internal_s": med, "overhead_s": 0.0,
        "wall_s_all": walls, "check_ok": check_ok, "gate_verdict": verdict,
        "pid": pid, "notes": "",
    }


# --------------------------------------------------------------------------
# The default must not change any published number
# --------------------------------------------------------------------------

def test_one_replicate_reproduces_the_single_process_summary():
    rec = _rec([3.0, 1.0, 2.0])
    row = aggregate_replicates([rec], runs_per_replicate=3)
    # Exactly what the sweep reported before replication existed.
    assert row["wall_s_median"] == 2.0
    assert row["wall_s_min"] == 1.0
    assert row["wall_s_max"] == 3.0
    assert row["n_runs"] == 3
    assert row["n_replicates"] == 1


def test_one_replicate_reports_no_across_process_spread():
    row = aggregate_replicates([_rec([3.0, 1.0, 2.0])], runs_per_replicate=3)
    # Empty, not zero. One process cannot measure a spread across processes,
    # and a 0.0 there would read as "measured, and it was nothing".
    assert row["wall_s_iqr_across"] == ""
    assert row["wall_s_p25_across"] == ""
    assert row["wall_s_p75_across"] == ""
    assert row["iqr_across_over_within"] == ""
    assert row["wall_s_iqr_within"] != ""


# --------------------------------------------------------------------------
# The two spreads are different quantities
# --------------------------------------------------------------------------

def test_headline_is_the_median_of_the_process_medians():
    recs = [_rec([1.0, 1.0], pid=1), _rec([2.0, 2.0], pid=2),
            _rec([10.0, 10.0], pid=3)]
    row = aggregate_replicates(recs, runs_per_replicate=2)
    assert row["wall_s_median"] == 2.0
    assert json.loads(row["replicate_medians"]) == [1.0, 2.0, 10.0]
    # Pooled extremes still span every timing.
    assert row["wall_s_min"] == 1.0 and row["wall_s_max"] == 10.0


def test_within_and_across_spreads_do_not_contaminate_each_other():
    # Each process is internally almost silent; the processes disagree wildly.
    recs = [_rec([1.00, 1.02], pid=1), _rec([2.00, 2.02], pid=2),
            _rec([10.00, 10.02], pid=3)]
    row = aggregate_replicates(recs, runs_per_replicate=2)
    # Within: half the 0.02 range of a two-run process (inclusive quartiles).
    assert abs(row["wall_s_iqr_within"] - 0.01) < 1e-9
    # Across: quartiles of the medians [1.01, 2.01, 10.01].
    assert abs(row["wall_s_p25_across"] - 1.51) < 1e-9
    assert abs(row["wall_s_p75_across"] - 6.01) < 1e-9
    assert abs(row["wall_s_iqr_across"] - 4.50) < 1e-9
    assert row["iqr_across_over_within"] == 450.0


def test_every_individual_timing_survives_twice():
    recs = [_rec([1.0, 1.5], pid=7), _rec([2.0, 2.5], pid=9)]
    row = aggregate_replicates(recs, runs_per_replicate=2)
    assert row["wall_s_all"] == [1.0, 1.5, 2.0, 2.5]
    assert json.loads(row["wall_s_by_replicate"]) == [[1.0, 1.5], [2.0, 2.5]]
    assert json.loads(row["replicate_pids"]) == [7, 9]


# --------------------------------------------------------------------------
# Disagreement between processes is a result, not noise to average away
# --------------------------------------------------------------------------

def test_a_dnf_keeps_the_replicates_that_finished():
    recs = [_rec([4.0, 4.2], pid=1),
            _rec([], dnf=True, check_ok="", verdict="", pid=2)]
    row = aggregate_replicates(recs, runs_per_replicate=2)
    assert row["dnf"] is True
    assert row["n_replicates"] == 2 and row["n_replicates_ok"] == 1
    assert json.loads(row["replicate_medians"]) == [4.1]
    assert "DNF on replicate 2" in row["notes"]


def test_a_verdict_that_flips_across_processes_is_flagged():
    recs = [_rec([1.0, 1.0], verdict="OK", pid=1),
            _rec([1.0, 1.0], verdict="BREACH", pid=2)]
    row = aggregate_replicates(recs, runs_per_replicate=2)
    assert "mandate verdict DISAGREES" in row["notes"]


def test_a_check_that_flips_across_processes_is_flagged():
    recs = [_rec([1.0, 1.0], check_ok=True, pid=1),
            _rec([1.0, 1.0], check_ok=False, pid=2)]
    row = aggregate_replicates(recs, runs_per_replicate=2)
    assert "check_ok DISAGREES" in row["notes"]


# --------------------------------------------------------------------------
# Resume
# --------------------------------------------------------------------------

def test_a_partial_replicate_set_is_not_a_finished_configuration():
    assert not row_is_complete({"dnf": "False", "n_replicates": "2"}, 5)
    assert row_is_complete({"dnf": "False", "n_replicates": "5"}, 5)
    assert row_is_complete({"dnf": "False", "n_replicates": "7"}, 5)


def test_a_row_written_before_replication_counts_as_one_process():
    legacy = {"dnf": "False", "wall_s_median": "5.2"}
    assert row_is_complete(legacy, 1)
    assert not row_is_complete(legacy, 3)


def test_a_dnf_is_a_finished_answer_at_any_replicate_count():
    # Re-establishing it costs two more time limits and changes nothing.
    assert row_is_complete({"dnf": "True", "n_replicates": "1"}, 5)
    assert row_is_complete({"dnf": "True", "n_replicates": ""}, 5)


def test_resume_recovers_the_job_from_a_solver_label():
    # The CSV label is built by the spec factory and carries the tolerance,
    # so the resume key cannot be the label. Rows written before the `job`
    # column existed still have to match.
    assert job_of({"job": "highs"}) == "highs"
    assert job_of({"solver": "highs-simplex"}) == "highs"
    assert job_of({"solver": "ortools-pdlp@0.0001"}) == "ortools_pdlp"
    assert job_of({"solver": "cuopt-pdlp@0.0001"}) == "cuopt_pdlp"
    assert job_of({"solver": "cuopt-pdlp+xover@0.0001"}) == "cuopt_pdlp+xover"
    # Memory-refusal rows already record the job name in `solver`.
    assert job_of({"solver": "cuopt_pdlp+xover"}) == "cuopt_pdlp+xover"

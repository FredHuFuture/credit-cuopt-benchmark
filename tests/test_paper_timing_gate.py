"""Publication gates over recorded CSVs only; no benchmark runs."""

import argparse
import csv
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import make_paper_numbers as numbers  # noqa: E402


TIMING_NAMES = {
    "PZeroSocpRefCuoptSeconds",
    "PZeroSocpRefClarabelSeconds",
    "PZeroSocpRefStackRatio",
}
SOLVERS = ("cuopt-barrier-socp", "clarabel-socp")


@pytest.fixture
def reference():
    header = {
        "gpu_contended": "no",
        "gpu_util_pct_at_capture": "0",
        "gpu_mem_used_mib_at_capture": "100",
    }
    rows = []
    for solver, endpoint, seconds in zip(
        SOLVERS, ("barrier", "interior-point"), (2.0, 6.0)
    ):
        rows.append({
            "case": "sigma@10bp", "te_mode": "sigma", "te_budget_bp": 10.0,
            "solver": solver, "method_class": "SOCP", "endpoint": endpoint,
            "status": "optimal", "status_source": "solver-returned",
            "check_ok": True, "purity_pass": True, "purity_worst_rel": 0.0,
            "respects_mandate": True, "desk_pass": True,
            "solver_objective": -0.05, "objective_recomputed": -0.05,
            "objective_discrepancy": 0.0, "te_sigma_bp": 10.0,
            "steady_state_median_s": seconds,
            "resource_before_gpu_util_pct": 0.0,
            "resource_before_gpu_mem_used_mib": 100.0,
            "resource_after_gpu_util_pct": 95.0,
            "resource_after_gpu_mem_used_mib": 150.0,
            "resource_note": "",
        })
    return header, rows


def render_reference(tmp_path, reference):
    header, rows = reference
    path = tmp_path / "p0_socp_reference.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        for key, value in header.items():
            stream.write(f"# {key}: {value}\n")
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    macros = [m for m in numbers.build_macros() if m.name in TIMING_NAMES]
    emitted, problems = numbers.render(macros, numbers.Ctx(tmp_path))
    assert not problems
    return {item.name: item for item in emitted}


def assert_excluded(item, reason):
    assert item.non_result, item.lines
    assert reason.lower() in item.non_result.lower()
    assert not item.missing_reason
    block = "\n".join(item.lines)
    assert "NON-RESULT:" in block
    assert numbers.MISSING_VALUE not in block
    assert item.lines[-1].startswith(
        rf"\newcommand{{\{item.name}}}{{\textit{{"
    )
    # The reason must reach the PDF, not just a source comment.
    assert numbers.latex_escape(item.non_result) in item.lines[-1]


def test_committed_contaminated_reference_cannot_publish_seconds_or_ratio():
    macros = [m for m in numbers.build_macros() if m.name in TIMING_NAMES]
    emitted, problems = numbers.render(macros, numbers.Ctx(REPO / "results"))
    assert not problems
    assert len(emitted) == 3
    for item in emitted:
        assert_excluded(item, "gpu_contended")
        block = "\n".join(item.lines)
        assert "p0_socp_reference.csv" in block
        assert "timings unreliable" in block


def test_clean_equivalent_socp_endpoints_can_publish_ratio(tmp_path, reference):
    emitted = render_reference(tmp_path, reference)
    for name, value in zip(
        ("PZeroSocpRefCuoptSeconds", "PZeroSocpRefClarabelSeconds",
         "PZeroSocpRefStackRatio"), ("2.00", "6.00", "3.00")
    ):
        assert not emitted[name].non_result
        assert emitted[name].lines[-1] == rf"\newcommand{{\{name}}}{{{value}}}"
    ratio_trace = "\n".join(emitted["PZeroSocpRefStackRatio"].lines)
    for field in ("gpu_contended", "resource_before_gpu_util_pct", "status",
                  "check_ok", "purity_pass", "objective_recomputed", "endpoint"):
        assert field in ratio_trace


@pytest.mark.parametrize("field,value", [
    ("gpu_contended", "yes -- timings unreliable"),
    ("gpu_contended", "unknown"),
    ("gpu_contended", None),
    ("gpu_util_pct_at_capture", None),
    ("gpu_util_pct_at_capture", "nan"),
    ("gpu_util_pct_at_capture", "busy"),
    ("gpu_util_pct_at_capture", "26"),
    ("gpu_mem_used_mib_at_capture", None),
    ("gpu_mem_used_mib_at_capture", "inf"),
])
def test_header_contamination_or_missing_telemetry_fails_closed(
    tmp_path, reference, field, value
):
    header, _ = reference
    if value is None:
        del header[field]
    else:
        header[field] = value
    for item in render_reference(tmp_path, reference).values():
        assert_excluded(item, field)


@pytest.mark.parametrize("solver_index", [0, 1])
@pytest.mark.parametrize("field,value", [
    ("resource_before_gpu_util_pct", 26.0),
    ("resource_before_gpu_util_pct", ""),
    ("resource_before_gpu_util_pct", "not available"),
    ("resource_before_gpu_mem_used_mib", "nan"),
    ("resource_after_gpu_util_pct", "nan"),
    ("resource_after_gpu_util_pct", 101.0),
    ("resource_after_gpu_mem_used_mib", -1.0),
    ("resource_note", "GPU telemetry unavailable from nvidia-smi; emitted NaN"),
    ("status", "time_limit"),
    ("status", "optimal_inaccurate"),
    ("status", ""),
    ("check_ok", False),
    ("check_ok", ""),
    ("purity_pass", False),
    ("purity_pass", ""),
    ("purity_worst_rel", 1e-3),
    ("respects_mandate", False),
    ("desk_pass", False),
    ("objective_recomputed", "nan"),
    ("solver_objective", -0.06),
    ("objective_discrepancy", 1e-3),
    ("te_sigma_bp", 11.0),
    ("steady_state_median_s", 0.0),
    ("steady_state_median_s", -1.0),
    ("steady_state_median_s", "inf"),
    ("steady_state_median_s", "nan"),
])
def test_each_ratio_operand_must_pass_the_independent_timing_gate(
    tmp_path, reference, solver_index, field, value
):
    reference[1][solver_index][field] = value
    emitted = render_reference(tmp_path, reference)
    assert_excluded(emitted["PZeroSocpRefStackRatio"], field)
    name = ("PZeroSocpRefCuoptSeconds", "PZeroSocpRefClarabelSeconds")[solver_index]
    assert_excluded(emitted[name], field)


@pytest.mark.parametrize("field", [
    "resource_before_gpu_util_pct", "resource_before_gpu_mem_used_mib",
    "resource_after_gpu_util_pct", "resource_after_gpu_mem_used_mib",
    "check_ok", "purity_pass", "objective_recomputed", "endpoint",
])
def test_missing_gate_columns_never_become_numeric_results(tmp_path, reference, field):
    for row in reference[1]:
        del row[field]
    assert_excluded(render_reference(tmp_path, reference)["PZeroSocpRefStackRatio"], field)


@pytest.mark.parametrize("solver_index", [0, 1])
@pytest.mark.parametrize("field,value", [
    ("endpoint", "repaired"),
    ("endpoint", "simplex"),
    ("endpoint", ""),
    ("method_class", "LP"),
    ("status_source", "author-repair"),
    ("te_mode", "mad"),
    ("te_budget_bp", 20.0),
])
def test_ratio_requires_equivalent_original_socp_endpoints(
    tmp_path, reference, solver_index, field, value
):
    reference[1][solver_index][field] = value
    assert_excluded(render_reference(tmp_path, reference)["PZeroSocpRefStackRatio"], field)


def test_ratio_rejects_different_achieved_objectives(tmp_path, reference):
    reference[1][1].update(solver_objective=-0.06, objective_recomputed=-0.06)
    assert_excluded(render_reference(tmp_path, reference)["PZeroSocpRefStackRatio"],
                    "objective")


def test_contamination_does_not_hide_valid_feasibility_numbers():
    names = {
        "PZeroSocpRefCuoptTeBp", "PZeroSocpRefClarabelTeBp",
        "PZeroSocpWorkingClarabelTeBp", "PZeroSocpWorkingClarabelObjective",
        "PZeroSelectedMadObjectiveLossBp",
    }
    emitted, problems = numbers.render(
        [m for m in numbers.build_macros() if m.name in names],
        numbers.Ctx(REPO / "results"),
    )
    assert not problems
    assert {item.name for item in emitted} == names
    assert all(not item.non_result and not item.missing_reason for item in emitted)
    for item in emitted[:3]:
        assert item.lines[-1].endswith("{10.00}")


def test_strict_generator_accepts_recorded_exclusions_without_placeholders(tmp_path):
    output = tmp_path / "numbers.tex"
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts/make_paper_numbers.py"),
         "--out", str(output)], cwd=REPO, text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    text = output.read_text(encoding="utf-8")
    assert numbers.MISSING_VALUE not in text
    for name in TIMING_NAMES:
        assert rf"\newcommand{{\{name}}}{{\textit{{" in text


def test_qp_producer_parses_to_the_recorded_matched_grid_and_strict_arms(monkeypatch):
    # Stop after the real parser, before main can touch the GPU or run a solve.
    import run_qp_branch

    class Parsed(Exception):
        pass

    captured = []
    real_parse = argparse.ArgumentParser.parse_args

    def parse_only(parser, *args, **kwargs):
        captured.append(real_parse(parser, *args, **kwargs))
        raise Parsed

    command = shlex.split(numbers.SOURCES["p0qp"].producer)
    assert command[:2] == ["python", "scripts/run_qp_branch.py"]
    monkeypatch.setattr(sys, "argv", command[1:])
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", parse_only)
    with pytest.raises(Parsed):
        run_qp_branch.main()
    args = captured[0]
    table = numbers.Ctx(REPO / "results").table("p0qp")
    assert args.grid == table.env["grid_n_bonds"]
    assert tuple(map(int, args.grid.split(","))) == (
        500, 1000, 2000, 3000, 5000, 7500, 10000, 15000, 29222,
    )
    assert set(args.arms) == set(table.df["arm"]) == {
        "clarabel_strict_dense", "clarabel_strict_factored",
        "cuopt_barrier_dense", "cuopt_barrier_factored",
    }
    assert args.repair_strict_clarabel
    assert args.runs == int(table.env["runs_per_config"])
    assert args.resume
    assert args.out == "results/p0_qp_matched.csv"

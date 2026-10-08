"""Paper macros over recorded P1 pilots only; no solver or GPU calls."""

import csv
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import make_paper_numbers as numbers  # noqa: E402


PRODUCERS = {
    "p1_bond_lots_cpu.csv":
        "python scripts/run_bond_lots.py --out results/p1_bond_lots_cpu.csv",
    "p1_request_timing_cpu.csv":
        "python scripts/run_request_bench.py --out results/p1_request_timing_cpu.csv",
    "p1_request_timing_gpu_smoke.csv":
        "python scripts/run_request_bench.py --allow-gpu "
        "--solvers cuopt-pdlp-1e-4 cuopt-pdlp-1e-6 cuopt-barrier-socp "
        "--time-limit 30 --out results/p1_request_timing_gpu_smoke.csv",
}

EXPECTED = {
    "POneLotOneRoundedSigmaBp": "58.6263",
    "POneLotTenRoundedSigmaBp": "9.8166",
    "POneLotHundredRoundedSigmaBp": "10.0081",
    "POneLotOneRepairSigmaBp": "13.1885",
    "POneLotTenRepairSigmaBp": "9.9482",
    "POneLotHundredRepairSigmaBp": "10.0025",
    "POneLotTenRepairGapPct": r"0.433\%",
    "POneLotHundredRepairSigmaExcessBp": "0.00245",
    "POneLotRowCount": "9",
    "POneGpuLooseSigmaBp": "28.29591",
    "POneGpuTightSigmaBp": "28.28596",
    "POneGpuSocpSigmaBp": "9.99997",
    "POneGpuSocpObjectiveGap": r"\ensuremath{1.084\times 10^{-6}}",
    "POneGpuSocpVsCpuObjectiveGap": r"\ensuremath{1.113\times 10^{-7}}",
    "POneCpuHighsSigmaBp": "28.28539",
    "POneCpuHighsSeconds": "2.29",
    "POneCpuSocpColdSigmaBp": "10.00000",
    "POneCpuSocpCachedSigmaBp": "10.00000",
    "POneCpuRowCount": "3",
    "POneGpuRowCount": "3",
    "POneGpuAcceptedCount": "0",
    "POneGpuRatioCount": "0",
    "POneCpuSocpColdSeconds": "0.787",
    "POneCpuSocpCachedSeconds": "0.657",
    "POneGpuLooseSeconds": "1.35",
    "POneGpuTightSeconds": "6.79",
    "POneGpuSocpSeconds": "1.25",
    "POneGpuLooseAccepted": "No",
    "POneGpuTightAccepted": "No",
    "POneGpuSocpAccepted": "No",
    "POneGpuLooseStatus": "optimal",
    "POneGpuTightStatus": "optimal",
    "POneGpuSocpStatus": "optimal",
}


def p1_macros(names=None):
    macros = [m for m in numbers.build_macros() if m.name.startswith("POne")]
    assert macros, "P1 pilot macros must be registered in build_macros"
    if names is not None:
        macros = [m for m in macros if m.name in names]
        assert {m.name for m in macros} == set(names)
    return macros


def rendered(results_dir, names=None):
    emitted, problems = numbers.render(p1_macros(names), numbers.Ctx(results_dir))
    assert not problems, problems
    return {item.name: item for item in emitted}


def assert_value(item, value):
    assert not item.missing_reason
    assert not item.non_result
    assert item.lines[-1] == rf"\newcommand{{\{item.name}}}{{{value}}}"


@pytest.fixture
def pilot_rows():
    rows = {}
    for filename in PRODUCERS:
        with (REPO / "results" / filename).open(newline="", encoding="utf-8") as stream:
            rows[filename] = list(csv.DictReader(stream))
    return rows


def write_pilots(tmp_path, rows):
    for filename, records in rows.items():
        with (tmp_path / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)


@pytest.mark.parametrize("filename,producer", PRODUCERS.items())
def test_missing_pilot_names_exact_producer(tmp_path, filename, producer):
    macros = p1_macros()
    _, problems = numbers.render(macros, numbers.Ctx(tmp_path))
    reasons = [reason for _, reason in problems if filename in reason]
    assert reasons, f"Missing {filename} must block strict generation"
    assert all(producer in reason for reason in reasons)


def test_committed_pilots_render_exact_values_with_provenance():
    emitted = rendered(REPO / "results")
    assert set(emitted) == set(EXPECTED)
    for name, value in EXPECTED.items():
        assert_value(emitted[name], value)
        block = "\n".join(emitted[name].lines)
        assert "%   src: results/p1_" in block
        assert "no CSV read" not in block
    gap = "\n".join(emitted["POneGpuSocpVsCpuObjectiveGap"].lines)
    for evidence in ("p1_request_timing_cpu.csv", "p1_request_timing_gpu_smoke.csv",
                     "model_hash", "instance_hash", "objective_recomputed"):
        assert evidence in gap


def test_slight_lot_sigma_failure_survives_rounding(pilot_rows):
    row = next(r for r in pilot_rows["p1_bond_lots_cpu.csv"]
               if float(r["nav_usd"]) == 1e8 and r["endpoint"] == "lot-milp-repair")
    assert row["sigma_pass"] == row["accepted_endpoint"] == "False"
    emitted = rendered(REPO / "results")
    assert_value(emitted["POneLotHundredRepairSigmaBp"], "10.0025")
    assert_value(emitted["POneLotHundredRepairSigmaExcessBp"], "0.00245")


@pytest.mark.parametrize("filename", [
    "p1_request_timing_cpu.csv", "p1_request_timing_gpu_smoke.csv",
])
@pytest.mark.parametrize("field", ["model_hash", "instance_hash", "formulation"])
@pytest.mark.parametrize("change", ["different", "", None])
def test_objective_comparison_rejects_unaligned_or_missing_model(
        tmp_path, pilot_rows, filename, field, change):
    records = pilot_rows[filename]
    for row in records:
        if change is None:
            row.pop(field)
        elif row["solver"] in {"clarabel-socp", "cuopt-barrier-socp"}:
            row[field] = change
    write_pilots(tmp_path, pilot_rows)
    emitted, problems = numbers.render(
        p1_macros({"POneGpuSocpVsCpuObjectiveGap"}), numbers.Ctx(tmp_path))
    assert problems, "An unaligned objective gap must fail strict generation"
    assert emitted[0].missing_reason
    assert field in emitted[0].missing_reason


def test_objective_comparison_uses_absolute_recomputed_cold_values(tmp_path, pilot_rows):
    for filename in ("p1_request_timing_cpu.csv", "p1_request_timing_gpu_smoke.csv"):
        for row in pilot_rows[filename]:
            if row["solver"] == "clarabel-socp":
                row["objective_recomputed"] = (
                    "-0.06" if row["request_kind"] == "fresh_model_cold" else "-0.9")
            elif row["solver"] == "cuopt-barrier-socp":
                row["objective_recomputed"] = "-0.05"
                row["solver_objective"] = "-0.8"
    write_pilots(tmp_path, pilot_rows)
    emitted = rendered(tmp_path, {"POneGpuSocpVsCpuObjectiveGap"})
    assert_value(emitted["POneGpuSocpVsCpuObjectiveGap"],
                 r"\ensuremath{1.000\times 10^{-2}}")


def test_cached_cpu_row_cannot_replace_missing_cold_reference(tmp_path, pilot_rows):
    filename = "p1_request_timing_cpu.csv"
    pilot_rows[filename] = [r for r in pilot_rows[filename]
                            if not (r["solver"] == "clarabel-socp"
                                    and r["request_kind"] == "fresh_model_cold")]
    write_pilots(tmp_path, pilot_rows)
    _, problems = numbers.render(p1_macros({"POneGpuSocpVsCpuObjectiveGap"}),
                                 numbers.Ctx(tmp_path))
    assert problems
    assert "fresh_model_cold" in problems[0][1]


def test_rejected_rows_never_become_accepted_or_timing_ratios(tmp_path, pilot_rows):
    # Even a stray numeric ratio on an optimal-but-rejected row is excluded.
    for row in pilot_rows["p1_request_timing_gpu_smoke.csv"]:
        row["request_ratio"] = "100"
    write_pilots(tmp_path, pilot_rows)
    emitted = rendered(tmp_path)
    assert_value(emitted["POneGpuAcceptedCount"], "0")
    assert_value(emitted["POneGpuRatioCount"], "0")
    for config in ("Loose", "Tight", "Socp"):
        assert_value(emitted[f"POneGpu{config}Accepted"], "No")
        assert_value(emitted[f"POneGpu{config}Status"], "optimal")
        seconds = emitted[f"POneGpu{config}Seconds"]
        assert_value(seconds, EXPECTED[seconds.name])
        block = "\n".join(seconds.lines)
        assert "diagnostic" in block
        assert "accepted = False" in block
    assert {m.name for m in p1_macros() if "Ratio" in m.name} == {"POneGpuRatioCount"}
    assert all(m.fmt != "ratio" for m in p1_macros())


def test_acceptance_text_and_count_follow_recorded_flag(tmp_path, pilot_rows):
    pilot_rows["p1_request_timing_gpu_smoke.csv"][0]["accepted"] = "True"
    write_pilots(tmp_path, pilot_rows)
    emitted = rendered(tmp_path)
    assert_value(emitted["POneGpuLooseAccepted"], "Yes")
    assert_value(emitted["POneGpuAcceptedCount"], "1")
    assert_value(emitted["POneGpuRatioCount"], "0")


@pytest.mark.parametrize("flag", ["", "unknown"])
def test_absent_or_invalid_acceptance_is_not_silently_no(tmp_path, pilot_rows, flag):
    pilot_rows["p1_request_timing_gpu_smoke.csv"][0]["accepted"] = flag
    write_pilots(tmp_path, pilot_rows)
    _, problems = numbers.render(p1_macros({"POneGpuLooseAccepted"}), numbers.Ctx(tmp_path))
    assert problems
    assert "accepted" in problems[0][1]


def test_main_strict_generates_all_numbers_to_scratch(tmp_path):
    out = tmp_path / "numbers.tex"
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts/make_paper_numbers.py"), "--out", str(out)],
        capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    content = out.read_text(encoding="utf-8")
    for name, value in EXPECTED.items():
        assert rf"\newcommand{{\{name}}}{{{value}}}" in content
    assert numbers.MISSING_VALUE not in content


@pytest.mark.parametrize("missing", PRODUCERS)
def test_main_strict_missing_pilot_does_not_write_output(tmp_path, missing):
    # Copy fixtures only; the committed results are never changed.
    results = tmp_path / "results"
    results.mkdir()
    for source in numbers.SOURCES.values():
        if source.filename != missing:
            shutil.copyfile(REPO / "results" / source.filename, results / source.filename)
    out = tmp_path / "numbers.tex"
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts/make_paper_numbers.py"),
         "--results-dir", str(results), "--out", str(out)],
        capture_output=True, text=True, check=False)
    assert result.returncode == 1, result.stderr
    assert not out.exists()
    assert missing in result.stderr
    assert PRODUCERS[missing] in result.stderr

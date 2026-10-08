import importlib.util
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "make_paper_numbers", REPO / "scripts" / "make_paper_numbers.py"
)
assert SPEC and SPEC.loader
paper_numbers = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = paper_numbers
SPEC.loader.exec_module(paper_numbers)
Ctx = paper_numbers.Ctx
build_macros = paper_numbers.build_macros
render = paper_numbers.render


def test_p0_macro_registry_is_complete_and_renderable():
    required = {
        "PZeroSocpRefCuoptSeconds",
        "PZeroSocpWorkingCuoptStatus",
        "PZeroNaiveMadOvershoot",
        "PZeroSelectedMadBudgetBp",
        "PZeroSelectedMadObjectiveLossBp",
        "PZeroEndpointTightRawSeconds",
        "PZeroEndpointBasicStatus",
        "PZeroTurnoverMarginalSeconds",
        "PZeroTurnoverMarginalSharePct",
        "PZeroQpEligibleRows",
        "PZeroQpRatioRows",
        "PZeroQpAmundiFactoredStatus",
    }
    macros = build_macros()
    selected = [m for m in macros if m.name in required]
    assert {m.name for m in selected} == required

    emitted, problems = render(selected, Ctx(REPO / "results"))
    assert not problems
    assert len(emitted) == len(required)


def test_p0_qp_ratio_macros_only_use_precomputed_eligible_rows():
    macros = [
        m for m in build_macros()
        if m.name.startswith("PZeroQpRawRatio")
    ]
    assert {m.name for m in macros} == {
        "PZeroQpRawRatioDenseFiveHundred",
        "PZeroQpRawRatioFactoredFiveHundred",
        "PZeroQpRawRatioDenseTwoThousand",
        "PZeroQpRawRatioFactoredTwoThousand",
        "PZeroQpRawRatioDenseThreeThousand",
        "PZeroQpRawRatioFactoredThreeThousand",
    }
    emitted, problems = render(macros, Ctx(REPO / "results"))
    assert not problems
    assert all(not item.non_result for item in emitted)

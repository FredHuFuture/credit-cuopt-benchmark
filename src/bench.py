"""Honest benchmark harness.

Protocol (decided in advance, before any numbers were seen):

- Monotonic clock (time.perf_counter) around prob.solve(); the first solve per
  (model, solver) is a discarded WARMUP (CUDA context + JIT + CVXPY
  canonicalization); reported wall time is the MEDIAN of the remaining runs,
  with min/max disclosed.
- Solver-internal time is recorded where the solver exposes it (cuOpt: yes,
  HiGHS: yes, OR-Tools PDLP via CVXPY: NOT exposed -> recorded as NaN and the
  wall column is authoritative). CVXPY canonicalization time is reported
  separately (compilation_time).
- Hard per-solve time limit; a run that exceeds it or returns any non-optimal
  status is a DNF: counted and reported, never silently dropped, never mixed
  into medians.
- CPU solvers get an explicit, disclosed thread count (default: all logical
  CPUs — the CPU's best showing).
- Any cross-size aggregate uses the shifted geometric mean (Mittelmann
  convention); single-size tables report raw medians.
- Every reported solve is validated by src/check.py (independent NumPy
  recomputation); check failures are flagged in the output row.

Tolerance mapping (the honest fine print):
- cuOpt PDLP and OR-Tools PDLP both terminate on residual criteria of the form
  `abs_tol + rel_tol * scale`, but the scales and norms differ in detail; we
  set both solvers' absolute and relative optimality tolerances to the same
  epsilon and DISCLOSE that the stopping rules are matched-as-closely-as-the-
  implementations-allow, not identical. Achieved solution quality is then
  verified independently by check.py, which is the comparison that matters.
- HiGHS simplex has no epsilon-optimality knob: it terminates at a vertex
  solution with feasibility tolerances (default 1e-7). It is reported as the
  "vertex-accuracy" baseline at its defaults, labeled as such — never as an
  equal-tolerance competitor to PDLP at 1e-4.
- HiGHS IPM stops on `ipm_optimality_tolerance` and, unless crossover runs,
  returns an INTERIOR point. That is the same distinction PDLP forces, so the
  two IPM configurations are reported separately (`highs-ipm`, without
  crossover, and `highs-ipm+xover`, with) rather than as one "interior-point"
  number: only the second is a mandate-signable vertex, and the position
  counts in the check columns are the evidence.

Algorithm provenance:
- A column labelled "IPM" is worth nothing unless the run can be shown to have
  been IPM. `highs_algorithm_evidence` reads HiGHS's own per-algorithm
  iteration counters back out of the info struct CVXPY forwards, and every
  HiGHS row carries the result in `algorithm_evidence`. See that function for
  what the counters can and cannot prove.
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import math
import os
import platform
import statistics
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import cvxpy as cp
import numpy as np

from .check import CheckResult, check_solution
from .model import CVaRModel

DEFAULT_TIME_LIMIT_S = 300.0
DEFAULT_THREADS = os.cpu_count() or 1


# --------------------------------------------------------------------------
# Environment capture
# --------------------------------------------------------------------------

def _pkg_version(name: str) -> str:
    try:
        import importlib.metadata as md
        return md.version(name)
    except Exception:  # noqa: BLE001
        return "n/a"


def gpu_contention(samples: int = 3, gap_s: float = 0.4) -> dict[str, float]:
    """Sample what else is using the GPU right now.

    Median-of-repeats defends against noise *within* a run. It does nothing
    against a sustained competing workload: a game or a video call pinning the
    card makes every configuration slower by an amount that is stable enough
    to look like a real measurement. This project measured the same solver
    configuration at 10s and at 77s on consecutive runs before noticing the
    GPU was at 98% utilisation for an unrelated reason.

    Returns utilisation and memory used by *everything*, so a caller can
    refuse to publish numbers taken on a busy card.
    """
    utils, mems = [], []
    for i in range(samples):
        try:
            out = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=utilization.gpu,memory.used,power.draw",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10)
            if out.returncode != 0:
                break
            u, m, _p = out.stdout.strip().splitlines()[0].split(",")
            utils.append(float(u))
            mems.append(float(m))
        except Exception:  # noqa: BLE001
            break
        if i + 1 < samples:
            time.sleep(gap_s)
    if not utils:
        return {"gpu_util_pct": float("nan"), "gpu_mem_used_mib": float("nan")}
    return {"gpu_util_pct": statistics.median(utils),
            "gpu_mem_used_mib": statistics.median(mems)}


def assert_gpu_idle(max_util_pct: float = 25.0, raise_on_busy: bool = False
                    ) -> dict[str, float]:
    """Warn (or refuse) when the GPU is already busy before timing anything."""
    c = gpu_contention()
    busy = c["gpu_util_pct"] == c["gpu_util_pct"] and \
        c["gpu_util_pct"] > max_util_pct
    if busy:
        msg = (f"GPU is {c['gpu_util_pct']:.0f}% busy and holding "
               f"{c['gpu_mem_used_mib']:.0f} MiB before this run starts. "
               f"Timings taken now are contaminated by whatever else is using "
               f"the card and must not be published.")
        if raise_on_busy:
            raise RuntimeError(msg)
        print(f"WARNING: {msg}", flush=True)
    c["gpu_busy_at_start"] = float(busy)
    return c


def capture_environment() -> dict[str, str]:
    env = {
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count_logical": str(os.cpu_count()),
        "cvxpy": _pkg_version("cvxpy"),
        "cuopt": _pkg_version("cuopt-cu12"),
        "cudf": _pkg_version("cudf-cu12"),
        "highspy": _pkg_version("highspy"),
        "ortools": _pkg_version("ortools"),
        "numpy": _pkg_version("numpy"),
    }
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    env["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        env["cpu_model"] = "n/a"
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        env["gpu"] = out.stdout.strip() if out.returncode == 0 else "n/a"
    except Exception:  # noqa: BLE001
        env["gpu"] = "n/a"
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    kb = int(line.split()[1])
                    env["ram_visible_gib"] = f"{kb / 1024 / 1024:.1f}"
                    break
    except OSError:
        env["ram_visible_gib"] = "n/a"
    env["wsl2"] = "yes" if "microsoft" in platform.release().lower() else "no"
    c = gpu_contention()
    env["gpu_util_pct_at_capture"] = f"{c['gpu_util_pct']:.0f}"
    env["gpu_mem_used_mib_at_capture"] = f"{c['gpu_mem_used_mib']:.0f}"
    env["gpu_contended"] = ("yes — timings unreliable"
                            if c["gpu_util_pct"] > 25 else "no")
    return env


# --------------------------------------------------------------------------
# Solver specifications
# --------------------------------------------------------------------------

@dataclass
class SolverSpec:
    """One benchmarked solver configuration."""

    label: str                       # e.g. "cuopt-pdlp@1e-4"
    cvxpy_solver: str                # cp.CUOPT / cp.HIGHS / cp.PDLP
    solver_opts: dict[str, Any]
    tolerance_desc: str              # human-readable stopping rule
    hardware: str                    # "GPU" | "CPU"
    threads: int | None = None       # None => not applicable (GPU)
    internal_time_exposed: bool = True
    # The checker verifies the solver met what it PROMISED: a solver asked
    # for eps must land within ~10*eps (allowance for residual norm scaling),
    # not within some unrelated fixed tolerance. Order-of-magnitude
    # violations (the documented PDLP failure mode) still fail loudly.
    check_feasibility_tol: float = 1e-6
    check_objective_tol: float = 1e-4
    # Complete effective configuration, reproducible from the CSV alone —
    # including defaults the CVXPY interface injects behind our back.
    config_full: str = ""
    # Optional post-solve probe that reports which algorithm the solver
    # ACTUALLY ran, read from the solver's own counters rather than inferred
    # from the options we set. Only HiGHS supplies one today; specs without it
    # leave the column empty rather than claiming something unverified.
    algorithm_probe: Callable[[cp.Problem], str] | None = None


# --------------------------------------------------------------------------
# cuOpt configuration introspection
#
# cuOpt's presolver values are C macros in the installed build's header
# (cuopt/linear_programming/constants.h: CUOPT_PRESOLVE_DEFAULT / _OFF /
# _PAPILO / _PSLP). Unlike the parameter NAMES, which the Python layer
# re-exports as CUOPT_* globals, those integers are not exposed to Python at
# all, so they are written out here — and then every value that reaches a
# solve is replayed through the installed C++ parser first, so a build that
# disagrees says so at spec construction rather than three cells into a sweep.
# --------------------------------------------------------------------------

CUOPT_PRESOLVE_DEFAULT = -1
CUOPT_PRESOLVE_OFF = 0
CUOPT_PRESOLVE_PAPILO = 1
CUOPT_PRESOLVE_PSLP = 2

_CUOPT_PRESOLVE_NAMES: dict[int, str] = {
    CUOPT_PRESOLVE_DEFAULT: "solver's choice",
    CUOPT_PRESOLVE_OFF: "off",
    CUOPT_PRESOLVE_PAPILO: "Papilo",
    CUOPT_PRESOLVE_PSLP: "PSLP",
}


def _cuopt_option(name: str) -> tuple[bool, Any]:
    """(exists, build default) for a cuOpt LP parameter, asked of THIS build.

    The cuOpt analogue of `_highs_option`, and it exists for a sharper reason.
    HiGHS rejects an option it does not have; cuOpt does not.
    `cuopt_conif._get_solver_settings` intersects the options dict with the
    build's parameter registry and DROPS every key that misses, without a
    warning. A renamed or misspelled option therefore leaves the solver at its
    default while `config_full` still records that the option was set — the
    exact failure mode this file is trying to close. Anything pinned below is
    looked up here first.

    The import is local because the CPU-only workers (highs, ortools) import
    this module too, and cuOpt must not become a hard dependency of theirs.
    """
    from cuopt.linear_programming import solver_settings as cs

    if name not in cs.solver_params:
        return False, None
    return True, cs.SolverSettings().get_parameter(name)


def _cuopt_assert_value(name: str, value: Any) -> None:
    """Fail now if the installed cuOpt would reject `value` for `name`.

    Value checking lives in the C++ layer and `SolverSettings.set_parameter`
    does not reach it: that method validates the NAME, stores the value, and
    returns. The range and type checks fire later, inside the solve, which in
    a benchmark means a subprocess dying after the problem was built. The
    setting is replayed here so the failure lands at spec construction.

    This is where the int-versus-bool trap gets caught. `presolve` is an
    integer-valued option, and on this build `True`, `False`, `"on"` and
    `"off"` are all rejected by the C++ parser ("value True is not an
    integer") even though `set_parameter` swallowed them without complaint.
    """
    from cuopt.linear_programming import solver_settings as cs

    probe = cs.SolverSettings()
    probe.set_parameter(name, value)
    probe.set_c_solver_settings()   # the C++ range/type check fires here


def _cuopt_enum_value(enum_cls: Any, token: str, what: str) -> int:
    """Resolve a cuOpt enum token the way `cuopt_conif` will, or raise.

    `cuopt_conif._solver_mode` / `._solver_method` call `token.isdigit()`,
    index the enum, and swallow EVERY failure into `None` — at which point the
    option is simply never set and the solve runs at the build default. On
    this build that default is Stable3 for `pdlp_solver_mode` and Concurrent
    for `method`, so a typo does not produce an error: it produces a
    differently-configured benchmark row wearing the right label. Passing a
    non-str is the same trap by another route, since `.isdigit()` raises
    AttributeError straight into that bare `except`.
    """
    if not isinstance(token, str):
        raise TypeError(
            f"cuOpt {what} must be a str: cuopt_conif calls .isdigit() on it "
            f"and silently ignores anything that raises, leaving the build "
            f"default in place (got {type(token).__name__})"
        )
    try:
        member = enum_cls(int(token)) if token.isdigit() else enum_cls[token]
    except (KeyError, ValueError) as e:
        known = ", ".join(m.name for m in enum_cls)
        raise ValueError(
            f"{token!r} is not a {what} the installed cuOpt knows; it has "
            f"{known}"
        ) from e
    return int(member)


def cuopt_convergence_evidence(problem: cp.Problem) -> str:
    """What cuOpt's own counters say about the solve it just finished.

    The cuOpt counterpart of `highs_algorithm_evidence`, and deliberately
    weaker, because cuOpt exposes less. `cuopt_conif` forwards
    `Solution.get_lp_stats()` as `solver_stats.extra_stats`, which on the LP
    path carries exactly four numbers: primal_residual, dual_residual, gap and
    nb_iterations. Those are worth recording — the ACHIEVED residuals are what
    this project's "matched tolerance is not a matched stopping rule" caveat
    turns on, and without them a row states only what was REQUESTED.

    What it cannot prove, and the reason presolve has to be pinned rather than
    measured: `presolve_time` is a MILP-only field on this build (it lives in
    `Solution.milp_stats`; `get_lp_stats` does not carry it), and
    `get_solved_by()` is never forwarded through CVXPY at all. So no cuOpt row
    can show which presolver ran, or whether one ran. The presolve state in
    `config_full` is a setting this harness pinned, not evidence the solver
    returned — unlike the HiGHS rows, where it is evidence.

    One shared caveat with the HiGHS probe: on a `solver_error` CVXPY raises
    before refreshing `solver_stats`, so a probe run against that record reads
    the PREVIOUS solve's counters. Rows reached that way are DNFs whose
    timings are discarded anyway, but the evidence string on them is stale.
    """
    stats = getattr(problem, "solver_stats", None)
    info = getattr(stats, "extra_stats", None)
    if not isinstance(info, dict) or not info:
        return "unavailable (solver exposed no LP stats dict)"
    order = ("primal_residual", "dual_residual", "gap", "nb_iterations")
    parts = []
    for key in order:
        if key not in info:
            continue
        v = info[key]
        parts.append(f"{key}={int(v)}" if key == "nb_iterations"
                     else f"{key}={float(v):.3e}")
    if not parts:
        return f"unavailable (LP stats dict carries none of {order})"
    return "cuopt lp_stats: " + " ".join(parts)


def cuopt_pdlp_spec(
    tol: float,
    time_limit_s: float = DEFAULT_TIME_LIMIT_S,
    mode: str = "Stable2",
    crossover: bool = False,
    presolve: int = CUOPT_PRESOLVE_PSLP,
) -> SolverSpec:
    """cuOpt PDLP on the GPU, with presolve and solver mode both PINNED.

    Presolve
    --------
    This spec used to set `pdlp_solver_mode` and leave `presolve` alone, which
    made cuOpt the one solver in the comparison whose most consequential
    setting was absent from every CSV row — while OR-Tools discloses its glop
    presolve in `config_full` and HiGHS runs presolve at a documented default.
    In a project that separately measured presolve at 9.2x on this problem
    class (README, warm-start trade-off: 91.7s without against 10.0s with),
    that is an uncontrolled variable larger than several of the effects being
    reported. It is now a named argument, validated against the installed
    build, and written into `config_full` on every row.

    MEASURED 2026-08-24, and the header was wrong. Past runs inherited the
    build default `presolve = -1` (`presolver_t::Default`), a SENTINEL rather
    than a state, and `utilities/internals.hpp` documents that sentinel as
    resolving to None. It does not. Four arms on one 300x2000 credit cell,
    three timed runs each, each arm in its own subprocess:

        presolve=-1 (the sentinel every committed row ran)   0.224s
        presolve=0  (OFF)                                    0.782s
        presolve=1  (Papilo, CPU)                            0.259s
        presolve=2  (PSLP, GPU)                              0.227s

    So the sentinel presolves, and pinning OFF would have made this project's
    re-run 3.5x slower than every number already committed, for a reason with
    nothing to do with anything the paper claims. That is why the runbook puts
    this measurement at hour zero of the GPU session rather than trusting the
    header.

    The pin is therefore PSLP, which is the only value that satisfies all
    three requirements at once. It is within noise of the sentinel (0.227
    against 0.224), so the session stays comparable with the committed CSVs.
    It is an explicit named value written into `config_full`, so the
    uncontrolled variable this whole change exists to remove is actually
    removed — "solver's choice" is not a state anyone can reproduce across
    builds. And it runs on the GPU, unlike Papilo, which is a CPU presolver
    linked into libcuopt through TBB with `num_cpu_threads` defaulting to all
    cores; switching Papilo on inside the column labelled "GPU", in a paper
    whose entire claim is GPU against CPU, would move an unmeasured amount of
    work onto the host under a GPU heading. `presolve=CUOPT_PRESOLVE_OFF` and
    `=CUOPT_PRESOLVE_PAPILO` remain one-argument experiments if a separately
    labelled arm is ever wanted.

    Note what this does NOT resolve: nothing on the LP result path reports
    which presolver actually ran (see `cuopt_convergence_evidence`), so the
    table above is inference from wall time, not a read of solver state. It
    is enough to choose a pin; it is not proof that -1 IS PSLP.

    Solver mode
    -----------
    `mode` stays Stable2, and the build default it OVERRIDES is now recorded
    alongside it. This build's default is Stable3, whose own enum docstring
    calls it the "best overall mode from experiments". So Stable2 is a
    conservative choice, these are not cuOpt's best numbers, and the CSV now
    carries the evidence for saying so instead of leaving a referee to
    discover it from the release notes.
    """
    from cuopt.linear_programming.solver_settings import (
        PDLPSolverMode,
        SolverMethod,
    )

    have_presolve, presolve_default = _cuopt_option("presolve")
    if not have_presolve:
        raise RuntimeError(
            "the installed cuOpt exposes no 'presolve' parameter, so this "
            "build's presolve state can be neither pinned nor disclosed; a "
            "cuOpt column cannot be published from it"
        )
    _cuopt_assert_value("presolve", presolve)

    # Resolved up front so a bad token fails here rather than silently
    # demoting the run to the build defaults (Stable3 / Concurrent).
    method = "PDLP"
    _cuopt_enum_value(SolverMethod, method, "method")
    _cuopt_enum_value(PDLPSolverMode, mode, "pdlp_solver_mode")
    have_mode, mode_default = _cuopt_option("pdlp_solver_mode")
    mode_default_name = (
        PDLPSolverMode(int(mode_default)).name if have_mode else "unknown"
    )

    opts: dict[str, Any] = {
        "solver_method": method,
        "pdlp_solver_mode": mode,
        # int, not bool: the C++ parser rejects True/False outright.
        "presolve": int(presolve),
        "absolute_primal_tolerance": tol,
        "relative_primal_tolerance": tol,
        "absolute_dual_tolerance": tol,
        "relative_dual_tolerance": tol,
        "absolute_gap_tolerance": tol,
        "relative_gap_tolerance": tol,
        "time_limit": time_limit_s,
        "crossover": 1 if crossover else 0,
    }
    pre_name = _CUOPT_PRESOLVE_NAMES.get(
        int(presolve), f"code {int(presolve)}")
    pre_default_name = _CUOPT_PRESOLVE_NAMES.get(
        int(presolve_default), f"code {int(presolve_default)}")
    label = f"cuopt-pdlp{'+xover' if crossover else ''}@{tol:g}"
    return SolverSpec(
        label=label,
        cvxpy_solver=cp.CUOPT,
        solver_opts=opts,
        tolerance_desc=(
            f"PDLP abs+rel primal/dual/gap tolerances = {tol:g}, "
            f"mode={mode} (build default {mode_default_name}), "
            f"presolve={int(presolve)} ({pre_name}; build default "
            f"{presolve_default} = {pre_default_name})"
            + (", crossover to basic solution" if crossover else "")
        ),
        hardware="GPU",
        internal_time_exposed=True,
        check_feasibility_tol=10 * tol,
        check_objective_tol=10 * tol,
        # Kept as valid JSON, with the overridden build defaults under a
        # marked key: a row has to show both what was pinned and what the pin
        # displaced, or "presolve=0" reads as a default rather than a choice.
        config_full=json.dumps({
            **opts,
            "_build_defaults": {
                "presolve": presolve_default,
                "pdlp_solver_mode": mode_default,
                "pdlp_solver_mode_name": mode_default_name,
            },
        }),
        algorithm_probe=cuopt_convergence_evidence,
    )


def ortools_pdlp_spec(
    tol: float,
    time_limit_s: float = DEFAULT_TIME_LIMIT_S,
    threads: int = DEFAULT_THREADS,
) -> SolverSpec:
    from ortools.pdlp import solvers_pb2

    from .pdlp_compat import PatchedPDLP  # upstream bounded-vars bug fix

    params = solvers_pb2.PrimalDualHybridGradientParams()
    params.num_threads = threads
    crit = params.termination_criteria.simple_optimality_criteria
    crit.eps_optimal_absolute = tol
    crit.eps_optimal_relative = tol
    opts = {"parameters_proto": params, "time_limit_sec": time_limit_s}
    return SolverSpec(
        label=f"ortools-pdlp@{tol:g}",
        cvxpy_solver=PatchedPDLP(),  # instance: CVXPY custom-solver path
        solver_opts=opts,
        tolerance_desc=f"PDLP eps_optimal abs+rel = {tol:g}, {threads} threads",
        hardware="CPU",
        threads=threads,
        internal_time_exposed=True,  # via PatchedPDLP solve_log capture
        check_feasibility_tol=10 * tol,
        check_objective_tol=10 * tol,
        config_full=(
            "PrimalDualHybridGradientParams{ "
            + " ".join(str(params).split())
            + " }; interface injects presolve_options.use_glop=True; "
            + f"time_limit_sec={time_limit_s}"
        ),
    )


def highs_algorithm_evidence(problem: cp.Problem) -> str:
    """Which algorithm HiGHS actually ran, from HiGHS's own counters.

    A column labelled "IPM" is not evidence that IPM ran; it is evidence that
    we set an option. The CVXPY HiGHS interface forwards the raw `HighsInfo`
    struct as `solver_stats.extra_stats`, and that struct keeps a SEPARATE
    iteration counter per algorithm (`ipm_iteration_count`,
    `crossover_iteration_count`, `simplex_iteration_count`,
    `pdlp_iteration_count`, `qp_iteration_count`) plus `basis_validity`. So
    the claim is checkable after the fact:

    - `solver="ipm"` must show ipm_iter > 0 with simplex_iter == 0. If simplex
      iterations appear instead, HiGHS fell back and the column is
      mislabelled. Measured on a 200x6,000 credit instance: 43 IPM iterations
      versus 3,442 simplex iterations for the same LP, so the two are not
      confusable.
    - `basis_validity` is what separates crossover from no crossover: 0
      (invalid, no basis) with `run_crossover="off"`, 1 with it on and 1 for
      simplex. No basis means no vertex, which is the whole point of the
      no-crossover row.
    - `crossover_iteration_count` is NOT that signal and must not be read as
      one. It counts PUSHES, and on the credit instance above a crossover
      that ran to completion needed zero of them — IPX's starting-basis
      construction had already reached the optimal basis — so the counter
      reads 0 whether crossover ran or not. Only `basis_validity` moved.
    - All counters zero AND an invalid basis means presolve disposed of the
      problem and no algorithm ran on it. That is the failure mode this probe
      exists to catch, because the options alone would still have said "ipm".

    What it cannot prove: the counters do not distinguish which of HiGHS's
    interior-point implementations executed, and they say nothing about how
    the work inside an iteration was spent.
    """
    stats = getattr(problem, "solver_stats", None)
    info = getattr(stats, "extra_stats", None)
    if info is None:
        return "unavailable (solver exposed no info struct)"
    counters = {
        "ipm": "ipm_iteration_count",
        "crossover": "crossover_iteration_count",
        "simplex": "simplex_iteration_count",
        "pdlp": "pdlp_iteration_count",
        "qp": "qp_iteration_count",
    }
    counts: dict[str, int] = {}
    for short, attr in counters.items():
        v = getattr(info, attr, None)
        if v is not None:
            counts[short] = int(v)
    if not counts:
        return "unavailable (info struct carries no iteration counters)"
    ran = [k for k, v in counts.items() if v > 0]
    basis = getattr(info, "basis_validity", None)
    if basis is None:
        basis_desc = "unknown"
    else:
        # highspy reports this as a plain int (0/1) rather than the C++ enum.
        basis_desc = {0: "invalid(no vertex)", 1: "valid(vertex)"}.get(
            int(basis), f"code {int(basis)}")
    return (
        f"ran={'+'.join(ran) if ran else 'none (presolve solved it)'}; "
        + " ".join(f"{k}_iter={v}" for k, v in counts.items())
        + f" basis={basis_desc}"
    )


def _highs_option(name: str) -> tuple[bool, Any]:
    """(exists, default value) for a HiGHS option, asked of THIS build.

    Hard-coded option names and hard-coded default tolerances are how a paper
    ends up describing a configuration the linked build never ran: HiGHS
    renames options and moves defaults between releases (`run_crossover` is a
    string here, not the boolean the name suggests). Anything that reaches
    `tolerance_desc` is therefore read back from the installed library, and an
    option this build does not have is reported as absent, never guessed.

    A missing highspy is left to raise ImportError on its own: "no HiGHS
    installed" and "this HiGHS lacks that option" are different facts and
    collapsing them into one silent False would hide the first.
    """
    import highspy

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    if h.getOptionType(name)[0] != highspy.HighsStatus.kOk:
        return False, None
    status, value = h.getOptionValue(name)
    if status != highspy.HighsStatus.kOk:
        return False, None
    return True, value


def _highs_presolve(native: dict[str, Any], presolve: str | None) -> str:
    """Pin HiGHS's presolve into `native` and return its disclosure text.

    Disclosure symmetry, not a tuning knob. `presolve` was absent from the
    HiGHS `config_full` for exactly the same reason it was absent from
    cuOpt's — nobody set it — so fixing only the cuOpt column would have
    swapped which solver a referee can hold to account rather than fixing
    anything. Both are now pinned and recorded.

    `presolve=None` pins the option to the value THIS build already defaults
    to, read back through `_highs_option`. Setting an option to the value it
    already holds cannot change a solve, so this puts the state in every CSV
    row without touching what the pending re-run measures. Only an explicit
    argument ("on" / "off" / "choose") is a behaviour change, and then it is
    one the caller asked for.
    """
    have, default = _highs_option("presolve")
    if not have:
        if presolve is not None:
            raise RuntimeError(
                "presolve was requested but this HiGHS build has no "
                "presolve option"
            )
        return ("presolve UNKNOWN — this HiGHS build exposes no presolve "
                "option")
    native["presolve"] = default if presolve is None else presolve
    return (f"presolve={native['presolve']!r} "
            f"(build default {default!r})")


def highs_spec(
    time_limit_s: float = DEFAULT_TIME_LIMIT_S,
    threads: int = DEFAULT_THREADS,
    solver: str = "simplex",
    feasibility_tol: float | None = None,
    seed: int = 0,
    presolve: str | None = None,
) -> SolverSpec:
    # HiGHS's native "solver" option clashes with CVXPY's solver= kwarg;
    # the interface provides the nested highs_options dict for exactly this.
    native: dict[str, Any] = {
        "solver": solver,
        "time_limit": time_limit_s,
        "threads": threads,
        "parallel": "on",
        "random_seed": seed,
    }
    pre_desc = _highs_presolve(native, presolve)
    tol_desc = "defaults (primal/dual feasibility 1e-7), vertex solution"
    if feasibility_tol is not None:
        native["primal_feasibility_tolerance"] = feasibility_tol
        native["dual_feasibility_tolerance"] = feasibility_tol
        tol_desc = f"primal/dual feasibility = {feasibility_tol:g}, vertex solution"
    tol_desc = f"{tol_desc}; {pre_desc}"
    return SolverSpec(
        label=f"highs-{solver}",
        cvxpy_solver=cp.HIGHS,
        solver_opts={"highs_options": native},
        tolerance_desc=tol_desc,
        hardware="CPU",
        threads=threads,
        internal_time_exposed=True,
        config_full=json.dumps(native),
        algorithm_probe=highs_algorithm_evidence,
    )


def highs_ipm_spec(
    time_limit_s: float = DEFAULT_TIME_LIMIT_S,
    threads: int = DEFAULT_THREADS,
    crossover: bool = False,
    optimality_tol: float | None = None,
    seed: int = 0,
    presolve: str | None = None,
) -> SolverSpec:
    """HiGHS interior-point, with the crossover switch made explicit.

    Interior-point is the CPU algorithm class whose hardware story is
    genuinely ambiguous for this problem shape — 20-50 iterations, each
    dominated by one factorization of a normal-equations system sized by the
    BOND count rather than the scenario count, which is dense and GEMM-shaped
    and therefore the one CPU baseline a GPU is not obviously going to beat.
    Benchmarking only simplex and PDLP on the CPU side leaves that class out
    of the comparison entirely.

    `crossover` is a first-class argument, mirroring `cuopt_pdlp_spec`,
    because the two settings answer different questions and this project is
    precisely about the gap between them. Without crossover HiGHS stops at an
    interior point where every bound is strictly slack, so the weight vector
    is DENSE; with crossover it finishes at a vertex. Reporting one number for
    "HiGHS IPM" without saying which would compare two different objects.

    How big that difference is in portfolio terms is an empirical question and
    must not be assumed to match PDLP's. Measured here on a 200-bond x 6,000-
    scenario credit instance at the default tolerance: 200/200 weights
    strictly positive without crossover against 109/200 with it, but the extra
    91 sat at 1e-16 to 1e-13 — dense in structure, machine noise in size, and
    a 2.4e-14 objective difference. Loosening the tolerance to 1e-6 lifted the
    smallest nonzero only to 4.1e-10. So on that instance the no-crossover
    portfolio is untradeable as a matter of form rather than of magnitude.
    Whether the full 500 x 20,000 mandate behaves the same way is unmeasured;
    `nonzero_weights` in the CSV is what settles it per row.

    HiGHS spells the switch `run_crossover`, and it is a STRING option
    ("on"/"off"/"choose"), not the boolean the name suggests — the name and
    its accepted values are checked against the installed build rather than
    assumed, and both end up verbatim in `config_full`.

    `optimality_tol` is left at the build default deliberately. Loosening it
    is not the free speed knob it looks like: at 1e-4 with crossover off,
    HiGHS 1.15.1 downgrades its own model status from Optimal to Unknown
    ("relative violation of tolerances"), and CVXPY then raises ValueError
    rather than returning a status, which `_solve_once` does not catch — the
    run becomes a worker failure, not a clean DNF. Measured, not assumed. Use
    the argument only when that outcome is the thing being studied.
    """
    have_solver, _ = _highs_option("solver")
    have_xover, xover_default = _highs_option("run_crossover")
    if not (have_solver and have_xover):
        missing = [n for n, ok in (("solver", have_solver),
                                   ("run_crossover", have_xover)) if not ok]
        raise RuntimeError(
            f"the installed HiGHS does not expose {missing}; an "
            f"interior-point column cannot be configured (let alone "
            f"labelled) on this build"
        )

    native: dict[str, Any] = {
        "solver": "ipm",
        "run_crossover": "on" if crossover else "off",
        "time_limit": time_limit_s,
        "threads": threads,
        "parallel": "on",
        "random_seed": seed,
    }
    pre_desc = _highs_presolve(native, presolve)
    have_tol, tol_default = _highs_option("ipm_optimality_tolerance")
    if optimality_tol is not None:
        if not have_tol:
            raise RuntimeError(
                "optimality_tol was requested but this HiGHS build has no "
                "ipm_optimality_tolerance option"
            )
        native["ipm_optimality_tolerance"] = optimality_tol
        tol_txt = f"ipm_optimality_tolerance = {optimality_tol:g}"
        eff_tol = float(optimality_tol)
    elif have_tol:
        tol_txt = (f"ipm_optimality_tolerance = {float(tol_default):g} "
                   f"(build default)")
        eff_tol = float(tol_default)
    else:
        tol_txt = ("ipm optimality tolerance UNKNOWN — this HiGHS build "
                   "exposes no ipm_optimality_tolerance option")
        eff_tol = math.nan

    if crossover:
        end_desc = (f"run_crossover=on (build default {xover_default!r}): "
                    f"crossover to a basic VERTEX solution")
    else:
        end_desc = (f"run_crossover=off (build default {xover_default!r}): "
                    f"terminates at an INTERIOR point, not a vertex")

    # An interior point is feasible to the IPM's own tolerance, not to
    # simplex's 1e-7, so the checker is told what this configuration actually
    # promised. The 1e-6 floor is the harness default and is kept as a floor
    # so a tight IPM tolerance can never LOOSEN the check.
    feas_tol = 1e-6 if math.isnan(eff_tol) else max(1e-6, 10.0 * eff_tol)

    return SolverSpec(
        label=f"highs-ipm{'+xover' if crossover else ''}",
        cvxpy_solver=cp.HIGHS,
        solver_opts={"highs_options": native},
        tolerance_desc=(
            f"HiGHS interior point, {tol_txt}; {end_desc}; {pre_desc}"),
        hardware="CPU",
        threads=threads,
        internal_time_exposed=True,
        check_feasibility_tol=feas_tol,
        config_full=json.dumps(native),
        algorithm_probe=highs_algorithm_evidence,
    )


def reference_spec(time_limit_s: float = 2 * DEFAULT_TIME_LIMIT_S) -> SolverSpec:
    """Tight-tolerance HiGHS run used as the ground-truth reference."""
    spec = highs_spec(
        time_limit_s=time_limit_s, feasibility_tol=1e-9, solver="simplex",
    )
    spec.label = "reference-highs@1e-9"
    spec.tolerance_desc = "HiGHS simplex, feasibility 1e-9 (reference)"
    return spec


# --------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------

@dataclass
class BenchRecord:
    spec_label: str
    hardware: str
    tolerance_desc: str
    threads: int | None
    n_assets: int
    n_scenarios: int
    nnz_estimate: int
    status: str
    dnf: bool
    n_runs: int
    wall_s_median: float
    wall_s_all: list[float]
    solver_internal_s_median: float   # NaN if not exposed
    compile_s: float                  # CVXPY canonicalization (first solve)
    objective: float
    check: CheckResult | None
    config_full: str = ""
    #: What the solver's own counters say it ran (see
    #: `highs_algorithm_evidence`); empty where the solver exposes nothing.
    algorithm_evidence: str = ""
    notes: str = ""

    def row(self) -> dict[str, Any]:
        c = self.check
        return {
            "solver": self.spec_label,
            "hardware": self.hardware,
            "tolerances": self.tolerance_desc,
            "threads": self.threads if self.threads is not None else "",
            "n_assets": self.n_assets,
            "n_scenarios": self.n_scenarios,
            "nnz_est": self.nnz_estimate,
            "status": self.status,
            "dnf": self.dnf,
            "n_runs": self.n_runs,
            "wall_s_median": round(self.wall_s_median, 4),
            "wall_s_min": round(min(self.wall_s_all), 4) if self.wall_s_all else "",
            "wall_s_max": round(max(self.wall_s_all), 4) if self.wall_s_all else "",
            "solver_internal_s": (
                round(self.solver_internal_s_median, 4)
                if not math.isnan(self.solver_internal_s_median) else ""
            ),
            "overhead_s": (
                round(self.wall_s_median - self.solver_internal_s_median, 4)
                if not (math.isnan(self.wall_s_median)
                        or math.isnan(self.solver_internal_s_median)) else ""
            ),
            "compile_s": round(self.compile_s, 4),
            "objective": self.objective,
            "check_ok": c.ok if c else "",
            "check_obj_gap": f"{c.objective_abs_gap:.2e}" if c else "",
            "check_budget_resid": f"{c.budget_residual:.2e}" if c else "",
            # Signed, so a reader can tell an over-invested book from an
            # under-invested one without rerunning the solve. The magnitude
            # above is what the gate uses; the direction is what the
            # write-up actually wants to talk about.
            "check_budget_signed": f"{c.budget_signed:+.2e}" if c else "",
            "gap_vs_ref_bp": (
                f"{c.gap_vs_reference_bp:+.4f}"
                if c and c.gap_vs_reference_bp is not None else ""
            ),
            "weight_l1_vs_ref": (
                f"{c.weight_l1_vs_reference:.4f}"
                if c and c.weight_l1_vs_reference is not None else ""
            ),
            "dust_positions": c.n_dust_positions if c else "",
            "positions": c.n_positions if c else "",
            # Interior-vs-vertex, at any magnitude: see CheckResult.
            "nonzero_weights": c.n_nonzero_weights if c else "",
            "config": self.config_full,
            "algorithm_evidence": self.algorithm_evidence,
            "notes": self.notes,
        }


def _solve_once(model: CVaRModel, spec: SolverSpec) -> tuple[float, float, str]:
    """One timed solve. Returns (wall_s, internal_s, cvxpy_status)."""
    t0 = time.perf_counter()
    try:
        # warm_start=False is load-bearing: CVXPY defaults it to True and the
        # HiGHS interface caches the previous solution, which would turn every
        # timed re-solve into a warm re-solve for HiGHS only (cuOpt ignores
        # the flag, PDLP has no cache) — silently timing different operations
        # per solver. Caught in review before the first published table.
        model.problem.solve(
            solver=spec.cvxpy_solver, verbose=False, warm_start=False,
            **spec.solver_opts
        )
        status = model.problem.status
    except cp.error.SolverError as e:
        return time.perf_counter() - t0, math.nan, f"solver_error: {e}"
    wall = time.perf_counter() - t0
    internal = math.nan
    stats = model.problem.solver_stats
    if (spec.internal_time_exposed and stats is not None
            and stats.solve_time is not None):
        internal = float(stats.solve_time)
    return wall, internal, status


def _join_notes(*parts: str) -> str:
    return "; ".join(p for p in parts if p)


def _probe_algorithm(model: CVaRModel, spec: SolverSpec) -> str:
    """Run the spec's algorithm probe, never letting it take the run down.

    The probe is provenance, not a result: a benchmark that crashed because it
    could not read an iteration counter would have thrown away the timings it
    already paid for. A failed probe is recorded as failed.
    """
    if spec.algorithm_probe is None:
        return ""
    try:
        return spec.algorithm_probe(model.problem)
    except Exception as e:  # noqa: BLE001
        return f"probe failed ({type(e).__name__}: {e})"


def run_spec(
    model: CVaRModel,
    spec: SolverSpec,
    n_runs: int = 3,
    reference: tuple[np.ndarray, float] | None = None,
) -> BenchRecord:
    """Warmup + n_runs timed solves of `model` with `spec`."""
    # Warmup (discarded): CUDA context, JIT, CVXPY canonicalization.
    warm_wall, _, warm_status = _solve_once(model, spec)
    compile_s = float(getattr(model.problem, "compilation_time", None) or math.nan)

    ok_statuses = {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
    retry_note = ""
    if warm_status not in ok_statuses:
        # One retry before the configuration is written off. The warmup solve
        # is the one that pays for CUDA context creation, JIT and CVXPY
        # canonicalization, so a time limit hit here can be startup cost
        # rather than an inability to converge — and a cell recorded as DNF
        # for that reason is a wrong answer, not a conservative one. The
        # second attempt runs warm; if it fails too the DNF is real. Both the
        # attempt and its outcome go into the notes either way, so a retried
        # cell is never indistinguishable from a first-try one in the CSV.
        first_status, first_wall = warm_status, warm_wall
        warm_wall, _, warm_status = _solve_once(model, spec)
        retry_note = (
            f"warmup retried once after status={first_status} "
            f"(wall={first_wall:.1f}s); retry status={warm_status} "
            f"(wall={warm_wall:.1f}s)"
        )

    if warm_status not in ok_statuses:
        return BenchRecord(
            spec_label=spec.label, hardware=spec.hardware,
            tolerance_desc=spec.tolerance_desc, threads=spec.threads,
            n_assets=model.n_assets, n_scenarios=model.n_scenarios,
            nnz_estimate=model.nnz_estimate, status=warm_status, dnf=True,
            n_runs=0, wall_s_median=math.nan, wall_s_all=[],
            solver_internal_s_median=math.nan, compile_s=compile_s,
            objective=math.nan, check=None,
            algorithm_evidence=_probe_algorithm(model, spec),
            notes=_join_notes(
                f"DNF on warmup (status={warm_status}, wall={warm_wall:.1f}s)",
                retry_note),
        )

    walls: list[float] = []
    internals: list[float] = []
    statuses: list[str] = []
    status = warm_status
    for _ in range(n_runs):
        wall, internal, status = _solve_once(model, spec)
        statuses.append(status)
        if status not in ok_statuses:
            return BenchRecord(
                spec_label=spec.label, hardware=spec.hardware,
                tolerance_desc=spec.tolerance_desc, threads=spec.threads,
                n_assets=model.n_assets, n_scenarios=model.n_scenarios,
                nnz_estimate=model.nnz_estimate, status=status, dnf=True,
                n_runs=len(walls), wall_s_median=math.nan, wall_s_all=walls,
                solver_internal_s_median=math.nan, compile_s=compile_s,
                objective=math.nan, check=None,
                algorithm_evidence=_probe_algorithm(model, spec),
                notes=_join_notes(f"DNF on timed run (status={status})",
                                  retry_note),
            )
        walls.append(wall)
        if not math.isnan(internal):
            internals.append(internal)

    evidence = _probe_algorithm(model, spec)
    weights = model.weights()
    objective = float(model.problem.value)
    ref_w, ref_obj = (reference if reference is not None else (None, None))
    chk = check_solution(
        w=weights, solver_objective=objective, R=model.R, mu=model.mu,
        alpha=model.alpha, risk_aversion=model.risk_aversion,
        w_max=model.w_max, reference_w=ref_w, reference_objective=ref_obj,
        feasibility_tol=spec.check_feasibility_tol,
        objective_tol=spec.check_objective_tol,
    )
    notes = "" if chk.ok else f"CHECK FAILED: {chk.notes}"
    n_inacc = sum(1 for st in statuses if st == cp.OPTIMAL_INACCURATE)
    if n_inacc:
        notes = (notes + "; " if notes else "") + \
            f"solver flagged inaccurate on {n_inacc}/{len(statuses)} runs"
    notes = _join_notes(notes, retry_note)

    return BenchRecord(
        spec_label=spec.label, hardware=spec.hardware,
        tolerance_desc=spec.tolerance_desc, threads=spec.threads,
        n_assets=model.n_assets, n_scenarios=model.n_scenarios,
        nnz_estimate=model.nnz_estimate, status=status, dnf=False,
        n_runs=len(walls), wall_s_median=statistics.median(walls),
        wall_s_all=walls,
        solver_internal_s_median=(
            statistics.median(internals) if internals else math.nan
        ),
        compile_s=compile_s, objective=objective, check=chk,
        config_full=spec.config_full, algorithm_evidence=evidence,
        notes=notes,
    )


def solve_reference(
    build: Callable[[], CVaRModel], spec: SolverSpec | None = None
) -> tuple[np.ndarray, float]:
    """Solve a fresh copy of the model at reference accuracy; return (w, obj).

    Uses a fresh model instance so the reference solve never pollutes the
    benchmarked problem object's CVXPY solver cache.
    """
    spec = spec or reference_spec()
    m = build()
    m.problem.solve(solver=spec.cvxpy_solver, verbose=False, **spec.solver_opts)
    if m.problem.status != cp.OPTIMAL:
        raise RuntimeError(f"reference solve not optimal: {m.problem.status}")
    return m.weights(), float(m.problem.value)


def shifted_geometric_mean(values: list[float], shift: float = 1.0) -> float:
    """Mittelmann-convention aggregate for cross-instance timings."""
    vals = [v + shift for v in values]
    return math.exp(sum(math.log(v) for v in vals) / len(vals)) - shift


def write_csv(path: str, rows: list[dict[str, Any]],
              env: dict[str, str]) -> None:
    fieldnames = list(rows[0].keys())
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        for k, v in env.items():
            f.write(f"# {k}: {v}\n")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------
# Spec factories (used by the subprocess worker)
#
# Each benchmarked solver runs in its OWN subprocess: highspy 1.15 and
# OR-Tools' bundled HiGHS export clashing C++ symbols, and loading both into
# one process is an ODR violation waiting to corrupt results. Process
# isolation also keeps CUDA context, allocator state, and OOMs contained —
# each timing starts from a clean slate.
# --------------------------------------------------------------------------

SPEC_FACTORIES: dict[str, Callable[..., SolverSpec]] = {
    "cuopt_pdlp": cuopt_pdlp_spec,
    "ortools_pdlp": ortools_pdlp_spec,
    "highs": highs_spec,
    "highs_ipm": highs_ipm_spec,
    "reference": reference_spec,
}

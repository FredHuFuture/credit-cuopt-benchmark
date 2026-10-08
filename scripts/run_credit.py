#!/usr/bin/env python
"""End-to-end credit Mean-CVaR run (M2).

universe -> scenarios (GPU and CPU, timed) -> credit LP -> solve on cuOpt,
OR-Tools PDLP and HiGHS -> independent checks -> OUT-OF-SAMPLE economics.

Two disciplines this script exists to enforce:

1. **Risk numbers are reported out of sample.** The optimizer sees scenario
   set A; every CVaR figure below is recomputed on an independent set B. An
   in-sample CVaR improvement is a fit statistic, not a risk reduction, and
   CVaR95 over 20,000 scenarios is decided by the 1,000 worst — exactly the
   ones the optimizer is fitting.
2. **Limits are checked in relative units, against two named gates.** A
   solver that returns a point inside its residual tolerance has not
   necessarily returned a portfolio inside the mandate — and "inside the
   mandate" itself has two defensible readings. Every configuration is scored
   against both: the repo's 1e-6 relative purity gate, and a desk-materiality
   gate at 1bp of portfolio weight (with each non-weight family scored at its
   own reporting granularity). Both verdicts go into the CSV. The economics
   are still reported from a purity-clean configuration whenever one exists,
   so this addition cannot loosen a published number by itself.

3. **The attribution baseline is a range, not a portfolio.** The min-DTS
   comparison portfolio is one vertex of a high-dimensional optimal face —
   see `min_risk_face` — so the run emits the max-return and min-return
   vertices of that face alongside it and quotes the attribution as a band.

Example:
    python scripts/run_credit.py --n-bonds 500 --n-scenarios 20000
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# NOTE: OR-Tools PDLP is deliberately absent from this script. highspy and
# OR-Tools bundle different HiGHS builds whose C++ symbols collide, and the
# clash is mutual: import cvxpy (which loads highspy) first and libortools
# fails on `setLocalOptionValue`; import OR-Tools first and highspy fails on
# `Highs::releaseMemory`. Exactly one of them can live in a process. The
# three-way solver comparison therefore belongs in the process-isolated
# harness (src/bench.py + scripts/bench_worker.py); this script exists to
# show the ECONOMICS and whether desk limits survive, using cuOpt and HiGHS.
from src.bench import (  # noqa: E402
    assert_gpu_idle, capture_environment, write_csv,
)
from src.check import check_solution, empirical_cvar  # noqa: E402
from src.credit import (  # noqa: E402
    DESK_WEIGHT_GATE_BP, NOTIONAL_USD, PURITY_REL_GATE,
    credit_constraint_builder, default_limits, exposure_report,
    gate_verdicts, min_risk_face, portfolio_stats, worst_violation,
)
from src.model import build_mean_cvar  # noqa: E402
from src.scenarios import credit_scenarios, warmup_gpu  # noqa: E402
from src.universe import make_universe, summarize  # noqa: E402

#: Solver configurations compared. Running cuOpt at two tolerances and with
#: crossover is not about speed — it is about whether the desk limits survive.
SOLVE_CONFIGS = (
    ("cuopt@1e-4", "cuopt", dict(tol=1e-4, crossover=False)),
    ("cuopt@1e-6", "cuopt", dict(tol=1e-6, crossover=False)),
    ("cuopt@1e-4+xover", "cuopt", dict(tol=1e-4, crossover=True)),
    ("highs-simplex", "highs", {}),
)


def solve_with(model, solver: str, tol: float = 1e-4, crossover: bool = False,
               threads: int = 16, runs: int = 3):
    """Warmup solve, then `runs` timed solves; return (median_s, status, obj).

    Median of repeats, not a single shot: observed run-to-run spread on these
    solves exceeded 40% (one crossover configuration measured 22.0s and 38.8s
    on consecutive runs), which is larger than the differences being compared.
    Same discipline as src/bench.py.
    """
    import cvxpy as cp

    if solver == "cuopt":
        opts = dict(
            solver_method="PDLP", pdlp_solver_mode="Stable2",
            absolute_primal_tolerance=tol, relative_primal_tolerance=tol,
            absolute_dual_tolerance=tol, relative_dual_tolerance=tol,
            absolute_gap_tolerance=tol, relative_gap_tolerance=tol,
            time_limit=300.0, crossover=1 if crossover else 0,
        )
        cvxpy_solver = cp.CUOPT
    elif solver == "highs":
        opts = {"highs_options": {
            "solver": "simplex", "time_limit": 300.0, "threads": threads,
            "parallel": "on", "random_seed": 0}}
        cvxpy_solver = cp.HIGHS
    else:
        raise ValueError(solver)

    # warm_start=False: CVXPY defaults it True and HiGHS caches solutions,
    # which would time warm re-solves for one solver only (see src/bench.py).
    model.problem.solve(solver=cvxpy_solver, verbose=False, warm_start=False,
                        **opts)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model.problem.solve(solver=cvxpy_solver, verbose=False,
                            warm_start=False, **opts)
        times.append(time.perf_counter() - t0)
    return statistics.median(times), model.problem.status, \
        float(model.problem.value), times


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-bonds", type=int, default=500)
    ap.add_argument("--n-scenarios", type=int, default=20_000)
    ap.add_argument("--alpha", type=float, default=0.95)
    ap.add_argument("--risk-aversion", type=float, default=3.0)
    ap.add_argument("--turnover", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--scen-seed", type=int, default=11)
    ap.add_argument("--oos-seed", type=int, default=101)
    ap.add_argument("--return-mode", choices=["excess", "total"],
                    default="excess")
    ap.add_argument("--runs", type=int, default=3,
                    help="timed solves per configuration (median reported)")
    ap.add_argument("--skip-cpu-scenarios", action="store_true")
    ap.add_argument("--face-cvar-lp", action="store_true",
                    help="also solve for the MINIMUM out-of-sample CVaR over "
                         "the min-DTS optimal face (one extra LP of full "
                         "scenario size; without it the face's CVaR range is "
                         "only sampled at the recovered vertices)")
    ap.add_argument("--allow-busy-gpu", action="store_true",
                    help="proceed even if another workload is using the GPU "
                         "(results will not be publishable)")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    # Check the card is free BEFORE spending the time, not after reading the
    # numbers: a competing workload makes every configuration slower by an
    # amount stable enough to look like a measurement. Same guard as
    # scripts/run_sweep.py.
    assert_gpu_idle(raise_on_busy=not args.allow_busy_gpu)

    pd.set_option("display.width", 130)
    pd.set_option("display.max_columns", 22)

    # ---- 1. universe --------------------------------------------------
    print(f"[1] Universe: {args.n_bonds} bonds (seed {args.seed})")
    uni = make_universe(n_bonds=args.n_bonds, seed=args.seed)
    b = uni.bonds
    print(f"    {uni.n_bonds} bonds / {b['issuer_id'].nunique()} issuers / "
          f"{b['sector'].nunique()} sectors")
    print(f"    benchmark: yield {uni.w_benchmark @ uni.ytm:.3%}, "
          f"OAS {uni.w_benchmark @ uni.oas_bp:.0f}bp, "
          f"duration {uni.w_benchmark @ uni.duration:.2f}y, "
          f"DTS {uni.w_benchmark @ uni.dts:,.0f}bp-y")
    d0 = uni.one_way_distance_to_benchmark()
    print(f"    incumbent w0 holds {(uni.w0 > 0).sum()} bonds; "
          f"{d0:.1%} one-way from the index "
          f"({'reachable' if d0 <= args.turnover else 'NOT reachable'} "
          f"within a {args.turnover:.0%} budget)")
    print("\n    Benchmark composition by rating:")
    print(summarize(uni).to_string(float_format=lambda v: f"{v:,.3f}"))

    # ---- 2. scenarios --------------------------------------------------
    print(f"\n[2] Scenarios: {args.n_scenarios:,} monthly paths "
          f"({args.return_mode} returns)")
    warmup_gpu(uni)   # real code path: context + this module's kernels
    R, mu, info = credit_scenarios(uni, n_scen=args.n_scenarios,
                                   seed=args.scen_seed, use_gpu=True,
                                   return_mode=args.return_mode)
    print(f"    {info['device']}: {info['seconds']:.3f}s  "
          f"({info['dtype']} sampling, incl. device->host transfer)")
    if not args.skip_cpu_scenarios:
        _, _, info_cpu = credit_scenarios(
            uni, n_scen=args.n_scenarios, seed=args.scen_seed, use_gpu=False,
            return_mode=args.return_mode)
        speedup = info_cpu["seconds"] / max(info["seconds"], 1e-9)
        tag = ("GPU" if info["device"].startswith("GPU")
               else "same-device (no GPU present)")
        print(f"    {info_cpu['device']}: {info_cpu['seconds']:.3f}s "
              f"-> {tag} {speedup:.1f}x")
    print(f"    default rate/bond {info['realized_default_rate_per_bond']:.4%}"
          f" vs target {info['target_default_rate_per_bond']:.4%} "
          f"(threshold calibrated on an independent draw); "
          f"{info['scenarios_with_any_default']:.1%} of scenarios "
          f"contain >=1 default")
    print(f"    mu vs realized scenario mean: max gap "
          f"{info['mu_vs_sample_mean_max_bp']:.2f}bp/month")

    # Independent evaluation set — every risk number below comes from this.
    R_oos, _, _ = credit_scenarios(uni, n_scen=args.n_scenarios,
                                   seed=args.oos_seed, use_gpu=True,
                                   return_mode=args.return_mode)
    print(f"    out-of-sample set generated (seed {args.oos_seed})")

    # ---- 3. credit LP --------------------------------------------------
    limits = default_limits(uni, turnover=args.turnover)
    print(f"\n[3] Limits: {limits.describe()}")
    model = build_mean_cvar(
        R, mu, alpha=args.alpha, risk_aversion=args.risk_aversion,
        w_max=limits.w_max,
        constraint_builder=credit_constraint_builder(uni, limits),
    )
    print(f"    LP: ~{model.n_assets + model.n_scenarios + 1:,} variables, "
          f"scenario block {model.nnz_estimate:,} nnz")
    print(f"    families: {', '.join(model.extra_constraint_names)}")

    # ---- 4. solve, and check whether the LIMITS actually hold ----------
    print("\n[4] Solve — speed AND whether desk limits survive")
    # `solve_rows`, not `rows`: these are the per-configuration timings and
    # they have to survive until the write step at the end, where `rows` is
    # reused for the economics table.
    results, solve_rows = {}, []
    for label, solver, kw in SOLVE_CONFIGS:
        try:
            secs, status, obj, all_t = solve_with(model, solver,
                                                  runs=args.runs, **kw)
            w = model.weights()
            chk = check_solution(
                w=w, solver_objective=obj, R=R, mu=mu, alpha=args.alpha,
                risk_aversion=args.risk_aversion, w_max=limits.w_max,
            )
            rep = exposure_report(uni, w, limits)
            viol, which = worst_violation(rep)
            gates = gate_verdicts(rep)
            results[label] = dict(secs=secs, status=status, obj=obj,
                                  w=w.copy(), check=chk, report=rep,
                                  violation=viol, gates=gates)
            solve_rows.append({
                "config": label, "status": status,
                "wall_s_median": secs, "wall_s_min": min(all_t),
                "wall_s_max": max(all_t),
                "objective": obj, "budget_resid": chk.budget_residual,
                "budget_signed": chk.budget_signed,
                "worst_rel_violation": viol,
                "worst_limit": which if viol > 0 else "-",
                # Both mandate verdicts, side by side, as columns. Neither is
                # derived from the other and neither is the headline on its
                # own; see the module docstring.
                **gates.as_row(),
            })
        except Exception as e:  # noqa: BLE001
            print(f"    {label:<18} FAILED: {type(e).__name__}: {e}")

    print(pd.DataFrame(solve_rows).set_index("config").to_string(
        float_format=lambda v: f"{v:,.6g}"))

    ref = "highs-simplex"
    if ref in results:
        print(f"\n    vs {ref} (exact vertex solution):")
        for label in results:
            if label == ref:
                continue
            d = np.abs(results[label]["w"] - results[ref]["w"]).sum()
            do = abs(results[label]["obj"] - results[ref]["obj"])
            print(f"      {label:<20} weight L1 {d:.4f}   "
                  f"objective diff {do:.2e}")

    # The two gates, side by side. They disagree on real solves — that
    # disagreement is a result, so it is printed rather than resolved.
    clean = [lbl for lbl in results
             if results[lbl]["gates"].purity_pass]
    desk_clean = [lbl for lbl in results if results[lbl]["gates"].desk_pass]
    print(f"\n    Mandate verdicts (purity gate {PURITY_REL_GATE:g} relative "
          f"vs desk gate {DESK_WEIGHT_GATE_BP:g}bp of weight on a "
          f"${NOTIONAL_USD / 1e9:.0f}bn book):")
    for lbl in results:
        g = results[lbl]["gates"]
        extra = ("" if np.isnan(g.desk_worst_usd)
                 else f"  (${g.desk_worst_usd:,.0f} on the worst row)")
        print(f"      {lbl:<20} {g.verdict:<8} purity "
              f"{g.purity_worst_rel:.2e} / desk {g.desk_worst_multiple:.2e}x"
              f"{extra}")
    only_desk = [lbl for lbl in desk_clean if lbl not in clean]
    if only_desk:
        print(f"      {len(only_desk)} configuration(s) fail the 1e-6 gate "
              f"and pass the desk gate: {', '.join(only_desk)}")

    # Fastest config whose limits actually hold, measured — not assumed from
    # a hardcoded tolerance ordering. The purity gate stays the default
    # selection rule: adding a looser gate must not silently relabel a
    # published portfolio as compliant.
    gate_used = "purity"
    if clean:
        pool = clean
    elif desk_clean:
        # Falling back is better than exiting with nothing measured, but it
        # changes what the economics mean, so it is recorded in the CSV
        # header and printed in capitals rather than mentioned in passing.
        pool, gate_used = desk_clean, "desk-only"
        print("\n    NO configuration passed the 1e-6 purity gate. Economics "
              "below come from a DESK-CLEAN-ONLY configuration; every number "
              "in section 5 inherits that caveat.")
    else:
        print("\nNo configuration respected every limit under either gate — "
              "stopping.")
        return 1
    chosen = min(pool, key=lambda lbl: results[lbl]["secs"])
    print(f"\n    Economics reported from: {chosen} "
          f"({results[chosen]['secs']:.2f}s — fastest config whose limits "
          f"hold under the {gate_used} gate; configs are ranked by measured "
          f"time, not tolerance)")

    # ---- 5. economics, OUT OF SAMPLE -----------------------------------
    w = results[chosen]["w"]
    print(f"\n[5] Does it make credit sense? "
          f"(CVaR/VaR on the independent set, seed {args.oos_seed})")
    # Scenario-free attribution baseline: how much of the tail reduction is
    # simply "the mandate permitted de-risking"? The baseline is a FACE, not
    # a portfolio — the limit set pins DTS at the floor the objective is
    # minimising — so the extremes are recovered and the attribution is
    # quoted as a band. `baseline_vertex` is exactly what `min_risk_baseline`
    # returns, so the historical row is unchanged and now has neighbours.
    face = min_risk_face(uni, limits, mu, R_eval=R_oos, alpha=args.alpha,
                         minimize_cvar_over_face=args.face_cvar_lp)
    print(f"    min-DTS optimal face: DTS pinned at {face.dts_target:,.4f} "
          f"bp-y (+/-{face.dts_band:.1e}), floor "
          f"{'binds' if face.floor_binds else 'does NOT bind'}; "
          f"{len(face.vertices)} vertices, max pairwise weight L1 "
          f"{face.max_vertex_l1:.4f}")
    if face.looks_like_a_point:
        print("      every recovered vertex coincided: the face may be a "
              "single point, so the band below is degenerate")
    face_rows = [dict(r) for r in face.as_rows()]
    w_minrisk = face.vertex("baseline_vertex").w

    face_labels = {"max_return": "min-DTS face: max return",
                   "min_return": "min-DTS face: min return",
                   "min_cvar": "min-DTS face: min CVaR_oos"}
    extra_portfolios = [(face_labels[v.label], v.w) for v in face.vertices
                        if v.label in face_labels]

    rows = []
    for label, ww in [("benchmark", uni.w_benchmark), ("incumbent w0", uni.w0),
                      ("min-DTS (no scenarios)", w_minrisk),
                      *extra_portfolios,
                      ("CVaR-optimal", w)]:
        c_is, _ = empirical_cvar(-(R @ ww), args.alpha)
        c_oos, v_oos = empirical_cvar(-(R_oos @ ww), args.alpha)
        st = portfolio_stats(uni, ww)
        rows.append({
            "portfolio": label, "exp.excess": float(mu @ ww),
            "CVaR_in": c_is, "CVaR_oos": c_oos, "VaR_oos": v_oos,
            "OAS(bp)": st["oas_bp"], "duration": st["duration"],
            "DTS": st["dts_bp_years"], "positions": st["n_positions"],
            "issuers": st["n_issuers"],
        })
    comp = pd.DataFrame(rows).set_index("portfolio")
    fmt = {"exp.excess": "{:.5f}".format}
    print(comp.to_string(formatters=fmt,
                         float_format=lambda v: f"{v:,.4f}"))

    base = comp.loc["benchmark"]
    opt = comp.loc["CVaR-optimal"]
    inc = comp.loc["incumbent w0"]
    mrb = comp.loc["min-DTS (no scenarios)"]
    tot = 1 - opt["CVaR_oos"] / base["CVaR_oos"]
    corner = 1 - mrb["CVaR_oos"] / base["CVaR_oos"]
    print(f"\n    Out-of-sample CVaR{args.alpha:.0%} vs benchmark: "
          f"{tot:+.1%}  (vs the incumbent book, where the trade actually "
          f"starts: {1 - opt['CVaR_oos'] / inc['CVaR_oos']:+.1%})")
    print(f"    In-sample would read {1 - opt['CVaR_in'] / base['CVaR_in']:+.1%}"
          f" — the gap is the overfit.")
    print(f"    ATTRIBUTION: the committed min-DTS vertex under the same "
          f"limits already gets {corner:+.1%}, i.e. {corner / tot:.0%} of the "
          f"reduction;")
    # ...but that vertex is one point of a face. The band is what the paper
    # quotes; the point above is kept only so the historical number stays
    # locatable inside it. Vertices whose search direction was computed on
    # the evaluation set are excluded here — see MinRiskFace.cvar_band.
    ex_lo, ex_hi = face.cvar_band(include_eval_selected=False)
    corner_lo = 1 - ex_hi / base["CVaR_oos"]
    corner_hi = 1 - ex_lo / base["CVaR_oos"]
    print(f"      ACROSS THE FACE that vertex was drawn from, the same "
          f"scenario-free rule gets {corner_lo:+.1%} to {corner_hi:+.1%} "
          f"({corner_lo / tot:.0%}-{corner_hi / tot:.0%} of the reduction) "
          f"at identical DTS. Neither end is an extremum of CVaR: these are "
          f"the vertices the return directions happened to land on, so the "
          f"true band is at least this wide.")
    print(f"      Expected excess over the face: "
          f"{face.exp_return_lo:.5f} to {face.exp_return_hi:.5f} per month "
          f"(EXACT range: both endpoints are LPs). The committed vertex "
          f"reads {mrb['exp.excess']:.5f}.")
    if face.cvar_lo_is_exact:
        hind_lo, _ = face.cvar_band(include_eval_selected=True)
        print(f"      HINDSIGHT BOUND: the face also contains a portfolio at "
              f"{1 - hind_lo / base['CVaR_oos']:+.1%}, found by minimising "
              f"CVaR over the face ON THE EVALUATION SET. That is a fact "
              f"about the face, not an achievable rule — the direction saw "
              f"the draw it is scored on.")
    else:
        print("      No min-CVaR-over-face solve was run (--face-cvar-lp); "
              "the de-risking end of the face is unmeasured.")
    print("      Maximising CVaR over a polytope is not an LP, so the worst "
          "tail on the face is not computed at all and the band can only "
          "widen from either direction.")
    print(f"    Expected excess given up vs benchmark: "
          f"{1 - opt['exp.excess'] / base['exp.excess']:.1%} "
          f"({base['exp.excess']:.5f} -> {opt['exp.excess']:.5f} per month)")

    print("\n    Binding constraints:")
    rep = results[chosen]["report"]
    binding = rep[rep["binding"]]
    cols = ["family", "item", "portfolio", "benchmark", "limit", "rel_slack"]
    print(binding[cols].to_string(index=False,
                                  float_format=lambda v: f"{v:,.4f}")
          if len(binding) else "    (none binding)")

    print("\n    Rating exposure, optimal vs benchmark:")
    tilt = pd.DataFrame({
        "optimal": uni.bonds.assign(x=w).groupby(
            "rating", observed=True)["x"].sum(),
        "benchmark": uni.bonds.assign(x=uni.w_benchmark).groupby(
            "rating", observed=True)["x"].sum()})
    tilt["active"] = tilt["optimal"] - tilt["benchmark"]
    print(tilt.to_string(float_format=lambda v: f"{v:,.4f}"))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        env = capture_environment()
        env["script"] = "scripts/run_credit.py (M2 end-to-end credit run)"
        env["n_bonds"] = str(args.n_bonds)
        env["n_scenarios"] = str(args.n_scenarios)
        env["alpha"] = str(args.alpha)
        env["risk_aversion"] = str(args.risk_aversion)
        env["turnover"] = str(args.turnover)
        env["return_mode"] = args.return_mode
        env["universe_seed"] = str(args.seed)
        env["scenario_seed"] = str(args.scen_seed)
        env["oos_seed"] = str(args.oos_seed)
        env["runs_per_config"] = str(args.runs)
        # ASCII-folded: the CSV header is written with the platform's default
        # encoding and describe() carries a plus-minus and an element-of sign.
        env["limits"] = (limits.describe()
                         .replace("±", "+/-").replace("∈", " in "))
        env["economics_reported_from"] = chosen
        env["economics_gate"] = gate_used
        env["purity_gate_rel"] = f"{PURITY_REL_GATE:g}"
        env["desk_gate_bp_weight"] = f"{DESK_WEIGHT_GATE_BP:g}"
        env["desk_gate_notional_usd"] = f"{NOTIONAL_USD:.0f}"
        env["face_cvar_lp"] = str(args.face_cvar_lp)
        env["scenario_gen_s"] = f"{info['seconds']:.3f}"
        env["scenario_device"] = info["device"]
        # All three frames go through the same environment-header writer the
        # sweep uses. A bare to_csv dump is a table with no record of which
        # GPU, driver, package versions or run parameters produced it, which
        # makes it unciteable a month later.
        # Derived from --out via the path parts, not a substring replace on
        # ".csv": with three files sharing a stem now, an --out that does not
        # end in .csv would have had them all write to the same name and the
        # last one would silently win.
        _p = Path(args.out)
        _sfx = _p.suffix or ".csv"
        limits_path = str(_p.with_name(f"{_p.stem}_limits{_sfx}"))
        solves_path = str(_p.with_name(f"{_p.stem}_solves{_sfx}"))
        face_path = str(_p.with_name(f"{_p.stem}_face{_sfx}"))
        write_csv(args.out, comp.reset_index().to_dict("records"), env)
        write_csv(limits_path, rep.to_dict("records"), env)
        written = [args.out, limits_path]
        # The timings were measured and printed but never committed until now;
        # the speed claims this script makes had no artifact behind them.
        if solve_rows:
            write_csv(solves_path, solve_rows, env)
            written.append(solves_path)
        # The face, as its own artifact: one row per recovered vertex, each
        # carrying the band summary, so the attribution range in the paper
        # cites a file rather than a paragraph.
        band = face.as_band_row()
        write_csv(face_path, [{**r, **band} for r in face_rows], env)
        written.append(face_path)
        print("\nWrote " + ", ".join(written))

    g = results[chosen]["gates"]
    # The exit code still answers the strict question. A desk-clean-only run
    # is not a pass, it is a documented near-miss, and the line says which.
    ok = results[chosen]["check"].ok and g.purity_pass
    device = "GPU" if chosen.startswith("cuopt") else "CPU"
    print(f"\n{'PASS' if ok else 'FAIL'}: credit CVaR LP solved on {device} "
          f"({chosen}), independently verified; mandate verdict "
          f"{g.verdict} (purity {g.purity_worst_rel:.2e}, "
          f"desk {g.desk_worst_multiple:.2e}x its tolerance).")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

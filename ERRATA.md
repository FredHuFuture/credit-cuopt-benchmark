# Errata — the corrections ledger

## Release corrections, 2026-10-07

- The former 7.67x reference SOCP timing ratio is withdrawn. Its source
  `p0_socp_reference.csv` flags `gpu_contended: yes — timings unreliable`.
  Feasibility evidence remains valid; the generator now emits a recorded
  timing exclusion rather than that number. No historical CSV is edited.
- The endpoint ladder's tight PDLP row passes the **MAD-form** mandate.
  Its 28.55bp sigma TE still fails the original 10bp target. Abstract,
  endpoint discussion and conclusion now make that distinction explicit.
- QP comparisons are described as matched-feasibility solver-stack results,
  not equal certified optimality accuracy or a pure hardware experiment.
- The model now states objective, units, annualization, seeds, gate thresholds,
  hardware, software and the synthetic provenance of headline experiments.
- README's nonexistent `environment.yml` is replaced by a tested CPU-only
  dependency path plus the archived full GPU lockfile. The QP producer command
  now passes the numeric grid and strict solver arms accepted by the runner.
- The endpoint table now distinguishes steady-state medians (plus separately
  timed repair) from timeout elapsed time; these are not end-to-end latencies.
  The archived simplex arm's repeated capped attempts are disclosed.
- Tail-sweep claims now identify the 120-bond, 4,000-scenario experiment and
  5bp budget. The 2.81x value is a minimum cell median, not a bound on draws.
  The figure distinguishes the Gaussian population identity from finite-sample
  selection effects.
- The synthetic model is stylized, not fitted. Its default probabilities,
  recovery, tail parameters, benchmark/incumbent construction and maturity-cell
  contribution constraints are now explicit in the appendix.
- A repaired QP point's objective is independently recomputed, but its zero
  discrepancy field is a self-comparison, not original-QP solver agreement.
  The published six raw/raw ratio pairs are unaffected by this clarification.

The following entries are historical disclosures, not scientific contributions
or current headline results.

Every number in this file is **historical**: it appeared in an earlier draft
of the README, has been superseded by a committed measurement, and survives
here so the correction is auditable instead of silent. The README links here
once; nothing in this file should be quoted as a current result.

The historical audit found that **four tables in the earlier README had no committed
CSV behind them, all four failed reproduction in the 2026-08-24 GPU session,
and every one had been wrong in the favourable direction.** Every number that
had a committed CSV behind it matched that audit's recomputation. The later
timing exclusion above illustrates why provenance alone is not enough.

Fresh numbers cited below live in the committed CSVs under `results/`
(environment headers included). Where a change has two causes — the
scenario-draw fix and genuine measurement error — that is said.

---

## 1. The M2 credit solve table (no CSV behind the old numbers)

| config | old draft | fresh (`m2_credit_500x20k_solves.csv`) |
|---|---:|---:|
| cuOpt PDLP @1e-4 | 2.45s, "9.8% over the turnover budget — trades 27.4% against a 25% limit" | 2.31s, turnover 4.2% over its limit, budget signed −1.0e-4 |
| cuOpt PDLP @1e-6 | 26.5s, breach 2.3e-6 | 35.9s, breach 1.3e-6 (3–7y cell), desk-materiality PASS |
| cuOpt @1e-4 + crossover | 39.0s, "44% slower than HiGHS" | **21.1s — faster than HiGHS** |
| HiGHS dual simplex | 27.2s | 22.2s |

The consequence is not cosmetic: the old table supported the draft's flagship
negative line — "the honest headline portfolio of a GPU-optimization project
was produced by the CPU." The fresh table inverts it at this size. The
fastest-compliant rule (unchanged) now selects **cuOpt+crossover**, and the
committed economics are reported from it (`economics_reported_from` in the
CSV header). The old crossover number was 1.85x too slow and the old HiGHS
number 1.23x too slow; both errors favoured the story being told.

## 2. Scenario generation (no CSV behind the old numbers)

Old draft: 5.2x / 8.1x / 8.6x / 7.5x at the four sizes, summarized as a
"5–9x" GPU win. Fresh (`rng_agreement.csv`): **1.36x / 3.03x / 5.66x /
4.85x** — a curve rising from near-parity as launch overhead amortizes, not a
flat multiple. Every old cell was high, by up to 3.8x. The distributional
agreement gate passes in both eras (CVaR within 0.2–1.94 SE).

## 3. The warm-start table (no CSV behind the old numbers) — inverted

Old draft: cold+presolve **10.0s** (the fast path), cold-no-presolve 91.7s,
warm-no-presolve 71.0s; conclusion "presolve alone is worth 9.2x, warm start
loses by 7x", and a declaration that warm-started PDLP for sequential
re-optimization "does not pay".

Fresh (`warmstart_tradeoff.csv`): cold+presolve 3.12s, cold-no-presolve
1.82s, **warm-no-presolve 1.72s** — presolve is a net 1.7x *cost*, warm start
is the fastest mode *and* lands an order of magnitude closer to feasible
(5.8e-4 vs 5.7e-3). Every directional claim in the old table was wrong, not
just the digits. The 9.2x presolve figure also propagated into two other
sections of the old draft (the roadmap and the presolve-pinning rationale) and
is retired everywhere.

## 4. The demo-latency table (no CSV behind the old numbers)

Old draft: hand-rolled 30.8 / 12.4 / 3.0s at 4k/10k/20k scenarios (CVXPY
22.1 / 2.5 / 6.5), summarized as "seconds to tens of seconds, 2.5–31s, and
not monotone in problem size", with a paragraph explaining the non-monotonicity
as conditioning. Fresh (`demo_latency.csv`): hand-rolled **0.47 / 1.45 /
3.02s** (CVXPY 0.76 / 2.69 / 7.10) — strictly monotone, and the explanation
explained an artifact. The old draft also disclosed a 3.3x self-contradiction
(this table's 3.0s vs the warm-start table's 10.0s on the same
configuration); measured back-to-back in one session the two scripts read
3.02s and 3.12s — the contradiction was cross-session noise, likely GPU
contention. One old observation survived intact: at 4,000 scenarios the
hand-rolled solve leaves a ~37% relative mandate violation.

## 5. The budget-direction error ("over-invested")

The old draft printed `sum(w)` = 1.0018 and 1.0054 on the failing sweep rows,
called them **over**-invested, and translated the second into "roughly $5.4mm
of portfolio that is not funded". The committed column (`check_budget_resid`)
carried `abs(sum(w) − 1)` — magnitude only; the direction was asserted, not
recorded. A later audit pass flagged this and added a signed column. The
fresh run answers it: the residuals are **negative** — −1.77e-3 / −3.85e-3 /
−2.32e-3 — the loose points are **under**-invested ($1.8–3.9mm uninvested on
$1bn). The magnitude was roughly right; the direction, the only part with an
economic story attached, was wrong.

## 6. Headline speedups and cell numbers

All old sweep numbers predate the scenario-draw fix (the default-trigger
calibration cache omitted seed and scenario count, so committed CSVs were
built on a warmup-warmed cache) and were additionally summarized wrong:

- "A GPU gets you an approximate credit portfolio **roughly 20x** faster",
  "**18–23x**", "**20–30x**": fresh raw-point speedups vs the best exact CPU
  option are **8.1x / 17.4x / 17.4x** at 5M/20M/20M nonzeros
  (`m3_sweep.csv`).
- "At 5M, where the cuOpt row is budget-clean, it is **6.9x** faster than
  HiGHS": the fresh 5M raw row is 8.1x faster and **no longer budget-clean**
  — all three raw rows now fail the checker.
- Old cells 1.02s / 4.06s / 3.17s (raw), 25.4s / DNF / 175.5s (crossover),
  35.7s / 93.9s / 210.5s (OR-Tools), 7.07s / 104.4s / 58.5s (HiGHS) → fresh
  0.80 / 4.11 / 3.17, 24.5 / 78.9 / 175.8, 47.9 / 73.3 / 208.3,
  6.44 / 71.4 / 55.1.
- **The crossover DNF at 1000×20,000 is gone**: the old draft reported
  `status=user_limit` on the warmup at 71.5s and no timed run; the fresh run
  converged in all five replicates at 78.9s median. The DNF was real and its
  cause was never identified; it did not reproduce.
- "0.28x / 0.33x — the GPU path takes 3–3.6x longer" (crossover vs HiGHS):
  fresh ratios are **3.81x / 1.10x / 3.19x** — and at the 500-bond working
  size the GPU path is *faster* (item 1).
- The old scaling-trend paragraph ("widens where bonds grow, shrinks where
  scenarios grow") was built on the old cells and is retired with them; three
  cells still cannot support a scaling law in either direction.

## 7. The attribution split (65/35) — not identified

The old draft attributed "roughly two thirds of the headline" CVaR reduction
to mandate-corner de-risking and one third to scenario-driven selection
(65/35), from a min-DTS baseline whose −9.9% was one arbitrary vertex of a
flat optimal face — a flaw the old draft itself disclosed in a warning box.
The face is now measured (`m2_credit_500x20k_face.csv`): the scenario-free
family spans −6.2% to +10.9% CVaR reduction ex ante, so the "corner share"
spans **−41% to +72%** of the optimizer's total, and a hindsight-selected
vertex of the face (−17.3%) beats the optimizer (−15.1%) outright. The split
is not identified; no single attribution number is quoted anywhere anymore.
A humbling coincidence, preserved deliberately: the fresh run's arbitrary
vertex again lands at almost exactly 65% of the optimizer's reduction — the
same unidentified number, reproduced twice, looking exactly as convincing the
second time.

The old draft's companion comparison ("the CVaR optimum carries a 5.5%
expected-return edge over the scenario-free book at identical DTS and
duration, and a 6.0% lower out-of-sample tail") dies with it: both margins
were properties of the arbitrary vertex, and the measured face's
expected-return range (0.000814–0.001421 monthly) contains the CVaR optimum
with room on both sides.

## 8. The M2 economics moved with the scenario draw

Old headline: out-of-sample CVaR₉₅ **2.05%**, −15.3% vs benchmark, −11.5% vs
incumbent, 270 positions. Fresh: 2.07%, −15.1%, −11.3%, 264 positions. This
is the predicted consequence of the scenario-cache fix (the old draft's
warning box said the digits would move and the direction would not; both
halves came true), not a measurement error — listed here so nobody diffs the
two and suspects worse.

## 9. The "17x" arithmetic error (pre-dates this session; preserved)

Investigating the ~25x disagreement between the CVXPY and hand-rolled cuOpt
paths, an early draft attributed ~8x to the crossover default and reported
the residual as "17x" — subtracting ratios that compose multiplicatively. The
residual is 25/8 ≈ 3x. Kept here because a project that audits its own
numbers does not get to hide the one it got wrong.

## 10. The presolve header

Every committed cuOpt row before 2026-08-24 ran `presolve = -1`, a sentinel
the installed build's header documents as "no presolve". Measured on
300×2,000 (recorded in `src/bench.py`): sentinel −1 = 0.224s, presolve
genuinely OFF = 0.782s, Papilo = 0.259s, PSLP = 0.227s. **The header is
wrong: the sentinel presolves** (behaviourally matching PSLP). Consequence
for the historical record: every committed cuOpt number was measured *with*
presolve on, while the config column recorded a value documented to mean off.
The spec now pins PSLP explicitly and discloses it per row.

## 11. The reference-gap pathology, resolved

The old draft disclosed that `gap_vs_ref_bp` was empty for the sweep and the
credit table, and that at 1000×5,000 the loose cuOpt row reported an
objective 0.61bp *better* than the HiGHS vertex while holding sum(w) = 1.0004
— buying return with portfolio it did not own, uncaught. The fresh sweep runs
a per-cell 1e-9 reference with a **symmetric** ±1bp gate: better-than-exact
now fails. On the fresh draw the same row's gap is +0.21bp (worse, inside the
gate) and the tall cell's is +1.47bp (worse, outside the gate, flagged).

## 12. The cross-process timing scare

The old draft reported the same crossover configuration at 22.0s and 38.8s on
consecutive runs and treated cross-process variance as a live threat to every
timing. The fresh sweep's five-process replicates measure it: across-process
IQR is 0.55–0.63x the *within*-process IQR on the crossover rows — smaller,
not larger. The scare does not reproduce on an idle card and was most likely
GPU contention (the same project once measured 10s vs 77s under a busy
browser tab). Exception, kept: HiGHS at 1000×5,000 shows 2.4x.

## 13. The M6 TE-mandate numbers moved with the instance size and the draw

This entry is a different species from items 1–4. Nothing here failed
reproduction — the M6 numbers always had a committed CSV behind them, and they
audited exact against it every time. What happened is that the **experiment was
re-run at a different size on a different scenario draw**, because the
committed run's timings were unusable as benchmarks (single runs, a contended
card, and a 300s cap that almost nothing finished inside). The re-run is
500 bonds × **20,000** scenarios at an **1,800s** cap, three runs per
configuration, on a card the header records as idle
(`gpu_util_pct_at_capture: 16`, `gpu_contended: no`). Both `m6_te_mandate.csv`
and `m6_te_sigma_ref.csv` were regenerated; the reference file's size and
values are unchanged.

**Every number in the left column is historical.** The finding reproduced in
full — the surrogate overshoots at every budget, monotonically worse as the
budget tightens — and the magnitudes shrank by roughly 10–30%:

| quantity | old (500 × 10,000, 300s cap) | new (500 × 20,000, 1,800s cap) |
|---|---:|---:|
| incumbent sigma-TE | 22.9bp | **22.68bp** |
| unconstrained sigma-TE | 75.8bp | **71.60bp** |
| realized sigma-TE @ 5bp | 25.8bp | **17.85bp** |
| overshoot @ 5bp | 5.17x | **3.57x** |
| realized sigma-TE @ 10bp | 36.4bp | **28.55bp** |
| overshoot @ 10bp | 3.64x | **2.85x** |
| realized sigma-TE @ 20bp | 49.0bp | **42.14bp** |
| overshoot @ 20bp | 2.45x | **2.11x** |
| realized sigma-TE @ 50bp | 72.5bp | **66.32bp** |
| overshoot @ 50bp | 1.45x | **1.33x** |
| realized sigma-TE @ 100bp (budget slack) | 75.8bp | **71.58bp** |
| summarized range in the README's lead | "1.5–5x" | **"1.3–3.6x"** |

The direction is worth naming rather than burying: the overshoot got
**smaller** at every budget. Doubling the scenario count gives the sigma
estimate more tail to work with and the draw is a different one, so the
surrogate's miss is less extreme than the committed run implied. The claim the
section makes is unchanged and its strongest single number is now 3.57x rather
than 5.17x — the correction runs against the story, which is the direction a
re-run most often does not run in.

Two more things changed with the re-run:

- **The DNF pattern.** Old: HiGHS simplex hit the 300s cap at all five
  budgets; cuOpt crossover hit it at 5/10/100bp and returned `SolverError` at
  20 and 50bp. New, at six times the cap: simplex hits 1,800s at **all five**
  budgets again (1,818–1,867s); crossover hits it at 5/10/50bp, reproduces the
  **same `SolverError` at the same 20bp budget**, and — new — **converged at
  100bp in 1,470.18s**, against 8.13s for the raw point, i.e. 181x. The old
  run produced no converged crossover cell at all, so the 181x has no
  predecessor to correct; it is an addition, not a revision.
- **The reference file's timings are off-limits, and the README no longer
  quotes any.** An earlier audit pass flagged three figures sourced from
  `m6_te_sigma_ref.csv` — **54.39s**, **94.61s** and an **88.5x** GPU-vs-CPU
  ratio. All three are historical; they predate even the current reference
  file, whose own walls read 104.26s (Clarabel), 118.86s (HiGHS) and 1.15s
  (cuOpt). The reason they are retired applies to their replacements too: that
  file's header reads `gpu_contended: yes — timings unreliable` on every run,
  because `capture_environment()` samples at *write* time, after every solve
  has already finished. The file's job is cross-solver agreement and the
  NOT-IN-CLASS demonstration. **No wall clock and no speed ratio should be
  quoted from it** — not the historical three, and not the ones it carries
  now. Neither artifact quotes one now: the paper's timing claims are all
  sourced from the working-size sweep, and its section on the mandate carries
  a paragraph stating what the reference file is for and why its walls cannot
  be used. The generator still defines the macros; they are simply unused.

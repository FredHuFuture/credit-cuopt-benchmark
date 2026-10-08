# Open items

## Unreleased extension (2026-10-08)

See `NEXT_VERSION.md` for the new request-timing lane and small CPU bond-lot
pilot. These do not upgrade v1 to a market-validated or executable trading
system. Real risk calibration, issuer-level quantity data, transaction costs,
liquidity, and integer sigma-constrained optimization remain open. Publication
is a manual action reserved for Fred.

The small cuOpt request-timing smoke has now completed. Its native SOCP
portfolio passes the constraint gates but misses the objective-consistency
gate (1.083584e-6 versus 1e-6); the two MAD portfolios fail true sigma.
Investigate barrier precision/auxiliary residuals before a GPU baseline or
matched timing comparison. Keep the original smoke rows and the checker.

## Current scope (2026-10-07)

The sections below are an August development ledger, not current publication
claims. In particular, use the corrected `p0_*.csv` series and release README
for conclusions about crossover, QP feasibility and TE. The initial public
version is a controlled credit/cuOpt benchmark. Real-data calibration,
out-of-time validation, practitioner mandate review, and an optional bond-lot
implementation layer are future extensions, not claimed completed work.

Do not treat the reference SOCP timing as publishable: its file-level
contention flag excludes it. MAD-form feasibility is not sigma feasibility.

From adversarial review of the README (credit PM, GPU engineer, hostile
referee), and now also from the 2026-08-24 GPU session that re-measured
everything and audited the README against its own artifacts. The audit's
meta-result — four unbacked tables, all failed reproduction, all in the
favourable direction — lives in `ERRATA.md`; this file is what remains to do.

Kept in the repo rather than a private list because several of these are
caveats a reader deserves to see, and because a public list of what is wrong
with your own project is cheaper than being told.

## Fixed

- [x] **The bond math is tested now, not just claimed.**
  `tests/test_bond_math.py` asserts the four closed-form claims against the
  public API. CPU-only, no solver, runs in under a second. (The suite has
  since grown to 95 tests: QP model, tracking-error forms, breach gates,
  attribution face — including one case checked against exhaustive
  enumeration — feasibility repair, N-PORT pipeline, sweep replicate
  bookkeeping.)
- [x] **Everything re-run in one session, CSVs committed** (2026-08-24). M2
  and M3 regenerated on one scenario draw with the fixed cache key, full
  environment headers, `assert_gpu_idle` in every timing script. The headline
  credit table finally has a CSV (`m2_credit_500x20k_solves.csv`) — and its
  numbers moved; see `ERRATA.md` §1. The fastest-compliant rule now selects
  cuOpt+crossover (21.1s vs HiGHS 22.2s): the old inversion dies at the
  working size.
- [x] **The ablation is run** (`results/ablation.csv`). Verdict: crossover
  cost tracks the auxiliary variables of the turnover epigraph (L1 jump 0.88s
  → 16.05s; the other five families move it 14%), not families, not mandate
  rows. Bonus findings: HiGHS IPM's crossover is free (multiples 0.86–1.12
  across all levels — the expensive thing is crossover *from a first-order
  iterate*), and HiGHS simplex is faster on the full mandate than on
  budget+box (20.8s vs 34.5s — constraints prune the search).
- [x] **The 1e-9 reference runs on the sweep**, per cell, with a symmetric
  ±1bp gate that fails better-than-exact. The old 0.61bp pathology can no
  longer pass quietly; the fresh tall cell flags +1.47bp worse-than-reference
  on the raw cuOpt row.
- [x] **The sweep is scored against the mandate.** Both gate verdicts
  (purity 1e-6, desk 1bp-of-weight) are per-row columns. Every raw cuOpt row
  is now BREACH; OR-Tools splits (BREACH/BREACH/DESK-OK).
- [x] **Replicates.** 5 processes × 3 solves on the crossover and HiGHS rows;
  across-process IQR is 0.55–0.63x within-process on the crossover rows —
  the historical cross-process scare does not reproduce (exception: HiGHS at
  1000×5000, 2.4x).
- [x] **M7, the QP branch, is run** (`results/qp_branch.csv`). Four arms,
  N=500..15,000 measured; dense refused from 10,000 up (host guard); factored
  15,000 in 1.56s; formulation worth 10.5–124.7x, hardware crossover between
  5,000 and 7,500 names. Q-convention calibration passed on the cuOpt path;
  CVXPY QP path reachable in the pinned wheel; canonicalization reported in
  `compile_s`. New defects came out of it — see below.
- [x] **The attribution baseline is fixed and measured.**
  `min_risk_face()` ran on the real M2 draw: the face spans −6.2%..+10.9%
  CVaR reduction ex ante → corner share −41%..+72% of the optimizer's total,
  and the hindsight vertex beats the optimizer. **The split is not
  identified**; the README quotes the band and refuses the ratio. (The fresh
  arbitrary vertex lands at ~65% again — see `ERRATA.md` §7 for why that
  means nothing.)
- [x] **Warm start re-measured and reconciled** (`warmstart_tradeoff.csv`):
  the old table inverted — presolve is a 1.7x net cost, warm is fastest and
  lands 10x closer to feasible. Demo latency re-measured back-to-back in the
  same session (`demo_latency.csv`): monotone 0.47/1.45/3.02s; the 3.3x
  self-contradiction was cross-session noise.
- [x] **cuOpt presolve pinned and disclosed** (`src/bench.py`): the build's
  header claims sentinel −1 = none; measured 0.224s vs 0.782s genuinely off
  — the sentinel presolves. Pinned PSLP (GPU-side), disclosed per row.
  Four-arm measurement recorded in the spec's docstring.
- [x] **RNG agreement reported as measured SE differences** rather than
  pass/fail (`rng_agreement.csv`: 0.20–1.94 SE across the four sizes).
- [x] **An ex-ante tracking-error budget** — `src/credit.py` family 7, three
  forms, exercised by `run_te_mandate.py` and 7 tests. The sigma form is one
  SOC of dimension S+1 (NOT-IN-CLASS for every LP solver); the MAD surrogate
  under-delivers by 1.33–3.57x at 50–5bp budgets, three solver families
  agreeing (`m6_te_mandate.csv` re-run at 500×20,000 with an 1,800s cap on an
  idle card, `m6_te_sigma_ref.csv`). The M9 tail sweep
  (`m9_tail_sweep.csv`) attributes the miss: jumps 68.1%, interaction 30.0%,
  t-tails alone 1.9% at the shipped point.

## New defects, found by the session

- [ ] **The generator still emits four timing macros from the contended
  reference file.** `m6_te_sigma_ref.csv` carries `gpu_contended: yes —
  timings unreliable` in its header, and no artifact quotes its walls any more:
  the paper's timing claims all come from the working-size sweep. But
  `make_paper_numbers.py` still defines `\MSixRefSigmaSeconds`,
  `\MSixRefMadSeconds`, `\MSixRefCuoptMadSeconds` and
  `\MSixRefCuoptVsClarabelMad` — the last a 90.8x GPU-vs-CPU ratio. They are
  unused, arithmetically correct, and one `\cite`-sized mistake away from
  being quoted again by someone who trusts the macro layer to only contain
  usable numbers. Drop them, or teach the generator to mark a macro as
  disclaimed by its own source header.

- [ ] **Device-side memory guard for the factored QP arm.** The host guard
  correctly refused the dense cells; the factored arm then died at N=29,222
  on a CUDA OOM in *scenario generation* — a 4,675,520,000-byte allocation
  (20,000 × 29,222 × 8) no guard covered. Add a device-side estimate to the
  refusal logic and recalibrate from this failure, the way the host guard was
  calibrated from its first real OOM.
- [ ] **Float64 promotion in the scenario path at scale.** That allocation is
  8 bytes per element: the scenario matrix is promoted from float32 to
  float64 somewhere in the generation path. Find the promoting op, pin the
  dtype end-to-end, and add a test that asserts the device array's dtype at a
  size where it matters. Halving this is the difference between the factored
  arm reaching 29,222 and not.
- [ ] **The QP crossover anomaly.** cuOpt accepts `crossover` on a QP (docs:
  unsupported) and the answer moves by L1 2.1e-9 — neither an error, nor a
  clean ignore, nor a real crossover. Reported upstream-worthy. Until
  explained, nothing may lean on QP crossover; the probe row in
  `qp_branch.csv` is the evidence.
- [ ] **A longer cap for the four M6 DNFs.** The at-scale re-run is done
  (500x20,000, 1,800s, idle card, three runs per configuration, `verdict` and
  the gate columns present), so the timing sentences are sourced. What remains
  is narrower: HiGHS simplex hit the cap at all five budgets and cuOpt
  crossover at four of five, so those nine cells are lower bounds rather than
  measurements. The 181x crossover cost rests on the single cell that
  converged. A longer cap would turn the bounds into numbers, at roughly an
  hour per cell — worth doing only if a reviewer asks for the distribution.
- [ ] **Stratified N-PORT trim.** The largest-by-held-value n=500 trim moves
  below-IG weight from 28.4% to 48.9%, changing the mandate the optimizer
  solves. Offer a stratified trim preserving the full universe's rating and
  maturity mix; until then an n=500 N-PORT run must not be called an IG
  mandate.
- [ ] **`run_ablation.py` differencing a structurally-zero column.** The
  probe counts every added constraint into `lp_eq_rows` (the CVXPY chain
  reports them there), so `lp_ineq_rows_added` is a difference of a column
  that is 0 at every level. Harmless to the findings (the row/variable deltas
  are read from `lp_eq_rows`/`lp_vars`), but the column is misleadingly named
  and should either be populated from the right probe field or dropped.

## Still needs a GPU (or a bigger box)

- [ ] **The SOC row on cuOpt's barrier.** The sigma-form TE budget is the one
  configuration where PDLP is not merely loose but inapplicable, and cuOpt
  ships a GPU barrier over cuDSS. `run_te_mandate.py` still lacks the
  barrier-on-SOC arm; M7's `--objective te` covers the QP-shaped variant
  only. This is now the most interesting unmeasured cell in the repo: the
  mandate as written, on the GPU, in class.
- [ ] **Feasibility repair on cuOpt's own iterates.** The M8 mechanism
  (89–115x cheaper than crossover on the weight-space mandate; fails against
  binding TE, 254x→276x) is measured on OR-Tools loose points with a HiGHS
  IPM+crossover stand-in. The npz path for a real cuOpt point + real cuOpt
  crossover is wired and unexercised.
- [ ] **A parallel CPU scenario baseline.** The GPU generation margin
  (1.36–5.66x) is one GPU against one NumPy thread. Rerun the CPU side
  threaded and requote; if the margin dies, that is the finding.
- [ ] **One datacenter-GPU column.** Every number is a consumer card under
  WSL2. The crossover economics and the QP hardware crossover could sit
  elsewhere on an H100; one sweep cell plus one QP cell would say whether the
  inversion is a property of the problem or of this hardware. Still the most
  load-bearing untested assumption in the writeup.
- [ ] **Time-to-matched-achieved-residual.** The two PDLPs stop in different
  places at the same requested 1e-4; tighten cuOpt until its achieved
  residual matches OR-Tools' and time that. Until then tolerance-matched
  margins are an upper bound.
- [ ] **BatchSolve on an efficient frontier** — dozens of related LPs, no
  single exact answer needed, so crossover is never paid: the case where the
  GPU may win outright.

## Measurement harness

- [ ] **The contention flag describes the wrong moment.** `capture_environment()`
  runs at write time, after every solve, so `gpu_util_pct_at_capture` and
  `gpu_contended` report what the machine was doing once the work had
  finished. `run_sweep.py` has the mirror-image problem: `assert_gpu_idle`
  fires once before the first cell and never again across a three-hour run.
  Neither samples DURING the window it claims to describe. This is not
  hypothetical: the M6 conic reference was stamped `gpu_contended: yes` twice
  from a post-run sample while the run itself was probably fine, and a sweep
  could be stamped clean and then contended for its whole duration. The fix is
  to sample on a timer for the life of the run and report min/median/max, so
  the header describes the run rather than an instant at one end of it.

- [ ] **Only the GPU was gated; the CPU baselines are timed on an unguarded
  CPU.** Found the expensive way. A WSL `unattended-upgrades` job put the load
  average at 3.65 while GPU utilisation looked fine, and the same reference
  instance that had taken Clarabel 94.0s took 143.1s -- a 52% inflation of a
  *competitive baseline*, invisible to every check in the repo. The TE values
  were byte-identical across both runs, so the determinism guarantee held and
  only the timings moved, which is exactly the failure mode that is easy to
  publish by accident. The session's `wait_idle` preamble now gates on load
  average per core as well as GPU utilisation; that logic belongs in
  `src/bench.py` next to `assert_gpu_idle` rather than in a session script,
  and the load average belongs in the environment header beside the GPU
  fields.

- [ ] **Progress output is lost to block buffering.** Redirect a runner's
  stdout to a file and Python buffers it in 8 KB blocks, so a five-hour run
  shows nothing but stderr warnings while every progress line sits in a
  buffer. A monitor watching that log reports silence and looks identical to a
  hang; the only way to tell them apart was counting CVXPY warning pairs to
  infer how many problems had been built. Either set `PYTHONUNBUFFERED=1` in
  every driver (the reference rerun does) or flush the progress prints, and
  say so in RUNBOOK.md.

## Model realism

- [ ] Rating-dependent per-issuer caps. The current flat 2.65% applies
  identically to AAA and single-B.
- [ ] Sector *floors*, not just caps — the same free-de-risking hole that was
  fixed for rating bands is still open one family over, and it feeds the
  mandate-corner face the attribution section now measures.
- [ ] Rating migration. Its absence means expected return has no issuer-level
  dispersion within a rating, which makes the widest bond in each bucket a
  modelled free lunch.
- [ ] Transaction costs — HY has the widest bid-ask, so costs would damp the
  up-in-quality tilt the results report. Note the ablation makes this item
  double-edged: a linear t-cost term canonicalizes exactly like turnover, so
  adding costs also adds the aux-variable block that dominates crossover.
- [ ] Model HY as HY: mostly callable, concentrated at 5–8y, spread duration
  materially shorter than modified duration. Currently it is IG paper with a
  wider spread, which overstates HY DTS for exactly the bucket being cut.
- [ ] Report the smallest held weight beside the position count
  (`DUST_THRESHOLD` is 1e-6, so counts include $1,000 positions on $1bn).
- [ ] Optionally add the dropped `C·Δy·Δs` cross term to the excess-return
  identity (~3% understatement of absolute CVaR; relative comparisons are
  unaffected).
- [ ] Real scenarios to go with the real universe: the N-PORT bonds still get
  model-generated risk. Per-field calibration sources are documented in
  `CalibrationConfig`; wiring FRED OAS histories into the factor calibration
  is the next step that keeps the licensing line intact.

## Measurement

- [ ] Reconcile the two multipliers quoted for the CVXPY-vs-DataModel gap
  (~8x from crossover alone, ~25x total) — the remainder is probably row
  conditioning, which was measured at >10x on the same path and therefore
  over-explains a ~3x residual; the experiment that settles it has not run.
- [ ] Hedge the warm-start result (now positive rather than negative) to the
  instance and perturbation actually tested: one problem, 10% risk-aversion
  steps. The demo's own use case is a small slider nudge — the case least
  covered — and the demo default should flip to warm once that case is
  measured.
- [ ] Extend `tests/` past what the session added: the scenario engine's
  calibration (realized default rate vs target) and the DTS/turnover unit
  conventions are still checked only by being printed during a run.

## Presentation

- [ ] Show the active positions the results table hides: the optimum is a
  front/long barbell against the benchmark, and the README reports only the
  aggregate tilts, which read as a mild de-risk.
- [ ] Characterize the synthetic benchmark honestly wherever it appears —
  ~15% below-IG makes it a crossover index, not an IG corporate index — and
  note that every limit is index-derived rather than absolute, which is not
  how mandates are written. (The N-PORT section now does this for the real
  universe; the synthetic sections should match.)
- [ ] The engineering-notes section is long again; if the README needs to
  shrink, move it to `NOTES.md` keeping the two highest-credibility traps
  (the CVXPY warm-start caching trap; the presolve sentinel) in the body.

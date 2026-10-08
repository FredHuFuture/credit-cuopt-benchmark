# Benchmark runbook

## Current release policy (2026-10-07)

Start with the CPU-only artifact checks in README.md. A successful number
generator establishes provenance, not publication readiness: inspect the
independent gates, exact mandate being checked, and timing exclusions as well.
The current paper studies credit portfolio optimization with cuOpt.

- Both `p0_socp_reference.csv` and `p0_socp_working.csv` support conic
  feasibility only. Do not publish their timing ratios.
- The endpoint ladder's MAD@10bp strict pass is **MAD-form LP feasibility**;
  the tight raw point has 28.55bp sigma TE, not 10bp sigma compliance.
- QP ratios match feasibility, representation and endpoint, not certified
  equal optimality accuracy. Report best-feasible objective gaps separately.
- Preserve all release measurements. Write new runs to new paths; never
  delete a CSV to make the number generator act as a to-do list.
- Long GPU sessions are optional future measurement work, not required for
  rebuilding the release PDF. Do not run them on a contended desktop.

The remaining sections document the **historical August sessions**. Their
intermediate interpretations, macro counts and completion criteria are
superseded by the current paper, README and this policy. Commands remain
useful for examining experiment design, but inspect each before rerunning.

## Historical GPU session

One sitting, on an idle machine, in the order below. Every headline table in
the paper comes out of a session like this one, and re-running half of it on
another day means measuring half the paper on a different scenario draw —
the defect this document exists to avoid.

Read it once before starting. The order is not arbitrary: the cheap probes at
the top exist to fail early, while there is still time to re-plan.

**Status: the session this was written for has been run.** Every table in the
paper now has a committed CSV behind it and the generator exits 0. What
follows is kept as the procedure for the next regeneration — a new scenario
draw, a datacenter card, a newer cuOpt — with each step annotated by what it
actually did the first time. Where a step's result changed the plan, that is
recorded in place rather than smoothed over.

---

## What decides when you are done

`scripts/make_paper_numbers.py` is the scoreboard, not this file. It reads the
committed CSVs and emits every number the paper prints; run it strict and it
either writes every macro or it refuses and names what is missing.

```bash
python scripts/make_paper_numbers.py
```

**The session is finished when it exits 0.** Nothing else is a completion
criterion — not this checklist, not a feeling that enough has run. It refuses
rather than emitting a placeholder, so an unfinished session cannot quietly
become a paper with a plausible-looking number in it.

On the run this document was written for it began at 113 unavailable macros
and reached 0; the count today is 421 defined, all backed. A fresh
regeneration starts by deleting or moving the CSVs it intends to replace, at
which point the generator's refusal lists exactly what the session still owes.

---

## Pre-flight

Run everything from the repository root, in the project venv.

```bash
git status --short                     # must be clean
.venv/bin/python -m pytest tests/ -q   # see the suite's current count
nvidia-smi                             # nothing else on the card
uptime                                 # load average near zero, see below
free -g                                # under WSL2, check the VM's allocation
```

**Gate on the CPU too, not just the card.** The GPU gate was the only one for
a long time and it let an expensive mistake through: a WSL
`unattended-upgrades` job held the load average at 3.65 while GPU utilisation
looked clean, and Clarabel took 143s on a reference instance it had solved in
94s minutes earlier. The CPU baselines here are competitive numbers, so an
unguarded CPU inflates exactly the column a reader would use to judge the
GPU. Before a timed step, both must hold: GPU utilisation under about 18% and
one-minute load average under roughly 0.15 per core. `wait_idle.py` polls both
and requires three consecutive quiet reads, because one lull is not a quiet
machine. Kill background package jobs first:

```bash
sudo systemctl stop unattended-upgrades
```

**Set `PYTHONUNBUFFERED=1` in every driver script.** Redirect a runner's
stdout to a log and Python buffers it in blocks: a five-hour run then shows
nothing but stderr warnings, and a monitor watching that log cannot tell a
working run from a hung one. This cost real diagnostic time; progress had to
be inferred by counting CVXPY warning pairs.

Close everything using the GPU. A browser with a hardware-accelerated video
tab is enough to invalidate a night: this project has already measured the
same configuration at 10s and 77s because the card was at 98% for an
unrelated reason. `assert_gpu_idle` now runs in every timing script and will
refuse, but do not make it do that work.

---

## Step 1 — probes (~20 min, never skipped)

Two cheap questions that change the plan if they go the wrong way.

**1a. Is cuOpt's QP path even reachable?** Five open M7 questions collapse
into one probe: whether the installed `cuopt-cu12==26.6.0` accepts a quadratic
objective through CVXPY, whether the Q convention needs the factor of two
(getting this wrong silently doubles the risk aversion — no crash, no
warning, every downstream timing attached to a problem nobody asked for),
what `crossover` does to a QP (error, warn, or silently ignored — the third
is the dangerous one, because an operator could believe they cleaned an
interior point up to a tradeable vertex when nothing ran), and what the
default `Concurrent` method does with a quadratic objective.

```bash
python scripts/run_qp_branch.py --probe-only --probe-crossover
```

If it reports the CVXPY path unreachable, the M7 GPU arm needs rebuilding on
the raw `DataModel` path and the identical-canonicalization guarantee that
justifies the four-arm comparison is lost. Knowing that at hour zero is the
whole point of running it first.

**1b. Does the presolve pin still match what the sentinel does?** This probe
earned its place at hour zero and should be repeated on any cuOpt upgrade.

Every cuOpt row committed before this session inherited `presolve = -1`, a
sentinel, and the installed header documents that sentinel as resolving to
"none". **The header is wrong.** Four arms on one 300x2000 credit cell, three
timed runs each, each arm in its own subprocess:

| presolve | wall |
|---|---:|
| `-1` (sentinel, what every earlier row ran) | 0.224s |
| `0` (off) | 0.782s |
| `1` (Papilo, CPU) | 0.259s |
| `2` (PSLP, GPU) | 0.227s |

The sentinel presolves. Pinning `0`, which is what this repo had just done on
the strength of the header, would have made the whole session 3.5x slower
than every committed number for a reason unconnected to anything the paper
claims — and it would have looked like a finding.

`src/bench.py` and `scripts/run_te_mandate.py` therefore pin **PSLP**: within
noise of the sentinel, explicit in `config_full` so it is reproducible across
builds, and running on the GPU. Papilo would move an unmeasured amount of
work onto the host under a column labelled "GPU", which is not a trade this
paper can make.

Nothing on this build's LP result path reports which presolver actually ran,
so the table above is inference from wall time. It is enough to choose a pin;
it is not proof that `-1` *is* PSLP. Re-measure rather than assume after any
version bump.

---

## Step 2 — M2 credit (~30-40 min, never skipped)

```bash
python scripts/run_credit.py --n-bonds 500 --n-scenarios 20000 --runs 3 \
    --out results/m2_credit_500x20k.csv
```

Produces the headline credit table that has never had a committed CSV, and
now also: the environment header those two files lack, both gate verdicts per
configuration, the signed budget residual, and the attribution face band from
`min_risk_face`.

Watch the face band. On a 60-bond toy it spanned 72% to 183% of the total
CVaR reduction, which if it survives at 500 bonds means the 65/35 attribution
split is **not identified at all** and the paper says so instead of quoting a
ratio.

---

## Step 3 — M3 sweep (~3-4 h, never skipped, same session as Step 2)

Non-negotiable: same sitting, same seeds, same scenario draw as Step 2. The
default-trigger cache key now includes the seed and scenario count, so
re-running one table and keeping the other leaves the paper built on two
different draws.

```bash
python scripts/run_sweep.py --grid full --resume \
    --runs 3 --replicates 5 --replicate-only cuopt_pdlp+xover,highs \
    --out results/m3_sweep.csv
python scripts/plot_sweep.py --csv results/m3_sweep.csv \
    --out results/m3_scaling.png
```

Replicates go only to the crossover and HiGHS rows because those carry the
central claim and the sweep already takes hours; that is 12 worker
invocations per cell instead of 4, plus one reference. It checkpoints per
configuration and `--resume` is now correct — the resume key was comparing a
CSV solver label against a factory name, two strings that can never be equal,
so before this only a hardcoded row count was doing any work.

Three things are new in the output and worth watching live:

- `gap_vs_ref_bp` is populated at last. The symmetric ±1bp gate now **fails**
  a row that beats the exact reference, so the committed 1000×5000 pathology
  (0.61bp "better" than the HiGHS vertex while holding `sum(w)` = 1.0004)
  will show as a check failure rather than passing quietly. That is correct.
- `wall_s_iqr_across` versus `wall_s_iqr_within`. If the across-process
  spread dwarfs the within-process one, every previously published median-of-3
  error bar was measuring the wrong thing, and the ratio
  `iqr_across_over_within` is the number that says so.
- Both gate verdicts per row. A `check_ok=True` row with `gate_verdict=BREACH`
  is the "portfolios of unknown admissibility" defect finally visible.

**Commit the M2 and M3 CSVs the moment both finish.** The non-negotiable pair
is then safe regardless of what happens later in the night.

---

## Step 4 — the small runs the generator is blocked on (~30 min)

Three short scripts, all currently blocking macros:

```bash
python scripts/check_rng_agreement.py     --out results/rng_agreement.csv
python scripts/check_warmstart_tradeoff.py 500 20000 \
                                          --out results/warmstart_tradeoff.csv
python scripts/check_demo_latency.py      --out results/demo_latency.csv
```

Run the last two **back to back, in this session**. They are the two halves of
a disclosed 3.3x self-contradiction: the warm-start table's "cold + presolve,
10.0s" and the demo table's "3.0s" are the same configuration, measured in
different sessions. One session settles whether that gap is real or was
always cross-session noise.

---

## Step 5 — the TE mandate at scale (~6 h, budget a night)

```bash
python scripts/run_te_mandate.py --out results/m6_te_mandate.csv
python scripts/run_te_mandate.py --n-bonds 300 --n-scenarios 3000 \
    --te-grid 10 --solvers clarabel cuopt@1e-4 highs-simplex \
    --out results/m6_te_sigma_ref.csv
```

Two files: the working-size sweep, then the small conic reference that panel B
of the M6 figure and the conic macros read. The reference needs Clarabel,
which is the only solver here that can express the second-order cone at all.

**Budget a night, not an hour.** The first attempt at this step died in a
three-hour window having written nothing, and the reason was a harness defect
rather than the problem being hard: `solve_once` ran its warmup and then three
timed solves unconditionally, so an arm that hits the cap burned four full time
limits proving what the warmup already knew. At the working size simplex hits
the cap at every budget, which is 4 x 1800s x 5 budgets for one arm alone. It
now bails after a failed warmup, the same rule `run_spec` in `src/bench.py` has
always used, and a DNF costs one time limit. Even so the full run took about
five and a half hours, essentially all of it in DNF arms burning their caps
honestly.

Do not shorten the cap to save time. The 1800s cap is the result: at the
working size, simplex fails to reach a vertex at every budget in six times the
old 300s limit, and *that* is the publishable sentence. Trimming to
`--te-grid 10 20 50` drops budgets, not the cap, and is the right economy if
one is needed.

What the run produced, as a sanity check for the next one: the MAD surrogate's
overshoot rises monotonically as the budget tightens; cuOpt at 1e-4 solves
every budget in 8-11s and at 1e-6 in 68-88s; crossover converges at exactly one
budget — the loosest, where it takes about 1470s against the raw point's 8s,
which is roughly 180x to clean up the arm the mandate constrains least. The
crossover arm returns a SolverError at 20bp, reproducibly, in both this run and
the previous one.

---

## Step 6 — the ablation (~2-2.5 h)

```bash
python scripts/run_ablation.py --out results/ablation.csv --resume
```

The paper's causal sentence. Pre-registered, so the reading is not chosen
after the numbers: at 500 bonds the turnover budget alone contributes 1,001 of
the 1,238 inequality rows the full mandate adds, because `norm1(w - w0)`
canonicalizes to ~2N+1 rows. If crossover cost tracks rows rather than
families, then "a credit mandate has six families of limits" is the wrong
explanation and "one family happens to be enormous" is the right one — and
that is the sharper paper, not the disappointing one.

Report against both `mandate_rows` and `lp_ineq_rows_added`, which the script
already records, so the two stories are distinguishable.

Behind schedule: drop the IPM row first, then `--levels L0,L1,L6` (~45 min).
L1 is turnover, the pre-registered suspect, so the trimmed version still
answers the question.

---

## Step 7 — feasibility repair on real cuOpt points (~30 min)

The CPU run already showed the mechanism: against the weight-space mandate an
L1 projection is 115x cheaper than crossover and buys full compliance for
0.22bp, and against a binding tracking-error limit it fails outright and can
make the breach worse. What the CPU cannot establish is that this holds for
**cuOpt's own iterates** — OR-Tools PDLP is the same algorithm but a different
implementation, with a different restart schedule and a GLOP presolve cuOpt
does not run, so part of the loose point's structure is simplex-derived.

Dump a loose cuOpt point and its crossover vertex to npz during Step 2 or 3,
then:

```bash
python scripts/run_feasibility_repair.py \
    --loose-source npz --loose-npz <loose.npz> --crossover-npz <xover.npz> \
    --out results/m8_feasibility_repair_gpu.csv
```

Supplying a crossover suppresses the CPU stand-in, so the cost ratio is not
formed against the wrong machine. One case per invocation until the npz path
has been exercised with a `--te-grid`.

---

## Step 8 — M7, the QP branch (~2-3 h, degrades gracefully)

```bash
python scripts/run_qp_branch.py --grid 2000,5000 --runs 1 --out results/qp_probe.csv
python scripts/run_qp_branch.py --out results/qp_branch.csv --resume
```

Ceiling probe first.

The dense arm was expected to die on **host** RAM, and it did: the dense Sigma
is built, densified and CSR-converted in host memory before anything crosses
the bus, so the guard refuses it from N = 10,000 up (estimated ~4.5 GiB at
10,000 rising to ~38 GiB at 29,222, against 5.9 GiB available). Those refusals
are recorded with their reason and the paper reports them as refusals. Do not
write M7 as if a dense index-scale data point exists.

**What was not expected, and what to fix before the next run: the grid died at
N = 29,222 on a CUDA out-of-memory in the FACTORED arm — the cheap one.** The
guard models the dense Hessian's host footprint and nothing guards device
allocations at all. The allocation that failed was 4,675,520,000 bytes, which
is 20,000 x 29,222 x 8: the scenario matrix, in the CuPy generator, not in the
solver. That size also says it was float64 while `credit_scenarios` defaults to
float32, so something on this path promotes and doubles the device footprint of
the largest array in the program for nothing. Two fixes before an index-scale
cell is attempted: a device-side guard beside the host one, and the promotion
tracked down.

`DENSE_Q_BYTES_PER_ENTRY_PEAK = 48` is arithmetic, not measurement — exactly
what `BYTES_PER_NNZ_PEAK` was before a real OOM corrected it by a factor of
two. Compare `host_peak_est_gib` against actual RSS and recalibrate from the
failure.

Also from the probe, and unresolved: cuOpt **accepts** a crossover request on a
QP, which its documentation lists as unsupported, and the answer moves by an L1
of 2.1e-9. It neither errors nor cleanly ignores it, and a real crossover would
move to a vertex — a large L1 change and a collapse in position count — so
whatever ran was not that. No result may lean on QP crossover until this is
explained.

Trim to `--grid 500,2000,5000,29222` if long: the 29,222 refusal is cheap and
is the punchline; 7,500/10,000/15,000 are the droppable middle.

---

## Drop order

If the card, the VM or the night gives out, drop in this order:

1. Step 8's middle cells, then Step 8 entirely
2. Step 7 (the CPU result already carries the mechanism)
3. Step 6 down to `L0,L1,L6`
4. Step 5 down to `--te-grid 10 20 50`
5. Step 4's demo-latency run

**Steps 1, 2 and 3 are never dropped.** Without them the publication
contradicts itself, which is worse than publishing less.

Commit CSVs after every step. WSL has eaten finished work three times on this
project.

---

## After the session

```bash
.venv/bin/python -m pytest tests/ -q
python scripts/make_paper_numbers.py          # must exit 0 now
python scripts/plot_te_mandate.py             # regenerate from fresh M6
./paper/build.sh                              # strict; refuses on any gap
git status --short
```

Then re-derive every README number from the fresh CSVs and nothing else, and
delete the two warning boxes the stale draw made necessary. If a conclusion
moved, the conclusion moved — the direction of every finding here is a
property of the model, but the digits never were.

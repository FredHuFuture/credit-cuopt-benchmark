# Supplemental pilots in the current revision candidate

At Fred's request, these local extensions are included in Appendix B of the
current paper and its prepared release package. They are not a separate paper.
The previously published v1.0.0 assets and historical measurements are retained.
Fred performs all final publication actions. Do not push, upload a release,
submit a paper, or post promotion from this workflow.

## Scope

Two independent improvements are implemented here:

1. A single-request timing/quality lane for the existing solver stacks.
2. A small CPU-only quantity experiment: continuous SOCP, naive rounding,
   and minimum-distance lot-aware MILP repair.

Neither requires a licensed production risk model. Neither claims one.
CuClarabel integration, issuer-specific quantity conventions, market-calibrated
risk, transaction costs, cardinality limits and liquidity remain outside this
pilot. No new market data is downloaded. There is no new large GPU grid.

The design references already in `paper/references.bib` are Amundi's *Bond
Portfolio Optimisation and Mixed Integer Programming* (2023), *Asymmetry PRISM*
(2026), and *CuClarabel* (2024). This pilot adopts explicit quantity constraints
and transparent timing/quality boundaries; it is not a reproduction of their
risk models, data or solver implementations. CuClarabel itself is not run.

## Quantity experiment

Implementation: `src/bond_lots.py`, `scripts/run_bond_lots.py`.

All arms share the same synthetic universe, scenario draw, linear mandate,
cash cap, and incumbent within each NAV case. To give minimum trades a valid
starting point, the original incumbent is floored to the lot lattice, small
positions are removed and residual cash is retained. This is a new starting
book, not the historical v1 incumbent. It can breach the final cash cap before
optimization. The benchmark itself is unchanged.

Hypothetical defaults (not asserted market conventions):

| Parameter | Value |
|---|---:|
| Bonds / scenarios | 60 / 500 |
| Universe / scenario seed | 7 / 11 |
| NAV cases | $1M, $10M, $100M |
| Par increment | $1,000 |
| Minimum nonzero holding | $10,000 par |
| Minimum nonzero trade | $5,000 par |
| Maximum cash | 2% of NAV |
| True sigma-TE target | 10bp annualized |
| Solver cap | 30s per solve; a capped case is not retried |

For lot counts `k_i`, settlement-price proxy `P_i` per $100 par, increment
`L`, and NAV `V`, bond weights are `w_i = k_i L P_i / (100 V)`.
Cash satisfies `sum(w) + cash = 1`; bond weights are never renormalized.
The synthetic price is treated as a settlement-price proxy with zero accrued
interest. Cash earns zero **excess** return. Cash has zero duration and DTS.
One-way turnover is `0.5 * (sum(abs(w - w0)) + abs(cash - cash0))`.

Nonzero holdings must exceed the holding minimum. Separate buy/sell binaries
require a trade to be either zero or at least the trade minimum; opposing
trades cannot be netted to evade that rule. The current implementation requires
both minima to be integer multiples of the increment. It is a disclosed
simplification, not a general security-master convention.

The continuous source maximizes `mu @ w - 3 * CVaR_0.95(w)` with a sigma cone.
The HiGHS MILP minimizes bond-plus-cash L1 distance to that source, enforcing
the same **linear** constraints plus quantities. It does **not** enforce the
sigma cone. True sigma, expected excess return, CVaR, quantity rules, cash and
every linear family are checked independently after solving.

A capped MILP can return a feasible incumbent without certifying the best
L1 distance. The output separates `independent_integer_feasible` from
`accepted_endpoint` and `optimality_proven`. The last means solver-reported
optimal within configured tolerances (relative MIP gap 1e-6, absolute 1e-9),
not a proof using exact arithmetic. No lot row earns a speed ratio. Negative
objective loss on an infeasible portfolio is not an economic improvement.

### Reproduce

From a Linux/WSL checkout using the pinned CPU environment:

```bash
python scripts/run_bond_lots.py --out results/p1_bond_lots_cpu.csv
python -m pytest tests/test_bond_lots.py -q
```

For the Windows checkout on this development machine:

```powershell
wsl -e bash -lc 'cd /mnt/c/Users/futur/localCodeBase/cuOpt && /home/future/bond-cvar-cuopt/.venv/bin/python scripts/run_bond_lots.py --out results/p1_bond_lots_cpu.csv'
```

Choose a **new** output filename for a rerun. Existing output and sidecars are
refused, not overwritten. Each completed endpoint is flushed immediately.
The `.portfolios.jsonl` sidecar retains holdings, quantities, residuals and
limits; `.metadata.json` records source hashes, versions and assumptions.
All scenarios and prices can be regenerated from the recorded seeds.

`total_request_s` on the continuous row includes model build, canonicalization,
solve, extraction and independent checking. On rounding/repair rows it is the
continuous request plus that endpoint's additional work. Scenario generation,
imports and file I/O are outside this boundary and are not smuggled into solver
time. One observation per NAV is a functional pilot, not a performance study.

### Recorded pilot result

`results/p1_bond_lots_cpu.csv` contains nine endpoint rows, all retained.
All three continuous references deliver approximately 10bp sigma TE.

| NAV | Naive rounded sigma TE | MILP repaired sigma TE | MILP linear/quantity gates | MILP conclusion |
|---|---:|---:|---|---|
| $1M | 58.6263bp | 13.1885bp | Pass / pass | L1 optimum, but sigma fails |
| $10M | 9.8166bp | 9.9482bp | Pass / pass | Feasible incumbent; 30s cap, not certified optimal |
| $100M | 10.0081bp | 10.0025bp | Pass / pass | L1 optimum, but strict sigma gate fails |

The $10M rounded portfolio is **not** feasible despite passing sigma: it has
negative cash (about $5,363 overspending) and a sub-minimum trade. The capped
$10M MILP incumbent passes the independent gates; its remaining reported MIP
gap is about 0.433%, on **L1 distance**, not on expected return or portfolio
CVaR. It is not included as a converged endpoint or used in a timing ratio.
The $100M repaired sigma excess is only 0.00245bp, below the historical 0.05bp
desk reporting tolerance, but it still fails this experiment's strict numerical
sigma gate. Do not silently switch gates to improve the success count.

This establishes a useful boundary in this one synthetic case: discrete repair
can restore the linear and quantity rules without preserving a tight sigma
constraint. It does not establish universal scaling with NAV, market execution
quality, or superiority of a particular integer solver. The new rounded
incumbent also changes with NAV, so this is not an isolated causal lot-size test.

## Single-request timing lane

Implementation: `src/request_timing.py`, `scripts/run_request_bench.py`.

This lane does not relabel historical `total_endpoint_s` fields, some of which
sum a first solve and an entire repeat batch. Each new row measures one request:
build, explicit canonicalization, solve call, repair (zero when absent), and
independent validation. The solve-call bucket includes adapter/transfer work;
it is not advertised as device-kernel time.

The first request creates a fresh model. Cached requests reuse the same CVXPY
Problem and may ask for warm start; this does not guarantee solver-state reuse.
Scenario generation and worker startup are separately identified. Resource
snapshots must not be described as peak usage or proof of uncontended hardware.

```bash
python scripts/run_request_bench.py --out results/p1_request_timing_cpu.csv
python -m pytest tests/test_request_timing.py -q
```

GPU paths require explicit `--allow-gpu`. Method-incompatible formulations
produce method-scoped non-results. A failed request is checkpointed without
repeating the failure. Only independently accepted rows with the same instance,
formulation, endpoint, request mode and objective-quality target may enter a
ratio. The default simplex/MAD and conic/sigma arms do not meet that condition.

### Recorded validation and GPU follow-up

`results/p1_request_timing_cpu.csv` contains three rows on 100 bonds and 1,000
CPU-generated scenarios. HiGHS solves MAD@10bp to `optimal`, but its true sigma
is 28.2854bp, so it is rejected and not repeated as a cached request. Clarabel
delivers approximately 10bp in both the cold and cached request and passes all
acceptance checks. No ratio is emitted. The single observations are a protocol
smoke test, not evidence of a stable cold/warm or CPU/GPU speed difference.

The cuOpt smoke completed on 2026-10-08 after Fred provided a quiet window.
An earlier attempt was held at 47% utilization. The successful launch followed
five preflight samples of 8%, 1%, 2%, 0%, and 18%, all below the 25% guard;
preceding samples had briefly reached 28-29%. This is a functional smoke, not
a controlled performance measurement. The command used a 30s per-request wall
cap and the same 100-by-1,000 input and seeds as the CPU smoke:

```bash
python scripts/run_request_bench.py --allow-gpu \
  --solvers cuopt-pdlp-1e-4 cuopt-pdlp-1e-6 cuopt-barrier-socp \
  --time-limit 30 --out results/p1_request_timing_gpu_smoke.csv
```

`results/p1_request_timing_gpu_smoke.csv` retains all three cold requests. All
returned solver status `optimal`, without timeout or error, but none passed
every independent acceptance gate. No cached requests or timing ratios were
produced. Use a new filename for any subsequent run.

| Configuration | True sigma TE | Purity gate | Objective consistency | Accepted |
|---|---:|---|---|---|
| PDLP 1e-4, MAD@10bp | 28.29591bp | Fail | Pass | No |
| PDLP 1e-6, MAD@10bp | 28.28596bp | Fail | Pass | No |
| Barrier, sigma@10bp | 9.99997bp | Pass | Fail | No |

Both PDLP portfolios pass the desk gate but slightly exceed their own MAD
constraint at the strict purity threshold. Independently of that small
numerical breach, their true sigma is approximately 2.83 times the target.

The barrier portfolio passes budget, bounds, purity, desk and true-sigma
checks. Its solver objective is -0.0522250161453, versus an independently
recomputed -0.0522239325611. The absolute difference, **1.083584e-6**, exceeds
the unchanged **1e-6** consistency gate. Its independently scored objective is
only 1.112544e-7 away from the same-model cold Clarabel reference; this does not
override the failed consistency gate or certify optimality. The barrier call
uses default solver tolerances. Tolerance and CVaR auxiliary-variable residuals
are hypotheses to investigate, not established causes: this CSV stores the
portfolio, not the returned CVaR auxiliary values. Do not relax the checker or
reinterpret `optimal` as an accepted measurement.

The CLI records instantaneous, device-wide GPU snapshots, not continuous
foreground-workload exclusion. Post-request readings of 65-86% may include
the solver itself and do not diagnose external contention. No stable CPU/GPU
or cold/warm speed claim follows from this smoke.

## Pilot verification before manuscript integration

The counts below record the pilot validation. The new manuscript/package build
has a separate current checklist in the private release handoff; do not treat
these earlier checks as verification of a later package.

- Full suite: **393 passed**, with 29 upstream CVXPY bounds warnings.
- After the GPU smoke: request-timing and bond-lot regression tests **55 passed**,
  with 16 upstream CVXPY bounds warnings. No solver code or gate was changed.
- Two independent code reviews, followed by focused re-reviews of all findings.
- Nine saved lot portfolios independently repriced and checked for budget,
  CVaR, objective and sigma; source hashes match.
- Six saved CPU/GPU request portfolios independently rescored for CVaR,
  objective, expected excess and sigma; saved objective discrepancies and
  mandate gates reconcile. Source hashes and single-request totals agree.
- Strict existing paper-number generation: **529 macros**, exit 0. The check
  wrote a private scratch output, not the published paper's inputs.
- At the pilot-only milestone, historical CSVs, the PDF and source archive
  were unchanged. The current manuscript integration regenerates the local PDF
  and source archive; it does not edit any measured result file.
- All work is local. No push, new public repository, release, submission or post.

## Remaining research and publication boundaries

- [x] Review the CPU pilot CSVs and full residuals, not just solver status.
- [x] Run the small new cuOpt timing lane after the quiet-window preflight.
- [ ] Diagnose the barrier objective discrepancy with auxiliary residuals and
      explicit solver tolerances before collecting an accepted GPU baseline.
      Preserve the first smoke and keep the independent gates unchanged.
- [ ] If integer sigma compliance is desired, separately design and test a
      conservative approximation or MISOCP approach. Linear repair is not one.
- [ ] Freeze the single-request timing protocol and run matched, isolated
      repeats on the exact cells chosen for a new paper claim.
- [ ] Keep timing observations separate from market or investment evidence.
- [x] Include the measured pilots and all failed gates in the current local
      manuscript candidate without adding a comparative timing claim.
- [ ] Fred reviews the candidate and personally performs final publication.

The existing published PDF and source archive remain unchanged. The current
local candidate includes these pilots with their limits; resolving the GPU
baseline is required for a new speed claim, not for reporting these outcomes.

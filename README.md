# Credit portfolio optimization with cuOpt

An open benchmark of **cuOpt for continuous corporate-bond portfolio
construction**, with spread/default scenarios and credit-specific mandates.

> Constraint delivery comes before solver speed.

The paper is titled **Credit Portfolio Optimization with cuOpt: A
Reproducible Benchmark of Risk, Constraints, and Solver Endpoints**. It studies a continuous corporate
bond Mean-CVaR problem with rating, sector, issuer, duration, DTS, turnover,
and tracking-error constraints. This is an application benchmark, not a new
solver or a real-market performance study. Main experiments use synthetic
securities and controlled scenarios; the N-PORT universe is an optional
real-instrument extension, not the source of the headline measurements.

[Paper and versioned artifacts](https://github.com/FredHuFuture/credit-cuopt-benchmark/releases)
are distributed with each release. This repository starts from a reviewed
snapshot; earlier development history and private correspondence are not
published.

The current local revision candidate includes the supplemental experiments in
[NEXT_VERSION.md](NEXT_VERSION.md): single-request timing and a hypothetical
bond-lot repair pilot, now reported in the paper's Appendix B. They add
implementation evidence, not a new speedup or real-market claim. The previously
published v1.0.0 assets are unchanged; this candidate awaits Fred's publication.

## What the P0 review changed

The original draft overreached. A sigma tracking-error mandate is an SOC
constraint, but that does not mean cuOpt or GPU optimization cannot represent
it. cuOpt 26.06 has a native conic barrier path, and classical lifted
polyhedral approximations also exist. MAD is one inexpensive LP proxy, not the
only alternative.

P0 therefore compares four endpoints:

- raw first-order point;
- minimum-distance repaired continuous point;
- crossover/basic solution;
- exact conic solution.

Public ratios compare the same formulation and endpoint, with matching
independent feasibility and objective-recomputation gates. They do not certify
identical distance to the mathematical optimum. Best-feasible objective gaps
are reported separately. An archived `gpu_contended` flag excludes SOCP
timings, even when the returned portfolio is valid.

## Main findings

### 1. Naive MAD calibration fails; recalibration works at a cost

On the 500-bond by 20,000-scenario working problem:

- MAD with a nominal 10bp budget delivers 28.55bp sigma tracking error, a
  2.85x miss.
- The selected recalibrated budget is 2.25bp and delivers 9.47bp in-model.
- Three fixed validation draws deliver 5.92bp to 7.32bp.
- Relative to exact SOCP, recalibrated MAD gives up 0.34bp/month of expected
  excess return; CVaR loss increases by 4.43bp and the scaled objective falls
  by 13.6bp. The objective is `mu @ w - 3 * CVaR_0.95(w)`; scaling means
  multiplication by 10,000, not an observed-return claim.

The conclusion is a cost-accuracy trade-off, not “the mandate is unavailable.”

Evidence: `results/p0_te_recalibration.csv`,
`results/p0_socp_working.csv`.

### 2. Native SOCP works at reference size

At 300 bonds by 3,000 scenarios, both cuOpt barrier and Clarabel independently
deliver 10bp sigma TE. The source file explicitly flags GPU contention, so
**its timings and the earlier 7.67x ratio are withdrawn**. This validates
native conic feasibility only.

At working size, Clarabel still delivers the exact 10bp mandate. Its repeat
session was contaminated by foreground gaming/video and swap pressure, so it
is feasibility evidence only. The native cuOpt path records a
`ERROR:NATIVE_SEGFAULT` in `libcudss.so.0`; it was run once and not converted
into a timing claim.

Evidence: `results/p0_socp_reference.csv`,
`results/p0_socp_working.csv`.

### 3. MAD-form LP feasibility does not require crossover here

For MAD@10bp:

- loose PDLP and its linear repair clear the desk gate but fail strict purity;
- tight PDLP returns a MAD-form LP-feasible continuous point with a 63.18s
  steady-state solve median (excluding generation, build and first solve);
- crossover/basic hits the 1,800s cap.

**That point still has 28.55bp sigma TE and fails the original 10bp target.**
The endpoint result concerns only the MAD proxy's constraints. This paper does
not call a vertex “tradeable” or claim that a basic solution is
a desk requirement. Bond lots, minimum trades, and cardinality are a separate
mixed-integer implementation layer.

Evidence: `results/p0_endpoint_ladder.csv`.

### 4. Turnover contributes, but does not explain all crossover cost

The paired leave-one-out rerun measures crossover overhead (crossover-enabled
minus raw solve medians, not an internal phase timer) at 1.03s for a
budget-and-box problem, 19.01s for the full mandate, and 6.53s after removing
turnover. Turnover’s paired marginal is 12.48s, or 65.7% of full overhead.

The pre-specified verdict is **partial contribution**. The old 87.8% causal
claim is retired.

Evidence: `results/p0_ablation_loo.csv`.

### 5. The QP result is a solver-stack comparison

The matched-feasibility sweep contains 28 independently eligible rows but only 12
rows with matched raw endpoints that may form ratios. Clarabel rows repaired
on CPU and raw cuOpt rows on GPU are reported separately, not divided.
At repaired weights, the objective is recomputed; the zero discrepancy field
does not establish agreement with a solver-reported original-QP objective.
Published ratios divide GPU by CPU steady-state solve medians.

Valid GPU/CPU ratios change with size and covariance representation. The
largest solved factored GPU cell has 15,000 names. At 29,222 names, dense arms
are refused by the harness’s host-memory policy and factored arms record
scenario-generation OOM. These are workstation outcomes, not claims that the
mathematical formulation ceases to exist.

Evidence: `results/p0_qp_matched.csv`.

## Supplemental pilots in this revision

- **Single-request timing:** separate build, canonicalization, solve, repair
  and validation, with a 30s request watchdog and explicit cold/cached modes.
  The 100-bond by 1,000-scenario GPU smoke retains all three `optimal` rows,
  but none passes all independent gates. PDLP/MAD delivers about 28.29bp
  against the 10bp sigma target. Native SOCP delivers 9.99997bp and satisfies
  the mandate, but its objective discrepancy is 1.083584e-6 against the 1e-6
  gate. No new GPU, warm-start or CPU/GPU speed ratio is claimed.
- **Cash-aware bond lots:** a 60-bond by 500-scenario CPU pilot compares
  continuous SOCP, rounding and L1 MILP repair at three NAVs. Quantity and
  linear compliance do not ensure sigma compliance. The $10M repair has an
  independently feasible capped incumbent, not certified L1 optimality.
  Lot sizes are hypothetical; this is not an executable bond-trading system.

Evidence: `results/p1_request_timing_cpu.csv`,
`results/p1_request_timing_gpu_smoke.csv`, and `results/p1_bond_lots_cpu.csv`
with its metadata/portfolio sidecars. Appendix B and `NEXT_VERSION.md` give
the complete outcomes, timing boundary, gates and remaining limitations.
The unresolved barrier objective discrepancy is disclosed as a failed
acceptance outcome, not hidden or used in a comparison.

## Real-instrument universe

`src/nport.py` and `scripts/fetch_nport.py` derive a bond universe from SEC
Form N-PORT public-domain filings. The pipeline records source URL, hashes,
filter stages, report date, fitted Treasury curve, proxy fields, and output
hashes. N-PORT supplies real instruments and characteristics, not a production
return covariance or transaction history.

Licensed TRACE, WRDS, FISD, ICE, Bloomberg, and ETF holdings extracts are not
redistributed. Optional runtime sources remain subject to their own terms.

## Reproduce

### CPU-only artifact checks (no GPU experiment rerun)

On Python 3.12 under Linux or WSL2:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-cpu.txt
source .venv/bin/activate
```

To typeset the paper, install TeX Live and Poppler (Ubuntu/Debian):

```bash
sudo apt install texlive-latex-base texlive-latex-recommended \
    texlive-fonts-recommended texlive-science lmodern poppler-utils
```

Then, from the repository root:

```bash
python scripts/make_paper_numbers.py
python -m pytest tests/ -q
bash paper/build.sh
bash paper/make_arxiv.sh
```

`make_paper_numbers.py` reads committed CSVs and generates
`paper/numbers.tex`. Strict mode exits nonzero if a requested measurement is
missing. Recorded DNF, OOM, native error, and policy refusal remain printable
non-results.

Strict PDF checks reject placeholders, undefined references/citations,
overfull boxes, missing figures, and Type 3 fonts. The source tarball is
rebuilt in an isolated temporary directory before packaging. This verifies
the saved artifact; it does not rerun the benchmark or independently validate
the risk model.

### GPU measurements (optional, potentially hours)

`requirements.lock` is the full measured Linux/WSL2 CUDA-12 environment,
including cuOpt 26.6.0 and CVXPY 1.9.2. It is not a portable Windows lockfile.
Use a separate Python 3.12 environment and install with NVIDIA's package
index available:

```bash
python -m pip install --extra-index-url https://pypi.nvidia.com -r requirements.lock
python scripts/run_te_mandate.py --help
python scripts/run_qp_branch.py --help
```

Consult `RUNBOOK.md` and the producer commands embedded in generated
`paper/numbers.tex`. Inspect caps and available RAM/VRAM first. Write reruns
under `results/tmp/` or a new result filename; do not overwrite release CSVs.
The published working-size native SOCP failure is not a reason to repeat an
unbounded failing session.

The six P0 evidence files are:

- `results/p0_socp_reference.csv`
- `results/p0_socp_working.csv`
- `results/p0_te_recalibration.csv`
- `results/p0_endpoint_ladder.csv`
- `results/p0_ablation_loo.csv`
- `results/p0_qp_matched.csv`

Historical result files remain committed for auditability. `ERRATA.md` records
claims that moved or were retired.

## Repository map

- `src/`: universe, credit model, scenarios, optimization, checking
- `scripts/`: experiment runners, plotting, provenance generation
- `tests/`: unit and regression tests
- `results/`: committed measurements and figures
- `paper/`: LaTeX source and strict build scripts
- `private/`: ignored reviewer material and draft PDFs

## Scope and limitations

- Main benchmark is a continuous relaxation. The supplemental lot-repair
  pilot does not enforce the sigma cone or model cardinality or executable
  trade lists.
- Controlled scenario generator, not a validated production risk model.
- One CPU/GPU workstation and one software stack.
- Solver-stack comparisons are not pure hardware causal effects.
- Independent scenario draws test generator robustness, not live-market
  validity.

## Publication status

This is a technical working paper and reproducible artifact, not a
peer-reviewed acceptance or a production investment system. GitHub release
does not imply arXiv or SSRN submission. Further empirical work may add
licensed bond-return/risk data, out-of-time market validation, and practitioner
mandates. Those extensions will be versioned separately.
Private correspondence is excluded from public releases.

## License

Code is licensed under Apache-2.0. Third-party data and software retain their
own licenses and terms.

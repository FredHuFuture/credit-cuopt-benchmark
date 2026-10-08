# Initial working paper and artifact, v1.0.0

**Credit Portfolio Optimization with cuOpt: A Reproducible Benchmark of Risk,
Constraints, and Solver Endpoints** — Fred Hu.

This release studies continuous corporate-bond portfolio construction with
credit-specific scenarios and limits. It is a technical working paper, not
peer-reviewed acceptance, an investment recommendation, or a production
trading system. No arXiv/SSRN publication or identifier is claimed.

## Scope

- Controlled monthly spread/default scenarios and rating, issuer, sector,
  duration, DTS and turnover constraints.
- Naive and recalibrated MAD versus a directly modeled sigma-TE constraint.
- Raw, repaired and basic LP endpoints, paired turnover leave-one-out, and
  matched-feasibility CPU/GPU QP comparisons.
- Independent scoring, original measurement CSVs, tests and reproducible paper
  source. Main experimental results use synthetic securities. The optional
  N-PORT adapter does not turn those results into real-data evidence.

## Important limitations and corrections

Both SOCP datasets are feasibility evidence only; contaminated timings are
excluded. A tight PDLP solve can satisfy the MAD-form LP while still failing
the original sigma target. QP ratios compare solver stacks and feasibility,
not pure hardware effects or equal certified optimality. Native failures,
resource limits and missing comparisons remain disclosed.

The repository is a reviewed snapshot, not a publication of private
development history or reviewer correspondence. The initial artifact leaves
licensed-data calibration, empirical out-of-time validation, practitioner
mandate review, and bond-lot implementation to later versions.

## Reproduction

See README.md for CPU-only checks and optional GPU reruns. The PDF and verified
LaTeX source tarball are attached to the release. `SHA256SUMS.txt` identifies
the release binaries. The source tarball is suitable for further submission
preparation; packaging it does not submit it to arXiv.

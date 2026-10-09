# Working paper and artifact revision, v1.1.0 candidate

**Credit Portfolio Optimization with cuOpt: A Reproducible Benchmark of Risk,
Constraints, and Solver Endpoints** by Fred Hu.

Prepared locally for Fred's review and final publication. This file does not
assert that v1.1.0 is live. The existing public v1.0.0 assets remain unchanged.
No arXiv/SSRN identifier, peer-reviewed acceptance, or practitioner endorsement
is claimed.

## What changed

- Appendix B incorporates a 100-bond, 1,000-scenario single-request timing
  pilot, with explicit build/canonicalization/solve/validation boundaries,
  cold/cached modes, per-request caps and saved portfolios.
- Three cuOpt smoke requests return `optimal`, but none passes every
  independent gate. MAD points breach true sigma; the SOCP portfolio satisfies
  the mandate but narrowly misses objective agreement. All failures remain
  visible. There is no new GPU or warm-start speedup claim.
- A 60-bond, 500-scenario CPU quantity pilot adds cash, hypothetical lot,
  holding and trade minima, naive rounding and L1 MILP repair at three NAVs.
  Repair enforces quantities and the linear mandate, not the sigma cone.
  A capped feasible incumbent is explicitly not certified optimal.
- Generator-derived paper numbers, regression tests, original result CSVs,
  metadata and portfolio sidecars accompany the new text. PDF date and
  document metadata are explicit; source packaging retains strict checks.
- The manuscript discloses substantive assistance from Claude/Claude Code
  and OpenAI Codex, including literature discovery, experimental planning,
  coding, analysis and writing. It distinguishes executed measurements from
  model-supplied numerical answers and retains author responsibility.
  The author line identifies an independent researcher.

## What did not change

The main P0 results, historical CSV bytes and prior release assets are
preserved. This remains a controlled credit/cuOpt application benchmark,
not a new solver, an executable bond-trading system or market-data evidence.
The optional N-PORT adapter does not supply the main experimental inputs.

The unresolved small-smoke barrier objective discrepancy requires further
diagnosis before an accepted GPU timing baseline can be claimed. It does not
invalidate reporting the failed outcome. Licensed risk models, security-level
quantity conventions, liquidity, costs, cardinality and integer sigma-cone
optimization remain future work.

## Included artifact and reproduction

The repository contains source, CPU/GPU runners, tests, archived results,
quantity sidecars and paper sources. README.md gives CPU-only checks and
optional GPU replication; NEXT_VERSION.md describes the supplemental protocol.
Do not overwrite archived result files or pool pilot times with the old grid.

The prepared release assets are the revised PDF, independently verified
minimal LaTeX source archive and SHA256SUMS.txt. The source archive is for
arXiv preparation; it is not the entire code artifact and does not submit the
paper. Fred performs all repository/release uploads and paper submissions.

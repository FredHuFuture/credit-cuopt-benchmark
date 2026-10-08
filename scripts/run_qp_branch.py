#!/usr/bin/env python
"""M7 — the QP branch: does a GPU change the answer to the Hessian problem?

THE HYPOTHESIS UNDER TEST
-------------------------
Everything this repo has measured so far lives on the LP branch: Mean-CVaR via
the Rockafellar-Uryasev linearization, solved with cuOpt's PDLP. That is the
branch where cuOpt's headline innovation actually is — a GPU first-order
method — and it is NOT the branch most production optimizers run. Industry
practice puts VARIANCE, or tracking error against a benchmark, in the
objective. Expected shortfall wins the report; variance wins the optimizer.

So the LP branch measures the solver NVIDIA is proud of on a problem desks
mostly do not solve, and this repo has zero measurement of the one they do.
That is the largest remaining hole in the evidence, and three facts make it
worth a controlled experiment rather than an assumption:

1. **cuOpt solves QP with BARRIER ONLY.** Mean-variance inherits none of the
   PDLP innovation. It gets an interior-point method — the same algorithm
   class CPU solvers have shipped for thirty years. Whatever advantage exists
   has to come from faster dense linear algebra inside each Newton step, not
   from a new algorithm.
2. **cuOpt QP does not support crossover.** On the LP branch this repo's
   central negative finding was that a first-order interior solution breaches
   the desk limits while reporting "Optimal", and that crossover is one way
   to ask for a basic-solution endpoint. Here there is no crossover to run. The
   mandate-compliance problem is therefore structurally WORSE on the QP
   branch, and that is a measurable outcome, not a caveat.
3. **NVIDIA has published no QP benchmark.** Their own cuFOLIO blueprint
   ships a mean-variance workflow built on cuOpt (`src/mean_variance_
   optimizer.py`), but every headline number — the speedup claims, the
   benchmark harness — is scoped to Mean-CVaR under `solver_method="PDLP"`.
   No named CPU solver, no asset count, no reproducible QP timing anywhere.
   This script measures something the vendor's own blueprint leaves unmeasured.

THE AMUNDI FRAMING (what makes this a question and not a drag race)
-------------------------------------------------------------------
Amundi's published position is that a covariance objective is not usable at
index scale, and the reason given is Hessian dimensionality: **29,222
securities**. A dense covariance over that universe is 6.36 GiB before any
solver touches it, and an interior-point method has to factor a system of that
order every Newton step. The industry answer has never been "buy a bigger
machine" — it is to stop forming the dense matrix at all and keep risk as
`Sigma = B F B' + D`, low-rank plus diagonal, which is what every commercial
risk vendor ships.

    So: does a GPU repeal the objection, or does the modelling choice?

THE THREE-WAY COMPARISON
------------------------
One universe. One limit sheet. One covariance. Only the representation and
the solver stack change:

    (a) cuopt_barrier_dense     dense Sigma, cuOpt barrier            GPU
    (b) clarabel_dense          dense Sigma, Clarabel interior-point  CPU
    (c) clarabel_factored       Sigma = B diag(f) B' + diag(d)        CPU
    (d) cuopt_barrier_factored  the same factored form, on the GPU

Arm (b) is the named CPU peer and it is an interior-point method, which makes
(a)-vs-(b) barrier-vs-barrier — the closest algorithmic match available in the
locked environment. Arm (c) is the vendor formulation. Arm (d) exists because
the factored form is still a QP, so it costs one extra configuration to learn
whether the GPU helps once the Hessian is no longer dense.

WHAT EACH POSSIBLE OUTCOME WOULD MEAN
-------------------------------------
- **(a) wins, and clears N = 29,222.** The cuOpt dense stack reaches the
  Amundi scale outright. This is the dense cuOpt-stack outcome, and the memory
  arithmetic says it is unreachable on this box (see below).
- **(a) wins at moderate N but dies before index scale.** The honest reading:
  the dense cuOpt stack moves the ceiling but does not remove it. The
  objection survives in weakened form.
- **(b) wins.** cuDSS's sparse direct machinery does not like a 100%-dense
  Hessian, and a mature CPU IPM on dense linear algebra is simply the better
  tool. A real result, and a boring one for the GPU.
- **(c) wins, and is the only arm that reaches index scale.** The most
  valuable outcome available, and the one to be most suspicious of wanting:
  it would say the answer is a MODELLING choice, not a solver-stack choice — the
  Amundi objection is repealed by the representation, on a CPU, at a fraction
  of the memory. That is on-brand for this repo and it is also what two
  independent 2026 GPU portfolio-optimization papers found by accident (both
  reach scale through factor compression, not through the card).
- **(d) beats (c).** The factored form is small enough that the GPU should
  have nothing to do; if it wins anyway the advantage was never about the
  Hessian.

**The design does not favour any of these.** Under the default
`--sigma-basis factor`, arms (a)-(d) solve the MATHEMATICALLY IDENTICAL
problem: the dense arms receive `B diag(f) B' + diag(d)` materialized as a
dense N x N matrix, and the factored arms receive the same three pieces
unmultiplied. Same optimum, same optimal weights, verified to 1e-9 in
`tests/test_qp_model.py`. Any timing difference is therefore a solver-stack
comparison at a fixed representation or a representation comparison within a
fixed solver stack, not a pure device-causal estimate. That is the whole
reason the
factored form is the DEFAULT basis rather than a competing risk model.

`--sigma-basis sample` runs the other, confounded comparison: dense arms get
the sample covariance of the scenario matrix, factored arms get the fitted
factor approximation of it. That version is closer to what a desk faces (the
factor model IS an approximation) but arm (c) then wins partly by solving an
easier, different problem. The confound is measured rather than argued about:
`sigma_rel_fro_err` and `portvol_rel_err_median` are written into every row,
so a (c) win under `sample` reads as "X times faster at a measured Y% risk-
model error", never as a free lunch.

WHERE THE COVARIANCE COMES FROM
-------------------------------
From the same scenario draw the CVaR branch consumes, which keeps the LP and
QP branches comparable (same universe, same seeds, same limits). The factor
structure is not invented for this script: `src/scenarios.py` IS an explicit
linear factor model, so the exposures are known analytically rather than
fitted blind. The credit return is

    r_i = carry_i - SD_i * OAS_i * (shock_i) + 0.5 * C_i * (...)^2

with `shock_i = (credit_sys + rating_factor + sector_factor + idio) * hy_mult`,
so bond i's loading on every systematic shock block is

    beta_i = -SD_i * OAS_i * hy_mult_i

which is -DTS_i/1e4 scaled by the HY multiplier — the same quantity the DTS
limit in `src/credit.py` is written on. The risk model and the limit sheet
therefore tell the same story, which is not something a bolted-on PCA would
give us.

F and D are ESTIMATED from the realized scenarios, never taken from the config
constants: the generator applies a mean-preserving widening skew that inflates
realized systematic vol roughly 18% above `credit_sys_vol`, so an analytic F
would be wrong by that much. F is then eigendecomposed and folded into B, so
the internal representation always has a DIAGONAL factor covariance. That is
exact (not an approximation) and it is what makes the factored Hessian
strictly diagonal — which matters far beyond this script: a diagonal Hessian
is the only quadratic objective OR-Tools' PDLP accepts, so the factored MVO is
natively solvable by a first-order method while the dense one is solvable by
no PDLP at all, cuOpt's included. The modelling choice is what puts the
problem back inside the fast solver class.

Two terms the linear factor model cannot carry, stated rather than hidden: the
convexity term is quadratic in the shocks, and issuer default jumps REPLACE
the return instead of perturbing it. Measured cost of both on a 300x8,000 CPU
draw: 14.6% relative Frobenius error against the sample covariance, and 3.4%
median relative error in portfolio volatility across 200 random portfolios.
Those are the numbers `--sigma-basis sample` writes into every row.

WHAT THIS DOES NOT CONTROL
--------------------------
- **(a) and (b) share a matrix, not an implementation.** cuOpt's barrier is a
  predictor-corrector over cuDSS sparse Cholesky; Clarabel is a homogeneous
  self-dual embedding over its own sparse LDL. Both are interior-point, and
  that is as close as the locked environment gets. This is the same fine
  print `src/bench.py` already carries for PDLP-vs-PDLP, and it is disclosed
  in the CSV header rather than glossed.
- **Raw solver settings are not pure device effects.** cuOpt's barrier
  and Clarabel are different implementations. P0 therefore reports a
  `solver_stack_ratio` only when both rows hit the same endpoint and quality
  target, and the strict Clarabel arms additionally run at 1e-9 feasibility
  and gap tolerances.
- **Canonicalization is measured but not eliminated.** One CVXPY model
  definition drives every arm, which is this repo's standing methodological
  commitment. The cost of that at a dense N x N P is real and grows as N^2
  (measured locally: 0.4s at N=1,500 for CLARABEL stuffing). It is reported
  in its own `compile_s` column, NOT folded into the solve time, and it is
  the reason a native-DataModel control run is pre-registered in TODO.md.
- **Arm (c) under `--sigma-basis sample` changes the risk model as well as
  the representation.** Confounded by construction; see the error columns.

HOW TO READ THE OUTPUT (results/qp_branch.csv)
----------------------------------------------
One row per (n_bonds, arm). The columns that carry the argument:

    n_bonds, arm, form         the cell; form is dense | factored
    hardware                   GPU | CPU
    status                     solver status, or a refusal (see below)
    wall_s_median              median of `--runs` timed solves, warmup discarded
    wall_s_min / wall_s_max    the spread, disclosed
    solver_internal_s          solver-reported time where exposed
    compile_s                  CVXPY canonicalization, NOT part of wall_s
    sigma_build_s              cost of materializing the dense Sigma (dense
                               arms only) — the price of the representation,
                               kept separate so neither side hides it
    objective                  solver-reported objective
    obj_recomputed             numpy recomputation from the weight vector
    obj_abs_gap                |recomputed - reported|; the accuracy column
    quad_nnz                   nonzeros in the Hessian actually handed over
    q_gib_dense / q_gib_arm    dense-equivalent vs what this arm really needs
    host_peak_est_gib          the memory guard's estimate for this cell
    n_positions, n_dust        an interior point with no crossover is DENSE.
                               A PM cannot trade 29,222 dust positions, so
                               these two columns are a mandate result, not a
                               diagnostic.
    worst_rel_violation        worst breach of the limit sheet (0 = clean)
    limits_verdict             "ok" or which limit broke and by how much
    respects_mandate           the headline. cuOpt QP cannot crossover, so
                               this column is where the no-crossover cost
                               shows up if it shows up anywhere.
    sigma_rel_fro_err          factored-vs-dense risk model error (0 under
    portvol_rel_err_median     the default factor basis, by construction)

Refusal statuses are RESULTS, not failures, and are never silently dropped:

    skipped_host_memory     the cell's dense Sigma would not fit in host RAM
    skipped_device_memory   ...or in the card's VRAM
    GPU-UNREACHABLE         cuOpt is not installed, or its CVXPY interface
                            does not accept a quadratic objective in this
                            pinned version. LOUD: the run exits non-zero.
    DNF                     ran, did not converge or timed out

The single most important cell is the one where the dense arms stop and the
factored arm keeps going. Plot all four arms against n_bonds with the dense
arms terminating at their refusal point; that picture is the experiment.

RUNNING IT (needs the GPU box for arms (a) and (d))
---------------------------------------------------
    python scripts/run_qp_branch.py --out results/qp_branch.csv

    # plumbing check first, minutes not hours, no GPU needed
    python scripts/run_qp_branch.py --grid 300,600 --runs 1 --cpu-only \
        --out results/qp_branch_small.csv

    # the crossover question on its own, one tiny solve
    python scripts/run_qp_branch.py --probe-only --grid 300

BEFORE TRUSTING ANY TIMING — the Q-convention calibration
----------------------------------------------------------
cuOpt's quadratic objective is `x'Qx` with NO 1/2 factor and Q internally
symmetrized to Q + Q^T; CVXPY's convention is `(1/2)x'Px`, so its cuOpt
interface passes P/2. Get that wrong in either direction and the risk aversion
is silently off by 2x — no crash, no warning, and every timing and every
mandate verdict in the CSV is then attached to a problem nobody asked for.

So the first thing each worker does is solve a 2-variable QP whose optimum is
known in closed form (derived in CALIBRATION below: p* = 0.2625, objective
-0.8121875 exactly) through the SAME code path the cell will use, and refuses
to proceed if the objective disagrees. A doubled risk aversion moves that
objective to -1.68734375, which is not a subtle miss. `--skip-calibration`
exists for debugging and should never be used for a published run.

ISOLATION AND THREAD PINNING
----------------------------
One clean subprocess per (cell, arm), self-spawned, following
`scripts/run_ablation.py` rather than `scripts/bench_worker.py` because this
experiment additionally needs the exposure report and a QP-specific checker
that the LP worker has no reason to carry. Timing is re-implemented locally
for a reason stated once: `src.bench.run_spec` is typed to `CVaRModel` and
unconditionally recomputes CVaR from a scenario matrix, which a scenario-free
QP does not have — it would report a spurious objective disagreement on every
row. Everything else comes from `src/bench.py` unchanged: `assert_gpu_idle`,
`capture_environment`, `write_csv`, the warmup-discard/median-of-N protocol
and the one-process-per-configuration rule.

Isolation matters MORE here than on the LP branch, not less. The dense arms
run near the host memory ceiling by design, so an arm that gets OOM-killed
must cost one configuration rather than the run. CPU thread counts are pinned
explicitly (OMP/OpenBLAS/MKL/Rayon) because Clarabel and the BLAS underneath
it will otherwise take the whole box, and an unpinned thread count silently
decides the solver-stack comparison.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

MARKER = "RESULT_JSON:"

#: Bond counts to sweep. The last entry is Amundi's published index-scale
#: number and is expected to be REFUSED by the dense arms on this hardware —
#: that refusal is the experiment's punchline, so the cell stays in the grid
#: and is reported with its reason rather than quietly omitted.
AMUNDI_INDEX_N = 29_222
DEFAULT_GRID: tuple[int, ...] = (
    500, 1_000, 2_000, 3_000, 5_000, 7_500, 10_000, 15_000, AMUNDI_INDEX_N,
)

#: Host bytes per dense-Hessian ENTRY at peak. Derivation: the fp64 matrix
#: itself (8 B), plus the scipy COO intermediate and the CSR copy CVXPY builds
#: on the way to the solver, both of which are alive simultaneously (~12 B per
#: nonzero, roughly three copies in flight). 48 is deliberately on the
#: pessimistic side of that arithmetic.
#:
#: This is an ESTIMATE and must be recalibrated from the first real OOM, the
#: way `scripts/run_sweep.py:BYTES_PER_NNZ_PEAK` was recalibrated from a
#: measured 8.5 GB RSS after the arithmetic that predicted 48 was wrong by
#: half. Until then, treat the refusal boundary as approximate and check
#: `host_peak_est_gib` against what the process actually reached.
DENSE_Q_BYTES_PER_ENTRY_PEAK = 48

#: Host bytes per nonzero for the factored form, same convention as
#: run_sweep.py's calibrated figure. The factored arms are three orders of
#: magnitude smaller than the dense ones, so this number never binds; it is
#: here so the guard is symmetric and no arm gets a free pass.
FACTORED_BYTES_PER_NNZ_PEAK = 120

#: Device-side multiple of the dense matrix: Q, plus a cuDSS factor of a fully
#: dense N x N block, plus KKT and workspace. NVIDIA documents barrier as
#: sparse Cholesky with dense-column elimination, but that machinery targets
#: the CONSTRAINT matrix, not a 100%-dense Hessian, so 3x is a floor and not a
#: promise. This is the single most likely way the GPU dense arm dies earlier
#: than predicted; see openQuestions.
DEVICE_Q_MULTIPLE = 3.0

#: Fraction of each budget a cell is allowed to claim. 0.50 for the host, and
#: for the same reason `run_sweep.py` uses 0.50 rather than 0.85: WSL2 grows
#: its VM on demand, so a cell that fits in guest-visible free memory can
#: still be killed when Windows cannot satisfy the growth — which takes the
#: orchestrator down with no in-guest OOM record.
HOST_BUDGET_FRACTION = 0.50
DEVICE_BUDGET_FRACTION = 0.60

#: Numerical floor for a factor variance to count as a live direction. The
#: analytic exposure matrix is rank-deficient by construction (the rating
#: dummies and the sector dummies each sum to the systematic column), so the
#: eigendecomposition of F returns a couple of exactly-zero directions.
#: Dropping them is EXACT — a direction with zero variance contributes
#: nothing to Sigma — and it makes the reported factor count honest.
FACTOR_EIG_FLOOR = 1e-14

#: Positions below this are dust a desk cannot trade. Same threshold the LP
#: branch's checker uses, imported rather than redefined.
from src.check import DUST_THRESHOLD  # noqa: E402

OBJECTIVE_REL_TOL = 1e-6

P0_SCHEMA_KEYS: tuple[str, ...] = (
    "n_bonds", "arm", "form", "hardware", "solver", "method_class",
    "endpoint", "requested_endpoint", "quality_target", "solver_config_id",
    "achieved_quality_target", "risk_model_id", "math_instance_id",
    "repair_mode", "resume_key", "status", "status_source", "dnf",
    "ratio_eligible", "ratio_exclusion_reason", "solver_stack_ratio",
    "solver_stack_ratio_peer", "solver_stack_ratio_exclusion_reason",
    "best_feasible_objective", "relative_best_objective_gap",
    "common_quality_time_s", "wall_s_median", "repair_s",
    "repair_status", "repaired_from_endpoint", "check_ok", "check_notes",
    "respects_mandate", "objective", "objective_source", "obj_recomputed",
    "obj_abs_gap", "objective_gate_rel_gap",
    "obj_rel_gap", "worst_rel_violation", "worst_limit", "limits_verdict",
)


# --------------------------------------------------------------------------
# The Q-scaling calibration
#
# max  mu'x - (lam/2) x'S x   s.t.  sum(x) = 1,  0 <= x <= 1
# with mu = (0.10, 0.05), S = [[4, 1], [1, 2]], lam = 1.
#
# Substituting x = (p, 1-p) and expanding:
#     x'Sx  = 4p^2 - 2p + 2
#     obj(p) = 0.05 + 0.05p - 0.5(4p^2 - 2p + 2) = -2p^2 + 1.05p - 0.95
#     obj'(p) = -4p + 1.05 = 0  ->  p* = 0.2625, interior, so the box and the
#                                   nonnegativity are both slack
#     obj(p*) = -0.8121875   exactly
#
# If a code path doubles the risk aversion (the classic x'Qx-vs-(1/2)x'Px
# mistake) the objective becomes -4p^2 + 2.05p - 1.95, minimized at
# p = 0.25625 with value -1.68734375. The WEIGHTS barely move; the OBJECTIVE
# moves by more than a factor of two. So the objective is the gate here, and
# the weight check is a secondary tell.
# --------------------------------------------------------------------------

CALIBRATION_MU = np.array([0.10, 0.05])
CALIBRATION_S = np.array([[4.0, 1.0], [1.0, 2.0]])
CALIBRATION_LAMBDA = 1.0
CALIBRATION_P_STAR = 0.2625
CALIBRATION_OBJ_STAR = -0.8121875
#: What the objective becomes if the 1/2 factor is applied twice or not at
#: all. Recorded so a failure message can say which way it went wrong.
CALIBRATION_OBJ_IF_DOUBLED = -1.68734375


# --------------------------------------------------------------------------
# Factor model — built from the structure src/scenarios.py actually uses
# --------------------------------------------------------------------------

@dataclass
class FactorModel:
    """Sigma = B diag(f) B' + diag(d), with f already diagonalized.

    `B` is (N, K) after the eigenvectors of the estimated factor covariance
    have been folded in, so `f` is a vector rather than a matrix and the
    quadratic form in (w, y=B'w) has a strictly DIAGONAL Hessian. Folding the
    eigenvectors into B is an exact rewrite, not an approximation.
    """

    B: np.ndarray                  # (N, K) exposures, orthogonalized
    f: np.ndarray                  # (K,) factor variances, nonneg, descending
    d: np.ndarray                  # (N,) specific variances, positive
    labels: list[str]              # original factor block names, pre-rotation
    n_dropped: int                 # zero-variance directions removed (exact)

    @property
    def n_assets(self) -> int:
        return self.B.shape[0]

    @property
    def n_factors(self) -> int:
        return self.B.shape[1]

    @property
    def quad_nnz(self) -> int:
        """Nonzeros in the Hessian the solver receives: diag(f) and diag(d)."""
        return self.n_factors + self.n_assets

    def dense(self) -> np.ndarray:
        """Materialize Sigma densely. This is the expensive representation."""
        S = (self.B * self.f) @ self.B.T
        S[np.diag_indices_from(S)] += self.d
        return 0.5 * (S + S.T)          # exact symmetry, not near-symmetry

    def portfolio_variance(self, w: np.ndarray) -> float:
        """w'Sigma w without ever forming Sigma. O(N K), not O(N^2)."""
        y = self.B.T @ w
        return float(y @ (self.f * y) + w @ (self.d * w))


def analytic_exposures(uni, cfg=None) -> tuple[np.ndarray, list[str]]:
    """Bond loadings on the scenario generator's own shock blocks.

    `src/scenarios.py` builds the relative spread shock as

        ds_rel = (credit_sys + rating_factor + sector_factor + idio) * hy_mult

    and the price return as `carry - SD * OAS * ds_rel + convexity term`. So
    the loading of bond i on any of those shock blocks is

        beta_i = -SD_i * OAS_i * hy_mult_i

    This is read off the generator, not fitted, which is why the factor model
    here is defensible as the risk model rather than as a convenient
    compression. The convexity term and the default jumps are genuinely
    outside a linear factor model and land in the residual; their cost is
    measured by `factor_fit_diagnostics`.

    Returns (B, labels) with K = 1 + n_ratings + n_sectors columns. B is
    rank-deficient by construction — the rating columns sum to the systematic
    column and so do the sector columns — which `fit_factor_model` resolves
    exactly rather than by dropping a reference category by hand.
    """
    from src.scenarios import CreditScenarioConfig

    cfg = cfg or CreditScenarioConfig()
    df = uni.bonds
    ratings = df["rating"].to_numpy()
    sectors = df["sector"].to_numpy()
    r_labels = sorted(set(str(x) for x in ratings))
    s_labels = sorted(set(str(x) for x in sectors))

    hy_mult = np.where(np.isin(ratings, ("BB", "B")), cfg.hy_vol_mult, 1.0)
    beta = -uni.spread_duration * uni.oas * hy_mult

    n = len(df)
    K = 1 + len(r_labels) + len(s_labels)
    B = np.zeros((n, K), dtype=np.float64)
    B[:, 0] = beta
    for j, r in enumerate(r_labels):
        m = ratings == r
        B[m, 1 + j] = beta[m]
    off = 1 + len(r_labels)
    for j, s in enumerate(s_labels):
        m = sectors == s
        B[m, off + j] = beta[m]

    labels = (["credit_sys"]
              + [f"rating:{r}" for r in r_labels]
              + [f"sector:{s}" for s in s_labels])
    return B, labels


def fit_factor_model(uni, R: np.ndarray, cfg=None) -> FactorModel:
    """Barra-style fit of Sigma = B F B' + D on the SAME scenarios the LP uses.

    Exposures are analytic (see `analytic_exposures`); only F and D are
    estimated. F must come from the realized factor returns and NOT from the
    config constants: the generator's mean-preserving widening skew inflates
    realized systematic volatility by roughly 18% over `credit_sys_vol`, so an
    analytic F would understate risk by that much.

    F is then eigendecomposed and the eigenvectors folded into B. That is an
    exact rewrite of the same Sigma, and it buys the property this whole
    experiment turns on: with F diagonal and D diagonal, substituting
    y = B'w makes the Hessian in (w, y) strictly diagonal.
    """
    Rc = R - R.mean(axis=0, keepdims=True)
    B, labels = analytic_exposures(uni, cfg)

    # Least squares per scenario. B is rank-deficient, so lstsq's minimum-norm
    # solution is the right choice: it puts zero weight on the null space,
    # which then shows up as exactly-zero eigenvalues below.
    g, *_ = np.linalg.lstsq(B, Rc.T, rcond=None)        # (K, S)
    g = g.T                                              # (S, K) factor returns
    resid = Rc - g @ B.T

    F = np.cov(g, rowvar=False)
    F = 0.5 * (F + F.T)
    d = resid.var(axis=0, ddof=0)

    ev, U = np.linalg.eigh(F)
    ev = np.clip(ev, 0.0, None)
    order = np.argsort(ev)[::-1]
    ev, U = ev[order], U[:, order]

    keep = ev > FACTOR_EIG_FLOOR * max(ev.max(), 1.0)
    n_dropped = int((~keep).sum())
    Bt = B @ U[:, keep]
    f = ev[keep]

    # A strictly positive specific variance keeps Sigma positive DEFINITE
    # rather than merely semi-definite. A singular Hessian is legal for both
    # solvers but it makes "which optimum did you get" ambiguous on a
    # degenerate face, and this experiment compares weight vectors.
    d = np.maximum(d, 1e-18)

    return FactorModel(B=np.ascontiguousarray(Bt), f=f, d=d,
                       labels=labels, n_dropped=n_dropped)


def factor_fit_diagnostics(fm: FactorModel, Sigma_sample: np.ndarray,
                           n_portfolios: int = 200,
                           seed: int = 3) -> dict[str, float]:
    """How much risk model is lost by factoring — measured, not asserted.

    Frobenius error is the headline a quant asks for; the portfolio-volatility
    error is the one that decides whether the difference matters, because that
    is the quantity the objective is built from. Both are reported so a
    factored-arm win under `--sigma-basis sample` can be stated as a speed
    number AT a quantified risk-model error rather than as a free lunch.
    """
    Sf = fm.dense()
    num = float(np.linalg.norm(Sf - Sigma_sample, "fro"))
    den = float(np.linalg.norm(Sigma_sample, "fro"))
    diag_err = float(
        np.abs(np.diag(Sf) - np.diag(Sigma_sample)).max()
        / max(float(np.abs(np.diag(Sigma_sample)).max()), 1e-300))

    rng = np.random.default_rng(seed)
    W = rng.dirichlet(np.ones(fm.n_assets), n_portfolios)
    vf = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", W, Sf, W), 0.0))
    vs = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", W, Sigma_sample, W), 0.0))
    rel = np.abs(vf - vs) / np.maximum(vs, 1e-300)
    return {
        "sigma_rel_fro_err": num / den if den > 0 else float("nan"),
        "sigma_max_rel_diag_err": diag_err,
        "portvol_rel_err_median": float(np.median(rel)),
        "portvol_rel_err_max": float(rel.max()),
    }


# --------------------------------------------------------------------------
# The QP model — one CVXPY definition, two representations
# --------------------------------------------------------------------------

@dataclass
class QPModel:
    """A built mean-variance (or tracking-error) QP plus what checks it."""

    problem: object                 # cp.Problem
    w: object                       # cp.Variable
    form: str                       # "dense" | "factored"
    objective_kind: str             # "mv" | "te"
    mu: np.ndarray
    risk_aversion: float
    w_max: float
    benchmark: np.ndarray | None    # the b in (w - b) for the te objective
    quad_nnz: int
    Sigma: np.ndarray | None        # dense arms only
    factors: FactorModel | None     # factored arms only

    @property
    def n_assets(self) -> int:
        return self.mu.shape[0]

    def weights(self) -> np.ndarray:
        if self.w.value is None:
            raise ValueError("problem has not been solved")
        return np.asarray(self.w.value).ravel()

    def quadratic_value(self, w: np.ndarray) -> float:
        """a'Sigma a for a = w or w - b, from whichever form we hold."""
        a = w if self.benchmark is None else w - self.benchmark
        if self.factors is not None:
            return self.factors.portfolio_variance(a)
        return float(a @ (self.Sigma @ a))

    def objective_value(self, w: np.ndarray) -> float:
        return float(self.mu @ w) - 0.5 * self.risk_aversion \
            * self.quadratic_value(w)


def _active(w, benchmark):
    return w if benchmark is None else w - benchmark


def build_qp_dense(
    Sigma: np.ndarray,
    mu: np.ndarray,
    risk_aversion: float,
    w_max: float,
    constraint_builder=None,
    benchmark: np.ndarray | None = None,
    objective_kind: str = "mv",
):
    """max mu'w - (lam/2) a'Sigma a with a dense Hessian, a = w (or w - b).

    `cp.psd_wrap` is load-bearing and not a shortcut. Plain `cp.quad_form` on
    a NumPy matrix runs an eigenvalue check to prove convexity, which is
    O(N^3) and would cost more than every solve in the sweep combined at
    N = 10,000. Sigma's positive semi-definiteness is guaranteed by
    construction here (it is B diag(f) B' + diag(d) with f, d >= 0, or a
    sample covariance) and asserted independently in tests/test_qp_model.py,
    so the check is redundant as well as ruinous.

    The matrix is symmetrized explicitly rather than trusted to be symmetric:
    cuOpt internally forms Q + Q^T, so a matrix that is symmetric only to
    floating-point noise would be handed over as a slightly different problem
    than the CPU arms receive.
    """
    import cvxpy as cp

    Sigma = np.asarray(Sigma, dtype=np.float64)
    Sigma = 0.5 * (Sigma + Sigma.T)
    N = Sigma.shape[0]
    if mu.shape != (N,):
        raise ValueError(f"mu has shape {mu.shape}, expected ({N},)")

    w = cp.Variable(N, name="w", nonneg=True)
    a = _active(w, benchmark)
    quad = cp.quad_form(a, cp.psd_wrap(Sigma))

    constraints = [cp.sum(w) == 1.0, w <= w_max]
    names = ["budget", "w_max"]
    if constraint_builder is not None:
        extra, extra_names = constraint_builder(w)
        constraints.extend(extra)
        names.extend(extra_names)

    problem = cp.Problem(
        cp.Maximize(mu @ w - 0.5 * risk_aversion * quad), constraints)
    return QPModel(problem=problem, w=w, form="dense",
                   objective_kind=objective_kind, mu=mu,
                   risk_aversion=risk_aversion, w_max=w_max,
                   benchmark=benchmark, quad_nnz=N * N, Sigma=Sigma,
                   factors=None), names


def build_qp_factored(
    fm: FactorModel,
    mu: np.ndarray,
    risk_aversion: float,
    w_max: float,
    constraint_builder=None,
    benchmark: np.ndarray | None = None,
    objective_kind: str = "mv",
):
    """The same QP with Sigma = B diag(f) B' + diag(d) kept unmultiplied.

    An auxiliary y = B'a turns the quadratic into

        a'Sigma a = y' diag(f) y + a' diag(d) a

    which is a DIAGONAL Hessian in (w, y) at the cost of K extra variables and
    K equality rows. Substituting B'a inline instead would re-form the dense
    product inside CVXPY and defeat the entire point, so the auxiliary
    variable is mandatory, not stylistic.

    Written as sums of squares of scaled variables rather than as a quad_form
    against a diagonal matrix: it is manifestly DCP-convex with no PSD check
    at all, and it canonicalizes to exactly the diagonal P the solvers want.
    Verified locally at N = 1,500: P has N + K nonzeros, all on the diagonal,
    and canonicalizes 26x faster than the dense form while agreeing with it
    to 2.4e-10 in objective.
    """
    import cvxpy as cp

    N, K = fm.B.shape
    if mu.shape != (N,):
        raise ValueError(f"mu has shape {mu.shape}, expected ({N},)")

    w = cp.Variable(N, name="w", nonneg=True)
    y = cp.Variable(K, name="y")
    a = _active(w, benchmark)

    quad = (cp.sum_squares(cp.multiply(np.sqrt(fm.f), y))
            + cp.sum_squares(cp.multiply(np.sqrt(fm.d), a)))

    constraints = [cp.sum(w) == 1.0, w <= w_max, y == fm.B.T @ a]
    names = ["budget", "w_max", "factor_definition"]
    if constraint_builder is not None:
        extra, extra_names = constraint_builder(w)
        constraints.extend(extra)
        names.extend(extra_names)

    problem = cp.Problem(
        cp.Maximize(mu @ w - 0.5 * risk_aversion * quad), constraints)
    return QPModel(problem=problem, w=w, form="factored",
                   objective_kind=objective_kind, mu=mu,
                   risk_aversion=risk_aversion, w_max=w_max,
                   benchmark=benchmark, quad_nnz=fm.quad_nnz, Sigma=None,
                   factors=fm), names


# --------------------------------------------------------------------------
# Arms
# --------------------------------------------------------------------------

#: (cvxpy solver attribute, form, hardware, needs_cuopt, description)
ARMS: dict[str, dict] = {
    "cuopt_barrier_dense": dict(
        solver="CUOPT", form="dense", hardware="GPU", gpu=True,
        method_class="QP barrier", endpoint="raw", quality_target="default",
        desc="cuOpt GPU barrier (cuDSS) on a dense N x N Hessian"),
    "cuopt_barrier_factored": dict(
        solver="CUOPT", form="factored", hardware="GPU", gpu=True,
        method_class="QP barrier", endpoint="raw", quality_target="default",
        desc="cuOpt GPU barrier on the low-rank-plus-diagonal Hessian"),
    "clarabel_dense": dict(
        solver="CLARABEL", form="dense", hardware="CPU", gpu=False,
        method_class="QP barrier", endpoint="raw", quality_target="default",
        desc="Clarabel CPU interior-point on a dense N x N Hessian"),
    "clarabel_factored": dict(
        solver="CLARABEL", form="factored", hardware="CPU", gpu=False,
        method_class="QP barrier", endpoint="raw", quality_target="default",
        desc="Clarabel CPU interior-point on the factored Hessian"),
    "clarabel_strict_dense": dict(
        solver="CLARABEL", form="dense", hardware="CPU", gpu=False,
        method_class="QP barrier", endpoint="raw",
        quality_target="strict-1e-9",
        desc="Clarabel CPU interior-point on a dense Hessian, 1e-9 feasibility/gap"),
    "clarabel_strict_factored": dict(
        solver="CLARABEL", form="factored", hardware="CPU", gpu=False,
        method_class="QP barrier", endpoint="raw",
        quality_target="strict-1e-9",
        desc="Clarabel CPU interior-point on the factored Hessian, 1e-9 feasibility/gap"),
    "osqp_dense": dict(
        solver="OSQP", form="dense", hardware="CPU", gpu=False,
        method_class="QP first-order", endpoint="raw", quality_target="default",
        desc="OSQP CPU first-order ADMM on a dense Hessian (counterpoint, "
             "a different algorithm class - not the CPU barrier)"),
    "osqp_factored": dict(
        solver="OSQP", form="factored", hardware="CPU", gpu=False,
        method_class="QP first-order", endpoint="raw", quality_target="default",
        desc="OSQP CPU first-order ADMM on the factored Hessian"),
    "highs_hipo_dense": dict(
        solver="HIGHS", form="dense", hardware="CPU", gpu=False,
        method_class="QP barrier", endpoint="raw", quality_target="default",
        desc="HiGHS HiPO parallel interior-point on a dense Hessian; needs "
             "the separate highspy-extras package, NOT in requirements.lock"),
}

#: The three-way comparison plus the free fourth arm. Everything else is
#: opt-in via --arms.
DEFAULT_ARMS = ("cuopt_barrier_dense", "clarabel_dense",
                "clarabel_factored", "cuopt_barrier_factored")


def solver_options(arm: str, time_limit_s: float, threads: int) -> dict:
    """Solver kwargs for one arm.

    The cuOpt line is the one that matters. `CUOPT_METHOD` defaults to
    Concurrent, and CVXPY's cuOpt interface only force-selects Barrier when
    SOC cones are present — so a plain quadratic objective left at the default
    runs under Concurrent, which is undocumented for QP and may spin up PDLP
    and dual simplex that cannot solve it at all. Pinning the method is not a
    tuning choice; leaving it unpinned is a methodology bug.
    """
    spec = ARMS[arm]
    if spec["solver"] == "CUOPT":
        return {"solver_method": "Barrier", "time_limit": time_limit_s}
    if spec["solver"] == "CLARABEL":
        opts = {"time_limit": time_limit_s}
        if spec.get("quality_target") == "strict-1e-9":
            opts.update({
                "tol_feas": 1e-9,
                "tol_gap_abs": 1e-9,
                "tol_gap_rel": 1e-9,
                "max_iter": 2_000,
            })
        return opts
    if spec["solver"] == "OSQP":
        return {"max_iter": 100_000}
    if spec["solver"] == "HIGHS":
        # "solver" collides with CVXPY's own solver= kwarg; the nested dict is
        # the interface's documented escape hatch. Without solver="hipo" this
        # is the qpasm active-set method, whose dense reduced-Hessian Cholesky
        # is capped by qp_nullspace_limit (default 4000) and which would be an
        # unfair straw man above N ~ 4,000.
        return {"highs_options": {"solver": "hipo", "time_limit": time_limit_s,
                                  "threads": threads}}
    raise ValueError(arm)


def arm_metadata(arm: str) -> dict:
    spec = ARMS.get(arm, {})
    return {
        "method_class": spec.get("method_class", ""),
        "endpoint": spec.get("endpoint", "raw"),
        "quality_target": spec.get("quality_target", "default"),
        "solver_config_id": solver_config_id(arm),
    }


def _is_true(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    return str(v).strip().lower() in {"true", "1", "yes", "y"}


def _float_or_nan(v) -> float:
    try:
        if v is None or v == "":
            return math.nan
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def _is_converged_status(status: str) -> bool:
    return str(status) in {"optimal", "optimal_inaccurate"}


def solver_config_id(arm: str) -> str:
    spec = ARMS.get(arm, {})
    solver = str(spec.get("solver", "")).lower()
    target = str(spec.get("quality_target", "default"))
    if solver == "cuopt":
        return "cuopt-barrier-default"
    if solver == "clarabel" and target == "strict-1e-9":
        return "clarabel-strict-1e-9"
    if solver:
        return f"{solver}-{target}"
    return ""


def objective_gate_rel_gap(row: dict) -> float:
    solver_obj = _float_or_nan(row.get("objective"))
    recomputed = _float_or_nan(row.get("obj_recomputed"))
    if not math.isfinite(solver_obj) or not math.isfinite(recomputed):
        return math.nan
    return abs(solver_obj - recomputed) / max(abs(recomputed), 1e-12)


def achieved_quality_target(row: dict,
                            objective_tol: float = OBJECTIVE_REL_TOL) -> str:
    if str(row.get("quality_target", "")) == "strict-1e-9" \
            and str(row.get("status", "")) != "optimal":
        return "failed-independent-gate"
    if not _is_converged_status(row.get("status", "")):
        return "failed-independent-gate"
    if _is_true(row.get("dnf", False)):
        return "failed-independent-gate"
    if not _is_true(row.get("check_ok", False)):
        return "failed-independent-gate"
    if not _is_true(row.get("respects_mandate", False)):
        return "failed-independent-gate"
    gap = objective_gate_rel_gap(row)
    if not math.isfinite(gap) or gap > objective_tol:
        return "failed-independent-gate"
    return "p0-independent-gate-v1"


def eligibility_reason(row: dict, endpoint: str | None = None,
                       quality_target: str | None = None,
                       objective_tol: float = OBJECTIVE_REL_TOL
                       ) -> tuple[bool, str]:
    """Mechanical gate for rows entering a common-quality ratio.

    A row is evidence only after the solver converged, the independent checker
    passed, the mandate gate passed, the recomputed objective agrees with the
    solver objective, and the requested endpoint/quality target matches.
    """
    reasons: list[str] = []
    if str(row.get("quality_target", "")) == "strict-1e-9":
        if str(row.get("status", "")) != "optimal":
            reasons.append(f"status={row.get('status', '')}, expected optimal")
    elif not _is_converged_status(row.get("status", "")):
        reasons.append(f"status={row.get('status', '')}")
    if _is_true(row.get("dnf", False)):
        reasons.append("dnf=True")
    if not _is_true(row.get("check_ok", False)):
        note = row.get("check_notes", "")
        reasons.append("check_ok=False" + (f" ({note})" if note else ""))
    if not _is_true(row.get("respects_mandate", False)):
        worst = row.get("worst_limit") or row.get("limits_verdict") or ""
        reasons.append("respects_mandate=False" + (f" ({worst})" if worst else ""))

    rel_gap = objective_gate_rel_gap(row)
    if not math.isfinite(rel_gap) or rel_gap > objective_tol:
        reasons.append(f"objective_gap>{objective_tol:g}")

    if endpoint is not None and str(row.get("endpoint", "")) != endpoint:
        reasons.append(f"endpoint={row.get('endpoint', '')}, expected {endpoint}")
    if quality_target is not None \
            and str(row.get("quality_target", "")) != quality_target:
        reasons.append(
            f"quality_target={row.get('quality_target', '')}, expected {quality_target}")

    return not reasons, "; ".join(reasons)


def risk_model_id(row: dict) -> str:
    explicit = str(row.get("risk_model_id", "")).strip()
    if explicit:
        return explicit
    basis = str(row.get("sigma_basis", "factor"))
    form = str(row.get("form", ""))
    if basis == "sample":
        return "sample-covariance" if form == "dense" else "factor-approximation"
    if basis == "factor":
        return "factor-exact"
    return f"{basis}:{form}"


def math_instance_id(row: dict) -> tuple:
    return (
        str(row.get("n_bonds", "")),
        str(row.get("objective_kind", "")),
        str(row.get("sigma_basis", "")),
        risk_model_id(row),
        str(row.get("risk_aversion", "")),
        str(row.get("universe_seed", "")),
        str(row.get("scenario_seed", "")),
        str(row.get("n_scenarios_for_sigma", "")),
        str(row.get("turnover", "")),
    )


def _instance_key(row: dict) -> tuple:
    return math_instance_id(row)


def repair_mode_for_config(args, n_bonds: int, arm: str) -> str:
    if getattr(args, "repair_strict_clarabel", False):
        spec = ARMS.get(arm, {})
        if spec.get("solver") == "CLARABEL" \
                and spec.get("quality_target") == "strict-1e-9":
            return "strict-clarabel-repair-all-sizes"
    return "none"


def requested_endpoint_for_config(args, n_bonds: int, arm: str) -> str:
    """Endpoint lifecycle requested by this run configuration.

    This is deliberately separate from the actual row `endpoint`.  With
    `--repair-strict-clarabel`, a strict Clarabel cell requests
    `repair-if-needed`: it may write either `endpoint=raw` when the raw solve
    already passes the independent gate, or `endpoint=repaired` when the
    minimum-distance endpoint was required.  Both outcomes complete the same
    lifecycle request for resume purposes.
    """
    if repair_mode_for_config(args, n_bonds, arm) != "none":
        return "repair-if-needed"
    return str(ARMS[arm].get("endpoint", "raw"))


def resume_key_for_config(args, n_bonds: int, arm: str) -> tuple:
    return (
        str(n_bonds), str(arm),
        requested_endpoint_for_config(args, n_bonds, arm),
        repair_mode_for_config(args, n_bonds, arm),
        str(getattr(args, "sigma_basis", "")),
        str(getattr(args, "objective", "")),
        str(getattr(args, "seed", "")),
        str(getattr(args, "scen_seed", "")),
        str(getattr(args, "n_scenarios", "")),
        str(getattr(args, "risk_aversion", "")),
        str(getattr(args, "turnover", "")),
        str(getattr(args, "runs", "")),
        solver_config_id(arm),
        str(getattr(args, "skip_calibration", False)),
    )


def resume_key_from_row(row: dict) -> tuple:
    arm = str(row.get("arm", ""))
    return (
        str(row.get("n_bonds", "")), arm,
        str(row.get("requested_endpoint", row.get("endpoint", "raw"))),
        str(row.get("repair_mode", "none") or "none"),
        str(row.get("sigma_basis", "")),
        str(row.get("objective_kind", "")),
        str(row.get("universe_seed", "")),
        str(row.get("scenario_seed", "")),
        str(row.get("n_scenarios_for_sigma", "")),
        str(row.get("risk_aversion", "")),
        str(row.get("turnover", "")),
        str(row.get("n_runs_requested", row.get("n_runs", ""))),
        str(row.get("solver_config_id", solver_config_id(arm))),
        str(row.get("skip_calibration", "False")),
    )


def grid_for_gate_mode(grid: tuple[int, ...], gate_n500_only: bool,
                       repair_n: int = 500) -> tuple[int, ...]:
    return (int(repair_n),) if gate_n500_only else grid


def row_config_context(args, n_bonds: int, arm: str) -> dict:
    repair_mode = repair_mode_for_config(args, n_bonds, arm)
    return {
        "n_scenarios_for_sigma": getattr(args, "n_scenarios", ""),
        "sigma_basis": getattr(args, "sigma_basis", ""),
        "objective_kind": getattr(args, "objective", ""),
        "risk_aversion": getattr(args, "risk_aversion", ""),
        "turnover": getattr(args, "turnover", ""),
        "universe_seed": getattr(args, "seed", ""),
        "scenario_seed": getattr(args, "scen_seed", ""),
        "n_runs_requested": getattr(args, "runs", ""),
        "skip_calibration": str(getattr(args, "skip_calibration", False)),
        "repair_mode": repair_mode,
        "requested_endpoint": requested_endpoint_for_config(args, n_bonds, arm),
        "resume_key": "|".join(resume_key_for_config(args, n_bonds, arm)),
        "solver_config_id": solver_config_id(arm),
    }


def _endpoint_time(row: dict) -> float:
    base = _float_or_nan(row.get("wall_s_median"))
    if not math.isfinite(base):
        return math.nan
    if row.get("endpoint") == "repaired":
        repair = _float_or_nan(row.get("repair_s"))
        if math.isfinite(repair):
            return base + repair
    return base


def annotate_matched_quality(rows: list[dict]) -> list[dict]:
    """Add P0 common-quality accounting columns to result rows.

    The input rows are copied.  Ratios are deliberately formed only between
    rows with the same mathematical instance, form, endpoint and quality
    target; ineligible or unmatched comparisons get NaN plus a printable
    exclusion reason.
    """
    out = [dict(r) for r in rows]
    for r in out:
        r.update({k: v for k, v in arm_metadata(str(r.get("arm", ""))).items()
                  if r.get(k, "") == "" or k not in r})
        r["risk_model_id"] = risk_model_id(r)
        r["math_instance_id"] = "|".join(math_instance_id(r))
        r["objective_gate_rel_gap"] = objective_gate_rel_gap(r)
        r["achieved_quality_target"] = achieved_quality_target(r)
        t = _endpoint_time(r)
        r["common_quality_time_s"] = t
        ok, why = eligibility_reason(r)
        r["ratio_eligible"] = ok
        r["ratio_exclusion_reason"] = "" if ok else why

    best_by_instance: dict[tuple, float] = {}
    for r in out:
        if not r.get("ratio_eligible"):
            continue
        obj = _float_or_nan(r.get("obj_recomputed"))
        if not math.isfinite(obj):
            continue
        key = _instance_key(r)
        best_by_instance[key] = max(obj, best_by_instance.get(key, -math.inf))

    for r in out:
        best = best_by_instance.get(_instance_key(r), math.nan)
        r["best_feasible_objective"] = best
        obj = _float_or_nan(r.get("obj_recomputed"))
        if math.isfinite(best) and math.isfinite(obj):
            r["relative_best_objective_gap"] = (best - obj) / max(abs(best), 1e-12)
        else:
            r["relative_best_objective_gap"] = math.nan

    for r in out:
        r["solver_stack_ratio"] = math.nan
        r["solver_stack_ratio_peer"] = ""
        r["solver_stack_ratio_exclusion_reason"] = ""
        if not r.get("ratio_eligible"):
            r["solver_stack_ratio_exclusion_reason"] = r["ratio_exclusion_reason"]
            continue
        target_hw = "CPU" if r.get("hardware") == "GPU" else "GPU"
        peers = [
            p for p in out
            if p is not r
            and p.get("ratio_eligible")
            and _instance_key(p) == _instance_key(r)
            and p.get("form") == r.get("form")
            and p.get("endpoint") == r.get("endpoint")
            and p.get("achieved_quality_target") == r.get("achieved_quality_target")
            and p.get("hardware") == target_hw
        ]
        if not peers:
            r["solver_stack_ratio_exclusion_reason"] = (
                "no matched CPU row with same form/endpoint/quality target"
                if r.get("hardware") == "GPU"
                else "no matched GPU row with same form/endpoint/quality target")
            continue
        peer = min(peers, key=lambda p: _endpoint_time(p))
        den = _endpoint_time(peer)
        num = _endpoint_time(r)
        if math.isfinite(num) and math.isfinite(den) and den > 0:
            r["solver_stack_ratio"] = num / den
            r["solver_stack_ratio_peer"] = peer.get("arm", "")
        else:
            r["solver_stack_ratio_exclusion_reason"] = "missing common-quality time"
    return out


def normalise_rows_for_write(rows: list[dict]) -> list[dict]:
    """Annotate rows and ensure every P0 accounting column exists."""
    out = annotate_matched_quality(rows)
    for r in out:
        for k in P0_SCHEMA_KEYS:
            r.setdefault(k, "")
    return out


def cuopt_qp_reachability() -> tuple[bool, str]:
    """Is cuOpt's quadratic-objective path actually reachable from here?

    Two things have to hold and neither is guaranteed by the lockfile alone:
    cuOpt must be installed and registered with CVXPY, and its CVXPY interface
    must declare `supports_quad_obj()`. If it does not, CVXPY canonicalizes a
    quadratic objective through the conic path into a second-order cone, and
    what gets measured is QP-as-SOCP rather than QP-as-QP.

    Never guessed at and never silently routed around: the answer goes in the
    CSV header and a False takes the run down with an actionable message.
    """
    try:
        import cvxpy as cp
    except Exception as e:                                    # noqa: BLE001
        return False, f"cvxpy import failed: {type(e).__name__}: {e}"

    if "CUOPT" not in cp.installed_solvers():
        return False, (
            "cuOpt is not registered with CVXPY. Expected cuopt-cu12==26.6.0 "
            "from requirements.lock on the WSL2 GPU box. Check: "
            "python -c \"import cvxpy; print(cvxpy.installed_solvers())\"")
    try:
        from cvxpy.reductions.solvers.defines import (
            SOLVER_MAP_CONIC, SOLVER_MAP_QP,
        )
        inst = SOLVER_MAP_QP.get("CUOPT") or SOLVER_MAP_CONIC.get("CUOPT")
        if inst is None:
            return False, "CUOPT is installed but has no CVXPY interface object"
        if not getattr(inst, "supports_quad_obj", lambda: False)():
            return False, (
                "CVXPY's CUOPT interface reports supports_quad_obj() == False "
                "in this pinned version, so a quadratic objective would be "
                "epigraph-lifted into a second-order cone instead of reaching "
                "DataModel.set_quadratic_objective_matrix. That measures "
                "QP-as-SOCP, not QP-as-QP. Either upgrade cvxpy (1.9.2's "
                "interface does declare it True) or build the GPU arm on the "
                "raw cuopt.linear_programming DataModel path, following "
                "src/interactive.py.")
    except Exception as e:                                    # noqa: BLE001
        return False, f"probing the CUOPT interface failed: {type(e).__name__}: {e}"
    return True, "CVXPY CUOPT interface accepts a quadratic objective"


# --------------------------------------------------------------------------
# Calibration and checking
# --------------------------------------------------------------------------

def run_calibration(cvxpy_solver, opts: dict, tol: float = 1e-6) -> dict:
    """Solve the 2-variable QP with the known optimum through this arm's path.

    Returns a dict; raises RuntimeError if the objective misses. Cheap enough
    (two variables) to run before every cell, and the only defence against a
    silently doubled risk aversion — which would corrupt every timing and
    every mandate verdict in the CSV without producing a single warning.
    """
    import cvxpy as cp

    w = cp.Variable(2, nonneg=True)
    quad = cp.quad_form(w, cp.psd_wrap(CALIBRATION_S))
    prob = cp.Problem(
        cp.Maximize(CALIBRATION_MU @ w
                    - 0.5 * CALIBRATION_LAMBDA * quad),
        [cp.sum(w) == 1.0, w <= 1.0])
    prob.solve(solver=cvxpy_solver, verbose=False, warm_start=False, **opts)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(
            f"Q-convention calibration did not solve: status={prob.status}. "
            f"Every timing downstream of this would be unverifiable.")
    obj = float(prob.value)
    p = float(np.asarray(w.value).ravel()[0])
    gap = abs(obj - CALIBRATION_OBJ_STAR)
    if gap > tol:
        doubled = abs(obj - CALIBRATION_OBJ_IF_DOUBLED) < 1e-3
        raise RuntimeError(
            f"Q-CONVENTION MISMATCH. The 2-variable calibration QP has a "
            f"closed-form optimum of {CALIBRATION_OBJ_STAR:.7f} at "
            f"p*={CALIBRATION_P_STAR}; this path returned {obj:.7f} at "
            f"p={p:.6f} (gap {gap:.2e}). "
            + ("That is exactly the value for a DOUBLED risk aversion "
               f"({CALIBRATION_OBJ_IF_DOUBLED:.7f}), i.e. the 1/2 factor is "
               "being applied on one side only. cuOpt's objective is x'Qx "
               "with no 1/2 and it symmetrizes Q + Q^T; CVXPY's is "
               "(1/2)x'Px and its cuOpt interface passes P/2."
               if doubled else
               "Do not trust any timing from this path until it is resolved.")
        )
    return {"calibration_obj": obj, "calibration_p": p,
            "calibration_abs_gap": gap}


def check_qp(model: QPModel, solver_objective: float,
             feasibility_tol: float = 1e-6,
             objective_tol: float = 1e-6) -> dict:
    """Independent NumPy recomputation of a QP solution.

    The LP branch's `src.check.check_solution` cannot be reused: it recomputes
    CVaR from a scenario matrix a scenario-free QP does not have, and would
    report a spurious objective disagreement on every row. What transfers
    verbatim is everything that is not the objective — budget residual, box
    violations, dust and position counts — and that part is reproduced here
    with the same thresholds and the same discipline (trust nothing from the
    solver but the weight vector).

    `n_positions` and `n_dust` are not diagnostics on this branch. A barrier
    solution with no crossover available is a dense interior point, so a
    position count near N is the EXPECTED signature and is itself a mandate
    result: nobody trades 29,222 dust positions.
    """
    w = np.asarray(model.weights(), dtype=np.float64).ravel()
    budget_residual = abs(float(w.sum()) - 1.0)
    lower_viol = float(np.maximum(-w, 0.0).max(initial=0.0))
    upper_viol = float(np.maximum(w - model.w_max, 0.0).max(initial=0.0))
    max_bound_violation = max(lower_viol, upper_viol)

    recomputed = model.objective_value(w)
    obj_gap = abs(recomputed - float(solver_objective))

    notes = []
    if budget_residual > feasibility_tol:
        notes.append("budget violated")
    if max_bound_violation > feasibility_tol:
        notes.append("bounds violated")
    # Relative gate: the objective is O(1e-3) on this problem, so an absolute
    # 1e-6 would be a 0.1% relative tolerance and would pass a solver that had
    # genuinely lost three digits.
    scale = max(abs(recomputed), 1e-12)
    if obj_gap / scale > objective_tol:
        notes.append("solver objective disagrees with recomputation")

    return {
        "check_ok": not notes,
        "check_notes": "; ".join(notes),
        "budget_residual": budget_residual,
        "max_bound_violation": max_bound_violation,
        "obj_recomputed": recomputed,
        "obj_abs_gap": obj_gap,
        "obj_rel_gap": obj_gap / scale,
        "portfolio_variance": model.quadratic_value(w),
        "n_positions": int((w >= DUST_THRESHOLD).sum()),
        "n_dust": int(((w > DUST_THRESHOLD * 1e-3)
                       & (w < DUST_THRESHOLD)).sum()),
        "max_weight": float(w.max()),
    }


def score_limits(uni, w: np.ndarray, limits) -> tuple[float, str]:
    """Worst relative breach of the limit sheet (0.0 if clean).

    `R=None`: every family except tracking error is a pure weight-space linear
    constraint and needs no scenario matrix. Rows for families that are
    switched off carry infinite or NaN relative slack and would poison
    `idxmin`, so they are filtered first — the same guard `run_ablation._score`
    uses, for the same reason.
    """
    from src.credit import exposure_report, worst_violation

    rep = exposure_report(uni, w, limits, R=None)
    live = rep[np.isfinite(rep["rel_slack"])]
    if live.empty:
        return 0.0, "no limits in force"
    return worst_violation(live)


def minimum_distance_repair(uni, limits, w_raw: np.ndarray,
                            time_limit_s: float) -> tuple[str, np.ndarray | None, float, str]:
    """Project a QP point back into the linear mandate, explicitly.

    This is an optional endpoint, not a hidden cleanup step.  The caller must
    label the row `endpoint=repaired` and add `repair_s` to the endpoint time.
    """
    import cvxpy as cp

    from src.credit import credit_constraint_builder

    w_raw = np.asarray(w_raw, dtype=np.float64).ravel()
    w = cp.Variable(w_raw.size, nonneg=True, name="w_repair")
    cons = [cp.sum(w) == 1.0, w <= limits.w_max]
    extra, _ = credit_constraint_builder(uni, limits, R=None)(w)
    cons.extend(extra)
    prob = cp.Problem(cp.Minimize(cp.norm1(w - w_raw)), cons)

    t0 = time.perf_counter()
    try:
        prob.solve(solver=cp.CLARABEL, verbose=False, warm_start=False,
                   time_limit=time_limit_s, tol_feas=1e-9,
                   tol_gap_abs=1e-9, tol_gap_rel=1e-9, max_iter=2_000)
    except Exception as e:                                    # noqa: BLE001
        return f"repair_error:{type(e).__name__}", None, time.perf_counter() - t0, str(e)[:300]
    wall = time.perf_counter() - t0
    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or w.value is None:
        return str(prob.status), None, wall, "repair produced no weights"
    return str(prob.status), np.asarray(w.value, dtype=np.float64).ravel(), wall, ""


def should_repair_strict_clarabel_row(
    arm: str,
    n_assets: int,
    repair_flag: bool,
    status: str,
    check_ok: bool,
    worst_rel_violation: float,
    repair_n: int = 500,
    purity_gate: float = 1e-6,
) -> bool:
    spec = ARMS.get(arm, {})
    return (
        bool(repair_flag)
        and spec.get("solver") == "CLARABEL"
        and spec.get("quality_target") == "strict-1e-9"
        and str(status) == "optimal"
        and bool(check_ok)
        and float(worst_rel_violation) > float(purity_gate)
    )


def scenario_generation_nonresults(
    args,
    n_bonds: int,
    runnable: list[tuple[str, float]],
    reason: str,
) -> list[dict]:
    """Represent a shared scenario-generation OOM at the arm boundary.

    The QP solver never ran, so these rows are runner-classified non-results,
    not solver failures. One row per pending arm makes resume deterministic and
    prevents an index-scale OOM from erasing the completed grid.
    """
    out: list[dict] = []
    for arm, peak in runnable:
        rec = {
            "n_bonds": n_bonds,
            "arm": arm,
            "form": ARMS[arm]["form"],
            "hardware": ARMS[arm]["hardware"],
            "solver": ARMS[arm]["solver"],
            "method_class": ARMS[arm].get("method_class", ""),
            "endpoint": ARMS[arm].get("endpoint", "raw"),
            "quality_target": ARMS[arm].get("quality_target", "default"),
            "status_source": "author-classification",
            "arm_desc": ARMS[arm]["desc"],
            "status": "scenario_generation_oom",
            "dnf": True,
            "wall_s_median": "",
            "q_gib_dense": round(dense_matrix_gib(n_bonds), 4),
            "host_peak_est_gib": round(peak, 3),
            "limits_verdict": "scenario_generation_oom",
            "check_ok": False,
            "check_notes": reason,
            "respects_mandate": False,
            "notes": reason,
        }
        rec.update(row_config_context(args, n_bonds, arm))
        out.append(rec)
    return out


# --------------------------------------------------------------------------
# Memory guards — a refused cell is a result
# --------------------------------------------------------------------------

def dense_host_peak_gib(n: int) -> float:
    return n * n * DENSE_Q_BYTES_PER_ENTRY_PEAK / 1024 ** 3


def dense_device_gib(n: int) -> float:
    return n * n * 8.0 * DEVICE_Q_MULTIPLE / 1024 ** 3


def factored_host_peak_gib(n: int, k: int) -> float:
    nnz = n * (k + 1) + k
    return nnz * FACTORED_BYTES_PER_NNZ_PEAK / 1024 ** 3


def dense_matrix_gib(n: int) -> float:
    return n * n * 8.0 / 1024 ** 3


def available_gib() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable"):
                    return int(line.split()[1]) / 1024 ** 2
    except OSError:
        pass
    return float("inf")


def memory_refusal(arm: str, n: int, k: int, have_host: float,
                   gpu_gib: float) -> tuple[str, str, float]:
    """(status, reason, host_peak_est_gib). Empty status means the cell runs."""
    spec = ARMS[arm]
    if spec["form"] == "dense":
        need = dense_host_peak_gib(n)
        if need > have_host * HOST_BUDGET_FRACTION:
            return ("skipped_host_memory",
                    f"dense Hessian needs ~{need:.1f} GiB host peak "
                    f"({dense_matrix_gib(n):.2f} GiB for the matrix alone) "
                    f"against {have_host:.1f} GiB available; budget is "
                    f"{HOST_BUDGET_FRACTION:.0%}", need)
        if spec["gpu"]:
            dev = dense_device_gib(n)
            if dev > gpu_gib * DEVICE_BUDGET_FRACTION:
                return ("skipped_device_memory",
                        f"dense Hessian needs ~{dev:.1f} GiB device "
                        f"({DEVICE_Q_MULTIPLE:g}x the {dense_matrix_gib(n):.2f} "
                        f"GiB matrix) against a {gpu_gib:.0f} GiB card; "
                        f"budget is {DEVICE_BUDGET_FRACTION:.0%}", need)
        return "", "", need

    need = factored_host_peak_gib(n, k)
    if need > have_host * HOST_BUDGET_FRACTION:
        return ("skipped_host_memory",
                f"factored Hessian needs ~{need:.2f} GiB host peak against "
                f"{have_host:.1f} GiB available", need)
    return "", "", need


# --------------------------------------------------------------------------
# Worker — one arm, one cell, one clean process
# --------------------------------------------------------------------------

def _timed_solve(model: QPModel, cvxpy_solver, opts: dict
                 ) -> tuple[float, float, str]:
    """One timed solve. Returns (wall_s, internal_s, status)."""
    import cvxpy as cp

    t0 = time.perf_counter()
    try:
        # warm_start=False for the same load-bearing reason src/bench.py gives:
        # CVXPY defaults it True and several interfaces cache the previous
        # solution, which would silently time warm re-solves for some arms and
        # cold ones for others.
        model.problem.solve(solver=cvxpy_solver, verbose=False,
                            warm_start=False, **opts)
        status = model.problem.status
    except cp.error.SolverError as e:
        return time.perf_counter() - t0, math.nan, f"solver_error: {e}"
    except Exception as e:                                    # noqa: BLE001
        return time.perf_counter() - t0, math.nan, \
            f"{type(e).__name__}: {str(e)[:200]}"
    wall = time.perf_counter() - t0
    internal = math.nan
    stats = getattr(model.problem, "solver_stats", None)
    if stats is not None and getattr(stats, "solve_time", None) is not None:
        internal = float(stats.solve_time)
    return wall, internal, status


def worker_main(args) -> int:
    """Build one arm's model on one cell, time it, score it, print RESULT_JSON.

    Self-spawned rather than reusing scripts/bench_worker.py because this
    experiment needs a QP-specific checker and the exposure report, neither of
    which the LP worker has any reason to carry — the same rationale
    run_ablation.py gives for spawning itself.
    """
    import cvxpy as cp

    from src.credit import credit_constraint_builder

    arm = args.arm
    spec = ARMS[arm]
    out: dict = {"arm": arm, "form": spec["form"],
                 "hardware": spec["hardware"], "solver": spec["solver"],
                 "method_class": spec.get("method_class", ""),
                 "endpoint": spec.get("endpoint", "raw"),
                 "quality_target": spec.get("quality_target", "default"),
                 "status_source": "solver-returned",
                 "arm_desc": spec["desc"],
                 "objective_source": "solver-reported",
                 "repair_s": "", "repair_status": "",
                 "repaired_from_endpoint": ""}

    if spec["gpu"]:
        ok, why = cuopt_qp_reachability()
        if not ok:
            out.update({"status": "GPU-UNREACHABLE", "dnf": True,
                        "status_source": "author-classification",
                        "wall_s_median": "", "limits_verdict": "GPU-UNREACHABLE",
                        "check_ok": False, "check_notes": why,
                        "respects_mandate": False, "notes": why})
            print(MARKER + json.dumps(out))
            return 3

    cvxpy_solver = getattr(cp, spec["solver"])
    opts = solver_options(arm, args.time_limit, args.threads)
    out["solver_opts"] = json.dumps(
        {k: v for k, v in opts.items() if k != "parameters_proto"})

    # ---- the Q-convention gate, before anything is timed -------------------
    if not args.skip_calibration:
        try:
            out.update(run_calibration(cvxpy_solver, opts))
        except Exception as e:                                # noqa: BLE001
            out.update({"status": "CALIBRATION-FAILED", "dnf": True,
                        "status_source": "author-classification",
                        "wall_s_median": "", "limits_verdict": "",
                        "check_ok": False,
                        "check_notes": f"{type(e).__name__}: {str(e)[:200]}",
                        "respects_mandate": False,
                        "notes": f"{type(e).__name__}: {str(e)[:400]}"})
            print(MARKER + json.dumps(out))
            return 4

    # ---- problem data ------------------------------------------------------
    with open(args.universe, "rb") as f:
        uni, limits = pickle.load(f)
    d = np.load(args.data)
    mu = d["mu"]
    fm = FactorModel(B=d["B"], f=d["f"], d=d["d"],
                     labels=list(d["labels"]), n_dropped=int(d["n_dropped"]))
    benchmark = uni.w_benchmark if args.objective == "te" else None

    builder = credit_constraint_builder(uni, limits, R=None)

    t0 = time.perf_counter()
    if spec["form"] == "dense":
        if args.sigma_npy:
            # Sample-covariance basis: every dense arm reads the SAME bytes
            # off disk rather than each recomputing np.cov and risking a
            # different rounding path.
            Sigma = np.load(args.sigma_npy)
        else:
            Sigma = fm.dense()
        model, names = build_qp_dense(
            Sigma, mu, args.risk_aversion, limits.w_max,
            constraint_builder=builder, benchmark=benchmark,
            objective_kind=args.objective)
    else:
        model, names = build_qp_factored(
            fm, mu, args.risk_aversion, limits.w_max,
            constraint_builder=builder, benchmark=benchmark,
            objective_kind=args.objective)
    sigma_build_s = time.perf_counter() - t0

    out.update({
        "n_bonds": int(model.n_assets),
        "n_factors": int(fm.n_factors),
        "quad_nnz": int(model.quad_nnz),
        "q_gib_dense": round(dense_matrix_gib(model.n_assets), 4),
        "q_gib_arm": round(
            dense_matrix_gib(model.n_assets) if spec["form"] == "dense"
            else fm.quad_nnz * 8.0 / 1024 ** 3, 6),
        "sigma_build_s": round(sigma_build_s, 4),
        "constraint_names": "|".join(names),
        "objective_kind": args.objective,
    })

    # ---- warmup (discarded), with one retry --------------------------------
    ok_statuses = {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
    warm_wall, _, warm_status = _timed_solve(model, cvxpy_solver, opts)
    compile_s = float(getattr(model.problem, "compilation_time", None)
                      or math.nan)
    retry_note = ""
    if warm_status not in ok_statuses:
        # One retry, same reasoning as src/bench.py run_spec: the warmup pays
        # CUDA context creation, cuDSS library init and CVXPY canonicalization,
        # so a first-attempt failure can be startup cost rather than an
        # inability to converge. It matters MORE for a barrier than for PDLP.
        first = (warm_status, warm_wall)
        warm_wall, _, warm_status = _timed_solve(model, cvxpy_solver, opts)
        retry_note = (f"warmup retried once after status={first[0]} "
                      f"(wall={first[1]:.1f}s); retry status={warm_status}")

    out["compile_s"] = round(compile_s, 4) if not math.isnan(compile_s) else ""

    if warm_status not in ok_statuses:
        out.update({"status": warm_status, "dnf": True, "wall_s_median": "",
                    "limits_verdict": "DNF", "check_ok": False,
                    "check_notes": f"DNF on warmup: {warm_status}",
                    "respects_mandate": False,
                    "notes": "; ".join(x for x in [
                        f"DNF on warmup (wall={warm_wall:.1f}s)",
                        retry_note] if x)})
        print(MARKER + json.dumps(out))
        return 0

    # ---- timed runs --------------------------------------------------------
    walls, internals, statuses = [], [], []
    status = warm_status
    for _ in range(args.runs):
        wall, internal, status = _timed_solve(model, cvxpy_solver, opts)
        statuses.append(status)
        if status not in ok_statuses:
            out.update({"status": status, "dnf": True, "wall_s_median": "",
                        "limits_verdict": "DNF", "check_ok": False,
                        "check_notes": f"DNF on timed run: {status}",
                        "respects_mandate": False,
                        "notes": "; ".join(x for x in [
                            f"DNF on timed run {len(walls) + 1}",
                            retry_note] if x)})
            print(MARKER + json.dumps(out))
            return 0
        walls.append(wall)
        if not math.isnan(internal):
            internals.append(internal)

    w = model.weights()
    objective = float(model.problem.value)
    chk = check_qp(model, objective)
    viol, which = score_limits(uni, w, limits)
    respects = bool(viol <= 1e-6 and chk["check_ok"])

    repair_s: float | str = ""
    repair_status = ""
    repaired_from_endpoint = ""
    raw_status = status
    raw_objective = objective
    raw_obj_recomputed = chk["obj_recomputed"]
    raw_worst_rel_violation = viol
    raw_limits_verdict = ("ok" if viol <= 1e-6
                          else f"VIOLATED {which} by {viol:.2e} of its scale")

    should_repair = (
        should_repair_strict_clarabel_row(
            arm, model.n_assets,
            bool(getattr(args, "worker_repair_strict_clarabel", False)),
            status, bool(chk["check_ok"]), viol,
            repair_n=int(getattr(args, "worker_repair_n", 500)))
    )
    repair_note = ""
    if should_repair:
        r_status, r_w, r_wall, r_note = minimum_distance_repair(
            uni, limits, w, args.time_limit)
        repair_s = round(r_wall, 4)
        repair_status = r_status
        repaired_from_endpoint = "raw"
        if r_w is not None:
            # Re-score the repaired endpoint with the same independent checker.
            model.w.value = r_w
            w = r_w
            objective = model.objective_value(w)
            chk = check_qp(model, objective)
            viol, which = score_limits(uni, w, limits)
            respects = bool(viol <= 1e-6 and chk["check_ok"])
            out["endpoint"] = "repaired"
            out["status_source"] = "author-repair"
            out["objective_source"] = "independent-recompute-at-repaired-weights"
            repair_note = f"minimum-distance repair status={r_status}"
        else:
            repair_note = f"minimum-distance repair failed: {r_status} {r_note}"

    n_inacc = sum(1 for s in statuses if s == cp.OPTIMAL_INACCURATE)
    notes = []
    if not chk["check_ok"]:
        notes.append(f"CHECK FAILED: {chk['check_notes']}")
    if n_inacc:
        # Expected to fire on a barrier without crossover, and reported rather
        # than tolerated: "inaccurate" is exactly the state the mandate
        # question is about.
        notes.append(f"solver flagged inaccurate on {n_inacc}/{len(statuses)}")
    if retry_note:
        notes.append(retry_note)
    if repair_note:
        notes.append(repair_note)

    out.update({
        "status": status, "dnf": False, "n_runs": len(walls),
        "wall_s_median": round(statistics.median(walls), 4),
        "wall_s_min": round(min(walls), 4),
        "wall_s_max": round(max(walls), 4),
        "wall_s_all": [round(x, 4) for x in walls],
        "solver_internal_s": (round(statistics.median(internals), 4)
                              if internals else ""),
        "objective": objective,
        "obj_recomputed": chk["obj_recomputed"],
        "obj_abs_gap": f"{chk['obj_abs_gap']:.3e}",
        "obj_rel_gap": f"{chk['obj_rel_gap']:.3e}",
        "portfolio_variance": chk["portfolio_variance"],
        "portfolio_vol": math.sqrt(max(chk["portfolio_variance"], 0.0)),
        "check_ok": chk["check_ok"],
        "check_notes": chk["check_notes"],
        "check_budget_resid": f"{chk['budget_residual']:.2e}",
        "max_bound_violation": f"{chk['max_bound_violation']:.2e}",
        "n_positions": chk["n_positions"],
        "n_dust": chk["n_dust"],
        "max_weight": round(chk["max_weight"], 8),
        "worst_rel_violation": f"{viol:.3e}",
        # Scientific, not percent. An interior point with no crossover
        # typically lands a few multiples of its own tolerance outside a
        # binding limit, and `{viol:.2%}` renders a real 4.6e-6 breach as
        # "0.00%" — which reads as a false alarm and is exactly the
        # distinction this branch exists to measure. The gate stays at 1e-6.
        "limits_verdict": ("ok" if viol <= 1e-6
                           else f"VIOLATED {which} by {viol:.2e} of its scale"),
        "worst_limit": which if viol > 1e-6 else "-",
        "respects_mandate": respects,
        "repair_s": repair_s,
        "repair_status": repair_status,
        "repaired_from_endpoint": repaired_from_endpoint,
        "raw_status": raw_status,
        "raw_objective": raw_objective,
        "raw_obj_recomputed": raw_obj_recomputed,
        "raw_worst_rel_violation": f"{raw_worst_rel_violation:.3e}",
        "raw_limits_verdict": raw_limits_verdict,
        "notes": "; ".join(notes),
    })
    print(MARKER + json.dumps(out))
    return 0


def probe_main(args) -> int:
    """Does cuOpt's QP path error, warn, or silently ignore `crossover`?

    The docs say only "crossover is not supported for QP problems", which
    leaves three very different behaviours on the table. It matters which:
    "silently ignored" means an operator can believe they requested a basic
    solution endpoint when nothing happened at all, which is a
    strictly worse failure than a hard error. Reported as a first-class
    result, not a footnote.
    """
    import cvxpy as cp

    from src.credit import credit_constraint_builder, default_limits
    from src.universe import make_universe

    out = {"arm": "probe_crossover", "form": "dense", "hardware": "GPU",
           "solver": "CUOPT", "n_bonds": args.n_bonds}
    ok, why = cuopt_qp_reachability()
    if not ok:
        out.update({"status": "GPU-UNREACHABLE", "dnf": True, "notes": why})
        print(MARKER + json.dumps(out))
        return 3

    uni = make_universe(n_bonds=args.n_bonds, seed=args.seed)
    limits = default_limits(uni, turnover=args.turnover)
    d = np.load(args.data)
    fm = FactorModel(B=d["B"], f=d["f"], d=d["d"],
                     labels=list(d["labels"]), n_dropped=int(d["n_dropped"]))
    mu = d["mu"]
    builder = credit_constraint_builder(uni, limits, R=None)

    results = {}
    for label, extra in (("crossover_off", {"crossover": 0}),
                         ("crossover_on", {"crossover": 1}),
                         ("crossover_unset", {})):
        model, _ = build_qp_dense(fm.dense(), mu, args.risk_aversion,
                                  limits.w_max, constraint_builder=builder)
        opts = dict(solver_method="Barrier", time_limit=args.time_limit)
        opts.update(extra)
        try:
            model.problem.solve(solver=cp.CUOPT, verbose=False,
                                warm_start=False, **opts)
            w = model.weights()
            results[label] = {
                "status": model.problem.status,
                "objective": float(model.problem.value),
                "n_positions": int((w >= DUST_THRESHOLD).sum()),
                "n_dust": int(((w > DUST_THRESHOLD * 1e-3)
                               & (w < DUST_THRESHOLD)).sum()),
                "w": w,
            }
        except Exception as e:                                # noqa: BLE001
            results[label] = {"status": f"RAISED {type(e).__name__}",
                              "error": str(e)[:300], "w": None}

    off, on = results["crossover_off"], results["crossover_on"]
    if on.get("w") is None:
        verdict = "ERRORS — cuOpt rejects crossover on a QP outright"
    elif off.get("w") is None:
        verdict = "inconclusive: the crossover=0 control itself failed"
    else:
        l1 = float(np.abs(on["w"] - off["w"]).sum())
        if l1 <= 1e-9:
            verdict = ("SILENTLY IGNORED - identical solution with crossover "
                       "on and off. An operator could believe they cleaned a "
                       "solution to a basic endpoint when nothing ran.")
        else:
            verdict = (f"ACCEPTED and changed the answer (weight L1 "
                       f"{l1:.3e}) - the docs say unsupported, so this needs "
                       f"explaining before any result relies on it")
    out.update({
        "status": "probe_complete",
        "crossover_verdict": verdict,
        "detail": json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "w"}
                              for k, v in results.items()}),
    })
    print(MARKER + json.dumps(out))
    return 0


# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------

def run_worker(argv: list[str], timeout_s: float, threads: int
               ) -> tuple[dict | None, str]:
    """Spawn this script as a worker with CPU threads pinned.

    Thread counts are set in the CHILD's environment, before its NumPy /
    OpenBLAS / Rayon import, because that is the only point at which they take
    effect. Left unpinned, Clarabel and the BLAS underneath the dense arms
    each grab the whole box and the solver-stack headline is decided by a
    default nobody chose.
    """
    env = os.environ.copy()
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS"):
        env[var] = str(threads)
    cmd = [sys.executable, str(Path(__file__).resolve())] + argv
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout_s, env=env)
    except subprocess.TimeoutExpired:
        return None, f"worker wall timeout after {timeout_s:.0f}s"
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(MARKER):
            return json.loads(line[len(MARKER):]), ""
    tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-6:]
    return None, f"worker exit {proc.returncode}: " + " | ".join(tail)


def gpu_total_gib() -> float:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            return float(out.stdout.strip().splitlines()[0]) / 1024.0
    except Exception:                                         # noqa: BLE001
        pass
    return 16.0        # RTX 4080, the box this repo is measured on


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", default="",
                    help="comma-separated bond counts (default: the full "
                         f"grid through the Amundi index scale "
                         f"{AMUNDI_INDEX_N:,})")
    ap.add_argument("--arms", nargs="*", default=list(DEFAULT_ARMS),
                    choices=sorted(ARMS), help="arms to run")
    ap.add_argument("--objective", choices=("mv", "te"), default="mv",
                    help="mv: max mu'w - (lam/2) w'Sigma w. te: the same "
                         "against the benchmark, (w-b)'Sigma(w-b), which is "
                         "the M6 tracking-error mandate as a QP objective "
                         "rather than an SOC constraint")
    ap.add_argument("--sigma-basis", choices=("factor", "sample"),
                    default="factor",
                    help="factor: dense arms get the factored Sigma "
                         "materialized, so every arm solves the IDENTICAL "
                         "problem and the comparison is representation-only. "
                         "sample: dense arms get cov(R) and the factored arms "
                         "get its factor approximation - closer to practice, "
                         "confounded by design, error columns quantify it")
    ap.add_argument("--n-scenarios", type=int, default=20_000,
                    help="scenarios used to ESTIMATE Sigma; the QP itself is "
                         "scenario-free, so this sets risk-model quality, not "
                         "problem size")
    ap.add_argument("--risk-aversion", type=float, default=3.0)
    ap.add_argument("--turnover", type=float, default=0.25)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--time-limit", type=float, default=300.0)
    ap.add_argument("--threads", type=int, default=16,
                    help="CPU threads, pinned in every worker's environment")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--scen-seed", type=int, default=11)
    ap.add_argument("--gpu-mem-gib", type=float, default=0.0,
                    help="override the detected VRAM budget")
    ap.add_argument("--host-mem-gib", type=float, default=0.0,
                    help="override the detected host memory budget")
    ap.add_argument("--cpu-only", action="store_true",
                    help="CPU scenario generation, no GPU-idle check, and GPU "
                         "arms are dropped WITH A RECORDED REASON rather than "
                         "silently; for formulation checks on a box with no "
                         "card")
    ap.add_argument("--allow-busy-gpu", action="store_true")
    ap.add_argument("--allow-missing-gpu", action="store_true",
                    help="exit 0 even if the cuOpt arms were unreachable "
                         "(default is to exit non-zero and say so loudly)")
    ap.add_argument("--skip-calibration", action="store_true",
                    help="skip the Q-convention gate. For debugging only - a "
                         "published run must never use this")
    ap.add_argument("--probe-crossover", action="store_true",
                    help="also run the crossover-behaviour probe")
    ap.add_argument("--probe-only", action="store_true",
                    help="run ONLY the crossover probe and exit")
    ap.add_argument("--resume", action="store_true",
                    help="skip (n_bonds, arm) pairs already present in --out")
    ap.add_argument("--gate-n500-only", action="store_true",
                    help="P0 gate mode: run only --repair-n (default 500) so "
                         "the strict raw/optional-repair gate can be checked "
                         "before a full grid")
    ap.add_argument("--repair-strict-clarabel", action="store_true",
                    help="if a strict Clarabel row at any grid size converges "
                         "but misses the independent mandate gate, report the "
                         "explicit minimum-distance repaired endpoint and use "
                         "solve + repair as common-quality time")
    ap.add_argument("--repair-n", type=int, default=500,
                    help="bond count selected by --gate-n500-only (default "
                         "500); repair itself applies at every grid size")
    ap.add_argument("--out", type=str, default="results/qp_branch.csv")
    # Worker mode (internal): this script spawns itself for process isolation.
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--probe", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--arm", default="", help=argparse.SUPPRESS)
    ap.add_argument("--data", default="", help=argparse.SUPPRESS)
    ap.add_argument("--universe", default="", help=argparse.SUPPRESS)
    ap.add_argument("--sigma-npy", default="", help=argparse.SUPPRESS)
    ap.add_argument("--n-bonds", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--worker-repair-strict-clarabel", action="store_true",
                    help=argparse.SUPPRESS)
    ap.add_argument("--worker-repair-n", type=int, default=500,
                    help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.probe:
        return probe_main(args)
    if args.worker:
        return worker_main(args)

    from src.bench import assert_gpu_idle
    from src.credit import default_limits
    from src.scenarios import credit_scenarios, warmup_gpu
    from src.universe import make_universe

    if not args.cpu_only:
        # Hours of solves are worth nothing if the card is busy.
        assert_gpu_idle(raise_on_busy=not args.allow_busy_gpu)

    arms = list(args.arms)
    dropped_gpu: list[str] = []
    if args.cpu_only:
        dropped_gpu = [a for a in arms if ARMS[a]["gpu"]]
        arms = [a for a in arms if not ARMS[a]["gpu"]]
    if not arms and not args.probe_only:
        print("no arms left to run", file=sys.stderr)
        return 1

    grid = (tuple(int(x) for x in args.grid.split(",") if x.strip())
            if args.grid else DEFAULT_GRID)
    grid = grid_for_gate_mode(grid, args.gate_n500_only, repair_n=args.repair_n)
    host_gib = args.host_mem_gib or available_gib()
    gpu_gib = args.gpu_mem_gib or gpu_total_gib()

    print(f"M7 QP branch: {len(grid)} cells x {len(arms)} arms, "
          f"objective={args.objective}, sigma-basis={args.sigma_basis}, "
          f"{args.runs} timed runs per configuration")
    print(f"  memory budgets: host {host_gib:.1f} GiB "
          f"(cells capped at {HOST_BUDGET_FRACTION:.0%}), "
          f"device {gpu_gib:.1f} GiB (capped at {DEVICE_BUDGET_FRACTION:.0%})")
    if dropped_gpu:
        print(f"  --cpu-only: GPU arms dropped and RECORDED: "
              f"{', '.join(dropped_gpu)}")

    gpu_ok, gpu_why = cuopt_qp_reachability()
    print(f"  cuOpt QP reachability: {'OK' if gpu_ok else 'NOT REACHABLE'} "
          f"- {gpu_why}")
    # Latched HERE, not only when a worker reports it. A GPU arm can also be
    # refused for memory before its worker ever runs, and in that case the
    # three-way comparison is still incomplete — so the promise the banner
    # below makes about the exit code has to hold either way.
    unreachable_seen = bool(not gpu_ok and any(ARMS[a]["gpu"] for a in arms))
    if not gpu_ok and any(ARMS[a]["gpu"] for a in arms):
        print("\n" + "!" * 78)
        print("The GPU arms of this experiment cannot run in this environment.")
        print(gpu_why)
        print("The CPU arms will still run and their rows are valid; every GPU")
        print("row will be recorded as GPU-UNREACHABLE and this script will")
        print("exit non-zero. Pass --allow-missing-gpu to accept that, or")
        print("--cpu-only to drop the GPU arms deliberately.")
        print("!" * 78 + "\n", flush=True)

    tmp = Path(tempfile.mkdtemp(prefix="qp_branch_"))
    rows: list[dict] = []
    done: set[tuple] = set()
    if args.resume and Path(args.out).exists():
        import csv as _csv
        with open(args.out) as f:
            body = [ln for ln in f if not ln.startswith("#")]
        for r in _csv.DictReader(body):
            rows.append(dict(r))
            done.add(resume_key_from_row(r))
        print(f"  resuming: {len(done)} configuration(s) already measured")

    t_start = time.perf_counter()

    # --probe-only answers one question (what does `crossover` do to a cuOpt
    # QP) and must not spend hours on the sweep to get there.
    sweep = () if args.probe_only else grid
    for ci, n_bonds in enumerate(sweep, 1):
        print(f"\n[cell {ci}/{len(sweep)}] {n_bonds:,} bonds  "
              f"(dense Sigma {dense_matrix_gib(n_bonds):.2f} GiB, "
              f"host peak est. {dense_host_peak_gib(n_bonds):.1f} GiB)",
              flush=True)

        # Cheap refusal FIRST: never build a universe or draw scenarios for a
        # cell no arm can run. A refused cell is a row, not an omission.
        pending = [
            a for a in arms
            if resume_key_for_config(args, n_bonds, a) not in done
        ]
        if not pending:
            print("  all arms already measured", flush=True)
            continue

        # A crude pre-check with k=0 so the factored guard is conservative;
        # the real k is known once the factor model is fitted.
        runnable = []
        for arm in pending:
            status, reason, peak = memory_refusal(arm, n_bonds, 64, host_gib,
                                                  gpu_gib)
            if status:
                print(f"  {arm:<26}{status}: {reason[:70]}", flush=True)
                rec = {
                    "n_bonds": n_bonds, "arm": arm,
                    "form": ARMS[arm]["form"],
                    "hardware": ARMS[arm]["hardware"],
                    "solver": ARMS[arm]["solver"],
                    "method_class": ARMS[arm].get("method_class", ""),
                    "endpoint": ARMS[arm].get("endpoint", "raw"),
                    "quality_target": ARMS[arm].get("quality_target", "default"),
                    "status_source": "author-classification",
                    "arm_desc": ARMS[arm]["desc"],
                    "status": status, "dnf": True, "wall_s_median": "",
                    "q_gib_dense": round(dense_matrix_gib(n_bonds), 4),
                    "host_peak_est_gib": round(peak, 3),
                    "limits_verdict": status, "check_ok": False,
                    "check_notes": reason, "respects_mandate": False,
                    "notes": reason,
                }
                rec.update(row_config_context(args, n_bonds, arm))
                rows.append(rec)
                _write(args, rows, grid, host_gib, gpu_gib, gpu_why)
            else:
                runnable.append((arm, peak))
        if not runnable:
            continue

        # ---- build the cell's problem data once, share it with every arm ---
        uni = make_universe(n_bonds=n_bonds, seed=args.seed)
        limits = default_limits(uni, turnover=args.turnover)
        if ci == 1:
            print(f"  limits: {limits.describe()}", flush=True)
        if not args.cpu_only:
            warmup_gpu(uni)
        try:
            R, mu, sinfo = credit_scenarios(
                uni, n_scen=args.n_scenarios, seed=args.scen_seed,
                use_gpu=not args.cpu_only)
        except MemoryError as exc:
            reason = ("scenario generation exhausted memory before any QP "
                      f"solver ran: {exc}")
            print(f"  scenario_generation_oom: {reason[:120]}", flush=True)
            rows.extend(scenario_generation_nonresults(
                args, n_bonds, runnable, reason))
            _write(args, rows, grid, host_gib, gpu_gib, gpu_why)
            continue
        fm = fit_factor_model(uni, R)
        print(f"  scenarios: {sinfo['seconds']:.2f}s on {sinfo['device']}; "
              f"factor model K={fm.n_factors} "
              f"({fm.n_dropped} zero-variance directions dropped, exactly), "
              f"Hessian nnz {fm.quad_nnz:,} vs dense {n_bonds ** 2:,}",
              flush=True)

        diag: dict[str, float] = {}
        sigma_npy = ""
        if args.sigma_basis == "sample":
            Sigma_sample = np.cov(R, rowvar=False)
            diag = factor_fit_diagnostics(fm, Sigma_sample)
            sigma_npy = str(tmp / f"sigma_{n_bonds}.npy")
            np.save(sigma_npy, Sigma_sample)
            print(f"  sample basis: factored-vs-sample rel Frobenius "
                  f"{diag['sigma_rel_fro_err']:.4f}, median portfolio-vol "
                  f"error {diag['portvol_rel_err_median']:.4f} - the "
                  f"factored arms are solving a DIFFERENT problem by this "
                  f"much", flush=True)
            del Sigma_sample
        else:
            # Identical problem across arms by construction; the columns exist
            # so a reader never has to remember which basis produced the file.
            diag = {"sigma_rel_fro_err": 0.0, "sigma_max_rel_diag_err": 0.0,
                    "portvol_rel_err_median": 0.0, "portvol_rel_err_max": 0.0}

        data_npz = str(tmp / f"data_{n_bonds}.npz")
        np.savez(data_npz, mu=mu, B=fm.B, f=fm.f, d=fm.d,
                 labels=np.array(fm.labels), n_dropped=fm.n_dropped)
        uni_pkl = str(tmp / f"uni_{n_bonds}.pkl")
        with open(uni_pkl, "wb") as f:
            pickle.dump((uni, limits), f)
        del R, mu

        for arm, peak in runnable:
            print(f"  {arm:<26}", end="", flush=True)
            budget = args.time_limit * (args.runs + 2) + 600
            argv = ["--worker", "--arm", arm, "--data", data_npz,
                    "--universe", uni_pkl, "--runs", str(args.runs),
                    "--time-limit", str(args.time_limit),
                    "--threads", str(args.threads),
                    "--risk-aversion", str(args.risk_aversion),
                    "--objective", args.objective]
            if args.repair_strict_clarabel:
                argv.append("--worker-repair-strict-clarabel")
                argv += ["--worker-repair-n", str(args.repair_n)]
            if args.skip_calibration:
                argv.append("--skip-calibration")
            if sigma_npy and ARMS[arm]["form"] == "dense":
                argv += ["--sigma-npy", sigma_npy]
            rec, why = run_worker(argv, timeout_s=budget, threads=args.threads)
            if rec is None:
                rec = {"arm": arm, "form": ARMS[arm]["form"],
                       "hardware": ARMS[arm]["hardware"],
                       "solver": ARMS[arm]["solver"],
                       "method_class": ARMS[arm].get("method_class", ""),
                       "endpoint": ARMS[arm].get("endpoint", "raw"),
                       "quality_target": ARMS[arm].get("quality_target", "default"),
                       "status_source": "author-classification",
                       "status": "worker_failure", "dnf": True,
                       "wall_s_median": "", "limits_verdict": "DNF",
                       "check_ok": False, "check_notes": why,
                       "respects_mandate": False, "notes": why}
                print(f"DNF ({why[:60]})", flush=True)
            else:
                if rec.get("status") == "GPU-UNREACHABLE":
                    unreachable_seen = True
                if rec.get("dnf"):
                    print(f"{rec.get('status', 'DNF')}  "
                          f"{str(rec.get('notes', ''))[:60]}", flush=True)
                else:
                    print(f"{float(rec['wall_s_median']):8.2f}s  "
                          f"pos={rec.get('n_positions', '')}  "
                          f"limits={str(rec.get('limits_verdict', ''))[:34]}",
                          flush=True)
            rec.setdefault("n_bonds", n_bonds)
            rec.update({
                "host_peak_est_gib": round(peak, 3),
                "scenario_gen_s": round(sinfo["seconds"], 3),
            })
            rec.update(row_config_context(args, n_bonds, arm))
            rec.update({k: round(v, 6) for k, v in diag.items()})
            rows.append(rec)
            # Checkpoint after every configuration: the dense cells are the
            # ones that will get OOM-killed, and losing the arms that already
            # succeeded to a crash in the one after is pure waste.
            _write(args, rows, grid, host_gib, gpu_gib, gpu_why)

        for p in (data_npz, uni_pkl, sigma_npy):
            if p:
                Path(p).unlink(missing_ok=True)

    # ---- crossover probe ---------------------------------------------------
    if args.probe_crossover or args.probe_only:
        n_probe = min(grid)
        print(f"\n[probe] crossover behaviour on a {n_probe:,}-bond QP",
              flush=True)
        uni = make_universe(n_bonds=n_probe, seed=args.seed)
        if not args.cpu_only:
            warmup_gpu(uni)
        R, mu, _ = credit_scenarios(uni, n_scen=args.n_scenarios,
                                    seed=args.scen_seed,
                                    use_gpu=not args.cpu_only)
        fm = fit_factor_model(uni, R)
        pnpz = str(tmp / "probe.npz")
        np.savez(pnpz, mu=mu, B=fm.B, f=fm.f, d=fm.d,
                 labels=np.array(fm.labels), n_dropped=fm.n_dropped)
        rec, why = run_worker(
            ["--probe", "--data", pnpz, "--n-bonds", str(n_probe),
             "--seed", str(args.seed), "--turnover", str(args.turnover),
             "--risk-aversion", str(args.risk_aversion),
             "--time-limit", str(args.time_limit)],
            timeout_s=args.time_limit * 6 + 600, threads=args.threads)
        if rec is None:
            rec = {"arm": "probe_crossover", "status": "worker_failure",
                   "dnf": True, "notes": why, "n_bonds": n_probe}
        print(f"  {rec.get('crossover_verdict', rec.get('notes', ''))}",
              flush=True)
        rows.append(rec)
        _write(args, rows, grid, host_gib, gpu_gib, gpu_why)

    _print_table(rows)
    print(f"\nM7 finished in {(time.perf_counter() - t_start) / 60:.1f} min")
    print(f"Wrote {args.out}")

    if unreachable_seen and not args.allow_missing_gpu:
        print("\n" + "!" * 78, file=sys.stderr)
        print("GPU ARMS DID NOT RUN. The CPU rows in the CSV are valid; the "
              "three-way", file=sys.stderr)
        print("comparison is NOT complete and must not be reported as one.",
              file=sys.stderr)
        print(gpu_why, file=sys.stderr)
        print("!" * 78, file=sys.stderr)
        return 2
    return 0


def _print_table(rows: list[dict]) -> None:
    print("\n" + "=" * 118)
    print(f"{'n_bonds':>8} {'arm':<26} {'status':<20} {'wall_med(s)':>11} "
          f"{'pos':>7} {'obj_rel_gap':>12} {'mandate':<9} {'limits':<20}")
    print("-" * 118)
    for r in rows:
        print(f"{str(r.get('n_bonds', '')):>8} {str(r.get('arm', '')):<26} "
              f"{str(r.get('status', ''))[:20]:<20} "
              f"{str(r.get('wall_s_median', '')):>11} "
              f"{str(r.get('n_positions', '')):>7} "
              f"{str(r.get('obj_rel_gap', '')):>12} "
              f"{str(r.get('respects_mandate', '')):<9} "
              f"{str(r.get('limits_verdict', ''))[:20]:<20}")
    print("=" * 118)


def _write(args, rows: list[dict], grid, host_gib: float, gpu_gib: float,
           gpu_why: str) -> None:
    from src.bench import capture_environment, write_csv

    env = capture_environment()
    # capture_environment version-stamps the LP branch's solvers; the QP
    # branch's CPU peers have to be added here or the CSV cannot reproduce
    # itself.
    from src.bench import _pkg_version
    for pkg in ("clarabel", "osqp", "scs", "scipy", "pandas"):
        env[pkg] = _pkg_version(pkg)

    env["experiment"] = (
        "M7 QP branch: mean-variance on cuOpt barrier (GPU) vs a CPU "
        "interior-point solver vs the factored low-rank-plus-diagonal "
        "formulation on CPU, on ONE universe and ONE limit sheet")
    env["hypothesis"] = (
        "cuOpt QP is barrier-only with no crossover, so mean-variance "
        "inherits none of the PDLP innovation; the open question is whether a "
        "cuOpt dense stack reaches Amundi's index-scale covariance objective "
        f"({AMUNDI_INDEX_N:,} securities) or whether the factored "
        "representation repeals it on a CPU")
    env["objective"] = (
        "max mu'w - (lam/2) w'Sigma w" if args.objective == "mv"
        else "max mu'w - (lam/2) (w-b)'Sigma (w-b), b = index weights")
    env["sigma_basis"] = args.sigma_basis
    env["sigma_source"] = (
        "Sigma = B diag(f) B' + diag(d) with B analytic from the scenario "
        "generator's own factor structure (beta_i = -SD_i * OAS_i * hy_mult_i) "
        "and f, d estimated from the same scenario draw the CVaR branch uses; "
        "F eigendecomposed and folded into B, which is exact and makes the "
        "factored Hessian strictly diagonal"
        + (". Dense arms received the sample covariance cov(R) instead, so "
           "the factored arms solve a DIFFERENT problem - see "
           "sigma_rel_fro_err and portvol_rel_err_median"
           if args.sigma_basis == "sample"
           else ". Dense arms received exactly that Sigma materialized, so "
                "every arm solves the IDENTICAL problem and the comparison is "
                "representation-and-solver-stack only"))
    env["grid_n_bonds"] = ",".join(str(n) for n in grid)
    env["gate_mode"] = (
        f"n500_only_at_{args.repair_n}" if getattr(args, "gate_n500_only", False)
        else "full_grid")
    env["amundi_index_scale_n"] = str(AMUNDI_INDEX_N)
    env["arms"] = ", ".join(f"{a} [{ARMS[a]['desc']}]" for a in args.arms)
    env["cuopt_method"] = ("pinned to Barrier on every cuOpt solve; the "
                           "CUOPT_METHOD default is Concurrent and CVXPY does "
                           "not force Barrier for a plain quadratic objective")
    env["cuopt_qp_reachability"] = gpu_why
    env["crossover"] = ("NOT AVAILABLE for QP in cuOpt. The LP branch's "
                        "basic-solution cleanup endpoint has no analogue "
                        "here, which is why n_positions, n_dust and "
                        "respects_mandate are the headline columns")
    env["tolerances"] = (
        "P0 ratio policy: a timing ratio is printable only when both rows "
        "converged, the independent checker passed, the mandate gate passed, "
        "the recomputed objective agrees with the solver objective, and "
        "endpoint plus achieved_quality_target match. Strict Clarabel arms set "
        "tol_feas=tol_gap_abs=tol_gap_rel=1e-9 and max_iter=2000; raw cuOpt "
        "barrier rows remain a separate solver-stack endpoint/quality target.")
    env["ratio_policy"] = (
        "solver_stack_ratio, not a pure device comparison. Rows outside the "
        "mechanical gate get NaN plus solver_stack_ratio_exclusion_reason.")
    env["q_convention_calibration"] = (
        "SKIPPED (--skip-calibration) - this run is not publishable"
        if args.skip_calibration else
        f"every arm solved a 2-variable QP with closed-form optimum "
        f"{CALIBRATION_OBJ_STAR} at p*={CALIBRATION_P_STAR} before any timing; "
        f"a doubled risk aversion would return "
        f"{CALIBRATION_OBJ_IF_DOUBLED}")
    env["n_scenarios_for_sigma"] = str(args.n_scenarios)
    env["risk_aversion"] = str(args.risk_aversion)
    env["turnover_one_way"] = str(args.turnover)
    env["universe_seed"] = str(args.seed)
    env["scenario_seed"] = str(args.scen_seed)
    env["runs_per_config"] = str(args.runs)
    env["time_limit_s"] = str(args.time_limit)
    env["threads_cpu_solvers"] = str(args.threads)
    env["thread_pinning"] = ("OMP/OPENBLAS/MKL/NUMEXPR/RAYON/VECLIB set in "
                             "every worker's environment before import")
    env["host_mem_budget_gib"] = f"{host_gib:.1f}"
    env["device_mem_budget_gib"] = f"{gpu_gib:.1f}"
    env["memory_model"] = (
        f"dense host peak estimated at {DENSE_Q_BYTES_PER_ENTRY_PEAK} B per "
        f"Hessian entry (fp64 matrix + COO + CSR in flight), device at "
        f"{DEVICE_Q_MULTIPLE:g}x the matrix; cells over "
        f"{HOST_BUDGET_FRACTION:.0%} of host or "
        f"{DEVICE_BUDGET_FRACTION:.0%} of device are REFUSED and recorded. "
        f"These are estimates and must be recalibrated from the first real "
        f"OOM, as run_sweep.py's BYTES_PER_NNZ_PEAK was")
    env["isolation"] = "one clean subprocess per (n_bonds, arm)"
    env["controls"] = ("identical universe, seeds, limit sheet, risk aversion "
                       "and CVXPY model definition across arms; only the "
                       "Hessian representation and solver stack change")
    env["not_controlled"] = (
        "cuOpt barrier and Clarabel are different interior-point "
        "implementations (predictor-corrector over cuDSS vs homogeneous "
        "self-dual embedding over sparse LDL), not the same algorithm; "
        "tolerances are each solver's own defaults; CVXPY canonicalization of "
        "a dense P grows as N^2 and is reported separately in compile_s "
        "rather than eliminated"
        + ("; under sigma-basis=sample the factored arms additionally solve a "
           "different risk model" if args.sigma_basis == "sample" else ""))

    annotated = normalise_rows_for_write(rows)
    keys: list[str] = list(P0_SCHEMA_KEYS)
    for r in annotated:
        for k in r:
            if k not in keys:
                keys.append(k)
    norm = [{k: r.get(k, "") for k in keys} for r in annotated]
    for r in norm:
        if isinstance(r.get("wall_s_all"), list):
            r["wall_s_all"] = json.dumps(r["wall_s_all"])
    if norm:
        write_csv(args.out, norm, env)


if __name__ == "__main__":
    sys.exit(main())

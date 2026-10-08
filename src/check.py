"""Independent solution checker.

Trusts nothing from the solver except the weight vector. CVaR is recomputed
directly from the scenario matrix by exact discrete minimization of the
Rockafellar-Uryasev objective — no LP algebra, no auxiliary variables — so a
solver that reports "Optimal" while violating its tolerances (a documented
failure mode of first-order methods at loose tolerances) is caught here.

Every number a benchmark table reports should have passed through this file.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DUST_THRESHOLD = 1e-6

#: Half-width of the acceptable band around the reference objective, in basis
#: points. The gate is SYMMETRIC on purpose. Worse than this means the solve
#: did not reach the reference optimum. Better than this means the point is
#: outside the feasible set — you cannot beat an exact optimum, so "better"
#: is a feasibility failure wearing a good number, and it used to pass here
#: silently while the mirror-image failure was flagged.
REFERENCE_GAP_TOL_BP = 1.0


def empirical_cvar(losses: np.ndarray, alpha: float) -> tuple[float, float]:
    """Exact discrete CVaR_alpha (uniform weights) and its minimizing t (VaR).

    Evaluates g(t) = t + mean(max(loss - t, 0)) / (1 - alpha) at every loss
    value via sorted prefix sums; the discrete minimizer is one of the loss
    values, so this is exact, O(S log S).
    """
    s = np.sort(losses)[::-1]  # descending: worst first
    S = s.shape[0]
    # suffix -> prefix sums of the sorted-descending array
    cum = np.concatenate(([0.0], np.cumsum(s)))  # cum[k] = sum of k worst
    k = np.arange(1, S + 1)
    # g evaluated at t = s[k-1]: t + (sum of losses above t) / ((1-a) S)
    g = s[k - 1] + (cum[k - 1] - (k - 1) * s[k - 1]) / ((1.0 - alpha) * S)
    i = int(np.argmin(g))
    return float(g[i]), float(s[i])


@dataclass
class CheckResult:
    ok: bool
    budget_residual: float
    #: sum(w) - 1, WITH ITS SIGN. `budget_residual` is the magnitude, which is
    #: the right thing to gate on but the wrong thing to report: an
    #: over-invested book (sum(w) > 1) is buying return with money it does not
    #: have, an under-invested one is leaving the mandate unfilled, and those
    #: are different findings. Every benchmark row here used to record only
    #: the magnitude while the write-up asserted the direction — a claim no
    #: committed artifact could support. Recorded separately so the gate keeps
    #: its old semantics and the direction stops being folklore.
    budget_signed: float
    max_bound_violation: float
    cvar_empirical: float
    var_empirical: float
    objective_recomputed: float
    objective_solver: float
    objective_abs_gap: float          # |recomputed - solver-reported|
    objective_renormalized: float     # objective of w / sum(w)
    n_dust_positions: int             # 0 < w < DUST_THRESHOLD
    n_positions: int
    # Strictly positive weights, at ANY magnitude. This is the vertex test,
    # and the dust counters cannot do its job: a HiGHS interior point stopped
    # at ipm_optimality_tolerance=1e-8 puts its "zero" weights at ~1e-16, far
    # under DUST_THRESHOLD, so n_positions and n_dust_positions are identical
    # for the interior point and the vertex it converged to while this count
    # separates them cleanly (measured, 200-bond credit instance: 200/200
    # strictly positive without crossover, 109/200 with it and 109/200 for
    # simplex). A vertex has at most as many nonzeros as it has active
    # constraints; an interior point has all of them.
    n_nonzero_weights: int
    # (ref_obj - renormalized_obj) * 1e4. Computed on BUDGET-RENORMALIZED
    # weights so a solution that "beats" the exact reference by holding
    # sum(w) = 1.0001 (feasibility-tolerance inflation) cannot show a
    # negative gap. Gated SYMMETRICALLY at +/-REFERENCE_GAP_TOL_BP: worse
    # than that => did not reach the optimum; better than that => the point
    # is infeasible. Both are failures.
    gap_vs_reference_bp: float | None
    weight_l1_vs_reference: float | None
    notes: str = ""

    def summary(self) -> str:
        parts = [
            f"ok={self.ok}",
            f"budget_resid={self.budget_residual:.2e}",
            f"bound_viol={self.max_bound_violation:.2e}",
            f"CVaR={self.cvar_empirical:.6f}",
            f"obj_gap={self.objective_abs_gap:.2e}",
            f"dust={self.n_dust_positions}",
        ]
        if self.gap_vs_reference_bp is not None:
            parts.append(f"vs_ref={self.gap_vs_reference_bp:+.3f}bp")
        return " ".join(parts)


def check_solution(
    w: np.ndarray,
    solver_objective: float,
    R: np.ndarray,
    mu: np.ndarray,
    alpha: float,
    risk_aversion: float,
    w_max: float,
    reference_w: np.ndarray | None = None,
    reference_objective: float | None = None,
    feasibility_tol: float = 1e-6,
    objective_tol: float = 1e-4,
) -> CheckResult:
    """Validate a portfolio produced by any solver, from first principles."""
    w = np.asarray(w, dtype=np.float64).ravel()

    budget_signed = float(w.sum()) - 1.0
    budget_residual = abs(budget_signed)
    lower_viol = float(np.maximum(-w, 0.0).max(initial=0.0))
    upper_viol = float(np.maximum(w - w_max, 0.0).max(initial=0.0))
    max_bound_violation = max(lower_viol, upper_viol)

    losses = -(R @ w)
    cvar, var = empirical_cvar(losses, alpha)
    objective = float(mu @ w) - risk_aversion * cvar
    obj_gap = abs(objective - solver_objective)

    # Budget-renormalized objective: divides out feasibility-tolerance
    # inflation so no tolerance-loose solution can appear to beat the
    # exact reference.
    w_sum = float(w.sum())
    if w_sum > 0.5:
        w_norm = w / w_sum
        cvar_n, _ = empirical_cvar(-(R @ w_norm), alpha)
        objective_renorm = float(mu @ w_norm) - risk_aversion * cvar_n
    else:
        objective_renorm = objective

    n_dust = int(((w > DUST_THRESHOLD * 1e-3) & (w < DUST_THRESHOLD)).sum())
    n_pos = int((w >= DUST_THRESHOLD).sum())
    n_nonzero = int((w > 0.0).sum())

    gap_bp = None
    l1 = None
    if reference_objective is not None:
        # positive => the reference found a better (higher) objective,
        # expressed in basis points of portfolio value
        gap_bp = float((reference_objective - objective_renorm) * 1e4)
    if reference_w is not None:
        l1 = float(np.abs(w - np.asarray(reference_w).ravel()).sum())

    notes = []
    if budget_residual > feasibility_tol:
        notes.append("budget violated")
    if max_bound_violation > feasibility_tol:
        notes.append("bounds violated")
    if obj_gap > objective_tol:
        notes.append("solver objective disagrees with recomputation")
    if gap_bp is not None and gap_bp > REFERENCE_GAP_TOL_BP:
        notes.append(f"more than {REFERENCE_GAP_TOL_BP:g}bp worse than "
                     f"reference")
    if gap_bp is not None and gap_bp < -REFERENCE_GAP_TOL_BP:
        notes.append(
            f"more than {REFERENCE_GAP_TOL_BP:g}bp BETTER than the reference "
            f"on renormalized weights: an exact optimum cannot be beaten, so "
            f"this is infeasibility, not solution quality")

    return CheckResult(
        ok=not notes,
        budget_residual=budget_residual,
        budget_signed=budget_signed,
        max_bound_violation=max_bound_violation,
        cvar_empirical=cvar,
        var_empirical=var,
        objective_recomputed=objective,
        objective_solver=float(solver_objective),
        objective_abs_gap=obj_gap,
        objective_renormalized=objective_renorm,
        n_dust_positions=n_dust,
        n_positions=n_pos,
        n_nonzero_weights=n_nonzero,
        gap_vs_reference_bp=gap_bp,
        weight_l1_vs_reference=l1,
        notes="; ".join(notes),
    )

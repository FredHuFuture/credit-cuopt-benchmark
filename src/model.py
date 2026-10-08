"""Mean-CVaR LP via the Rockafellar-Uryasev linearization, in CVXPY.

One model definition drives every solver in the benchmark (cuOpt GPU PDLP,
HiGHS, OR-Tools CPU PDLP) so the comparison is apples-to-apples: identical
canonicalization, identical matrices.

    max  mu'w - lam * CVaR_alpha(-R w)
    s.t. CVaR linearization:  u_s >= -(R w)_s - t,  u >= 0
         sum(w) = 1,  0 <= w <= w_max
         [+ any extra linear constraints, e.g. the M2 credit set]

CVaR_alpha = t + (1 / ((1-alpha) S)) * sum(u)  for uniform scenario weights.

Modeling rule (learned the hard way in M0): never use big-M sentinels for
"unbounded" — bad scaling makes first-order methods ~10x slower. CVXPY
variables are unbounded by default; keep them that way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import cvxpy as cp
import numpy as np

#: A constraint builder receives the freshly created weight variable and
#: returns (constraints, names). It is a callback rather than a list because
#: `w` is created inside `build_mean_cvar` — see `src/credit.py` for the
#: credit constraint set built this way.
ConstraintBuilder = Callable[[cp.Variable], "tuple[list, list[str]]"]


@dataclass
class CVaRModel:
    """A built Mean-CVaR problem plus everything needed to check its solution."""

    problem: cp.Problem
    w: cp.Variable
    t: cp.Variable
    u: cp.Variable
    R: np.ndarray
    mu: np.ndarray
    alpha: float
    risk_aversion: float
    w_max: float
    extra_constraint_names: list[str] = field(default_factory=list)

    @property
    def n_assets(self) -> int:
        return self.R.shape[1]

    @property
    def n_scenarios(self) -> int:
        return self.R.shape[0]

    @property
    def nnz_estimate(self) -> int:
        """Nonzeros of the dominant scenario block, S*(N+2)."""
        s, n = self.R.shape
        return s * (n + 2)

    def weights(self) -> np.ndarray:
        if self.w.value is None:
            raise ValueError("problem has not been solved")
        return np.asarray(self.w.value).ravel()


def build_mean_cvar(
    R: np.ndarray,
    mu: np.ndarray,
    alpha: float = 0.95,
    risk_aversion: float = 1.0,
    w_max: float = 0.05,
    constraint_builder: ConstraintBuilder | None = None,
) -> CVaRModel:
    """Build the Rockafellar-Uryasev Mean-CVaR LP.

    `constraint_builder` is the credit hook: `src/credit.py` supplies rating
    bands, sector caps, issuer limits, duration-bucket matching, a DTS cap and
    a turnover budget, all built on the `w` variable created here.
    """
    S, N = R.shape
    if mu.shape != (N,):
        raise ValueError(f"mu has shape {mu.shape}, expected ({N},)")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")

    w = cp.Variable(N, name="w", nonneg=True)
    t = cp.Variable(name="t")           # VaR level (free)
    u = cp.Variable(S, name="u", nonneg=True)  # tail excess losses

    cvar = t + cp.sum(u) / ((1.0 - alpha) * S)

    constraints: list[cp.Constraint] = [
        cp.sum(w) == 1.0,
        w <= w_max,
        u + t + R @ w >= 0.0,   # u_s >= loss_s - t with loss = -(R w)
    ]
    names = ["budget", "w_max", "cvar_tail"]
    if constraint_builder is not None:
        extra, extra_names = constraint_builder(w)
        constraints.extend(extra)
        names.extend(extra_names)

    objective = cp.Maximize(mu @ w - risk_aversion * cvar)
    problem = cp.Problem(objective, constraints)

    return CVaRModel(
        problem=problem, w=w, t=t, u=u, R=R, mu=mu, alpha=alpha,
        risk_aversion=risk_aversion, w_max=w_max,
        extra_constraint_names=names,
    )

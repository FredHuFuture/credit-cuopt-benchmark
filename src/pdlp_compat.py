"""Workaround for an upstream CVXPY bug in the OR-Tools PDLP interface.

cvxpy 1.9.2's ``pdlp_conif.PDLP.apply`` evaluates ``if problem.lower_bounds:``
where ``lower_bounds`` is a NumPy array whenever the model has bounded
variables (e.g. ``nonneg=True``) — raising "truth value of an array is
ambiguous". Portfolio problems always have bounded weights, so the stock
interface cannot solve them at all.

``PatchedPDLP`` subclasses the stock interface and overrides ``apply`` with
the two-character fix (``is not None``). Everything else — solve path, status
mapping, option handling — is inherited unchanged, so benchmark behavior is
identical to what stock CVXPY would do if the bug were fixed upstream.

The ``apply`` body below is copied from cvxpy 1.9.2 (Apache-2.0, the CVXPY
authors) with only the truthiness fix. TODO: report upstream and drop this
module once a fixed cvxpy ships.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_array

import cvxpy.settings as s
from cvxpy import Zero
from cvxpy.constraints import NonNeg
from cvxpy.reductions.solvers.conic_solvers.pdlp_conif import PDLP


class PatchedPDLP(PDLP):
    """OR-Tools PDLP interface with the bounded-variables fix, plus
    solver-internal time capture (stock CVXPY discards solve_log entirely,
    leaving the CPU-PDLP row without the one number the tier-1 GPU-vs-CPU
    comparison needs)."""

    def name(self) -> str:
        # CVXPY's custom-solver path requires a name distinct from the
        # built-in registry; the underlying solver is still OR-Tools PDLP.
        return "PDLP_FIXED"

    def solve_via_data(self, data, warm_start, verbose, solver_opts,
                       solver_cache=None):
        from ortools.pdlp import solvers_pb2
        from ortools.pdlp.python import pdlp

        parameters = solvers_pb2.PrimalDualHybridGradientParams()
        parameters.verbosity_level = 3 if verbose else 0
        parameters.presolve_options.use_glop = True  # stock CVXPY default
        if "parameters_proto" in solver_opts:
            proto = solver_opts["parameters_proto"]
            if not isinstance(proto, solvers_pb2.PrimalDualHybridGradientParams):
                return {"status": s.SOLVER_ERROR}
            parameters.MergeFrom(proto)
        if "time_limit_sec" in solver_opts:
            parameters.termination_criteria.time_sec_limit = float(
                solver_opts["time_limit_sec"])

        result = pdlp.primal_dual_hybrid_gradient(
            data[self.PDLP_MODEL], parameters)
        solution = {
            "primal": result.primal_solution,
            "dual": result.dual_solution,
            "status": self._status_map(result.solve_log),
            # The addition vs stock: keep the solver's own timer.
            "solve_time_sec": result.solve_log.solve_time_sec,
        }
        info = self._get_convergence_info(
            result.solve_log.solution_stats, result.solve_log.solution_type)
        solution["value"] = (info.primal_objective if info is not None
                             else -np.inf)
        return solution

    def invert(self, solution, inverse_data):
        sol = super().invert(solution, inverse_data)
        if isinstance(solution, dict) and "solve_time_sec" in solution:
            sol.attr[s.SOLVE_TIME] = solution["solve_time_sec"]
        return sol

    def apply(self, problem):
        from ortools.pdlp.python import pdlp

        data = {}
        inv_data = {self.VAR_ID: problem.x.id}
        if not problem.formatted:
            problem = self.format_constraints(problem, None)
        data[s.PARAM_PROB] = problem
        data[self.DIMS] = problem.cone_dims

        constr_map = problem.constr_map
        inv_data["constraints"] = constr_map[Zero] + constr_map[NonNeg]

        c, d, A, b = problem.apply_parameters()
        A = csr_array(A)
        data["num_constraints"], data["num_vars"] = A.shape

        model = pdlp.QuadraticProgram()
        model.objective_offset = d.item() if isinstance(d, np.ndarray) else d
        model.objective_vector = c
        # Upstream writes `if problem.lower_bounds:` which raises on ndarray.
        if problem.lower_bounds is not None:
            model.variable_lower_bounds = problem.lower_bounds
        else:
            model.variable_lower_bounds = np.full_like(c, -np.inf)
        if problem.upper_bounds is not None:
            model.variable_upper_bounds = problem.upper_bounds
        else:
            model.variable_upper_bounds = np.full_like(c, np.inf)

        model.constraint_matrix = A
        constraint_lower_bounds = np.full_like(b, -np.inf)
        constraint_upper_bounds = np.full_like(b, np.inf)
        num_eq = problem.cone_dims.zero
        constraint_lower_bounds[:num_eq] = -b[:num_eq]
        constraint_upper_bounds[:num_eq] = -b[:num_eq]
        constraint_lower_bounds[num_eq:] = -b[num_eq:]

        model.constraint_lower_bounds = constraint_lower_bounds
        model.constraint_upper_bounds = constraint_upper_bounds

        data[self.PDLP_MODEL] = model
        data['lb'] = problem.lower_bounds
        data['ub'] = problem.upper_bounds
        data['l'] = constraint_lower_bounds
        data['u'] = constraint_upper_bounds
        data['A'], data['c'] = A, c
        return data, inv_data

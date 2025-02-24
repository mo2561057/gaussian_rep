from typing import Callable, List

import numpy as np
import estimagic as em
import cvxpy as cp

from gaussian_rep.utils import get_kronecker_function, get_objective_function, get_derivative_objective_function
from gaussian_rep.utils import get_dual_objective_function, get_dual_constraint, solve_dual_problem

def estimate_model_primal(
        params: np.ndarray,
        objective_function : Callable[[np.ndarray], float],
        optimagic_options : dict
    ) -> dict:
    res = em.maximize(
        objective_function,
        params,
        **optimagic_options
    )
    return res


def recursively_estimate_multivariate_dual(
        S:Callable[[np.ndarray], np.ndarray],
        s:Callable[[np.ndarray], np.ndarray],
        W:Callable[[np.ndarray], np.ndarray],
        y: np.ndarray,
        X: np.ndarray,
        beta: List[np.ndarray]=[],
):
    S = [S] if callable(S) else S
    s = [s] if callable(s) else s

    # Estimate the first problem
    S_current = S if callable(S) else S[0]
    s_current = s if callable(s) else s[0]

    y_current = y[:,0] if y.ndim > 1 else y

    T = get_kronecker_function(S_current, W)
    t = get_kronecker_function(s_current, W)
    dual_objective_function = get_dual_objective_function()
    dual_constraint = get_dual_constraint(T, t, y_current, X)

    rslt = solve_dual_problem(dual_objective_function, dual_constraint, len(y_current), algorithm=cp.SCS)
    beta.append(rslt)

    if len(S) > 1:
        return recursively_estimate_multivariate_dual(S[1:],s[1:], T, y[:,1:], (y_current,X), beta)    
    else:
        return beta
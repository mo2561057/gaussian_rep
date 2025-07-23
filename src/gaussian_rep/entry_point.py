from typing import Callable, List

import numpy as np
import cvxpy as cp

from gaussian_rep.utils import get_kronecker_function, get_objective_function, get_derivative_objective_function
from gaussian_rep.utils import get_dual_objective_function, get_dual_constraint, solve_dual_problem

def recursively_estimate_multivariate_dual(
        S:Callable[[np.ndarray], np.ndarray],
        s:Callable[[np.ndarray], np.ndarray],
        W:Callable[[np.ndarray], np.ndarray],
        y: np.ndarray,
        X: np.ndarray,
        beta: List[np.ndarray]=[],
        algorithm=cp.SCS,
        algorithm_options={},
        regularization=None,
        tol: float = 1e-05
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

    if regularization is not None:
        regularization_procedure_current = regularization(T, y_current, X)
    else:
        regularization_procedure_current = None    
    
    rslt = solve_dual_problem(
        dual_objective_function,
        dual_constraint,
        len(y_current),
        algorithm=algorithm,
        algorithm_options=algorithm_options,
        regularization=regularization_procedure_current,
        tol=tol)
    beta.append(rslt)

    if len(S) > 1:
        T_ = get_kronecker_function(S_current, W)
        return recursively_estimate_multivariate_dual(
            S[1:],s[1:], T_, y[:,1:], np.concatenate([y_current.reshape(y_current.shape[0],1),X],axis=1), beta, algorithm=algorithm, algorithm_options=algorithm_options)    
    else:
        return beta
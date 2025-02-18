from typing import Callable, List

import numpy as np
import estimagic as em
import cvxpy as cp

from gaussian_rep.utils import get_kronecker_function, get_objective_function, get_derivative_objective_function


def estimate_model(
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

def recursively_estimate_model_multiple_outcomes(
        S : List[Callable[[np.ndarray], np.ndarray]],
        s : List[Callable[[np.ndarray], np.ndarray]],
        W : List[Callable[[np.ndarray], np.ndarray]],
        X : np.ndarray,
        y : np.ndarray):
    pass

def get_full_likelihood_function(
        S : Callable[[np.ndarray], np.ndarray],
        s : Callable[[np.ndarray], np.ndarray],
        W : Callable[[np.ndarray], np.ndarray],
        y : np.ndarray,
        X : np.ndarray
    ) -> Callable[[np.ndarray], np.ndarray]:
    """
    Build full likelihood function from components.

    Args:
        S : Outcome component of Gaussian representation
        s : Gradient of S
        W : Covariate component of Gaussian representation
        X : numpy array of covariates
        Y : numpy array of outcomes

    Returns:
        Callable[[np.ndarray], np.ndarray]: Full likelihood function
    """
    T = get_kronecker_function(S, W)

    #Not sure if that is correct
    t = get_kronecker_function(s, W)
    return get_objective_function(T, t, y, X)

def get_derivative_likelihood_function(
        S : Callable[[np.ndarray], np.ndarray],
        s : Callable[[np.ndarray], np.ndarray],
        W : Callable[[np.ndarray], np.ndarray],
        y : np.ndarray,
        X : np.ndarray
    ) -> Callable[[np.ndarray], np.ndarray]:
    """
    Build full likelihood function from components.

    Args:
        S : Outcome component of Gaussian representation
        s : Gradient of S
        W : Covariate component of Gaussian representation
        X : numpy array of covariates
        Y : numpy array of outcomes

    Returns:
        Callable[[np.ndarray], np.ndarray]: Full likelihood function
    """
    T = get_kronecker_function(S, W)
    t = get_kronecker_function(s, W)


    #Not sure if that is correct
    return get_derivative_objective_function(T, t, y, X)


def solve_dual_problem(dual_objective_function : Callable[[np.ndarray, np.ndarray], float], 
                       foc_gradient : Callable[[np.ndarray, np.ndarray], np.ndarray],
                       n : int,
                       algorithm : str = "ECOS"
                       ):
                       
    u = cp.Variable(n)
    v = cp.Variable(n)
    condition_1 = foc_gradient(u,v)==0
    constraints = [condition_1]
    objective = cp.Minimize(dual_objective_function(u,v))
    problem = cp.Problem(objective, constraints)

    problem.solve(solver=algorithm, verbose=True, max_iters=1000)
    b_hat = -condition_1.dual_value
    return b_hat

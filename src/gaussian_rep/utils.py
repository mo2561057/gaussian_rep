import numpy as np
import math
from typing import Callable, List, Tuple

import cvxpy as cp


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


def get_kronecker_function(
    S: List[Callable[[float], np.ndarray]],
    W: tuple(List[Callable[[np.ndarray], np.ndarray]]),
) -> callable:
    """
    Compute the Kronecker product of two vectors of functions and multiply by coefficients
    using vectorized operations.
    
    Args:
        W: List of functions taking numpy array argument
        S: List of functions taking scalar argument
        
    Returns:
        float: Result of kronecker product multiplied by coefficients
    """

    def kronecker_function(*args):
        y,X = args if len(args)==2 else (args[0][:,0], args[0][:,1:])
        W_values = W(X)
        S_values = S(y)
        return np.einsum(
            "nj,nk->njk", W_values, S_values).reshape(X.shape[0], W_values.shape[1]*S_values.shape[1])

    return kronecker_function


def get_objective_function(
    T: List[Callable[[np.ndarray, float], np.ndarray]],
    t: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
):
    """
    Build the complete likelihood function from combined function and derivatives.
    First pre-compute function values over T and t. 
    Then build the objective function.

    Args:
        T: List of base functions for gaussian representation
        t: Derivative of base functions T
    """
    
    T_hat = T(y, X)
    t_hat = t(y, X)

    def objective_function(coefficients):
        rslt = (-0.5*(
            np.log(2*math.pi) + np.einsum("nk,k->n", T_hat, coefficients["value"])**2)
             + np.log(np.einsum("nk,k->n",t_hat, coefficients["value"]))).mean()
        if math.isnan(rslt):
            return -1e20
        return rslt

    return objective_function



def get_dual_objective_function():
    """
    Build the dual of the likelihood function.

    Args:
        T: List of base functions for gaussian representation
        t: Derivative of base functions T
    """

    def dual_objective_function(u,v):
        n = u.shape[0]
        rslt = -n*(0.5*math.log(2*math.pi) + 1)  +  (u**2 - cp.log(-v)).sum() 
        return rslt

    return dual_objective_function


def get_dual_constraint(T, t, y, X):
    """
    Build the dual of the likelihood function.

    Args:
        T: List of base functions for gaussian representation
        t: Derivative of base functions T
    """
    T_hat = T(y, X)
    t_hat = t(y, X)

    def foc_constraint(u,v):
        rslt = T_hat.T@u + t_hat.T@v
        return rslt

    return foc_constraint



def get_derivative_objective_function(
    T: List[Callable[[np.ndarray, float], np.ndarray]],
    t: Callable[[np.ndarray, float], np.ndarray],
    y: np.ndarray,
    X: np.ndarray,
):
    T_hat = T(y, X)
    t_hat = t(y, X)
  
    def derivative_objective_function(coefficients):
        base_1 = np.einsum("nk,k->n", T_hat, coefficients["value"])
        term_1 = np.einsum("n,nk->nk", base_1, T_hat)
        base_2 = 1/(np.einsum(
            "nk,k->n", t_hat, coefficients["value"]))
        term_2 = np.einsum("n,nk->nk", base_2, t_hat)
        return -term_1.mean(axis=0) + term_2.mean(axis=0)
    return derivative_objective_function


def get_initial_bounds(derivative, input_params, magnitude=1000):
    params = input_params.copy()
    deviation = np.abs(magnitude/derivative)
    params["upper_bound"] = params["value"] + deviation
    params["lower_bound"] = params["value"] - deviation
    return params 


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
                       algorithm : str = "ECOS",
                       algorithm_options : dict = {"max_iters": 1000, "verbose": True},
                       regularization: np.ndarray = None,
                       ):
                       
    u = cp.Variable(n)
    v = cp.Variable(n)
    if regularization is not None:
        # We need to know the size of T
        condition_1 = foc_gradient(u,v)<=regularization
        condition_2 = foc_gradient(u,v)>=-regularization
        condition_3 = v<0 # Ensure v is negative
        constraints = [condition_1, condition_2, condition_3]
    else:    
        condition_1 = foc_gradient(u,v)==0
        condition_2 = v < 0  # Ensure v is negative
        constraints = [condition_1, condition_2]
    objective = cp.Minimize(dual_objective_function(u,v))
    problem = cp.Problem(objective, constraints)

    problem.solve(solver=algorithm, **algorithm_options)
    print("Objective value:", problem.value)
    print("Status:", problem.status)
    print("Primal variables:", u.value, v.value)
    print("Eval:",dual_objective_function(u, v).value)
    print("Constraint residual:", foc_gradient(u, v).value)
    if regularization is not None:
        # For the regularized case, we need to combine dual values from both constraints
        # The dual value for the upper bound minus the dual value for the lower bound
        b_hat = condition_2.dual_value - condition_1.dual_value
    else:
        # For the non-regularized case, just get the dual value as before
        b_hat = -condition_1.dual_value

    return b_hat
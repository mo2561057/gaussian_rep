import numpy as np
import scipy
import pandas as pd
import math
from typing import Callable, List

def get_kronecker_function(
    S: List[Callable[[float], np.ndarray]],
    W: List[Callable[[np.ndarray], np.ndarray]],
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

    def kronecker_function(X,y):
        W_values = W(X)
        S_values = S(y)
        return np.einsum(
            "nj,nk->njk", W_values, S_values).reshape(X.shape[0], W_values.shape[1]*S_values.shape[1])

    # Multiply by coefficients and sum
    return kronecker_function




def get_objective_function(
    T: List[Callable[[np.ndarray, float], np.ndarray]],
    t: Callable[[np.ndarray, np.ndarray], np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
):
    """
    Build the complete likelihood function from combined function and derivatives.
    First pre-compute function values over T and t. 
    Then build the objective function.

    Args:
        T: List of base functions for gaussian representation
        t: Derivative of base functions T
    """
    
    T_hat = T(X,y)
    t_hat = t(X,y)

    def objective_function(coefficients):
        return (-0.5*(
            np.log(2*math.pi) + np.einsum("nk,k->n", T_hat, coefficients)**2
             + np.log(np.einsum("nk,k->n",t_hat, coefficients)))).mean()

    return objective_function
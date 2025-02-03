from utils import get_kronecker_function, get_objective_function
from typing import Callable, List

import numpy as np
import estimagic as em



def estimate_model(
        params: np.ndarray,
        objective_function : Callable[[np.ndarray], float]
    ) -> dict:
    res = em.maximize(
        objective_function,
        params,
        algorithm="nag_pybobyqa"
    )
    return res

def recursively_estimate_model_multiple_outcomes():
    pass

def get_full_likelihood_function(
        S : List[Callable[[float], np.ndarray]],
        s : Callable[[np.ndarray, float], np.ndarray],
        W : List[Callable[[np.ndarray], np.ndarray]],
        X : np.ndarray,
        y : np.ndarray
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
    return get_objective_function(T, t, X, y)


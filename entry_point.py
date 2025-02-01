from utils import get_kronecker_function, get_objective_function
from typing import Callable, List
import numpy as np




def estimate_model():
    pass

def recursively_estimate_model_multiple_outcomes():
    pass

def get_full_likelihood_function(
        S : List[Callable[[np.float], np.ndarray]],
        W : List[Callable[[np.ndarray], np.ndarray]],
        s : Callable[[np.ndarray, np.float], np.ndarray],
        X : np.ndarray,
        Y : np.ndarray
    ) -> Callable[[np.ndarray], np.ndarray]:
    """
    Build full likelihood function from components.

    Args:
        S : Outcome component of Gaussian representation
        W : Covariate component of Gaussian representation
        s : Gradient of S
        X : numpy array of covariates
        Y : numpy array of outcomes

    Returns:
        Callable[[np.ndarray], np.ndarray]: Full likelihood function
    """
    T = get_kronecker_function(W, S)
    t = get_kronecker_function(s, T)
    return get_objective_function(T, t, X, Y)


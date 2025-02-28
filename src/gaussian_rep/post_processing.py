"""Post processing of results. Add code to convert results in a probability distribution."""
import functools

from scipy.stats import norm
import numpy as np

from gaussian_rep.utils import get_kronecker_function

def convert_to_distribution(S, W, coefficients):
    T = get_kronecker_function(S, W)
    def conditional_cdf(y, x):
        input_y, input_x = _broadcast_arguments(y, x)
        return norm.cdf(T(input_y,input_x)@coefficients)
    return conditional_cdf


def _broadcast_arguments(y, x):
    if isinstance(y, (float,int)):
        out_y = np.array([y])
    else:
        out_y = y
    if type(x) is tuple:
        out_x = tuple((x_[np.newaxis,:] if x_.ndim == 1 else x_ for x_ in x))
    elif x.ndim == 1:
        out_x = x[np.newaxis,:]
    else:
        out_x = x
    return out_y, out_x

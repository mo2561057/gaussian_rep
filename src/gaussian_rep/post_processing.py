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

def convert_to_distribution_multivariate(S, W, coefficients):
    Ts = [get_kronecker_function(S[0],W)]
    for S_ in S[1:]:
        Ts.append(get_kronecker_function(S_,Ts[-1]))
    
    def conditional_cdf(y, x, prob=lambda y: np.ones(y.shape[0]) if y.ndim > 1 else 1):
        input_y, input_x = _broadcast_arguments_multivariate(y, x)
        prob = prob(input_y) if callable(prob) else prob
        y_current = input_y[:,0]
        current_position = input_y.shape[1]
        prob_dim = norm.cdf(Ts[-current_position](y_current,input_x)@coefficients[-current_position])
        prob = prob*prob_dim
        if input_y.shape[1] == 1:
            return prob
        else:
            return conditional_cdf(input_y[:,1:],(y_current, input_x),prob)

    return conditional_cdf



def _broadcast_arguments_multivariate(y, x):
    if y.ndim == 1:
        out_y = y[np.newaxis,:]
    else:
        out_y = y
    
    if type(x) is tuple:
        out_x = tuple((x_[np.newaxis,:] if x_.ndim == 1 else x_ for x_ in x))
    
    elif x.ndim == 1:
        out_x = x[np.newaxis,:]
    else:
        out_x = x

    return out_y, out_x



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

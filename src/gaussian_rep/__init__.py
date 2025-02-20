# src/__init__.py
from gaussian_rep.utils import (
    get_kronecker_function,
    get_objective_function,
    
)
from gaussian_rep.specification import (
    spline_linear,
    spline_spline,
    combine_functions
)
from gaussian_rep.entry_point import (
    get_full_likelihood_function,
    estimate_model
)

__all__ = [
    'get_kronecker_function',
    'get_objective_function',
    'combine_functions',
    'spline_linear',
    'spline_spline',
    'get_full_likelihood_function',
    'estimate_model'
]
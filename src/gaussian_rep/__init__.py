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
    recursively_estimate_multivariate_dual,
    estimate_model_primal
)
from gaussian_rep.post_processing import (
    convert_to_distribution,
    convert_to_distribution_multivariate
)
__all__ = [
    'get_kronecker_function',
    'get_objective_function',
    'combine_functions',
    'spline_linear',
    'spline_spline',
    'estimate_model_primal',
    'recursively_estimate_multivariate_dual',
    'convert_to_distribution',
    'convert_to_distribution_multivariate'
    ]
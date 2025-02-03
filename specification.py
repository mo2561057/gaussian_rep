"""
Tools to specify diferent model functions.
This creates a coupe of basic specifications that can be used within the framework.
"""
from functools import wraps, reduce
from itertools import product

import numpy as np
from scipy.interpolate import BSpline



def _constant_function(X):
    return np.ones(X.shape[0])


def linear_linear():
    # How do I 
    S = [lambda y: _constant_function(y), lambda y: y]
    s = [lambda y: _constant_function(y)*0, lambda y: _constant_function(y)]
    W = [lambda x: _constant_function(x), lambda x: x]
    
    return combine_functions(S), combine_functions(s), combine_functions(W)


def spline_linear(spec_y):
    
    splines_y = get_spline_basis(**spec_y)
    S = [lambda y: _constant_function(y), lambda y: y, *splines_y]

    s = [lambda y: _constant_function(y)*0, lambda y: _constant_function(y), *[func_.derivative(1) for func_ in splines_y]]
    W = [lambda x: _constant_function(x), lambda x: x]

    return combine_functions(S), combine_functions(s), combine_functions(W)

def spline_spline(spec_y, spec_x):
    splines_y = get_spline_basis(**spec_y)
    splines_x = get_spline_basis(**spec_x)

    S = [lambda y: _constant_function(y), lambda y: y, *splines_y]
    s = [lambda y: _constant_function(y)*0, lambda y: _constant_function(y), *[func_.derivative(1) for func_ in splines_y]]

    W = [lambda x: _constant_function(x), lambda x: x, *splines_x]
    return combine_functions(S), combine_functions(s), combine_functions(W)

def combine_functions(functions):

    def out_function(x):
        """
        Efficiently stack arrays horizontally, broadcasting 1D arrays to 2D if needed.
        
        Args:
            arrays: List of numpy arrays with same first dimension,
                can be mix of 1D and 2D arrays
        
        Returns:
            2D numpy array with horizontally stacked inputs
        """
        arrays = [func_(x) for func_ in functions]

        # Convert 1D arrays to 2D columns only when needed
        shaped_arrays = [
            arr[:, np.newaxis] if arr.ndim == 1 else arr 
            for arr in arrays
        ]
        # Use np.hstack for efficient concatenation
        return np.hstack(shaped_arrays)
    return out_function

def get_spline_basis(domains, n_bases, degree):
    # Get knots in each dimension
    knot_sequences = get_all_knot_sequences(
            domains, degree, n_bases)
    return [return_basis_function(knots)
            for knots in knot_sequences]


def wrap_return_basis_functions(func):
    @wraps(func)
    def wrapper(knots):
        if isinstance(knots[0], (int, float)):
            return func(knots)
        
        # Get knot sequences for each n_bases
        functions = [func(knots_dim) for knots_dim in knots]
        
        def out_function(x):
            # How should this deal with vectors? 
            # Should we vectorize at this level already?
            return reduce(
                lambda a,b: a*b,
                [f(x[:,i]) for i,f in enumerate(functions)])
        # Return cartesian product of all sequences
        return out_function
    return wrapper

@wrap_return_basis_functions
def return_basis_function(knots):
    return BSpline.basis_element(knots)


def wrap_knot_sequences(func):
    """
    Decorator that modifies get_all_knot_sequences to handle lists of n_bases.
    
    If n_bases is scalar: behaves normally
    If n_bases is list: returns product of knot sequences for each n_bases value
    """

    @wraps(func)
    def wrapper(domain, degree, n_bases):
        if isinstance(n_bases, (int, float)):
            return func(domain, n_bases, degree)
        
        # Get knot sequences for each n_bases
        sequences = [
            func(domain[i], n,degree) 
            for i,n in enumerate(n_bases)
        ]
        
        # Return cartesian product of all sequences
        return [
            list(seq) 
            for seq in product(*sequences)
        ]

    return wrapper

@wrap_knot_sequences
def get_all_knot_sequences(
        domain, n_bases, degree):
    
    min_val, max_val = domain

    knots = np.r_[
        [min_val] * (degree),
        np.linspace(min_val, max_val, n_bases - degree + 1)[1:-1],
        [max_val] * (degree)
    ]
    knot_sequences = [knots[i:i+degree+2] for i in range(len(knots) - degree - 1)]

    return knot_sequences

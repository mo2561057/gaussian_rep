"""This file tests the whole prcocedure with simulated data."""
import numpy as np
import pandas as pd

from gaussian_rep.entry_point import solve_dual_problem
from gaussian_rep.specification import spline_linear, linear_linear, spline_spline
from gaussian_rep.post_processing import convert_to_distribution
from gaussian_rep.utils import get_kronecker_function
from gaussian_rep.utils import get_dual_constraint, get_dual_objective_function


def simulated_sample_univariate_spline_spline(
        
):
    X = np.zeros((5000,1))
    X[:, 0] = np.random.uniform(-1, 1, 5000)

    y = np.random.normal(X[:,0],1)
    y = np.clip(y, -3, 3)

    domain_y = (-3,3)
    domain_x = (-1, 1)
    n_bases_y = 10
    n_bases_x = 6
    degree_y = 3
    degree_x = 3
    spec_y = {'domains': domain_y, 'n_bases': n_bases_y, 'degree': degree_y}
    spec_x = {'domains': domain_x, 'n_bases': n_bases_x, 'degree': degree_x}

    # Generate base functions
    S, s, W = spline_spline(spec_y, spec_x)

    return X, y, S, s, W


def test_procedure_univariate(test_runs=1):
    X, y, S, s, W = simulated_sample_univariate_spline_spline()    
    T = get_kronecker_function(S, W)
    t = get_kronecker_function(s, W)

    dual_objective_function = get_dual_objective_function()
    foc_gradient = get_dual_constraint(T, t, y, X)

    rslt = solve_dual_problem(
        dual_objective_function, foc_gradient, 5000)

    conditional_cdf = convert_to_distribution(S, W, rslt)

    for i in range(test_runs):
        check_number = np.random.uniform(-1,1)
        evaluate = [check_number-2 , check_number-1, check_number, check_number+1, check_number+2]
        actual = np.array(
            [conditional_cdf(check_number, np.array([check_number])) for check_number in evaluate])
        
        expected = np.array(
            [0.025, 0.16,0.5, 0.84, 0.975])

        np.testing.assert_allclose(actual, expected, rtol=1e-1)






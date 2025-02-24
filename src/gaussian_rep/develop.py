import math

import numpy as np
import pandas as pd
import pickle
import cvxpy as cp

from gaussian_rep.entry_point import recursively_estimate_multivariate_dual
from gaussian_rep.specification import spline_linear, linear_linear, spline_spline
from gaussian_rep.post_processing import convert_to_distribution, convert_to_distribution_multivariate
from gaussian_rep.utils import get_kronecker_function
from gaussian_rep.utils import get_initial_bounds
from gaussian_rep.utils import get_dual_constraint, get_dual_objective_function

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

######## Dual Approach #########
# Get dual functions
rslt = recursively_estimate_multivariate_dual(S,s,W,y,X)

conditional_cdf = convert_to_distribution(S, W, rslt[0])


######## Multivariate Version #########
y_2 = np.random.normal((X[:,0] + y)/2,1)
domain_y = (-2,2)
n_bases_y = 10
degree_y = 3
degree_x = 3
spec_y = {'domains': domain_y, 'n_bases': n_bases_y, 'degree': degree_y}
S_2, s_2, _ = spline_spline(spec_y, spec_x)
S_multi = [S, S_2]
s_multi = [s, s_2]
full_y = np.concatenate( [y[:,np.newaxis], y_2[:,np.newaxis]], axis=1)
rslt_2 = recursively_estimate_multivariate_dual(S_multi,s_multi,W,full_y,X,[])
Xy_1 = np.concatenate([y[:,np.newaxis], X], axis=1)
T = get_kronecker_function(S, W)
cdf = convert_to_distribution(S_2, T, rslt_2[1])
cdf(0, (np.array([-0.5]), np.array([0.5])))





multivariate_cdf = convert_to_distribution_multivariate(
    S_multi, W, rslt_2
)

multivariate_cdf(np.array([0.5,0.5]), np.array([0.5]))



y_check = np.array([[0, 0.1, 0.2, -0.1, -0,2, 0, 0.1, 0.2, -0.1, -0,2]]).reshape(6,2)
X_check = np.array([[0,0.1,0.2,0.3,0.4,0.5]]).reshape(6,1)
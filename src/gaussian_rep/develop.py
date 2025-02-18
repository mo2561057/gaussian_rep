import math

import numpy as np
import pandas as pd
import pickle
import cvxpy as cp

from gaussian_rep.entry_point import get_full_likelihood_function, estimate_model
from gaussian_rep.entry_point import solve_dual_problem
from gaussian_rep.specification import spline_linear, linear_linear, spline_spline
from gaussian_rep.post_processing import convert_to_distribution
from gaussian_rep.utils import get_kronecker_function
from gaussian_rep.utils import get_initial_bounds
from gaussian_rep.utils import get_dual_constraint, get_dual_objective_function
from gaussian_rep.entry_point import get_derivative_likelihood_function


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


T = get_kronecker_function(S, W)
t = get_kronecker_function(s, W)


######## Dual Approach #########

# Get dual functions
dual_objective_function = get_dual_objective_function()
foc_gradient = get_dual_constraint(T, t, y, X)

rslt = solve_dual_problem(
    dual_objective_function, foc_gradient, 5000)

conditional_cdf = convert_to_distribution(S, W, rslt)

##########  Primal Approach ##########
# Generate objective function
objective_function = get_full_likelihood_function(S, s, W, y, X)
derivative_objective_function = get_derivative_likelihood_function(S, s, W, y, X)
container = np.nan

for i in range(1000):
    params = pd.DataFrame({"value":np.random.uniform(-0.1,0.1,1024), "upper_bound":1, "lower_bound":-1})
    rslt = objective_function(params)
    print(rslt)
    if rslt>-100000000000:
        container = rslt
        break

params = get_initial_bounds(
    derivative_objective_function(params), params)

print(container)

with open("params.pkl", "wb") as f:
    pickle.dump(params, f)

optimagic_options = {
    "algorithm": "scipy_lbfgsb",
    "scaling": True,
    "scaling_options": {"method":"bounds"},
    "algo_options": {
        'convergence.ftol_rel': 0,
        "trustregion.initial_radius":1,
                },
    "jac": derivative_objective_function
}

rslt = estimate_model(
    params, objective_function, optimagic_options=optimagic_options)

for i in range(100):
    rslt = estimate_model(
    rslt.params, objective_function, optimagic_options=optimagic_options)

params["value"] = rslt.params["value"]

conditional_cdf = convert_to_distribution(S, W, rslt.params["value"])

T = get_kronecker_function(S, W)
t = get_kronecker_function(s, W)

T_hat = T(y,X)
t_hat = t(y,X)

# Check where the nans come from:
pos =[pos for pos,x in enumerate(rslt.history["criterion"]) if math.isnan(x)][0]
params = rslt.history["params"][pos]

t_hat_check = np.einsum("nk,k->n", t_hat, params["value"])
T_hat_check = np.einsum("nk,k->n", T_hat, rslt.params["value"])
test = (-0.5*(np.log(2*math.pi) + (T_hat_check)**2) + t_hat_check).mean()

check = objective_function(params)
derivative = derivative_objective_function(params)[14]

params.loc[14, "value"] = params.loc[14, "value"] + 0.1
check2 = objective_function(params)

deriv = (check2 - check) /(0.1)




def test_derivative():
    pass
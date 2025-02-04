import math

import numpy as np
import pickle

from gaussian_rep.entry_point import get_full_likelihood_function, estimate_model
from gaussian_rep.specification import spline_linear


y = np.random.uniform(8,9.2,5000)

params = np.random.uniform(0,1,80)


X = np.zeros((5000, 3))
X[:, 0] = np.random.uniform(8, 9.2, 5000)
X[:, 1] = np.random.uniform(8.5, 9.6, 5000)
X[:, 2] = np.random.uniform(11, 18, 5000)


domain_y = (8, 9.2)
domain_x = [(8, 9.2),(8.5, 9.6),(11, 18)]
n_bases_y = 20
n_bases_x = [6, 6, 6]
degree_y = 3
degree_x = 3
spec_y = {'domains': domain_y, 'n_bases': n_bases_y, 'degree': degree_y}
spec_x = {'domains': domain_x, 'n_bases': n_bases_x, 'degree': degree_x}

# Generate base functions
S, s, W = spline_linear(spec_y)

# Generate objective function
objective_function = get_full_likelihood_function(S, s, W, X, y)

container = np.nan

for i in range(1000):
    params = np.random.uniform(-100,100,80)
    rslt = objective_function(params)
    if math.isnan(rslt) is False:
        container = rslt
        break

print(container)

with open("params.pkl", "wb") as f:
    pickle.dump(container, f)


rslt = estimate_model(params, objective_function)
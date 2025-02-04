import pytest
import numpy as np
from scipy.interpolate import BSpline
from gaussian_rep.utils import (
    get_kronecker_function,
    get_objective_function,
)
from gaussian_rep.specification import combine_functions, get_spline_basis, get_all_knot_sequences

def test_kronecker_function():
    """Test the kronecker function with simple basis functions."""
    # Define simple test functions
    W = combine_functions([lambda x: np.ones_like(x), lambda x: x])
    S = combine_functions([lambda y: np.ones_like(y), lambda y: y])
    
    # Create test data
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([0.5, 1.5])
    
    # Get kronecker function
    kron_func = get_kronecker_function(S, W)
    result = kron_func(X, y)
    
    # Check shape
    assert result.shape == (2, 8)  # 2 points × (2×2) functions
    
    # Check specific values
    expected = np.array([
        [1.0, 0.5, 1, 0.5, 1, 0.5, 2, 1.0],
        [1.0, 1.5, 1.0, 1.5, 3, 4.5, 4, 6.0],
    ])
    np.testing.assert_array_almost_equal(result, expected)

def test_combine_functions():
    """Test combining functions with different output dimensions.
       Combine functions works on functions that takes vectors or matrices as input to handle data.
       None of the functions must have a scalar as output.
    """
    # Test case 1: All vector
    funcs1 = [lambda x: np.ones_like(x), lambda x: 2*np.ones_like(x)]
    x1 = np.array([1.0, 2.0])
    combined1 = combine_functions(funcs1)
    result1 = combined1(x1)
    assert result1.shape == (2, 2)
    
    # Test case 2: Mix of vector and matrix outputs
    funcs2 = [
        lambda x: np.ones(len(x)),
        lambda x: np.column_stack([x, x**2])
    ]
    x2 = np.array([1.0, 2.0])
    combined2 = combine_functions(funcs2)
    result2 = combined2(x2)
    assert result2.shape == (2, 3)

def test_spline_basis():
    """Test creation and evaluation of spline basis functions."""
    # Test parameters
    domain = (0, 1)
    n_bases = 4
    degree = 3
    
    # Get basis functions
    splines = get_spline_basis(domain, n_bases, degree)
    
    # Test points
    x = np.linspace(0, 1, 100)
    
    # Check properties
    for spline in splines:
        # Check if function handles arrays
        values = spline(x)
        assert len(values) == len(x)
        
        # Check non-negativity
        assert np.all(values >= 0)
        
        # Check derivatives
        deriv = spline.derivative()
        deriv_values = deriv(x)
        assert len(deriv_values) == len(x)

def test_objective_function():
    """Test the objective function construction and evaluation."""
    # Create simple test functions
    def T(X, y):
        return np.column_stack([np.ones_like(y), y])
    
    def t(X, y):
        return np.column_stack([np.zeros_like(y), np.ones_like(y)])
    
    # Test data
    X = np.random.rand(10, 2)
    y = np.random.rand(10)
    
    # Get objective function
    obj_func = get_objective_function(T, t, X, y)
    
    # Test with some coefficients
    coeffs = np.array([1.0, 0.5])
    result = obj_func(coeffs)
    
    # Check if result is scalar
    assert np.isscalar(result)
    
    # Check if result is finite
    assert np.isfinite(result)

def test_knot_sequences():
    """Test generation of knot sequences."""
    # Test single dimension
    domain = (0, 1)
    n_bases = 4
    degree = 3
    
    sequences = get_all_knot_sequences(domain, n_bases, degree)
    
    # Check number of sequences
    assert len(sequences) == n_bases
    
    # Check sequence properties
    for seq in sequences:
        # Check length
        assert len(seq) == degree + 1
        
        # Check ordering
        assert np.all(np.diff(seq) >= 0)
        
        # Check domain bounds
        assert seq[0] >= domain[0]
        assert seq[-1] <= domain[1]

def test_edge_cases():
    """Test edge cases and error handling."""
    # Test empty input
    with pytest.raises(ValueError):
        get_all_knot_sequences((0, 1), 0, 3)
    
    # Test invalid degree
    with pytest.raises(ValueError):
        get_all_knot_sequences((0, 1), 4, -1)
    
    # Test invalid domain
    with pytest.raises(ValueError):
        get_all_knot_sequences((1, 0), 4, 3)  # min > max

def test_broadcasting():
    """Test broadcasting behavior of functions."""
    # Create test functions
    W = combine_functions([lambda x: x])
    S = combine_functions([lambda y: y])
    
    kron_func = get_kronecker_function(S, W)
    
    # Test with different array shapes
    X1 = np.array([[1.0], [2.0]])
    y1 = np.array([0.5, 1.5])
    result1 = kron_func(X1, y1)
    
    X2 = np.random.rand(100, 1)
    y2 = np.random.rand(100)
    result2 = kron_func(X2, y2)
    
    assert result1.shape == (2, 1)
    assert result2.shape == (100, 1)
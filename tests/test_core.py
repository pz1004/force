"""
Tests for the core FORCE estimator.
"""
import numpy as np
import pytest

from force import ForceEstimator
from force.data import generate_synthetic_data

@pytest.fixture
def synthetic_data():
    """Provides a small synthetic dataset for testing."""
    X, _ = generate_synthetic_data(n_samples=100, n_features=5, contamination=0.1)
    return X

def test_force_estimator_runs(synthetic_data):
    """
    Test that the ForceEstimator can be initialized and fit to data
    without raising an exception.
    """
    try:
        estimator = ForceEstimator()
        result = estimator.fit(synthetic_data)
        
        # Check output shape and basic properties
        assert result.shape == (5, 5), "Output should be a 5x5 matrix"
        assert np.all(np.diag(result) == 1.0), "Diagonal elements should be 1"
        assert np.all(result >= -1.0) and np.all(result <= 1.0), "Values should be in [-1, 1]"

    except Exception as e:
        pytest.fail(f"ForceEstimator().fit() raised an exception: {e}")

def test_force_input_validation():
    """Test the input validation for the estimator."""
    estimator = ForceEstimator()
    
    # Must be 2D array
    with pytest.raises(ValueError, match="must be a 2D NumPy array"):
        estimator.fit(np.random.rand(10))
        
    # Must have enough samples
    with pytest.raises(ValueError, match="must have at least 5 samples"):
        estimator.fit(np.random.rand(4, 3))
        
    # Must have enough features
    with pytest.raises(ValueError, match="must have at least 2 features"):
        estimator.fit(np.random.rand(10, 1))

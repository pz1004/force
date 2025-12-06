"""
Tests for the comparative estimators.
"""
import numpy as np
import pytest

from force import (
    PearsonEstimator,
    SpearmanEstimator,
    WinsorizedEstimator,
    FastMCDEstimator,
    ExactTrimmedEstimator,
)
from force.data import generate_synthetic_data

@pytest.fixture
def synthetic_data():
    """Provides a small synthetic dataset for testing."""
    X, _ = generate_synthetic_data(n_samples=100, n_features=4, contamination=0.1)
    return X

# A list of all estimator classes to be tested
ESTIMATOR_CLASSES = [
    PearsonEstimator,
    SpearmanEstimator,
    WinsorizedEstimator,
    FastMCDEstimator,
    ExactTrimmedEstimator,
]

@pytest.mark.parametrize("EstimatorClass", ESTIMATOR_CLASSES)
def test_estimator_runs(EstimatorClass, synthetic_data):
    """
    Tests that each comparative estimator can be initialized and run.
    This acts as a basic smoke test.
    """
    try:
        estimator = EstimatorClass()
        result = estimator.fit(synthetic_data)
        
        # Basic validation of the output
        assert isinstance(result, np.ndarray), "Estimator must return a NumPy array"
        assert result.shape == (4, 4), "Output correlation matrix shape is incorrect"
        assert np.allclose(np.diag(result), 1.0), "Diagonal of a corr matrix must be 1"
        assert np.all(result >= -1.0) and np.all(result <= 1.0), "All values must be in [-1, 1]"
        
    except Exception as e:
        pytest.fail(f"{EstimatorClass.__name__} raised an exception during fit: {e}")

def test_spearman_for_two_features():
    """
    `scipy.stats.spearmanr` returns a float for 2 features, not a matrix.
    This test ensures our wrapper handles that case correctly.
    """
    X = np.random.rand(10, 2)
    estimator = SpearmanEstimator()
    result = estimator.fit(X)
    assert result.shape == (2, 2), "Spearman should return a 2x2 matrix for 2 features"
    assert np.allclose(np.diag(result), 1.0), "Diagonal should be 1"

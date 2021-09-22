import unittest

import numpy as np

from girth.factoranalysis import principal_components_analysis as pca
from girth.factoranalysis import principal_axis_factor as paf

class TestPrincipalAxis(unittest.TestCase):
    """Test fixture for principal components."""

    def test_principal_axis_factor(self):
        """Testing Principle Axis Factor."""
        rng = np.random.default_rng(2016)

        data = rng.uniform(-2, 2, size=(10, 100))
        unique_var = rng.uniform(0.2, 2, size=10)

        # Create 2 Factor Data
        cov_matrix = np.cov(data)
        loadings, eigenvalues, _ = pca(cov_matrix, 2)

        # Add Unique variance
        cov_matrix2 = loadings @ loadings.T + np.diag(unique_var)

        loadings_paf, eigenvalues2, variance = paf(cov_matrix2, 2)

        np.testing.assert_allclose(loadings, loadings_paf, rtol=1e-4)
        np.testing.assert_allclose(eigenvalues, eigenvalues2, rtol=1e-4)
        np.testing.assert_allclose(unique_var, variance, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
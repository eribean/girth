import unittest

import numpy as np

from girth.factoranalysis import principal_components_analysis as pca


class TestPrincipalComponents(unittest.TestCase):
    """Test fixture for principal components."""

    def test_principal_component_analysis(self):
        """Testing Principle Component Analysis."""
        rng = np.random.default_rng(4646)

        data = rng.uniform(-1, 1, size=(10, 100))

        # Covariance
        cov_matrix = np.cov(data)
        loadings, _, _ = pca(cov_matrix, 2)

        # Compare to SVD version
        u, s, vt = np.linalg.svd(cov_matrix, hermitian=True)
        loadings_compare = u[:, :2] @ np.diag(np.sqrt(s[:2]))

        np.testing.assert_allclose(loadings, loadings_compare, rtol=1e-4)

    def test_principal_component_analysis_full(self):
        """Testing Principle Component Analysis no Factors."""
        rng = np.random.default_rng(4646)

        data = rng.uniform(-1, 1, size=(10, 100))

        # Covariance
        cov_matrix = np.cov(data)
        loadings, eigs, uvars = pca(cov_matrix)

        # Compare to SVD version
        u, s, _ = np.linalg.svd(cov_matrix, hermitian=True)
        loadings_compare = u @ np.diag(np.sqrt(s))

        np.testing.assert_allclose(eigs, s, rtol=1e-4)
        np.testing.assert_allclose(uvars, np.zeros_like((uvars)), rtol=1e-4)
        np.testing.assert_allclose(loadings, loadings_compare, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
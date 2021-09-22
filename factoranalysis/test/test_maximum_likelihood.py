import unittest

import numpy as np

from girth.factoranalysis import principal_components_analysis as pca
from girth.factoranalysis import maximum_likelihood_factor_analysis as mlfa

from girth.common import procrustes_rotation


class TestMaximumLikelihood(unittest.TestCase):
    """Test fixture for maximum likelihood factor analysis."""

    #TODO: Need algorithm validity test

    def test_maximum_likelihood_recovery(self):
        """Testing Maximum Likelihood Recovery Factor."""
        rng = np.random.default_rng(18434)

        data = rng.uniform(-2, 2, size=(10, 100))
        unique_var = rng.uniform(0.2, 2, size=10)

        # Create 3 Factor Data
        cor_matrix = np.cov(data)
        loadings, eigenvalues, _ = pca(cor_matrix, 3)

        # Add Unique variance
        cor_matrix2 = loadings @ loadings.T + np.diag(unique_var)

        initial_guess = np.ones((10,)) *.5
        loadings_paf, _, variance = mlfa(cor_matrix2, 3, initial_guess=initial_guess)

        # Remove any rotation
        rotation = procrustes_rotation(loadings, loadings_paf)
        updated_loadings = loadings_paf @ rotation
        updated_eigs = np.square(updated_loadings).sum(0)

        # Did I Recover initial values (upto a rotation)
        np.testing.assert_allclose(loadings, updated_loadings, rtol=1e-3)
        np.testing.assert_allclose(eigenvalues, updated_eigs, rtol=1e-3)
        np.testing.assert_allclose(unique_var, variance, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
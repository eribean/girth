import unittest

import numpy as np
from scipy.optimize import minimize

from girth.factoranalysis import principal_components_analysis as pca
from girth.factoranalysis import minimum_rank_factor_analysis as mrfa
from girth.factoranalysis.minimum_rank import _mrfa_min_func


class TestMinimumRank(unittest.TestCase):
    """Test fixture for minimum rank."""

    def test_minimum_rank_recovery(self):
        """Testing Minimum Rank Recovery."""
        rng = np.random.default_rng(5487341)

        data = rng.uniform(-2, 2, size=(10, 200))
        unique_var = rng.uniform(0.2, .5, size=10)

        # Create 3 Factor Data
        cor_matrix = np.corrcoef(data)
        loadings, eigenvalues, _ = pca(cor_matrix, 3)

        # Add Unique variance
        cor_matrix2 = loadings @ loadings.T + np.diag(unique_var)

        initial_guess = np.ones((10,)) * 2
        loadings_paf, eigenvalues2, variance = mrfa(cor_matrix2, 3, n_iter=5000,
                                                    initial_guess=initial_guess)

        # Did I Recover initial values?
        np.testing.assert_allclose(loadings, -loadings_paf, rtol=1e-3)
        np.testing.assert_allclose(eigenvalues, eigenvalues2, rtol=1e-3)
        np.testing.assert_allclose(unique_var, variance, rtol=1e-3)

    def test_minimum_rank_derivative(self):
        """Testing the derivative calculation in minimum rank."""
        def no_derivative(inverse_half_variance, correlation_cholesky, n_factors):
            return _mrfa_min_func(inverse_half_variance, 
                                  correlation_cholesky, 
                                  n_factors)[0]

        rng = np.random.default_rng(216857371353)
        data = rng.uniform(-2, 2, size=(10, 200))

        # Create Data
        cor_matrix = np.corrcoef(data)
        cholesky_corr = np.linalg.cholesky(cor_matrix)

        initial_guess = rng.uniform(.1, .9, 10)
        initial_guess = 1 / np.sqrt(initial_guess)
        bounds = [(1, 100)] * 10

        # Compare numerical to analytical derivatives
        for n_factors in range(1, 5):
            result = minimize(no_derivative, 
                              initial_guess, 
                              args=(cholesky_corr, n_factors),
                              method='SLSQP',
                              bounds=bounds,
                              options={'maxiter': 1})

            derivative_calc = _mrfa_min_func(initial_guess, 
                                             cholesky_corr, 
                                             n_factors)
            np.testing.assert_allclose(result['jac'], derivative_calc[1], atol=1e-5)





    def test_minimum_zero_eigenvalue(self):
        """Testing Forced Semi-Positive Definite."""
        rng = np.random.default_rng(12473)

        data = rng.uniform(-2, 2, size=(10, 100))

        # Create 2 Factor Data
        cor_matrix = np.corrcoef(data)

        _, _, variance = mrfa(cor_matrix, 3)

        eigens = np.linalg.eigvalsh(cor_matrix - np.diag(variance))
        
        # Is the last eigenvalue zero?
        self.assertAlmostEqual(eigens[0], 0, places=5)


if __name__ == "__main__":
    unittest.main()
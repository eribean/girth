import unittest

import numpy as np

from girth import create_synthetic_irt_dichotomous
from girth import create_synthetic_mirt_dichotomous
from girth import create_correlated_abilities


class TestSynthetic(unittest.TestCase):
    """Testing the creation of synthetic irt function."""

    def test_synthetic_irt_creation(self):
        """Testing the creation of synthetic data."""
        seed = 31

        # Regression test
        expected = np.array([[False, False, False, False,  True,  True],
                             [False, False,  True,  True,  True,  True],
                             [False, False, False,  True,  True,  True]])

        value = create_synthetic_irt_dichotomous(np.array([1.2, -0.2, 1.3]),
                                                 1.31, np.linspace(-6, 6, 6),
                                                 seed)

        np.testing.assert_array_equal(expected, value)


    def test_synthetic_mirt_creation(self):
        """Testing the creation of synthetic data."""
        seed = 164
        np.random.seed(seed-1)
        # Regression test
        expected = np.array([[False, False, False, False, False, False],
                             [ True, False, False,  True, False,  True],
                             [ True,  True, False, False, False, False],
                             [ True,  True,  True,  True, False,  True],
                             [ True,  True,  True,  True,  True,  True]])

        n_factors = 3
        n_items = 5
        n_people = 6
        discrimination = np.random.randn(n_items, n_factors)
        difficulty = np.linspace(-5, 5, n_items)
        thetas = np.random.randn(n_factors, n_people)
        value = create_synthetic_mirt_dichotomous(difficulty, discrimination,
                                                  thetas, seed)

        np.testing.assert_array_equal(expected, value)


    def test_synthetic_mirt_creation_single(self):
        """Testing the creation of synthetic data, common discrimination."""
        seed = 546
        np.random.seed(seed-1)
        # Regression test
        expected = np.array([[False, False, False, False, False, False],
                             [False, False, False, False, False, False],
                             [False, False, False,  True,  True,  True],
                             [False,  True,  True,  True,  True,  True],
                             [ True,  True,  True,  True,  True,  True]])

        n_factors = 3
        n_items = 5
        n_people = 6
        discrimination = np.random.randn(1, n_factors)
        difficulty = np.linspace(-5, 5, n_items)
        thetas = np.random.randn(n_factors, n_people)
        value = create_synthetic_mirt_dichotomous(difficulty, discrimination,
                                                  thetas, seed)

        np.testing.assert_array_equal(expected, value)


    def test_correlated_abilities(self):
        """Testing the creation of correlated abilities."""
        np.random.seed(120)
        n_participants = 1000
        rho = 0.73
        correlation_matrix = np.array([[1, rho], [rho, 1]])

        output = create_correlated_abilities(correlation_matrix, n_participants)
        output_corr = np.corrcoef(output)

        np.testing.assert_almost_equal(output_corr, correlation_matrix, decimal=1)


if __name__ == '__main__':
    unittest.main()

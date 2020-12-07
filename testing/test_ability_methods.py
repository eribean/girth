import unittest # pylint: disable=cyclic-import

import numpy as np
from scipy.stats import skewnorm

from girth.ability_methods import _ability_eap_abstract
from girth import (ability_eap, ability_map, 
                   ability_mle)

from girth import create_synthetic_irt_dichotomous                   


def _rmse(expected, result):
    """Helper function to compute rmse."""
    return np.sqrt(np.nanmean(np.square(expected - result)))


class TestAbilityEstimates(unittest.TestCase):
    """Tests the estimation of ability parameters."""

    @classmethod
    def setUp(self):
        """Sets up synthetic data to use for estimation."""
        np.random.seed(7)
        self.difficulty = np.linspace(-2.4, 1.7, 15)
        self.discrimination = 0.5 + np.random.rand(15) * 2
        self.discrimination_single = 1.702
        self.expected_theta = np.random.randn(300)
        distribution = skewnorm(0.5, 0.2, 1.1)
        self.expected_skew = distribution.rvs(size=300)
        self.skew_expected_theta_func = distribution.pdf

        # Create first synthetic data test
        self.set_one = create_synthetic_irt_dichotomous(self.difficulty, self.discrimination,
                                                        self.expected_theta, seed=312)

        # Create second synthetic data test
        self.set_two = create_synthetic_irt_dichotomous(self.difficulty, self.discrimination_single,
                                                        self.expected_theta, seed=547)

        # Create Skewed data set
        self.set_three = create_synthetic_irt_dichotomous(self.difficulty, self.discrimination,
                                                          self.expected_skew, seed=872)

        # Add missing values
        dataset = create_synthetic_irt_dichotomous(self.difficulty, self.discrimination,
                                                   self.expected_theta, seed=312)
        dataset = dataset.astype('float')
        mask = np.random.rand(*dataset.shape) < 0.1
        dataset[mask] = np.nan
        self.set_four = dataset

        # Regression Test
        self.regression_difficulty = np.linspace(-1.5, 1.5, 5)
        self.regression_discrimination = np.linspace(0.8, 1.8, 5)
        self.regression_theta = np.random.randn(10)
        self.set_five = create_synthetic_irt_dichotomous(self.regression_difficulty,
                                                         self.regression_discrimination,
                                                         self.regression_theta, seed=422)
        
    def test_ability_mle(self):
        """Testing Maximum Likelihood estimates."""

        ## Regression tests for various types of measurements

        # Full discrimination
        theta1 = ability_mle(self.set_one, self.difficulty, self.discrimination)
        self.assertAlmostEqual(_rmse(theta1, self.expected_theta), 0.516, places=3)

        # Single discrimination
        theta2 = ability_mle(self.set_two, self.difficulty, self.discrimination_single)
        self.assertAlmostEqual(_rmse(theta2, self.expected_theta), 0.443, places=3)

        # Skewed distribution
        theta3 = ability_mle(self.set_three, self.difficulty, self.discrimination)
        self.assertAlmostEqual(_rmse(theta3, self.expected_skew), 0.582, places=3)

        # Missing Values
        theta4 = ability_mle(self.set_four, self.difficulty, self.discrimination)
        self.assertAlmostEqual(_rmse(theta4, self.expected_theta), 0.542, places=3)

        # Regression
        expected = [-1.73287257, -0.48635278,  0.45113559, np.nan, -0.08638913, 
                     1.69245051, -1.03434564, -1.3481655, -1.3481655, 0.45113559]
        theta5 = ability_mle(self.set_five, self.regression_difficulty, self.regression_discrimination)
        np.testing.assert_array_almost_equal(theta5, expected,decimal=5)

    def test_ability_map(self):
        """Testing Maximum a posteriori estimates."""
        # Full discrimination
        theta1 = ability_map(self.set_one, self.difficulty, self.discrimination)
        self.assertAlmostEqual(_rmse(theta1, self.expected_theta), 0.437, places=3)
        
        # Single discrimination
        theta2 = ability_map(self.set_two, self.difficulty, self.discrimination_single)
        self.assertAlmostEqual(_rmse(theta2, self.expected_theta), 0.419, places=3)

        # Skewed distribution
        options = {'distribution': self.skew_expected_theta_func}
        theta3 = ability_map(self.set_three, self.difficulty, self.discrimination,
                             options)
        self.assertAlmostEqual(_rmse(theta3, self.expected_skew), 0.506, places=3)

        # Missing Values
        theta4 = ability_map(self.set_four, self.difficulty, self.discrimination)
        self.assertAlmostEqual(_rmse(theta4, self.expected_theta), 0.462, places=3)

        # Regression
        expected = [-0.75820877, -0.26554781,  0.27579622, -1.17914526, -0.04986583,
                     1.04563432, -0.51614533, -0.63534464, -0.63534464,  0.27579622]

        theta5 = ability_map(self.set_five, self.regression_difficulty, self.regression_discrimination)
        np.testing.assert_array_almost_equal(theta5, expected,decimal=5)

    def test_ability_eap(self):
        """Testing Expected a posteriori estimates."""
        # Full discrimination
        theta1 = ability_eap(self.set_one, self.difficulty, self.discrimination)
        self.assertAlmostEqual(_rmse(theta1, self.expected_theta), 0.436, places=3)
         
        # Single discrimination
        theta2 = ability_eap(self.set_two, self.difficulty, self.discrimination_single)
        self.assertAlmostEqual(_rmse(theta2, self.expected_theta), 0.418, places=3)

        # Skewed distribution
        options = {'distribution': self.skew_expected_theta_func}
        theta3 = ability_eap(self.set_three, self.difficulty, self.discrimination,
                             options)
        self.assertAlmostEqual(_rmse(theta3, self.expected_skew), 0.501, places=3)

        # Missing Values
        theta4 = ability_eap(self.set_four, self.difficulty, self.discrimination)
        self.assertAlmostEqual(_rmse(theta4, self.expected_theta), 0.462, places=3)

        # Regression
        expected = [-0.81893966, -0.31526459,  0.24361885, -1.2459348,  -0.09338352,
                     1.05603316, -0.57198605, -0.69371833, -0.69371833,  0.24361885]

        theta5 = ability_eap(self.set_five, self.regression_difficulty, self.regression_discrimination)
        np.testing.assert_allclose(theta5, expected, atol=1e-3, rtol=1e-3)

    def test_ability_eap_abstract(self):
        """Testing eap computation."""
        np.random.seed(1002124)
        partial_int = np.random.randn(1000, 41)
        weight = np.random.randn(41)
        theta = np.linspace(-3, 3, 41)

        result = _ability_eap_abstract(partial_int, weight, theta)

        denom = (partial_int * (weight)).sum(1)
        numer = (partial_int * (weight * theta)).sum(1)
        expected = numer / denom
        
        np.testing.assert_allclose(result, expected, atol=1e-5, rtol=1e-3)


if __name__ == '__main__':
    unittest.main()

import unittest # pylint: disable=cyclic-import

import numpy as np
from scipy.stats import skewnorm

from girth import (
    ability_eap, ability_map, ability_mle, INVALID_RESPONSE)
from girth.unidimensional.polytomous.ability_estimation_poly import _ability_eap_abstract
from girth.synthetic import create_synthetic_irt_dichotomous


def _rmse(expected, result):
    """Helper function to compute rmse."""
    return np.sqrt(np.nanmean(np.square(expected - result)))


class TestAbilityEstimates(unittest.TestCase):
    """Tests the estimation of ability parameters."""

    @classmethod
    def setUp(self):
        """Sets up synthetic data to use for estimation."""
        rng = np.random.default_rng(55546546448096)
        self.difficulty = np.linspace(-2.4, 1.7, 15)
        self.discrimination = 0.5 + rng.uniform(0, 2, size=15)
        self.discrimination_single = 1.702
        self.expected_theta = rng.standard_normal(300)
        distribution = skewnorm(0.5, 0.2, 1.1)
        self.expected_skew = distribution.rvs(size=300, random_state=rng)
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
        mask = rng.uniform(0, 1, size=dataset.shape) < 0.1
        dataset[mask] = INVALID_RESPONSE
        self.set_four = dataset

        # Regression Test
        self.regression_difficulty = np.linspace(-1.5, 1.5, 5)
        self.regression_discrimination = np.linspace(0.8, 1.8, 5)
        self.regression_theta = rng.standard_normal(10)
        self.set_five = create_synthetic_irt_dichotomous(self.regression_difficulty,
                                                         self.regression_discrimination,
                                                         self.regression_theta, seed=422)
        
    def test_ability_mle(self):
        """Testing Maximum Likelihood estimates."""

        ## Regression tests for various types of measurements

        # Full discrimination
        theta1 = ability_mle(self.set_one, self.difficulty, self.discrimination)
        self.assertAlmostEqual(_rmse(theta1, self.expected_theta), 0.463, places=3)

        # Single discrimination
        theta2 = ability_mle(self.set_two, self.difficulty, self.discrimination_single)
        self.assertAlmostEqual(_rmse(theta2, self.expected_theta), 0.437, places=3)

        # Skewed distribution
        theta3 = ability_mle(self.set_three, self.difficulty, self.discrimination)
        self.assertAlmostEqual(_rmse(theta3, self.expected_skew), 0.449, places=3)

        # Missing Values
        theta4 = ability_mle(self.set_four, self.difficulty, self.discrimination)
        self.assertAlmostEqual(_rmse(theta4, self.expected_theta), 0.478, places=3)

        # Regression
        expected = [-1.73287, -1.73287,  0.45114, -0.48635, -0.48635, -0.27791,
                    np.nan,  1.52444, -1.34817, -1.34817]
        theta5 = ability_mle(self.set_five, self.regression_difficulty, self.regression_discrimination)
        np.testing.assert_array_almost_equal(theta5, expected, decimal=5)

    def test_ability_map(self):
        """Testing Maximum a posteriori estimates."""
        # Full discrimination
        theta1 = ability_map(self.set_one, self.difficulty, self.discrimination)
        self.assertAlmostEqual(_rmse(theta1, self.expected_theta), 0.412, places=3)
        
        # Single discrimination
        theta2 = ability_map(self.set_two, self.difficulty, self.discrimination_single)
        self.assertAlmostEqual(_rmse(theta2, self.expected_theta), 0.412, places=3)

        # Skewed distribution
        options = {'distribution': self.skew_expected_theta_func}
        theta3 = ability_map(self.set_three, self.difficulty, self.discrimination,
                             options)
        self.assertAlmostEqual(_rmse(theta3, self.expected_skew), 0.436, places=3)

        # Missing Values
        theta4 = ability_map(self.set_four, self.difficulty, self.discrimination)
        self.assertAlmostEqual(_rmse(theta4, self.expected_theta), 0.429, places=3)

        # Regression
        expected = [-0.75821, -0.75821,  0.2758 , -0.26555, -0.26555, -0.15639,
                    1.53448,  0.95394, -0.63534, -0.63534]

        theta5 = ability_map(self.set_five, self.regression_difficulty, self.regression_discrimination)
        np.testing.assert_array_almost_equal(theta5, expected,decimal=5)

    def test_ability_eap(self):
        """Testing Expected a posteriori estimates."""
        # Full discrimination
        theta1 = ability_eap(self.set_one, self.difficulty, self.discrimination)
        self.assertAlmostEqual(_rmse(theta1, self.expected_theta), 0.411, places=3)
         
        # Single discrimination
        theta2 = ability_eap(self.set_two, self.difficulty, self.discrimination_single)
        self.assertAlmostEqual(_rmse(theta2, self.expected_theta), 0.411, places=3)

        # Skewed distribution
        options = {'distribution': self.skew_expected_theta_func}
        theta3 = ability_eap(self.set_three, self.difficulty, self.discrimination,
                             options)
        self.assertAlmostEqual(_rmse(theta3, self.expected_skew), 0.436, places=3)

        # Missing Values
        theta4 = ability_eap(self.set_four, self.difficulty, self.discrimination)
        self.assertAlmostEqual(_rmse(theta4, self.expected_theta), 0.429, places=3)

        # Regression
        expected = [-0.818932, -0.818932,  0.243619, -0.315264, -0.315264, -0.20308 ,
                    1.58863 ,  0.957627, -0.693715, -0.693715]

        theta5 = ability_eap(self.set_five, self.regression_difficulty, self.regression_discrimination)
        np.testing.assert_allclose(theta5, expected, atol=1e-3, rtol=1e-3)

    def test_ability_eap_abstract(self):
        """Testing eap computation."""
        rng = np.random.default_rng(21357489413518)
    
        partial_int = rng.standard_normal((1000, 41))
        weight = rng.standard_normal(41)
        theta = np.linspace(-3, 3, 41)

        result = _ability_eap_abstract(partial_int, weight, theta)

        denom = (partial_int * (weight)).sum(1)
        numer = (partial_int * (weight * theta)).sum(1)
        expected = numer / denom
        
        np.testing.assert_allclose(result, expected, atol=1e-5, rtol=1e-3)


if __name__ == '__main__':
    unittest.main()

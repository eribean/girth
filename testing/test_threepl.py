import unittest  # pylint: disable=cyclic-import

import numpy as np

from girth import create_synthetic_irt_dichotomous
from girth.three_pl import (threepl_full, threepl_mml, 
                            ability_3pl_eap, ability_3pl_map,
                            ability_3pl_mle)


class TestMMLThreePLMethods(unittest.TestCase):

    # REGRESSION TESTS
    # The 3PL doesn't have good convergence properties
    # so don't test them

    """Setup synthetic data."""

    def setUp(self):
        """Setup synthetic data for tests."""
        np.random.seed(846)
        difficulty = np.linspace(-1.5, 1.5, 5)
        discrimination = np.random.rand(5) + 0.5
        thetas = np.random.randn(600)
        guessing = np.random.rand(5) * .3
        syn_data = create_synthetic_irt_dichotomous(difficulty, discrimination,
                                                    thetas, guessing=guessing)
        self.data = syn_data
        self.guessing = guessing

    def test_twopl_regression_mml(self):
        """Testing twopl separate methods."""
        syn_data = self.data.copy()
        output = threepl_mml(syn_data)

        expected_discrimination = np.array([0.595478, 0.539002, 1.03748 , 4., 4.])
        expected_output = np.array([-0.235392, -1.996236,  0.09334 ,  0.932623,  1.432708])
        expected_guess = np.array([3.300000e-01, 5.572645e-08, 2.432607e-01, 3.274185e-01,
                                   1.818037e-01])

        np.testing.assert_allclose(
            expected_discrimination, output[0], rtol=1e-2, atol=1e-2)
        np.testing.assert_allclose(expected_output, output[1], rtol=1e-2, atol=1e-2)
        np.testing.assert_allclose(expected_guess, output[2], rtol=1e-2, atol=1e-2)

    def test_twopl_regression_full(self):
        """Testing twopl full methods."""
        syn_data = self.data.copy()
        output = threepl_full(syn_data)

        expected_discrimination = np.array([0.595896, 0.538749, 1.035768, 4., 4.])
        expected_output = np.array([-0.235112, -1.997106,  0.090057,  0.932323,  1.432479])
        expected_guess = np.array([3.300000e-01, 6.282188e-18, 2.422328e-01, 3.274102e-01,
                                   1.817438e-01])

        np.testing.assert_allclose(
            expected_discrimination, output[0], rtol=1e-2, atol=1e-2)
        np.testing.assert_allclose(expected_output, output[1], rtol=1e-2, atol=1e-2)
        np.testing.assert_allclose(expected_guess, output[2], rtol=1e-2, atol=1e-2)


class TestAbilityEstimates3PL(unittest.TestCase):
    """Tests the estimation of ability parameters."""

    def setUp(self):
        """Setup synthetic data for tests."""
        np.random.seed(45184)
        difficulty = np.linspace(-1.5, 1.5, 10)
        discrimination = np.random.rand(10) + 0.5
        thetas = np.random.randn(1000)
        guessing = np.random.rand(10) * .25
        syn_data = create_synthetic_irt_dichotomous(difficulty, discrimination,
                                                    thetas, guessing=guessing)

        self.data = syn_data
        self.guessing = guessing
        self.discrimination = discrimination
        self.difficulty = difficulty

    def test_ability_mle(self):
        """Testing Maximum Likelihood estimates."""

        ## Regression tests for various types of measurements
        recovered_theta = ability_3pl_mle(self.data, self.difficulty,
                                          self.discrimination, self.guessing)

        # Get stats on the theta
        mean = np.nanmean(recovered_theta)
        std = np.nanstd(recovered_theta)
        minimum = np.nanmin(recovered_theta)
        maximum = np.nanmax(recovered_theta)

        self.assertAlmostEqual(mean, -0.088980088, places=3)
        self.assertAlmostEqual(std, 1.572094414, places=3)
        self.assertAlmostEqual(minimum, -5.999995, places=3)
        self.assertAlmostEqual(maximum, 3.2975309, places=3)

    def test_ability_map(self):
        """Testing Maximum A Posteriori."""

        ## Regression tests for various types of measurements
        recovered_theta = ability_3pl_map(self.data, self.difficulty,
                                          self.discrimination, self.guessing)

        # Get stats on the theta
        mean = np.nanmean(recovered_theta)
        std = np.nanstd(recovered_theta)
        minimum = np.nanmin(recovered_theta)
        maximum = np.nanmax(recovered_theta)

        self.assertAlmostEqual(mean, 0.0348392, places=3)
        self.assertAlmostEqual(std,0.71308818659, places=3)
        self.assertAlmostEqual(minimum, -1.7259849459, places=3)
        self.assertAlmostEqual(maximum, 1.64162059967, places=3)

    def test_ability_eap(self):
        """Testing Expected A Posteriori."""

        ## Regression tests for various types of measurements
        recovered_theta = ability_3pl_eap(self.data, self.difficulty,
                                          self.discrimination, self.guessing)

        # Get stats on the theta
        mean = np.nanmean(recovered_theta)
        std = np.nanstd(recovered_theta)
        minimum = np.nanmin(recovered_theta)
        maximum = np.nanmax(recovered_theta)

        self.assertAlmostEqual(mean, 0.004648932, places=3)
        self.assertAlmostEqual(std, 0.732529437, places=3)
        self.assertAlmostEqual(minimum, -1.7767052989, places=3)
        self.assertAlmostEqual(maximum, 1.67935964, places=3)


if __name__ == '__main__':
    unittest.main()

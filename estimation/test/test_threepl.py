import unittest  # pylint: disable=cyclic-import

import numpy as np

from girth.synthetic import create_synthetic_irt_dichotomous
from girth import (threepl_mml,ability_3pl_eap, ability_3pl_map,
    ability_3pl_mle)


class TestMMLThreePLMethods(unittest.TestCase):

    # REGRESSION TESTS
    # The 3PL doesn't have good convergence properties
    # so don't test them

    """Setup synthetic data."""

    def setUp(self):
        """Setup synthetic data for tests."""
        rng = np.random.default_rng(783162340587963094862)
        difficulty = np.linspace(-1.5, 1.5, 5)
        discrimination = rng.uniform(0.5, 1.5, 5)
        thetas = rng.standard_normal(600)
        guessing = rng.uniform(0, .3, 5)
        syn_data = create_synthetic_irt_dichotomous(difficulty, discrimination,
                                                    thetas, guessing=guessing, seed=rng)
        self.data = syn_data
        self.guessing = guessing

    def test_threepl_regression_mml(self):
        """Testing threepl separate methods."""
        syn_data = self.data.copy()
        output = threepl_mml(syn_data)

        expected_discrimination = np.array([0.500858, 0.896211, 0.673024, 0.474641, 1.148911])
        expected_difficulty = np.array([-2.522691, -1.291617, -0.2958  , -0.355891,  1.186779])
        expected_guess = np.array([0, 0, 0, 0, .18262])

        np.testing.assert_allclose(
            expected_discrimination, output['Discrimination'], rtol=1e-2, atol=1e-2)
        np.testing.assert_allclose(expected_difficulty, output['Difficulty'], rtol=1e-2, atol=1e-2)
        np.testing.assert_allclose(expected_guess, output['Guessing'], rtol=1e-2, atol=1e-2)


class TestAbilityEstimates3PL(unittest.TestCase):
    """Tests the estimation of ability parameters."""

    def setUp(self):
        """Setup synthetic data for tests."""
        rng = np.random.default_rng(342543289078524332)
        
        difficulty = np.linspace(-1.5, 1.5, 10)
        discrimination = rng.uniform(0.5, 1.5, 10)
        thetas = rng.standard_normal(1000)
        guessing = rng.uniform(0, .25, 10)
        syn_data = create_synthetic_irt_dichotomous(difficulty, discrimination,
                                                    thetas, guessing=guessing, seed=rng)

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

        self.assertAlmostEqual(mean, -0.04757676, places=3)
        self.assertAlmostEqual(std, 1.54440249, places=3)
        self.assertAlmostEqual(minimum, -5.999995, places=3)
        self.assertAlmostEqual(maximum, 3.2913410, places=3)

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

        self.assertAlmostEqual(mean, -0.02026042, places=3)
        self.assertAlmostEqual(std, 0.6800486, places=3)
        self.assertAlmostEqual(minimum, -1.8349905, places=3)
        self.assertAlmostEqual(maximum, 1.431140027, places=3)

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

        self.assertAlmostEqual(mean, -0.024376, places=3)
        self.assertAlmostEqual(std, 0.7086291, places=3)
        self.assertAlmostEqual(minimum, -1.8949730, places=3)
        self.assertAlmostEqual(maximum, 1.47836656, places=3)


if __name__ == '__main__':
    unittest.main()

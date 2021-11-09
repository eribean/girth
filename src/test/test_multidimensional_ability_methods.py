import unittest

import numpy as np

from girth import multidimensional_ability_eap, multidimensional_ability_map
from girth.synthetic import create_synthetic_irt_dichotomous, create_synthetic_irt_polytomous


class TestMultidimensionalMethods(unittest.TestCase):
    """Test Fixture for multidimensional ability methods."""

    def test_map_eap_fail_for_one_factor(self):
        """Testing ability fails with 1 factor inputs."""
        discrimination = np.zeros((10, 1))
        difficulty = np.zeros((10, 1))

        with self.assertRaises(AssertionError):
            multidimensional_ability_map(np.ones((10, 10)), difficulty, discrimination)

        with self.assertRaises(AssertionError):
            multidimensional_ability_eap(np.ones((10, 10)), difficulty, discrimination)

    def test_twopl_abilities(self):
        """Testing ability recovery for 2PL models."""
        rng = np.random.default_rng(3625345623544674874913623165)

        discrimination = rng.uniform(-2, 2, (10, 3))
        difficulty = np.linspace(-1.5, 1.5, 10)
        thetas = rng.standard_normal((3, 250))

        syn_data = create_synthetic_irt_dichotomous(difficulty, discrimination, 
                                                    thetas, seed=rng)
        
        abilities_eap = multidimensional_ability_eap(syn_data, difficulty, 
                                                     discrimination, {'quadrature_n': 21})

        abilities_map = multidimensional_ability_map(syn_data, difficulty, 
                                                     discrimination)
        
        rmse_eap = np.sqrt(np.square(abilities_eap - thetas).mean())
        rmse_map = np.sqrt(np.square(abilities_map - thetas).mean())

        # Regression Tests
        self.assertGreater(rmse_map, rmse_eap)
        self.assertAlmostEqual(rmse_eap, .6734, places=4)
        self.assertAlmostEqual(rmse_map, .6857, places=4)

    def test_grm_abilities(self):
        """Testing ability recovery for 2PL models."""
        rng = np.random.default_rng(3625345623544674874913623165)

        discrimination = rng.uniform(-2, 2, (10, 3))
        difficulty = np.sort(rng.uniform(-1, 1, (10, 3)))
        thetas = rng.standard_normal((3, 250))

        syn_data = create_synthetic_irt_polytomous(difficulty, discrimination, 
                                                   thetas, seed=rng, model='grm_md')
        
        abilities_eap = multidimensional_ability_eap(syn_data, difficulty, 
                                                     discrimination, {'quadrature_n': 21})

        abilities_map = multidimensional_ability_map(syn_data, difficulty, 
                                                     discrimination)
        rmse_eap = np.sqrt(np.square(abilities_eap - thetas).mean())
        rmse_map = np.sqrt(np.square(abilities_map - thetas).mean())

        # Regression Tests
        self.assertGreater(rmse_map, rmse_eap)
        self.assertAlmostEqual(rmse_eap, .65225, places=4)
        self.assertAlmostEqual(rmse_map, .65531, places=4)
       

if __name__ == "__main__":
    unittest.main()
import unittest  # pylint: disable=cyclic-import

import numpy as np
from scipy import stats

from girth import create_synthetic_irt_dichotomous
from girth import twopl_mml_eap

from girth import create_synthetic_irt_polytomous
from girth import grm_mml_eap


class TestMMLEAPMethods(unittest.TestCase):
    """Testing running eap methods."""
    
    def test_2pl_mml_eap_method(self):
        """Testing the 2PL EAP/MML Method."""
        rng = np.random.default_rng(68213328964)

        n_items = 5
        n_people = 150
        difficulty = rng.standard_normal(n_items)
        discrimination = rng.rayleigh(scale=.8, size=n_items) + 0.25
        thetas = rng.standard_normal(n_people)
        syn_data = create_synthetic_irt_dichotomous(difficulty, discrimination,
                                                    thetas, seed=rng)

        result = twopl_mml_eap(syn_data, {'hyper_quadrature_n': 21})

        # Smoke Tests / Regression Tests
        expected_difficulty = np.array([-0.388077, -0.620283,  0.512279,  
                                       0.413742, -0.99742])
        expected_discrimination = np.array([1.653249, 0.934447, 2.077619, 
                                            1.033233, 1.43353])
        expected_rayleigh_scale = 0.942502691

        np.testing.assert_allclose(result['Difficulty'], expected_difficulty, 
                                   atol=1e-3, rtol=1e-3)
        np.testing.assert_allclose(result['Discrimination'], expected_discrimination, 
                                   atol=1e-3, rtol=1e-3)
        self.assertAlmostEqual(result['Rayleigh_Scale'], expected_rayleigh_scale, 3)
    
    def test_2pl_mml_eap_method_csirt(self):
        """Testing the 2PL EAP/MML Method with CSIRT."""
        rng = np.random.default_rng(55584684359412)

        n_items = 10
        n_people = 300
        difficulty = rng.standard_normal(n_items)
        discrimination = rng.rayleigh(scale=.8, size=n_items) + 0.25
        thetas = rng.standard_normal(n_people)
        syn_data = create_synthetic_irt_dichotomous(difficulty, discrimination,
                                                    thetas, seed=rng)

        result = twopl_mml_eap(syn_data, {'estimate_distribution': True})

        # Smoke Tests / Regression Tests
        expected_difficulty = np.array([0.164805, -0.847592,  0.702926,  
                                        1.680903,  1.668845, -0.505642,
                                        -0.041614,  2.329605, -0.426624, 
                                        0.487625])
        expected_discrimination = np.array([1.880526, 1.261701, 2.02081, 
                                            0.613751, 0.927996, 1.551378,
                                            1.323135, 0.962349, 1.048494, 
                                            0.853246])
        expected_rayleigh_scale = 0.814573286388

        np.testing.assert_allclose(result['Difficulty'], expected_difficulty, 
                                   atol=1e-3, rtol=1e-3)
        np.testing.assert_allclose(result['Discrimination'], expected_discrimination, 
                                   atol=1e-3, rtol=1e-3)
        self.assertAlmostEqual(result['Rayleigh_Scale'], expected_rayleigh_scale, 3)

    def test_grm_mml_eap_method(self):
        """Testing the GRM EAP/MML Method."""
        rng = np.random.default_rng(49831588423129843218)

        n_items = 10
        n_people = 300
        difficulty = rng.standard_normal((n_items, 3))
        difficulty = np.sort(difficulty, axis=1)
        discrimination = rng.rayleigh(scale=.8, size=n_items) + 0.25
        thetas = rng.standard_normal(n_people)
        syn_data = create_synthetic_irt_polytomous(difficulty, discrimination,
                                                   thetas, seed=rng)

        result = grm_mml_eap(syn_data, {'hyper_quadrature_n': 21})
        # Smoke Tests / Regression Tests
        expected_difficulty = np.array([[-.823933242, -.681746823,  .709577506],
                                        [-1.19660316, -.591194824,  .238794870],
                                        [-1.57623244,  1.02585861,  1.31222183],
                                        [-1.26761219,   0.000,  1.00491272],
                                        [-.976097754,  .504520875,  .696167966],
                                        [-.101893742,  .866864508,  1.03823366],
                                        [-.0342219350,  .148527684,  1.04351431],
                                        [-.0793099062,  .697370885,  2.10740616],
                                        [-2.36175151, -1.23090629,  .457447133],
                                        [-1.90991042,  .0161508466,  2.24006109]])
        expected_discrimination = np.array([1.54167 , 0.679827, 1.730017, 
                                            0.604388, 2.163533, 1.168009,
                                            1.837987, 1.370698, 1.050023, 0.99699])
        expected_rayleigh_scale = 0.9128964575265555

        np.testing.assert_allclose(result['Difficulty'], expected_difficulty, 
                                   atol=1e-3, rtol=1e-3)
        np.testing.assert_allclose(result['Discrimination'], expected_discrimination, 
                                   atol=1e-3, rtol=1e-3)
        self.assertAlmostEqual(result['Rayleigh_Scale'], expected_rayleigh_scale, 3)

    


if __name__ == '__main__':
    unittest.main()
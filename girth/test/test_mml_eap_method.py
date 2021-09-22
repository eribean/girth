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
        rng = np.random.default_rng(498134161633318511141265412)

        n_items = 5
        n_people = 150
        difficulty = rng.standard_normal(n_items)
        discrimination = rng.rayleigh(scale=.8, size=n_items) + 0.25
        thetas = rng.standard_normal(n_people)
        syn_data = create_synthetic_irt_dichotomous(difficulty, discrimination,
                                                    thetas, seed=rng)

        result = twopl_mml_eap(syn_data, {'hyper_quadrature_n': 21})

        # Smoke Tests / Regression Tests
        expected_difficulty = np.array([-2.46188484,  0.1742868 ,  0.39955514,
                                        0.26342169, -0.48505039])
        expected_discrimination = np.array([1.0578002 , 1.53611834, 0.85715942, 
                                            0.81414815, 1.87899111])
        expected_rayleigh_scale = 0.84603275509

        np.testing.assert_allclose(result['Difficulty'], expected_difficulty, 
                                   atol=1e-3, rtol=1e-3)
        np.testing.assert_allclose(result['Discrimination'], expected_discrimination, 
                                   atol=1e-3, rtol=1e-3)
        self.assertAlmostEqual(result['Rayleigh_Scale'], expected_rayleigh_scale, 3)
    
    def test_2pl_mml_eap_method_csirt(self):
        """Testing the 2PL EAP/MML Method with CSIRT."""
        rng = np.random.default_rng(2443562564766554184345564)

        n_items = 10
        n_people = 300
        difficulty = rng.standard_normal(n_items)
        discrimination = rng.rayleigh(scale=.8, size=n_items) + 0.25
        thetas = rng.standard_normal(n_people)
        syn_data = create_synthetic_irt_dichotomous(difficulty, discrimination,
                                                    thetas, seed=rng)

        result = twopl_mml_eap(syn_data, {'estimate_distribution': True})

        # Smoke Tests / Regression Tests
        expected_difficulty = np.array([2.23559897, 0.29362425, 1.38100264,  
                                        0.08279119,  0.08444424, 1.18819404, 
                                        -1.07454426,  0.39896597,  1.8653525 , -1.84060226])

        expected_discrimination = np.array([1.50841056, 0.99998227, 0.83592715, 
                                            0.66005477, 1.4747552, 0.77039347, 1.05569968, 
                                            0.87996685, 0.65185882, 0.58027912])

        expected_rayleigh_scale = 0.6113984943713622

        np.testing.assert_allclose(result['Difficulty'], expected_difficulty, 
                                   atol=1e-3, rtol=1e-3)
        np.testing.assert_allclose(result['Discrimination'], expected_discrimination, 
                                   atol=1e-3, rtol=1e-3)
        self.assertAlmostEqual(result['Rayleigh_Scale'], expected_rayleigh_scale, 3)

    def test_grm_mml_eap_method(self):
        """Testing the GRM EAP/MML Method."""
        rng = np.random.default_rng(236472098334423445254346234514216354152)

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
        expected_difficulty = np.array([
            [-3.82592436e-01, -3.32902099e-02,  1.17205422e+00],
            [ 7.16137937e-01,  2.28463703e+00,  2.39706723e+00],
            [-9.57180145e-01,  8.92447121e-01,  1.66519245e+00],
            [-6.59495671e-01, -1.26141755e-01,  2.10861683e-01],
            [ 3.92071879e-01,  1.16292199e+00,  1.50486048e+00],
            [-4.73496122e-01, -1.61617298e-02,  7.64450828e-01],
            [-1.35482731e+00, -3.79201843e-01,  1.77660394e+00],
            [-1.89738605e-05,  1.30368580e-01,  7.61251849e-01],
            [-9.37302462e-01,  7.30546564e-01,  1.13065092e+00],
            [-7.32189206e-01, -3.28669948e-01,  9.13926636e-01]])
            
        expected_discrimination = np.array([1.96220993, 1.17821202, 0.76327225, 
                                            2.27400557, 1.51836504, 0.99894561, 
                                            1.62168688, 0.82284504, 1.51384642, 1.40226151])
        expected_rayleigh_scale = 0.91585679394

        np.testing.assert_allclose(result['Difficulty'], expected_difficulty, 
                                   atol=1e-3, rtol=1e-3)
        np.testing.assert_allclose(result['Discrimination'], expected_discrimination, 
                                   atol=1e-3, rtol=1e-3)
        self.assertAlmostEqual(result['Rayleigh_Scale'], expected_rayleigh_scale, 3)

    


if __name__ == '__main__':
    unittest.main()
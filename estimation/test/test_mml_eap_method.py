import unittest  # pylint: disable=cyclic-import

import numpy as np
from scipy import stats

from girth.synthetic import (create_synthetic_irt_dichotomous, 
    create_synthetic_irt_polytomous)
from girth import twopl_mml_eap, grm_mml_eap


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
        expected_difficulty = np.array([-2.59843406,  0.16349846,  0.47781903, 
                                         0.28155578, -0.56089445])
        expected_discrimination = np.array([1.25226578, 1.30324285, 1.03646604, 
                                            0.89044465, 1.53304179])
        expected_rayleigh_scale = 0.7854159806618226

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
        expected_difficulty = np.array([2.50209149,  0.29538839,  1.77631619, -0.08186472,
                                        0.10324231,  1.2814035 , -1.18880568,  0.41636641,  
                                        1.93354785, -2.02509822])

        expected_discrimination = np.array([1.45225193, 1.05471218, 0.85484699, 0.57281053, 
                                            1.38938894, 1.19435566, 0.94578246, 0.83194594, 
                                            0.78164372, 0.48282339])

        expected_rayleigh_scale = 0.6185686206694863

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
            [-0.38879266, -0.0676688 ,  1.17476981],
            [ 0.80428071,  2.52643442,  2.64529057],
            [-0.90612171,  0.95480765,  1.62234167],
            [-0.71455876, -0.2025803 ,  0.10107777],
            [ 0.40495752,  1.20726801,  1.52233384],
            [-0.44724955, -0.04744493,  0.81796385],
            [-1.39214136, -0.41827662,  1.65457775],
            [ 0.07009911,  0.21055447,  0.68580568],
            [-0.86960408,  0.71323535,  1.1572905 ],
            [-0.71543594, -0.32119227,  0.90824854]])
            
        expected_discrimination = np.array([1.88806681, 0.9014368, 1.30044468, 
                                            1.18238355, 1.34671981, 1.02987716, 
                                            1.20451242, 0.62125472, 1.27234125, 1.46234896])

        expected_rayleigh_scale = 0.6919453387630914

        np.testing.assert_allclose(result['Difficulty'], expected_difficulty, 
                                   atol=1e-3, rtol=1e-3)
        np.testing.assert_allclose(result['Discrimination'], expected_discrimination, 
                                   atol=1e-3, rtol=1e-3)
        self.assertAlmostEqual(result['Rayleigh_Scale'], expected_rayleigh_scale, 3)

    


if __name__ == '__main__':
    unittest.main()
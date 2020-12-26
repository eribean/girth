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
        np.random.seed(618331)

        n_items = 5
        n_people = 150
        difficulty = stats.norm(0, 1).rvs(n_items)
        discrimination = stats.rayleigh(loc=0.25, scale=.8).rvs(n_items)
        thetas = np.random.randn(n_people)
        syn_data = create_synthetic_irt_dichotomous(difficulty, discrimination,
                                                    thetas)

        result = twopl_mml_eap(syn_data, {'hyper_quadrature_n': 21})

        # Smoke Tests / Regression Tests
        expected_difficulty = np.array([-0.2436698, 0.66299148, 1.3451037, 
                                        -0.68059041, 0.40516614])
        expected_discrimination = np.array([1.99859796, 0.67420679, 1.18591025, 
                                            1.60937911, 1.19672389])
        expected_rayleigh_scale = 0.9106036068099617

        np.testing.assert_allclose(result['Difficulty'], expected_difficulty, 
                                   atol=1e-3, rtol=1e-3)
        np.testing.assert_allclose(result['Discrimination'], expected_discrimination, 
                                   atol=1e-3, rtol=1e-3)
        self.assertAlmostEqual(result['Rayleigh_Scale'], expected_rayleigh_scale, 3)


    def test_2pl_mml_eap_method_csirt(self):
        """Testing the 2PL EAP/MML Method with CSIRT."""
        np.random.seed(779841)

        n_items = 10
        n_people = 300
        difficulty = stats.norm(0, 1).rvs(n_items)
        discrimination = stats.rayleigh(loc=0.25, scale=.8).rvs(n_items)
        thetas = np.random.randn(n_people)
        syn_data = create_synthetic_irt_dichotomous(difficulty, discrimination,
                                                    thetas)

        result = twopl_mml_eap(syn_data, {'estimate_distribution': True})

        # Smoke Tests / Regression Tests
        expected_difficulty = np.array([-0.91561408, 1.29631473, 1.01751178,
                                        -0.10536047, -0.02235909, -0.56510317,
                                        -1.67564893, -1.45646904,  1.89544833, 
                                        -0.78602385])
        expected_discrimination = np.array([0.9224411, 0.88102312, 0.86716565, 
                                            1.38012222, 0.67176012, 1.84035622,
                                            1.58453053, 1.11488035, 1.07633054, 
                                            1.44767879])
        expected_rayleigh_scale = 0.7591785686427288

        np.testing.assert_allclose(result['Difficulty'], expected_difficulty, 
                                   atol=1e-3, rtol=1e-3)
        np.testing.assert_allclose(result['Discrimination'], expected_discrimination, 
                                   atol=1e-3, rtol=1e-3)
        self.assertAlmostEqual(result['Rayleigh_Scale'], expected_rayleigh_scale, 3)

    
    def test_grm_mml_eap_method(self):
        """Testing the GRM EAP/MML Method."""
        np.random.seed(99854)

        n_items = 10
        n_people = 300
        difficulty = stats.norm(0, 1).rvs(n_items*3).reshape(n_items, -1)
        difficulty = np.sort(difficulty, axis=1)
        discrimination = stats.rayleigh(loc=0.25, scale=.8).rvs(n_items)
        thetas = np.random.randn(n_people)
        syn_data = create_synthetic_irt_polytomous(difficulty, discrimination,
                                                   thetas)

        result = grm_mml_eap(syn_data, {'hyper_quadrature_n': 21})

        # Smoke Tests / Regression Tests
        expected_difficulty = np.array([[-0.68705911,  0.25370937,  0.62872705],
                                        [-1.81331475, -1.52607597,  0.01957819],
                                        [-2.16305964, -0.51648053,  0.20447022],
                                        [-1.51064069, -1.18709807,  1.74368598],
                                        [-2.44714587, -1.01438472, -0.44406173],
                                        [-1.38622596, -0.11417447,  1.14001425],
                                        [-0.92724279, -0.11335446,  1.30273993],
                                        [-0.55972331, -0.28527674,  0.01131112],
                                        [-1.72941028, -0.34732405,  1.17681916],
                                        [-1.73346085, -0.12292641,  0.91797906]])
        expected_discrimination = np.array([1.35572245, 0.77018004, 0.92848851, 1.6339604, 
                                            0.79229545, 2.35881697, 0.64452994, 1.86795956, 
                                            1.56986454, 1.93426233])
        expected_rayleigh_scale = 0.9161607303681261

        np.testing.assert_allclose(result['Difficulty'], expected_difficulty, 
                                   atol=1e-3, rtol=1e-3)
        np.testing.assert_allclose(result['Discrimination'], expected_discrimination, 
                                   atol=1e-3, rtol=1e-3)
        self.assertAlmostEqual(result['Rayleigh_Scale'], expected_rayleigh_scale, 3)

    


if __name__ == '__main__':
    unittest.main()
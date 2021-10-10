import unittest

import numpy as np

from girth import create_synthetic_mirt_dichotomous, create_synthetic_irt_polytomous
from girth import multidimensional_grm_mml, multidimensional_twopl_mml
from girth import initial_guess_md

from girth.multidimensional.multidimensional_mml_methods import _build_einsum_string
from girth.multidimensional.multidimensional_initial_guess import _constrained_rotation

class TestMultiDimensionalIRT(unittest.TestCase):
    """Test fixture for multidimensional IRT."""

    def test_multidimensional_2pl(self):
        """Testing Multidimensional 2PL Model."""
        rng = np.random.default_rng(15132185498151515176351456531457133245)
        difficulty = np.linspace(-1.5, 1, 20)
        discrimination = rng.uniform(-2, 2, (20, 2))
        discrimination[-1, -1] = 0
        discrimination[-1, 0] *= np.sign(discrimination[-1, 0])
        discrimination[-2, -1] *= np.sign(discrimination[-2, -1])
        thetas = rng.standard_normal((2, 1000))
        syn_data = create_synthetic_mirt_dichotomous(difficulty, 
                                                     discrimination, 
                                                     thetas, seed=rng)
        
        results = multidimensional_twopl_mml(syn_data, 2, 
                                            {'quadrature_n': 31,
                                             'max_iteration': 750})

        rmse_discrimination = np.sqrt((np.square(discrimination - 
                                                 results['Discrimination']).mean(1)))
        
        rmse_difficulty = np.sqrt((np.square(difficulty - 
                                             results['Difficulty']).mean()))
        
        expected_discrimination = [0.12546215, 0.07612413, 0.1016446,  0.03714225, 0.21972059, 
                                   0.2208587, 0.14856666, 0.07090967, 0.2034743, 0.05039998, 
                                   0.21632571, 0.05843558, 0.03358012, 0.06995068, 0.19614291,
                                   0.12359983, 0.14229356, 0.10547182, 0.18112039, 0.00171321]

        self.assertAlmostEqual(rmse_difficulty, 0.093267, places=4)
        np.testing.assert_allclose(expected_discrimination, rmse_discrimination, atol=1e-4)

        # smoke test
        multidimensional_twopl_mml(syn_data, 2, 
                                   {'quadrature_n': 31,
                                    'initial_guess': False,
                                    'max_iteration': 2})
    
    def test_multidimensional_grm(self):
        """Testing Multidimensional GRM Model."""
        rng = np.random.default_rng(93209819094946739803948765)
        difficulty = -1*np.sort(rng.uniform(-1.5, 1.5, (5, 3)), axis=1)
        discrimination = rng.uniform(-2, 2, (5, 3))
        discrimination[-1] = [1, 0, 0]
        discrimination[-2] = [1, 1, 0]
        discrimination[-3] = [1, 1, 1]
        thetas = rng.standard_normal((3, 500))

        syn_data = create_synthetic_irt_polytomous(difficulty, 
                                                   discrimination, 
                                                   thetas, model='grm_md',
                                                   seed=rng)

        # Smoke Test        
        results = multidimensional_grm_mml(syn_data, 3, 
                                            {'quadrature_n': 15,
                                             'use_LUT': True,
                                             'max_iteration': 2})

        with self.assertRaises(AssertionError):
            results = multidimensional_grm_mml(syn_data, 1, 
                                                {'quadrature_n': 15,
                                                'use_LUT': True,
                                                'max_iteration': 2})                                             

    def test_initial_guess(self):
        """Testing the initial guess estimation."""
        rng = np.random.default_rng(6113654132541626534123241358462635552)
        difficulty = np.linspace(-1.5, 1, 20)
        discrimination = rng.uniform(-2, 2, (20, 2))
        discrimination[-1, -1] = 0
        discrimination[-1, 0] *= np.sign(discrimination[-1, 0])
        discrimination[-2, -1] *= np.sign(discrimination[-2, -1])
        thetas = rng.standard_normal((2, 2000))
        syn_data = create_synthetic_mirt_dichotomous(difficulty, 
                                                     discrimination, 
                                                     thetas, seed=rng)

        estimated_discrimination = initial_guess_md(syn_data, 2)

        rmse_discrimination = np.sqrt((np.square(estimated_discrimination - 
                                       discrimination).mean(1)))
        
        expected = [0.0464029, 0.09418929, 0.12875905, 0.07286165, 0.12253134, 
                    0.08039802, 0.10111907, 0.03280063, 0.05492763, 0.11300778, 
                    0.13379047, 0.04615303, 0.01944146, 0.0531284, 0.02966092, 
                    0.17807948, 0.05713511, 0.10060402, 0.07125093, 0.16767814]

        np.testing.assert_allclose(expected, rmse_discrimination, atol=1e-4)

    def test_constrained_rotation(self):
        """Testing constrained rotation."""
        n_items = 20

        for n_factors in range(1, 11):
            # These matrices help with the discrimination constraints  
            lower_indicies = np.triu_indices(n_items, k=1, m=n_factors)
            diagonal_indices = np.diag_indices(n_factors)
            lower_length = lower_indicies[0].shape[0]
            compare_zero = np.zeros(lower_length)

            # Set constraints to be the final items
            lower_indicies = (n_items - 1 - lower_indicies[0], lower_indicies[1])
            diagonal_indices = (n_items - 1 - diagonal_indices[0], diagonal_indices[1])            

            discrimination = np.random.uniform(-2, 2, (n_items, n_factors))
            new_discrimination = _constrained_rotation(discrimination)

            self.assertTrue(np.all(new_discrimination[diagonal_indices] > 0))

            if n_factors > 1:
                np.testing.assert_allclose(new_discrimination[lower_indicies], 
                                           compare_zero, atol=1e-6)








if __name__ == '__main__':
    unittest.main()

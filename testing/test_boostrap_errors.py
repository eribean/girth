import unittest

import numpy as np

from girth import standard_errors_bootstrap
from girth.synthetic import create_synthetic_irt_dichotomous, create_synthetic_irt_polytomous
from girth import (jml_methods, mml_methods, mml_eap_methods, mml_full_methods,
                   conditional_methods)

def _contains_keys(results, identifier):
    """Checks for standard keys in bootstrap result."""
    for key in ['Standard Errors', '95th CI', 'Bias', 'Solution']:
        if key not in results.keys():
            raise AssertionError(f"Key: {key} not found in return argument." 
                                 f"Error in {identifier}")

    for key in results['95th CI']:
        if np.any(results['95th CI'][key][1] < results['95th CI'][key][0]):
            raise AssertionError(f"Confidence Interval Error. {key} " 
                                 f"Error in {identifier}")
   

class TestBootstrapStandardErrors(unittest.TestCase):
    """Test Fixture for Bootstrap Standard Errors."""
    
    # Smoke Tests to make sure they give an output

    # Tests bootstrap errors
    def setUp(self):
        rng = np.random.default_rng(48725309847520)
        self.discrimination = 0.25 + rng.rayleigh(.7, 5)
        self.difficulty = np.linspace(-1.5, 1.5, 5)
        self.difficulty_poly = np.sort(rng.standard_normal((5, 3)), axis=1)
        self.theta = rng.standard_normal(1000)
        self.options = {'max_iteration': 2}
        self.boot_iter = 10

    def test_jml_methods_dichotomous(self):
        """Testing Bootstrap on JML Methods Dichotomous."""
        rng = np.random.default_rng(39485720394875)
        dataset = create_synthetic_irt_dichotomous(self.difficulty, self.discrimination, 
                                                   self.theta, seed=rng)

        result = standard_errors_bootstrap(dataset, jml_methods.rasch_jml, n_processors=1,
                                           bootstrap_iterations=self.boot_iter, 
                                           options=self.options)
        self.assertEqual(result['Standard Errors']['Discrimination'][0], 0)
        _contains_keys(result, 'Rasch JML')

        result = standard_errors_bootstrap(dataset, jml_methods.onepl_jml, n_processors=2,
                                           bootstrap_iterations=self.boot_iter, 
                                           options=self.options)
        _contains_keys(result, '1PL JML')

        result = standard_errors_bootstrap(dataset, jml_methods.twopl_jml, n_processors=2,
                                           bootstrap_iterations=self.boot_iter, 
                                           options=self.options)
        _contains_keys(result, '2PL JML')

    def test_jml_methods_polytomous(self):
        """Testing Bootstrap on JML Methods Polytomous."""
        rng = np.random.default_rng(8672379287302651089)
        
        dataset = create_synthetic_irt_polytomous(self.difficulty_poly, self.discrimination, 
                                                  self.theta, seed=rng)

        result = standard_errors_bootstrap(dataset, jml_methods.grm_jml, n_processors=2,
                                           bootstrap_iterations=self.boot_iter, 
                                           options=self.options)
        self.assertTupleEqual(result['95th CI']['Difficulty'][0].shape, 
                              self.difficulty_poly.shape)
        _contains_keys(result, 'GRM JML')

        dataset = create_synthetic_irt_polytomous(self.difficulty_poly, self.discrimination, 
                                                  self.theta, seed=rng, model='pcm')

        result = standard_errors_bootstrap(dataset, jml_methods.pcm_jml, n_processors=2,
                                           bootstrap_iterations=self.boot_iter, 
                                           options=self.options)
        self.assertTupleEqual(result['95th CI']['Difficulty'][0].shape, 
                              self.difficulty_poly.shape)
        _contains_keys(result, 'PCM JML')

    def test_rasch_conditional(self):
        """Testing rasch conditional methods."""
        rng = np.random.default_rng(426376867989075563)
        dataset = create_synthetic_irt_dichotomous(self.difficulty, self.discrimination, 
                                                   self.theta, seed=rng)

        result = standard_errors_bootstrap(dataset, conditional_methods.rasch_conditional, 
                                           n_processors=2,
                                           bootstrap_iterations=self.boot_iter, 
                                           options=self.options)
        self.assertEqual(result['Standard Errors']['Discrimination'][0], 0)
        _contains_keys(result, 'Rasch MML')

    def test_mml_methods_dichotomous(self):
        """Testing Bootstrap on MML Methods Dichotomous."""
        rng = np.random.default_rng(8764328976187234)
        dataset = create_synthetic_irt_dichotomous(self.difficulty, self.discrimination, 
                                                   self.theta, seed=rng)

        result = standard_errors_bootstrap(dataset, mml_methods.rasch_mml, n_processors=2,
                                           bootstrap_iterations=self.boot_iter, 
                                           options=self.options)
        self.assertEqual(result['Standard Errors']['Discrimination'][0], 0)
        _contains_keys(result, 'Rasch MML')

        result = standard_errors_bootstrap(dataset, mml_methods.onepl_mml, n_processors=2,
                                           bootstrap_iterations=self.boot_iter, 
                                           options=self.options)
        _contains_keys(result, '1PL MML')

        result = standard_errors_bootstrap(dataset, mml_methods.twopl_mml, n_processors=2,
                                           bootstrap_iterations=self.boot_iter, 
                                           options=self.options)
        _contains_keys(result, '2PL MML')

    def test_mml_methods_polytomous(self):
        """Testing Bootstrap on MML Methods Polytomous."""
        rng = np.random.default_rng(4347621232345345696)
        
        dataset = create_synthetic_irt_polytomous(self.difficulty_poly, self.discrimination, 
                                                  self.theta, seed=rng)

        result = standard_errors_bootstrap(dataset, mml_methods.grm_mml, n_processors=2,
                                           bootstrap_iterations=self.boot_iter, 
                                           options=self.options)
        self.assertTupleEqual(result['95th CI']['Difficulty'][0].shape, 
                              self.difficulty_poly.shape)
        _contains_keys(result, 'GRM MML')

        dataset = create_synthetic_irt_polytomous(self.difficulty_poly, self.discrimination, 
                                                  self.theta, seed=rng, model='pcm')

        result = standard_errors_bootstrap(dataset, mml_full_methods.pcm_mml, n_processors=2,
                                           bootstrap_iterations=self.boot_iter, 
                                           options=self.options)
        self.assertTupleEqual(result['95th CI']['Difficulty'][0].shape, 
                              self.difficulty_poly.shape)
        _contains_keys(result, 'PCM MML')

    def test_eap_mml_methods(self):
        """Testing Bootstrap on eap methods."""
        rng = np.random.default_rng(66739234876520981)
        dataset = create_synthetic_irt_dichotomous(self.difficulty, self.discrimination, 
                                                   self.theta, seed=rng)

        result = standard_errors_bootstrap(dataset, mml_eap_methods.twopl_mml_eap, n_processors=2,
                                           bootstrap_iterations=self.boot_iter, 
                                           options=self.options)
        _contains_keys(result, '2PL EAP-MML')        
        
        dataset = create_synthetic_irt_polytomous(self.difficulty_poly, self.discrimination, 
                                                  self.theta, seed=rng)

        result = standard_errors_bootstrap(dataset, mml_eap_methods.grm_mml_eap, n_processors=2,
                                           bootstrap_iterations=self.boot_iter, 
                                           options=self.options)
        self.assertTupleEqual(result['95th CI']['Difficulty'][0].shape, 
                              self.difficulty_poly.shape)
        _contains_keys(result, 'GRM EAP-MML')



if __name__ == "__main__":
    unittest.main()
import unittest

import numpy as np

from girth import rasch_conditional
from girth import create_synthetic_irt_dichotomous
from girth.conditional_methods import _symmetric_functions


class TestConditionalRasch(unittest.TestCase):
    """Regression tests to make sure output is working."""

    def test_conditional_regression(self):
        """Testing conditional rasch model."""
        rng = np.random.default_rng(1643242198124)
        difficuly = np.linspace(-1.5, 1.5, 5)
        discrimination = 1
        thetas = rng.standard_normal(600)
        syn_data = create_synthetic_irt_dichotomous(difficuly, discrimination,
                                                    thetas, seed=rng)

        output = rasch_conditional(syn_data)['Difficulty']
        expected_output = np.array([-1.543584, -0.732148, -0.01494,  
                                    0.768816,  1.521855])

        np.testing.assert_allclose(expected_output, output, atol=1e-6)


    def test_conditional_regression_discrimination(self):
        """Testing conditional rasch model with non-unity discrimination."""
        rng = np.random.default_rng(88743218879951231)
        difficuly = np.linspace(-1.5, 1.5, 5)
        discrimination = 1.7
        thetas = rng.standard_normal(600)
        syn_data = create_synthetic_irt_dichotomous(difficuly, discrimination,
                                                    thetas, seed=rng)

        output = rasch_conditional(syn_data, discrimination)['Difficulty']
        expected_output = np.array([-1.565442, -0.905947,  0.118824,  
                                     0.767591,  1.584975])

        np.testing.assert_allclose(expected_output, output, atol=1e-6)


    def test_conditional_close(self):
        """Testing conditional rasch model for accuracy."""
        rng = np.random.default_rng(468135249816547)
        difficuly = np.linspace(-1.5, 1.5, 5)
        discrimination = 1.2
        thetas = rng.standard_normal(600)
        syn_data = create_synthetic_irt_dichotomous(difficuly, discrimination,
                                                    thetas, seed=rng)
        output = rasch_conditional(syn_data, discrimination)['Difficulty']

        np.testing.assert_array_almost_equal(difficuly, output, decimal=1)


    def test_symmetric_function(self):
        """Testing the generation of symmetric functions."""
        rng = np.random.default_rng(587731189834)
        betas = rng.standard_normal(10)

        # Compare by checking against fft method
        fft_size = betas.size + 1
        fit = np.c_[np.ones_like(betas), np.exp(-betas)]
        expected = np.fft.ifft(np.fft.fft(fit, fft_size, axis=1).prod(axis=0)).real

        # Output of function
        output = _symmetric_functions(betas)

        np.testing.assert_almost_equal(output, expected)


if __name__ == '__main__':
    unittest.main()

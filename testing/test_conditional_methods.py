import unittest

import numpy as np

from girth import rasch_conditional
from girth import create_synthetic_irt_dichotomous


class TestConditionalRasch(unittest.TestCase):
    """Regression tests to make sure output is working."""

    def test_conditional_regression(self):
        """Testing conditional rasch model."""
        np.random.seed(91)
        difficuly = np.linspace(-1.5, 1.5, 5)
        discrimination = 1
        thetas = np.random.randn(600)
        syn_data = create_synthetic_irt_dichotomous(difficuly, discrimination,
                                                    thetas)

        output = rasch_conditional(syn_data)
        expected_output = np.array([-1.39893814, -0.80083855,
                                    -0.00947712,  0.61415543,  1.59509838])

        np.testing.assert_allclose(expected_output, output)

    def test_conditional_regression_discrimination(self):
        """Testing conditional rasch model."""
        np.random.seed(142)
        difficuly = np.linspace(-1.5, 1.5, 5)
        discrimination = 1.7
        thetas = np.random.randn(600)
        syn_data = create_synthetic_irt_dichotomous(difficuly, discrimination,
                                                    thetas)

        output = rasch_conditional(syn_data, discrimination)
        expected_output = np.array([-1.38086088, -0.74781933,
                                    -0.01267694,  0.77906715,  1.36228999])

        np.testing.assert_allclose(expected_output, output)

    def test_conditional_close(self):
        """Testing conditional rasch model."""
        np.random.seed(574)
        difficuly = np.linspace(-1.5, 1.5, 5)
        discrimination = 1.2
        thetas = np.random.randn(1600)
        syn_data = create_synthetic_irt_dichotomous(difficuly, discrimination,
                                                    thetas)
        output = rasch_conditional(syn_data, discrimination)

        np.testing.assert_array_almost_equal(difficuly, output, decimal=1)


if __name__ == '__main__':
    unittest.main()

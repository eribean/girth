import unittest

import numpy as np

from girth.utilities.utils import _get_quadrature_points
from girth import validate_estimation_options, create_beta_LUT
from girth.utilities.look_up_table import _array_LUT


class LookUpTableTests(unittest.TestCase):
    """Test Fixture for look-up table."""
    
    def test_array_LUT(self):
        """Test the creation of the array look up table."""
        alpha = np.linspace(.2, 4, 500)
        beta = np.linspace(-6, 6, 500)
        theta, weights = _get_quadrature_points(41, -5, 5)
        output = _array_LUT(alpha, beta, theta, weights)

        #Expected
        z = alpha[:, None, None] * (beta[None, :, None] - theta[None, None, :])
        expected = np.sum(1.0  / (1. + np.exp(z)) * weights[None, None, :], axis=2)        

        np.testing.assert_allclose(output, expected, atol=1e-4, rtol=1e-3)

    def test_lut_creation(self):
        """Test the lookup table creation function."""
        lut_func = create_beta_LUT((0.5, 2, 500), (-3, 3, 500))

        # do two values
        options = validate_estimation_options(None)
        quad_start, quad_stop = options['quadrature_bounds']
        quad_n = options['quadrature_n']
        
        theta, weight = _get_quadrature_points(quad_n, quad_start, quad_stop)
        distribution = options['distribution'](theta)

        alpha1 = 0.89
        beta1 = 1.76

        p_value1 = ((weight * distribution) / (1.0 + np.exp(-alpha1*(theta - beta1)))).sum()
        estimated_beta = lut_func(alpha1, p_value1)
        self.assertAlmostEqual(beta1, estimated_beta, places=4)

        alpha1 = 1.89
        beta1 = -2.34

        p_value1 = ((weight * distribution) / (1.0 + np.exp(-alpha1*(theta - beta1)))).sum()
        estimated_beta = lut_func(alpha1, p_value1)
        self.assertAlmostEqual(beta1, estimated_beta, places=4)


if __name__ == '__main__':
    unittest.main()

import unittest

import numpy as np
import numba as nb

from girth.utils import _get_quadrature_points
from girth.numba_functions import _compute_partial_integral, _array_LUT

# These don't register as covered tests in nose but
# The tests do run

class NumbaTests(unittest.TestCase):

    # Testing numba functions
    
    def test_array_LUT(self):
        """Test the creation of the array look up table."""
        alpha = np.linspace(.2, 4, 500)
        beta = np.linspace(-6, 6, 500)
        theta, weights = _get_quadrature_points(41, -5, 5)
        output = np.zeros((alpha.size, beta.size))
        _array_LUT(alpha, beta, theta, weights, output)

        #Expected
        z = alpha[:, None, None] * (beta[None, :, None] - theta[None, None, :])
        expected = np.sum(1.0  / (1. + np.exp(z)) * weights[None, None, :], axis=2)        

        np.testing.assert_allclose(output, expected, atol=1e-4, rtol=1e-3)

if __name__ == '__main__':
    unittest.main()

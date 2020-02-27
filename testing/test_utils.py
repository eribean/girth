import unittest

import numpy as np
from scipy.special import roots_legendre
from scipy import integrate

from girth import irt_evaluation
from girth.utils import _get_quadrature_points, _compute_partial_integral


class TestUtilitiesMethods(unittest.TestCase):
    """Tests the utilities functions in girth."""

    def test_irt_evaluation_single_discrimination(self):
        """Testing the IRT evaluation method when discrimination is scalar."""
        difficuly = np.array([-1, 1])
        theta = np.array([1., 2.])
        discrimination = 4.0

        # Expected output
        expected_output = 1.0 / (1.0 + np.exp(discrimination * (difficuly[:, None] - theta)))
        output = irt_evaluation(difficuly, discrimination, theta)

        np.testing.assert_allclose(output, expected_output)

    def test_irt_evaluation_array_discrimination(self):
        """Testing the IRT evaluation method when discrimination is array."""
        difficuly = np.array([-1, 1])
        theta = np.array([1., 2.])
        discrimination = np.array([1.7, 2.3])

        # Expected output
        expected_output = 1.0 / (1.0 + np.exp(discrimination[:, None] * (difficuly[:, None] - theta)))
        output = irt_evaluation(difficuly, discrimination, theta)

        np.testing.assert_allclose(output, expected_output)

    def test_quadrature_points(self):
        """Testing the creation of quadrtature points"""
        n_points = 11

        # A smoke test to make sure it's running properly
        quad_points = _get_quadrature_points(n_points, -1, 1)

        x, _ = roots_legendre(n_points)

        np.testing.assert_allclose(x, quad_points)

    def test_partial_integration_single(self):
        """Tests the integration quadrature function."""

        # Set seed for repeatability
        np.random.seed(154)

        discrimination = 1.32
        difficuly = np.linspace(-1.3, 1.3, 5)
        the_sign = (-1)**np.random.randint(low=0, high=2, size=(5, 1))

        quad_points = _get_quadrature_points(61, -6, 6)
        dataset = _compute_partial_integral(quad_points, difficuly, discrimination,
                                            the_sign)

        value = integrate.fixed_quad(lambda x: dataset, -6, 6, n=61)[0]

        discrrm = discrimination * the_sign * -1
        xx = np.linspace(-6, 6, 1001)
        yy = irt_evaluation(difficuly, discrrm.squeeze(), xx)
        yy = yy.prod(axis=0)
        yy *= np.exp(-np.square(xx) / 2) / np.sqrt(2*np.pi)
        expected = yy.sum() * 12 / 1001

        self.assertAlmostEqual(value[0], expected.sum(), places=3)

    def test_partial_integration_array(self):
        """Tests the integration quadrature function on array."""

        # Set seed for repeatability
        np.random.seed(121)

        discrimination = np.random.rand(5) + 0.5
        difficuly = np.linspace(-1.3, 1.3, 5)
        the_sign = (-1)**np.random.randint(low=0, high=2, size=(5, 1))

        quad_points = _get_quadrature_points(61, -6, 6)
        dataset = _compute_partial_integral(quad_points, difficuly, discrimination,
                                            the_sign)

        value = integrate.fixed_quad(lambda x: dataset, -6, 6, n=61)[0]

        discrrm = discrimination * the_sign.squeeze() * -1
        xx = np.linspace(-6, 6, 1001)
        yy = irt_evaluation(difficuly, discrrm, xx)
        yy = yy.prod(axis=0)
        yy *= np.exp(-np.square(xx) / 2) / np.sqrt(2*np.pi)
        expected = yy.sum() * 12 / 1001

        self.assertAlmostEqual(value[0], expected.sum(), places=3)



if __name__ == '__main__':
    unittest.main()

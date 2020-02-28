import unittest

import numpy as np

from girth import create_synthetic_irt_dichotomous
from girth import rauch_approx, onepl_approx, twopl_approx
from girth import rauch_separate, onepl_separate, twopl_separate
from girth import rauch_full, onepl_full, twopl_full


class TestMMLRaschMethods(unittest.TestCase):

    ### REGRESSION TESTS

    """Setup synthetic data."""
    def setUp(self):
        """Setup synthetic data for tests."""
        np.random.seed(3)
        difficulty = np.linspace(-1.5, 1.5, 5)
        discrimination = 1.12
        thetas = np.random.randn(600)
        syn_data = create_synthetic_irt_dichotomous(difficulty, discrimination,
                                                    thetas)
        self.data = syn_data
        self.discrimination = discrimination


    def test_rasch_regression_approximate(self):
        """Testing rasch approximation methods."""
        syn_data = self.data.copy()
        output = rauch_approx(syn_data, self.discrimination)
        expected_output = np.array([-1.27477108, -0.7771253 , -0.07800756,
                                    0.62717748,  1.41945661])

        np.testing.assert_allclose(expected_output, output)


    def test_rasch_regression_separate(self):
        """Testing rasch separate methods."""
        syn_data = self.data.copy()
        output = rauch_separate(syn_data, self.discrimination)
        expected_output = np.array([-1.32474665, -0.81460991, -0.08221992,
                                     0.65867573,  1.47055368])

        np.testing.assert_allclose(expected_output, output)


    def test_rasch_regression_full(self):
        """Testing rasch full methods."""
        syn_data = self.data.copy()
        output = rauch_full(syn_data, self.discrimination)
        expected_output = np.array([-1.3221573 , -0.81445556, -0.08485538,
                                    0.65457445,  1.4664268])

        np.testing.assert_allclose(expected_output, output)

    def test_rasch_close(self):
        """Testing rasch converging methods."""
        np.random.seed(333)
        difficulty = np.linspace(-1.25, 1.25, 5)
        discrimination = 0.87
        thetas = np.random.randn(2000)
        syn_data = create_synthetic_irt_dichotomous(difficulty, discrimination,
                                                    thetas)

        output = rauch_separate(syn_data, discrimination)
        np.testing.assert_array_almost_equal(difficulty, output, decimal=1)


if __name__ == '__main__':
    unittest.main()

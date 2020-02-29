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


class TestMMLOnePLMethods(unittest.TestCase):

    ### REGRESSION TESTS

    """Setup synthetic data."""
    def setUp(self):
        """Setup synthetic data for tests."""
        np.random.seed(873)
        difficulty = np.linspace(-1.5, 1.5, 5)
        discrimination = 1.843
        thetas = np.random.randn(600)
        syn_data = create_synthetic_irt_dichotomous(difficulty, discrimination,
                                                    thetas)
        self.data = syn_data
        self.discrimination = discrimination


    def test_onepl_regression_approximate(self):
        """Testing onepl approximation methods."""
        syn_data = self.data.copy()
        output = onepl_approx(syn_data)
        expected_output = np.array([-1.31442222, -0.6020974 , -0.03617573,
                                    0.72561239,  1.32432846])

        self.assertAlmostEqual(output[0], 1.9339261148822318)
        np.testing.assert_allclose(expected_output, output[1], rtol=1e-6)


    def test_onepl_regression_separate(self):
        """Testing onepl separate methods."""
        syn_data = self.data.copy()
        output = onepl_separate(syn_data)
        expected_output = np.array([-1.37650768, -0.64900385, -0.0393339 ,
                                    0.7791904 ,  1.38618721])

        self.assertAlmostEqual(output[0], 1.901703384)
        np.testing.assert_allclose(expected_output, output[1], rtol=1e-6)


    def test_onepl_regression_full(self):
        """Testing onepl full methods."""
        syn_data = self.data.copy()
        output = onepl_full(syn_data)
        expected_output = np.array([-1.37891489, -0.64731397, -0.03576614,
                                     0.78093483,  1.38451727])

        self.assertAlmostEqual(output[0], 1.9017531986)
        np.testing.assert_allclose(expected_output, output[1], rtol=1e-6)


    def test_oneple_close(self):
        """Testing rasch converging methods."""
        np.random.seed(843)
        difficulty = np.linspace(-1.25, 1.25, 10)
        discrimination = 0.87
        thetas = np.random.randn(1000)
        syn_data = create_synthetic_irt_dichotomous(difficulty, discrimination,
                                                    thetas)

        output = onepl_separate(syn_data)
        self.assertLess(np.abs(output[0] - discrimination).max(), 0.1)
        self.assertLess(np.abs(output[1] - difficulty).max(), 0.2)


class TestMMLTwoPLMethods(unittest.TestCase):

    ### REGRESSION TESTS

    """Setup synthetic data."""
    def setUp(self):
        """Setup synthetic data for tests."""
        np.random.seed(247)
        difficulty = np.linspace(-1.5, 1.5, 5)
        discrimination = np.random.rand(5) + 0.5
        thetas = np.random.randn(600)
        syn_data = create_synthetic_irt_dichotomous(difficulty, discrimination,
                                                    thetas)
        self.data = syn_data
        self.discrimination = discrimination


    def test_twopl_regression_approximate(self):
        """Testing onepl approximation methods."""
        syn_data = self.data.copy()
        output = twopl_approx(syn_data)

        expected_discrimination = np.array([0.99992967, 1.86001065, 1.36698371, 
                                            0.53176185, 0.90445937])
        expected_output = np.array([-1.26863512, -0.60677766, -0.07459365,  
                                    0.75672082,  1.62766257])

        np.testing.assert_allclose(expected_discrimination, output[0], rtol=1e-6)
        np.testing.assert_allclose(expected_output, output[1], rtol=1e-6)


    def test_twopl_regression_separate(self):
        """Testing onepl separate methods."""
        syn_data = self.data.copy()
        output = twopl_separate(syn_data)

        expected_discrimination = np.array([0.99973958, 1.86359282, 1.35543477, 
                                            0.52934393, 0.90911332])
        expected_output = np.array([-1.31515501, -0.64828062, -0.07980151,  
                                     0.77407585,  1.66774338])

        np.testing.assert_allclose(expected_discrimination, output[0], rtol=1e-6)
        np.testing.assert_allclose(expected_output, output[1], rtol=1e-6)


    def test_twopl_regression_full(self):
        """Testing onepl full methods."""
        syn_data = self.data.copy()
        output = twopl_full(syn_data)

        expected_discrimination = np.array([0.99976507, 1.86290853, 1.35540523, 
                                            0.5293102 , 0.91224903])
        expected_output = np.array([-1.31492755, -0.64812777, -0.08017468,  
                                     0.77399901,  1.66321445])

        np.testing.assert_allclose(expected_discrimination, output[0], rtol=1e-6)
        np.testing.assert_allclose(expected_output, output[1], rtol=1e-6)


    def test_twopl_close(self):
        """Testing rasch converging methods."""
        np.random.seed(43)
        difficulty = np.linspace(-1.25, 1.25, 10)
        discrimination = 0.5 + np.random.rand(10)
        thetas = np.random.randn(2000)
        syn_data = create_synthetic_irt_dichotomous(difficulty, discrimination,
                                                    thetas)

        output = twopl_separate(syn_data)
        self.assertLess(np.abs(output[0] - discrimination).mean(), 0.1)
        self.assertLess(np.abs(output[1] - difficulty).mean(), 0.1)        


if __name__ == '__main__':
    unittest.main()

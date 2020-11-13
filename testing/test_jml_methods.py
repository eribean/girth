import unittest

import numpy as np

from girth import create_synthetic_irt_dichotomous
from girth import create_synthetic_irt_polytomous
from girth import rasch_jml, onepl_jml, twopl_jml, grm_jml, pcm_jml
from girth.jml_methods import _jml_inequality


class TestJointMaximum(unittest.TestCase):
    """Tests the joint maximum functions in girth."""

    ## REGRESSION TESTS, JML isn't a greate estimator so there aren't any
    ## closeness tests.

    def test_joint_regression_rasch(self):
        """Testing joint maximum rasch model."""
        np.random.seed(91)
        difficuly = np.linspace(-1.5, 1.5, 5)
        discrimination = 1
        thetas = np.random.randn(600)
        syn_data = create_synthetic_irt_dichotomous(difficuly, discrimination,
                                                    thetas)

        output = rasch_jml(syn_data)
        expected_output = np.array([-1.61394095, -0.88286827,
                                     0.04830973,  0.77146166,  1.95939084])

        np.testing.assert_allclose(expected_output, output)


    def test_joint_regression_onepl(self):
        """Testing joint maximum onepl model."""
        np.random.seed(118)
        difficuly = np.linspace(-1.5, 1.5, 5)
        discrimination = 1.34
        thetas = np.random.randn(600)
        syn_data = create_synthetic_irt_dichotomous(difficuly, discrimination,
                                                    thetas)

        output = onepl_jml(syn_data)
        expected_output = np.array([-1.70585219, -1.03551581,
                                    -0.09329877,  0.92320069,  1.77063402])

        np.testing.assert_allclose(expected_output, output[1], 1e-3)
        self.assertAlmostEqual(1.5316042, output[0])


    def test_joint_regression_twopl(self):
        """Testing joint maximum twopl model."""
        np.random.seed(138)
        difficuly = np.linspace(-1.5, 1.5, 5)
        discrimination = 0.5 + np.random.rand(5)
        thetas = np.random.randn(600)
        syn_data = create_synthetic_irt_dichotomous(difficuly, discrimination,
                                                    thetas)

        output = twopl_jml(syn_data)

        # Expected Outputs
        alphas = np.array([0.29202081, 4., 0.91621924, 
                           4., 0.27536785])

        betas = np.array([-6., -0.39644246, -0.00862153,
                          0.3869096, 6.])

        np.testing.assert_allclose(alphas, output[0], rtol=1e-3)
        np.testing.assert_allclose(betas, output[1], rtol=1e-3)


class TestPolytomousJMLMethods(unittest.TestCase):
    """Test the joint maximum likelihood methods for polytomous data."""

    def test_jml_inequality(self):
        """Testing inequality constraint."""
        estimates = [1.0, 2.0, 1.0, 3.0]
        output = _jml_inequality(estimates)
        np.testing.assert_equal(output, [1.0, 1.0, -1.0, 2.0])

        estimates = [1.0, 2.0, 2.5, 2.5]
        output = _jml_inequality(estimates)
        np.testing.assert_equal(output, [1.0, 1.0, 0.5, 0.0])

    def test_graded_jml_regression(self):
        """Testing joint maximum grm model."""
        np.random.seed(1022)
        difficulty = np.sort(np.random.randn(5, 3), axis=1)
        discrimination = 0.5 + np.random.rand(5)
        thetas = np.random.randn(50) 

        syn_data = create_synthetic_irt_polytomous(difficulty, discrimination,
                                                   thetas)

        output = grm_jml(syn_data)

        # Expected Outputs (Basically a smoke test)
        alphas = np.array([4., 0.79558366, 0.25, 4., 1.84876057])

        betas = np.array([[-0.06567396,  0.00834646,  0.04343122],
                          [-1.72554323, -1.56602067, -0.87385678],
                          [-3.30647782,  1.86102210,      np.nan],
                          [-0.47923614,  0.31797999,  0.89676892],
                          [-0.67769087,  0.49737400,      np.nan]])

        np.testing.assert_allclose(alphas, output[0], rtol=1e-3)
        np.testing.assert_allclose(betas, output[1], rtol=1e-3)

    def test_partial_credit_jml_regression(self):
        """Testing joint maximum partial credit model."""
        np.random.seed(3)
        difficulty = np.random.randn(5, 3)
        discrimination = 0.5 + np.random.rand(5)
        thetas = np.random.randn(50) 

        syn_data = create_synthetic_irt_polytomous(difficulty, discrimination,
                                                   thetas, model='pcm')
       
        output = pcm_jml(syn_data)

        # Expected Outputs (Basically a smoke test)
        alphas = np.array([0.41826845, 4., 0.356021, 0.429537, 4.])
        betas = [[ 6.,         -1.83001497,  0.57618739],
                 [-1.34063642, -0.36478753,  0.3783893 ],
                 [ 3.32581876, -1.63421762,  0.93340153],
                 [ 2.57971531, -3.65201053,  1.80513887],
                 [ 0.55722782,  1.01035442,  0.74398655]]

        np.testing.assert_allclose(alphas, output[0], rtol=1e-3)
        np.testing.assert_allclose(betas, output[1], rtol=1e-3)
        

if __name__ == '__main__':
    unittest.main()

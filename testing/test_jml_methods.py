import unittest

import numpy as np

from girth import create_synthetic_irt_dichotomous
from girth import rasch_jml, onepl_jml, twopl_jml


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

        np.testing.assert_allclose(expected_output, output[1])
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
        alphas = np.array([1.13276949, 1.45060249, 1.3071953,
                           1.89628754, 1.2134972])

        betas = np.array([-1.97170647, -0.78656027,  0.00478144,
                          0.76341106,  1.69786748])

        np.testing.assert_allclose(alphas, output[0])
        np.testing.assert_allclose(betas, output[1], rtol=1e-6)


if __name__ == '__main__':
    unittest.main()

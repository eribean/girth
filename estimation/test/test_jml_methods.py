import unittest

import numpy as np

from girth.synthetic import (create_synthetic_irt_dichotomous,
    create_synthetic_irt_polytomous)
from girth import rasch_jml, onepl_jml, twopl_jml, grm_jml, pcm_jml
from girth.unidimensional.polytomous.grm_jml import _jml_inequality


class TestJointMaximum(unittest.TestCase):
    """Tests the joint maximum functions in girth."""

    ## REGRESSION TESTS, JML isn't a greate estimator so there aren't any
    ## closeness tests.

    def test_joint_regression_rasch(self):
        """Testing joint maximum rasch model."""
        rng = np.random.default_rng(77953183513228)
        difficuly = np.linspace(-1.5, 1.5, 5)
        discrimination = 1
        thetas = rng.standard_normal(600)
        syn_data = create_synthetic_irt_dichotomous(difficuly, discrimination,
                                                    thetas, seed=rng)

        output = rasch_jml(syn_data)['Difficulty']
        expected_output = np.array([-1.947492, -0.801259,  0.053089,  
                                    1.098387,  1.938692])

        np.testing.assert_allclose(expected_output, output, atol=1e-6)


    def test_joint_regression_onepl(self):
        """Testing joint maximum onepl model."""
        rng = np.random.default_rng(79465412498218)
        difficuly = np.linspace(-1.5, 1.5, 5)
        discrimination = 1.34
        thetas = rng.standard_normal(600)
        syn_data = create_synthetic_irt_dichotomous(difficuly, discrimination,
                                                    thetas, seed=rng)

        output = onepl_jml(syn_data)
        expected_output = np.array([-1.830353, -0.740702,  0.116478,  
                                     0.845492,  1.594863])

        np.testing.assert_allclose(expected_output, output['Difficulty'], 1e-3)
        self.assertAlmostEqual(1.553676352, output['Discrimination'])


    def test_joint_regression_twopl(self):
        """Testing joint maximum twopl model."""
        rng = np.random.default_rng(7432843158)
        difficuly = np.linspace(-1.5, 1.5, 5)
        discrimination = 0.5 + rng.uniform(0, 1, 5)
        thetas = rng.standard_normal(600)
        syn_data = create_synthetic_irt_dichotomous(difficuly, discrimination,
                                                    thetas, seed=rng)

        output = twopl_jml(syn_data)

        # Expected Outputs
        alphas = np.array([4., 0.25    , 0.25,
                           4., 1.279453])

        betas = np.array([-0.714757, -2.817987,  0.363446,  
                          0.27692 ,  1.219496])

        np.testing.assert_allclose(alphas, output['Discrimination'], rtol=1e-3)
        np.testing.assert_allclose(betas, output['Difficulty'], rtol=1e-3)


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
        rng = np.random.default_rng(794423824483)

        difficulty = np.sort(rng.standard_normal((5, 3)), axis=1)
        discrimination = 0.5 + rng.uniform(0, 1, 5)
        thetas = rng.standard_normal(50)

        syn_data = create_synthetic_irt_polytomous(difficulty, discrimination,
                                                   thetas, seed=rng)

        output = grm_jml(syn_data)

        # Expected Outputs (Basically a smoke test)
        alphas = np.array([4., 4., 0.25, 0.77995, 1.196129])

        betas = np.array([[-0.17792238,  0.24824511,  0.28731164],
                          [0.19936216,  0.23620655,  0.31278439],
                          [-2.55652279,  1.9247382,   3.01879088],
                          [-2.04153357, -0.66666598,  np.nan],
                          [-0.89040827, -0.14637313,  0.01385897]])

        np.testing.assert_allclose(alphas, output['Discrimination'], rtol=1e-3)
        np.testing.assert_allclose(betas, output['Difficulty'], rtol=1e-3)

    def test_partial_credit_jml_regression(self):
        """Testing joint maximum partial credit model."""
        rng = np.random.default_rng(794423824483)
        difficulty = rng.standard_normal((5, 3))
        discrimination = 0.5 + rng.uniform(0, 1, 5)
        thetas = rng.standard_normal(50)

        syn_data = create_synthetic_irt_polytomous(difficulty, discrimination,
                                                   thetas, model='pcm', seed=rng)
       
        output = pcm_jml(syn_data)

        # Expected Outputs (Basically a smoke test)
        alphas = np.array([0.825756, 4.      , 0.307507, 0.634488, 1.824519])
        betas = [[ 0.50703512, -0.54701056,  0.33005929],
                 [-0.35882868,  0.18148919,  0.45435675],
                 [-1.57347383,  2.00685349,  3.70221183],
                 [ 0.36730398, -2.06545728, -2.85900132],
                 [-0.36114401, -0.10069452, -0.46412294]]

        np.testing.assert_allclose(alphas, output['Discrimination'], rtol=1e-3)
        np.testing.assert_allclose(betas, output['Difficulty'], rtol=1e-3)
        

if __name__ == '__main__':
    unittest.main()

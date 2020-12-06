import unittest

import numpy as np
from scipy.stats import norm as gaussian

import girth.latent_ability_distribution as glad


class TestLatentDistribution(unittest.TestCase):

    def test_parameter_constraints(self):
        """Testing parameter identification constraints."""
        current_parameters = np.random.randn(10)
        sample_space = np.linspace(-4, 4, 10)
        delta_sample = sample_space[5] - sample_space[4]

        result = glad._parameter_constraints(current_parameters, sample_space)

        expected_results = np.zeros((3,))
        expected_results[0] = np.sum(current_parameters) - 1
        expected_results[1] = sample_space.dot(current_parameters)
        expected_results[2] = (np.square(sample_space).dot(current_parameters) - 
                               (3 - np.square(delta_sample)) / 3)

        # Check result
        np.testing.assert_allclose(result, expected_results, rtol=1e-5)

    def test_resample(self):
        """Testing cubic spline resample method."""
        cs_test = glad.CubicSplinePDF({'number_of_samples': 7, 
                                       'quadrature_bounds': (-4.5, 4.5)})

        with self.assertRaises(AssertionError):
            glad.resample(cs_test, 7)

        # Smoke Test
        cs_test.update_coefficients(np.random.rand(7))

        # Test Upsample
        output_up = glad.resample(cs_test, 9)

        self.assertEqual(output_up.number_of_samples, 9)
        result = glad._parameter_constraints(output_up.coefficients, 
                                             output_up.sample_space)
        np.testing.assert_allclose(result, np.zeros((3,)), 
                                   atol= 1e-5, rtol=1e-5)

        # Test Downsample
        output_down = glad.resample(cs_test, 5)

        self.assertEqual(output_down.number_of_samples, 5)
        result = glad._parameter_constraints(output_down.coefficients, 
                                             output_down.sample_space)
        np.testing.assert_allclose(result, np.zeros((3,)), 
                                   atol= 1e-5, rtol=1e-5)

    def test_cubic_spline_class(self):
        """Testing the CubicSpline class."""
        # Test the cubic spline static call
        y_test = glad.CubicSplinePDF.cubic_spline([-2, -1, 0., 1, 2])
        np.testing.assert_allclose(y_test, [0, 1/6, 2/3, 1/6, 0],
                                   atol=1e-6)

        cubic_spline = glad.CubicSplinePDF({'number_of_samples': 7, 
                                            'quadrature_bounds': (-4.5, 4.5)})

        # Test Update Coefficients
        new_coefficients = np.random.rand(9)
        with self.assertRaises(AssertionError):
           cubic_spline.update_coefficients(new_coefficients)
        
        new_coefficients = np.random.rand(7)
        cubic_spline.update_coefficients(new_coefficients)
        np.testing.assert_array_equal(new_coefficients, 
                                      cubic_spline.coefficients[2:-2])
        np.testing.assert_array_equal([0, 0], 
                                      cubic_spline.coefficients[:2])
        np.testing.assert_array_equal([0, 0], 
                                      cubic_spline.coefficients[-2:])

        # Testing continuous pdf creation
        new_coefficients = [0, 0, 0, 1, 0, 0, 0]
        cubic_spline.update_coefficients(new_coefficients)
        _, y = cubic_spline.continuous_pdf((-5, 5), 1001)

        x_loc = np.linspace(-5, 5, 1001) / cubic_spline.delta_sample
        y_loc = cubic_spline.cubic_spline(x_loc)

        np.testing.assert_allclose(y * cubic_spline.delta_sample, 
                                   y_loc, atol=1e-6)

        # Test filter matrix (smoke-test only)
        # 11 = number_of_samples + 4 (edges)
        x_loc = np.linspace(-2, 2, 10)
        filter_matrix = cubic_spline.filter_matrix(x_loc)
        self.assertTupleEqual(filter_matrix.shape, (10, 11))

        # Test Call
        y_values = cubic_spline(x_loc) * cubic_spline.delta_sample
        expected_y = glad.CubicSplinePDF.cubic_spline(x_loc / cubic_spline.delta_sample)
        np.testing.assert_allclose(y_values, expected_y,
                                   atol=1e-4, rtol=1e-4)

    def test_latent_pdf_class(self):
        """Tests the LatentPDF class."""
        # Testing no distribution estimation
        latent_pdf = glad.LatentPDF()
        cs_test = latent_pdf._init_cubic_spline()
        self.assertEqual(cs_test.number_of_samples, 3)

        # Check output
        dist_x_weights = latent_pdf(None, None)
        np.testing.assert_allclose(dist_x_weights, 
                                   latent_pdf.weights * 
                                   gaussian(0, 1).pdf(latent_pdf.quadrature_locations))


        ## Testing parameter estimation
        latent_pdf = glad.LatentPDF({"estimate_distribution": True,
                                     "number_of_samples": 7})
        cs_test = latent_pdf._init_cubic_spline()
        self.assertEqual(cs_test.number_of_samples, 7)

        # Check output at iteration 0
        dist_x_weights = latent_pdf(None, 0)
        np.testing.assert_allclose(dist_x_weights, 
                                   latent_pdf.weights * 
                                   gaussian(0, 1).pdf(latent_pdf.quadrature_locations))


        # Create a dummy distribution and integration
        np.random.seed(66871)
        coeffs = np.random.rand(9)
        cs_dist = glad.CubicSplinePDF({'number_of_samples': 9, 
                                       'quadrature_bounds': [-4.5, 4.5]})
        cs_dist.update_coefficients(coeffs)
        cs_fixed = glad.resample(cs_dist, 7)
        result = glad._parameter_constraints(cs_fixed.coefficients, 
                                             cs_fixed.sample_space)
        np.testing.assert_allclose(result, np.zeros((3,)), 
                                   atol= 1e-5, rtol=1e-5)

        # dummy integration method (smoke-test to make sure it runs)
        unweighted_integration = np.random.rand(1000,
                                    latent_pdf.quadrature_locations.size)

        dist_x_weights = latent_pdf(unweighted_integration, 1)
        np.testing.assert_allclose(latent_pdf.cubic_splines[0].coefficients[2:-2],
                                   [0., 0.057376, 0.264267, 0.313561, 0.350573, 0.014223, 0.], 
                                   atol=1e-4)
        # test aic / bic call
        aic, bic = latent_pdf.compute_metrics(unweighted_integration, 
                                              dist_x_weights, 4)
        self.assertAlmostEqual(bic - aic, 4 * (np.log(1000) - 2))


if __name__ == '__main__':
    unittest.main()

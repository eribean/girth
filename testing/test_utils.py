import unittest # pylint: disable=cyclic-import


import numpy as np
from scipy.special import roots_legendre
from scipy import integrate

from girth import (create_synthetic_irt_dichotomous,
                   irt_evaluation, trim_response_set_and_counts,
                   convert_responses_to_kernel_sign)
from girth import condition_polytomous_response, get_true_false_counts
from girth.utils import _get_quadrature_points, _compute_partial_integral
from girth.polytomous_utils import (_graded_partial_integral, _solve_for_constants, 
                                    _solve_integral_equations, _credit_partial_integral)


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

    def test_trim_response_set(self):
        """Testing trim of all yes/no values."""
        np.random.seed(439)
        dataset = np.random.rand(10, 300)
        counts = np.random.rand(300)

        # Pass through
        new_set, new_counts = trim_response_set_and_counts(dataset, counts)
        np.testing.assert_array_equal(dataset, new_set)
        np.testing.assert_array_equal(counts, new_counts)

        # Make first column zeros
        dataset[:, 0] = 0
        new_set, new_counts = trim_response_set_and_counts(dataset, counts)

        np.testing.assert_array_equal(dataset[:, 1:], new_set)
        np.testing.assert_array_equal(counts[1:], new_counts)

        # Make last column all 1
        dataset[:, -1] = 1
        new_set, new_counts = trim_response_set_and_counts(dataset, counts)

        np.testing.assert_array_equal(dataset[:, 1:-1], new_set)
        np.testing.assert_array_equal(counts[1:-1], new_counts)

        # Test when array contains nans
        mask = np.random.rand(*dataset.shape) < 0.1
        dataset[mask] = np.nan

        # There are responses with zero variance
        locations = np.where(np.nanstd(dataset, axis=0) == 0)
        self.assertTrue(locations[0].size > 0)
        
        new_set, new_counts = trim_response_set_and_counts(dataset, counts)
        locations = np.where(np.nanstd(new_set, axis=0) == 0)
        self.assertTrue(locations[0].size == 0)        

    def test_get_true_false_counts(self):
        """Testing the counting of true and false."""
        test_array = np.array([[1., 0., 4.3, 2.1, np.nan],
                               [np.nan, np.nan, -1, 3.2, 1.2],
                               [0.0, 1.0, 1.0, 0.0, 1.0]])

        n_false, n_true = get_true_false_counts(test_array)
        
        expected_true = [1, 0, 3]
        expected_false = [1, 0, 2]

        np.testing.assert_array_equal(expected_true, n_true)
        np.testing.assert_array_equal(expected_false, n_false)

    def test_convert_response_to_kernel_sign(self):
        """Testing conversion of response to kernel sign."""
        test_array = np.array([[1., 0., 4.3, 2.1, np.nan],
                               [np.nan, np.nan, -1, 3.2, 1.2],
                               [0.0, 1.0, 1.0, 0.0, 1.0]])

        expected_output = np.array([[-1, 1, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [1, -1, -1, 1, -1]])

        result = convert_responses_to_kernel_sign(test_array)                                   

        np.testing.assert_array_equal(expected_output, result)



class TestPolytomousUtilities(unittest.TestCase):
    """Tests the polytomous utilities"""

    def test_condition_polytomous_response(self):
        """Testing polytomous response conditioning."""
        dataset = np.random.randint(1, 6, (15, 100))
        dataset[7] = np.random.randint(1, 3, (1, 100))
        output = condition_polytomous_response(dataset, trim_ends=False)

        self.assertTupleEqual(output[0].shape, dataset.shape)
        row_max_start = np.cumsum(dataset.max(axis=1))
        row_max_end = row_max_start.copy() - 1
        row_max_start = np.roll(row_max_start, 1)
        row_max_start[0] = 0

        for ndx in range(dataset.shape[0]):
            otpts = np.unique(dataset[ndx])
            otpts2 = np.unique(output[0][ndx])
            self.assertEqual(otpts.size, output[1][ndx])
            self.assertTrue(otpts2.min() == row_max_start[ndx])
            self.assertTrue(otpts2.max() == row_max_end[ndx])

        # Trim First Column but not last
        dataset[:, 0] = 1
        output = condition_polytomous_response(dataset, trim_ends=True)
        self.assertTupleEqual(output[0].shape, (dataset.shape[0], dataset.shape[1]-1))
        self.assertTrue(output[0].std(axis=0)[0] != 0)
 
        # Trim First/Last Column but not last
        dataset[:, -1] = 1
        output = condition_polytomous_response(dataset, trim_ends=True)
        self.assertTupleEqual(output[0].shape, (dataset.shape[0], dataset.shape[1]-2))
        self.assertTrue(output[0].std(axis=0)[0] != 0)
        self.assertTrue(output[0].std(axis=0)[-1] != 0)


    def test_solve_for_constants(self):
        """Testing solving for boundary constants."""
        np.random.seed(73)
        dataset = np.random.randint(0, 4, (1, 100))
        _, counts = np.unique(dataset, return_counts=True)

        output = _solve_for_constants(dataset)

        #Compare to hand calculations
        b11 = counts[0] + counts[1]
        b12 = -counts[0]
        b13 = 0
        b21 = -counts[2]
        b22 = counts[1] + counts[2]
        b23 = -counts[1]
        b31 = 0
        b32 = -counts[3]
        b33 = counts[2] + counts[3]
        b_matrix = np.array([[b11, b12, b13],
                             [b21, b22, b23],
                             [b31, b32, b33]])
        hand_calcs = np.linalg.inv(b_matrix) @ np.array([counts[1], 0, 0]).T
        np.testing.assert_array_equal(output, hand_calcs)


    def test_integral_equations(self):
        """Tests solving for integral given a ratio."""
        np.random.seed(786)
        theta = np.random.randn(50000)
        discrimination = 1.43
        difficulty = np.array([-.4, .1, .5])

        # Compare against dichotomous data
        syn_data = create_synthetic_irt_dichotomous(difficulty, discrimination, theta)
        n0 = np.count_nonzero(~syn_data, axis=1)
        n1 = np.count_nonzero(syn_data, axis=1)
        ratio = n1 / (n1 + n0)

        theta = _get_quadrature_points(61, -5, 5)
        distribution = np.exp(-np.square(theta) / 2) / np.sqrt(2 * np.pi)
        results = _solve_integral_equations(discrimination, ratio, distribution, theta)
        np.testing.assert_array_almost_equal(results, difficulty, decimal=2)


    def test_graded_partial_integral(self):
        """Testing the partial integral in the graded model."""
        theta = _get_quadrature_points(61, -5, 5)
        distribution = np.exp(-np.square(theta) / 2) / np.sqrt(2 * np.pi)
        responses = np.random.randint(0, 3, (10, 100))
        betas = np.array([-10000, -.3, 0.1, 1.2])
        betas_roll = np.roll(betas, -1)
        betas_roll[-1] = 10000

        output = _graded_partial_integral(theta, betas, betas_roll,
                                          1.0, responses, distribution)

        # Compare to hand calculations
        hand_calc = list()
        for ndx in range(responses.shape[1]):
            left_betas = betas[responses[:, ndx]]
            right_betas = betas_roll[responses[:, ndx]]
            probability = (1.0 / (1.0 + np.exp(left_betas[:, None] - theta[None, :])) - 
                           1.0 / (1.0 + np.exp(right_betas[:, None] - theta[None, :])))
            hand_calc.append(probability.prod(0))

        hand_calc = np.asarray(hand_calc)
        hand_calc *= distribution

        np.testing.assert_array_equal(hand_calc, output)


    def test_credit_partial_integration(self):
        """Testing the partial integral in the graded model."""
        theta = _get_quadrature_points(61, -5, 5)
        response_set = np.array([0, 1, 2, 2, 1, 0, 3, 1, 3, 2, 2, 2])
        betas = np.array([0, -0.4, 0.94, -.37])
        discrimination = 1.42

        # Hand calculations
        offsets = np.cumsum(betas)[1:]
        first_pos = np.ones_like(theta)
        second_pos = np.exp(discrimination * (theta - offsets[0]))
        third_pos = np.exp(2*discrimination * (theta - offsets[1]/2))
        last_pos = np.exp(3*discrimination * (theta - offsets[2]/3))
        norm_term = first_pos + second_pos + third_pos + last_pos

        probability_values = [first_pos / norm_term, second_pos / norm_term,
                              third_pos / norm_term, last_pos / norm_term]
        expected = np.zeros((response_set.size, theta.size))
        for ndx, response in enumerate(response_set):
            expected[ndx] = probability_values[response]

        result = _credit_partial_integral(theta, betas, discrimination,
                                          response_set)

        np.testing.assert_array_almost_equal(result, expected)
        

if __name__ == '__main__':
    unittest.main()

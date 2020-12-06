import unittest  # pylint: disable=cyclic-import


import numpy as np
from scipy.special import roots_legendre
from scipy import integrate, stats

from girth import (create_synthetic_irt_dichotomous,
                   irt_evaluation, trim_response_set_and_counts,
                   convert_responses_to_kernel_sign,
                   validate_estimation_options, condition_polytomous_response,
                   get_true_false_counts, mml_approx)
from girth.utils import _get_quadrature_points, default_options, create_beta_LUT
from girth.numba_functions import _compute_partial_integral
from girth.polytomous_utils import (_graded_partial_integral, _solve_for_constants,
                                    _solve_integral_equations, _credit_partial_integral,
                                    _unfold_partial_integral)
from girth.synthetic import _unfold_func


class TestUtilitiesMethods(unittest.TestCase):
    """Tests the utilities functions in girth."""

    def test_irt_evaluation_single_discrimination(self):
        """Testing the IRT evaluation method when discrimination is scalar."""
        difficuly = np.array([-1, 1])
        theta = np.array([1., 2.])
        discrimination = 4.0

        # Expected output
        expected_output = 1.0 / \
            (1.0 + np.exp(discrimination * (difficuly[:, None] - theta)))
        output = irt_evaluation(difficuly, discrimination, theta)

        np.testing.assert_allclose(output, expected_output)

    def test_irt_evaluation_array_discrimination(self):
        """Testing the IRT evaluation method when discrimination is array."""
        difficuly = np.array([-1, 1])
        theta = np.array([1., 2.])
        discrimination = np.array([1.7, 2.3])

        # Expected output
        expected_output = 1.0 / \
            (1.0 + np.exp(discrimination[:, None]
                          * (difficuly[:, None] - theta)))
        output = irt_evaluation(difficuly, discrimination, theta)

        np.testing.assert_allclose(output, expected_output)

    def test_quadrature_points(self):
        """Testing the creation of quadrtature points"""
        n_points = 11

        # A smoke test to make sure it's running properly
        quad_points, weights = _get_quadrature_points(n_points, -1, 1)

        x, w = roots_legendre(n_points)

        np.testing.assert_allclose(x, quad_points)
        np.testing.assert_allclose(w, weights)

    def test_partial_integration_single(self):
        """Tests the integration quadrature function."""

        # Set seed for repeatability
        np.random.seed(154)

        discrimination = 1.32
        difficulty = .67
        the_sign = (-1)**np.random.randint(low=0, high=2, size=(1, 10))

        quad_points, _ = _get_quadrature_points(61, -6, 6)
        the_output = np.zeros((the_sign.shape[1], quad_points.size))        
        value = _compute_partial_integral(quad_points, difficulty, 
                                          discrimination, the_sign[0],
                                          the_output)

        discrrm = discrimination * the_sign
        expected = 1.0 / (1 + np.exp(np.outer(discrrm, (quad_points - difficulty))))
        np.testing.assert_allclose(value, expected)

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

    def test_mml_approx(self):
        """Testing approximation of rasch model with normal distribution."""
        dataset = np.random.randint(0, 2, (3, 100))
        discrimination = 2.31
        result = mml_approx(dataset, discrimination)

        n_no = np.count_nonzero(dataset == 0, axis=1)
        n_yes = np.count_nonzero(dataset, axis=1)

        scalar = np.log(n_no / n_yes)

        expected = (np.sqrt(1 + discrimination**2 / 3) *
                    scalar / discrimination)
        result2 = mml_approx(dataset, discrimination, scalar)

        np.testing.assert_array_almost_equal(expected, result)
        np.testing.assert_array_almost_equal(expected, result2)


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
        self.assertTupleEqual(
            output[0].shape, (dataset.shape[0], dataset.shape[1]-1))
        self.assertTrue(output[0].std(axis=0)[0] != 0)

        # Trim First/Last Column but not last
        dataset[:, -1] = 1
        output = condition_polytomous_response(dataset, trim_ends=True)
        self.assertTupleEqual(
            output[0].shape, (dataset.shape[0], dataset.shape[1]-2))
        self.assertTrue(output[0].std(axis=0)[0] != 0)
        self.assertTrue(output[0].std(axis=0)[-1] != 0)

    def test_solve_for_constants(self):
        """Testing solving for boundary constants."""
        np.random.seed(73)
        dataset = np.random.randint(0, 4, (1, 100))
        _, counts = np.unique(dataset, return_counts=True)

        output = _solve_for_constants(dataset)

        # Compare to hand calculations
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
        syn_data = create_synthetic_irt_dichotomous(
            difficulty, discrimination, theta)
        n0 = np.count_nonzero(~syn_data, axis=1)
        n1 = np.count_nonzero(syn_data, axis=1)
        ratio = n1 / (n1 + n0)

        theta, weights = _get_quadrature_points(61, -5, 5)
        distribution = np.exp(-np.square(theta) / 2) / np.sqrt(2 * np.pi)
        results = _solve_integral_equations(
            discrimination, ratio, distribution * weights, theta, None)
        np.testing.assert_array_almost_equal(results, difficulty, decimal=2)

    def test_graded_partial_integral(self):
        """Testing the partial integral in the graded model."""
        theta, _ = _get_quadrature_points(61, -5, 5)
        responses = np.random.randint(0, 3, (10, 100))
        betas = np.array([-10000, -.3, 0.1, 1.2])
        betas_roll = np.roll(betas, -1)
        betas_roll[-1] = 10000

        output = np.ones((responses.shape[1], theta.size))
        for ndx in range(responses.shape[0]):
            output *= _graded_partial_integral(theta, betas, betas_roll,
                                               np.array([1,]), responses[ndx])

        # Compare to hand calculations
        hand_calc = list()
        for ndx in range(responses.shape[1]):
            left_betas = betas[responses[:, ndx]]
            right_betas = betas_roll[responses[:, ndx]]
            probability = (1.0 / (1.0 + np.exp(left_betas[:, None] - theta[None, :])) -
                           1.0 / (1.0 + np.exp(right_betas[:, None] - theta[None, :])))
            hand_calc.append(probability.prod(0))

        hand_calc = np.asarray(hand_calc)

        np.testing.assert_array_equal(hand_calc, output)

    def test_credit_partial_integration(self):
        """Testing the partial integral in the graded model."""
        theta, _ = _get_quadrature_points(61, -5, 5)
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

    def test_unfold_partial_integration(self):
        """Testing the unfolding integral."""
        theta, _ = _get_quadrature_points(61, -5, 5)
        response_set = np.array([0, 1, 2, 2, 1, 0, 3, 1, 3, 2, 2, 2])
        betas = np.array([-1.3, -.4, 0.2])
        delta = -0.76
        # (2N -1) / 2 - n
        folding = 3.5 - np.arange(4)
        discrimination = 1.42

        # Convert to PCM thresholds
        full = np.concatenate((betas, [0], -betas[::-1]))
        full += delta
        scratch = np.zeros((full.size + 1, theta.size))
        _unfold_func(full, discrimination, theta, scratch)

        expected = np.zeros((response_set.size, theta.size))
        for ndx, response in enumerate(response_set):
            expected[ndx] = scratch[response]

        result = _unfold_partial_integral(theta, delta, betas,
                                          discrimination, folding,
                                          response_set)
        np.testing.assert_array_almost_equal(result, expected)

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


class TestOptions(unittest.TestCase):
    """Tests default options."""

    def test_default_creation(self):
        """Testing the default options."""
        output = default_options()
        x = np.linspace(-3, 3, 101)
        expected = stats.norm(0, 1).pdf(x)

        self.assertEqual(output['max_iteration'], 25)
        self.assertEqual(output['quadrature_n'], 61)
        self.assertEqual(output['use_LUT'], True)
        self.assertEqual(output['estimate_distribution'], False)
        self.assertEqual(output['number_of_samples'], 9)
        self.assertTupleEqual(output['quadrature_bounds'], (-5, 5))
        result = output['distribution'](x)
        np.testing.assert_array_almost_equal(expected,
                                             result, decimal=6)
        self.assertEqual(len(output.keys()), 7)

    def test_no_input(self):
        """Testing validation for No input."""
        result = validate_estimation_options(None)
        x = np.linspace(-3, 3, 101)
        expected = stats.norm(0, 1).pdf(x)

        self.assertEqual(len(result.keys()), 7)
        self.assertEqual(result['max_iteration'], 25)
        self.assertEqual(result['quadrature_n'], 61)
        self.assertEqual(result['use_LUT'], True)
        self.assertEqual(result['estimate_distribution'], False)
        self.assertEqual(result['number_of_samples'], 9)
        self.assertTupleEqual(result['quadrature_bounds'], (-5, 5))
        result = result['distribution'](x)
        np.testing.assert_array_almost_equal(expected,
                                             result, decimal=6)

    def test_warnings(self):
        """Testing validation when inputs are bad."""
        test = {'Bad Key': "Come at me Bro"}
        with self.assertRaises(KeyError):
            validate_estimation_options(test)

        test = [21.0]
        with self.assertRaises(AssertionError):
            validate_estimation_options(test)

        test = {'max_iteration': 12.0}
        with self.assertRaises(AssertionError):
            validate_estimation_options(test)

        test = {'max_iteration': -2}
        with self.assertRaises(AssertionError):
            validate_estimation_options(test)

        test = {'distribution': stats.norm(0, 1)}
        with self.assertRaises(AssertionError):
            validate_estimation_options(test)

        test = {'quadrature_bounds': 4.3}
        with self.assertRaises(AssertionError):
            validate_estimation_options(test)

        test = {'quadrature_bounds': (4, -3)}
        with self.assertRaises(AssertionError):
            validate_estimation_options(test)

        test = {'quadrature_n': 12.2}
        with self.assertRaises(AssertionError):
            validate_estimation_options(test)

        test = {'quadrature_bounds': 2}
        with self.assertRaises(AssertionError):
            validate_estimation_options(test)

        test = {'use_LUT': 1}
        with self.assertRaises(AssertionError):
            validate_estimation_options(test)

        test = {'estimate_distribution': 1}
        with self.assertRaises(AssertionError):
            validate_estimation_options(test)

        test = {'number_of_samples':3}
        with self.assertRaises(AssertionError):
            validate_estimation_options(test)

    def test_population_update(self):
        """Testing update to options."""
        x = np.linspace(-3, 3, 101)
        expected = stats.norm(2, 1).pdf(x)

        new_parameters = {'distribution': stats.norm(2, 1).pdf}
        output = validate_estimation_options(new_parameters)
        self.assertEqual(len(output.keys()), 7)
        result = output['distribution'](x)
        np.testing.assert_array_almost_equal(expected,
                                             result, decimal=6)

        new_parameters = {'quadrature_bounds': (-7, -5),
                          'quadrature_n': 13,
                          'estimate_distribution': True}
        output = validate_estimation_options(new_parameters)
        self.assertEqual(output['max_iteration'], 25)
        self.assertEqual(output['quadrature_n'], 13)
        self.assertEqual(output['estimate_distribution'], True)

        self.assertTupleEqual(output['quadrature_bounds'], (-7, -5))
        self.assertEqual(len(output.keys()), 7)

        new_parameters = {'max_iteration': 43}
        output = validate_estimation_options(new_parameters)
        self.assertEqual(output['max_iteration'], 43)
        self.assertEqual(len(output.keys()), 7)

        new_parameters = {'use_LUT': False,
                          'number_of_samples': 142}
        output = validate_estimation_options(new_parameters)
        self.assertEqual(output['use_LUT'], False)
        self.assertEqual(output['number_of_samples'], 142)        
        self.assertEqual(len(output.keys()), 7)        


if __name__ == '__main__':
    unittest.main()

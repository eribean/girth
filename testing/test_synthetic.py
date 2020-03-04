import unittest

import numpy as np

from girth import create_synthetic_irt_dichotomous
from girth import create_synthetic_mirt_dichotomous
from girth import create_synthetic_irt_polytomous
from girth import create_correlated_abilities

from girth.synthetic import (_my_digitize, _credit_func, 
                             _graded_func)

class TestSynthetic(unittest.TestCase):
    """Testing the creation of synthetic irt function."""

    def test_synthetic_irt_creation(self):
        """Testing the creation of synthetic data."""
        seed = 31

        # Regression test
        expected = np.array([[False, False, False, False,  True,  True],
                             [False, False,  True,  True,  True,  True],
                             [False, False, False,  True,  True,  True]])

        value = create_synthetic_irt_dichotomous(np.array([1.2, -0.2, 1.3]),
                                                 1.31, np.linspace(-6, 6, 6),
                                                 seed=seed)

        np.testing.assert_array_equal(expected, value)


    def test_synthetic_mirt_creation(self):
        """Testing the creation of synthetic data."""
        seed = 164
        np.random.seed(seed-1)
        # Regression test
        expected = np.array([[False, False, False, False, False, False],
                             [ True, False, False,  True, False,  True],
                             [ True,  True, False, False, False, False],
                             [ True,  True,  True,  True, False,  True],
                             [ True,  True,  True,  True,  True,  True]])

        n_factors = 3
        n_items = 5
        n_people = 6
        discrimination = np.random.randn(n_items, n_factors)
        difficulty = np.linspace(-5, 5, n_items)
        thetas = np.random.randn(n_factors, n_people)
        value = create_synthetic_mirt_dichotomous(difficulty, discrimination,
                                                  thetas, seed)

        np.testing.assert_array_equal(expected, value)


    def test_synthetic_mirt_creation_single(self):
        """Testing the creation of synthetic data, common discrimination."""
        seed = 546
        np.random.seed(seed-1)
        # Regression test
        expected = np.array([[False, False, False, False, False, False],
                             [False, False, False, False, False, False],
                             [False, False, False,  True,  True,  True],
                             [False,  True,  True,  True,  True,  True],
                             [ True,  True,  True,  True,  True,  True]])

        n_factors = 3
        n_items = 5
        n_people = 6
        discrimination = np.random.randn(1, n_factors)
        difficulty = np.linspace(-5, 5, n_items)
        thetas = np.random.randn(n_factors, n_people)
        value = create_synthetic_mirt_dichotomous(difficulty, discrimination,
                                                  thetas, seed)

        np.testing.assert_array_equal(expected, value)


    def test_correlated_abilities(self):
        """Testing the creation of correlated abilities."""
        np.random.seed(120)
        n_participants = 1000
        rho = 0.73
        correlation_matrix = np.array([[1, rho], [rho, 1]])

        output = create_correlated_abilities(correlation_matrix, n_participants)
        output_corr = np.corrcoef(output)

        np.testing.assert_almost_equal(output_corr, correlation_matrix, decimal=1)


class TestPolytomousSynthetic(unittest.TestCase):
    """Testing the creation of synthetic polytomous irt function."""

    def test_my_digititize(self):
        """Testing local digitize function."""
        test_candidate = np.zeros((3,))

        # Zero position
        test_candidate[:] = [0.25, .3, .6]
        output = _my_digitize(test_candidate)

        self.assertEquals(output, 0)

        # One position
        test_candidate[:] = [0.3, .25, .6]
        output = _my_digitize(test_candidate)

        self.assertEquals(output, 1)

        # One position boundary
        test_candidate[:] = [0.3, .25, .3]
        output = _my_digitize(test_candidate)

        self.assertEquals(output, 1)

        # Two position 
        test_candidate[:] = [0.35, .25, .3]
        output = _my_digitize(test_candidate)

        self.assertEquals(output, 2)


    def test_graded_function(self):
        """Testing the graded response model computation"""
        # Create basic data
        difficulties = np.array([-2.3, .3, 1.2])
        discrimination = 0.78
        thetas = np.linspace(-3, 3, 100)

        # Initialize output variable and call data
        output = np.zeros((difficulties.size + 1, thetas.size))
        _graded_func(difficulties, discrimination, thetas, output)

        # Compare to hand computations
        first_position = 1.0 / (1.0 + np.exp(discrimination * (thetas - difficulties[0])))
        second_position = 1.0 / (1.0 + np.exp(discrimination * (thetas - difficulties[1])))
        last_position = 1.0 / (1.0 + np.exp(discrimination * (thetas - difficulties[2])))

        np.testing.assert_array_almost_equal(output[0], first_position)
        np.testing.assert_array_almost_equal(output[1], second_position - first_position)
        np.testing.assert_array_almost_equal(output[2], last_position - second_position)
        np.testing.assert_array_almost_equal(output[3], 1 - last_position)
        

    def test_credit_function(self):
        """Testing the partial credit computation"""
        # Create basic data
        difficulties = np.array([-0.67, .24, .84])
        discrimination = 1.24
        thetas = np.linspace(-3, 3, 100)

        # Initialize output variable and call data
        output = np.zeros((difficulties.size + 1, thetas.size))
        _credit_func(difficulties, discrimination, thetas, output)

        # Compare to hand computations
        first_position = 1.0 
        second_position = np.exp(discrimination * (thetas - difficulties[0]))
        third_position = np.exp(discrimination * (thetas - difficulties[1])) * second_position
        last_position = np.exp(discrimination * (thetas - difficulties[2])) * third_position

        normalizing = first_position + second_position + third_position + last_position

        np.testing.assert_array_almost_equal(output[0], first_position / normalizing)
        np.testing.assert_array_almost_equal(output[1], second_position / normalizing)
        np.testing.assert_array_almost_equal(output[2], third_position / normalizing)
        np.testing.assert_array_almost_equal(output[3], last_position / normalizing)


    def test_create_polytomous_data_fail(self):
        """Testing synthetic polytomous function fails with 1 level"""
        difficulty = np.array([[1.0]])

        with self.assertRaises(AssertionError):
            create_synthetic_irt_polytomous(difficulty, difficulty, difficulty)

        with self.assertRaises(KeyError):
            create_synthetic_irt_polytomous([1, 2, 3], difficulty, 
                                            difficulty, model='boom')
        


    def test_check_polytomous_discrimination(self):
        """Smoke tests if a single value for discrimination passes"""
        difficulty = np.array([[1.0, 2., 3.]])
        discrimination = 3

        # Simple Smoke tests
        create_synthetic_irt_polytomous(difficulty, discrimination, difficulty)
        create_synthetic_irt_polytomous(difficulty, np.array([discrimination]), 
                                        difficulty)


    def test_check_polytomous_regression(self):
        """Regression testing graded and credit polytomous functions"""
        seed = 876
        np.random.seed(seed)
        difficulty = np.random.randn(5, 4)
        discrimination = 1.23
        thetas = np.random.randn(8)

        # Simple Smoke tests
        poly_data_graded = create_synthetic_irt_polytomous(difficulty, 
                                                           discrimination,
                                                           thetas,
                                                           model='grm',
                                                           seed=seed)

        poly_data_credit = create_synthetic_irt_polytomous(difficulty, 
                                                           discrimination,
                                                           thetas,
                                                           model='pcm',
                                                           seed=seed)

        expected_graded = np.array([[5, 1, 1, 5, 1, 1, 5, 5],
                                    [5, 5, 1, 5, 5, 1, 3, 1],
                                    [5, 5, 1, 5, 1, 1, 5, 1],
                                    [5, 5, 5, 5, 5, 1, 5, 1],
                                    [5, 5, 4, 5, 4, 4, 5, 2]])

        expected_partial = np.array([[5, 5, 2, 5, 2, 1, 5, 3],
                                     [5, 5, 1, 5, 4, 1, 4, 1],
                                     [5, 5, 1, 3, 1, 1, 5, 1],
                                     [5, 5, 1, 5, 4, 1, 5, 1],
                                     [5, 5, 2, 5, 4, 2, 5, 2]])

        np.testing.assert_array_equal(poly_data_graded, expected_graded)
        np.testing.assert_array_equal(poly_data_credit, expected_partial)

if __name__ == '__main__':
    unittest.main()

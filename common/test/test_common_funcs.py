import unittest

import numpy as np

from girth.utils import INVALID_RESPONSE
from girth.common import (entropy, hypersphere, procrustes_rotation, 
                          reverse_score, cronbach_alpha)


class TestCommonFunctions(unittest.TestCase):
    """Tests fixture for the common functions."""

    def test_entropy(self):
        """Testing entropy calculations."""

        # One value in each column should have zero entropy
        dataset = np.eye((10))
        result = entropy(dataset, axis=1)
        self.assertAlmostEqual(result, 0.0, delta=1e-4)

        result = entropy(dataset, axis=0)
        self.assertAlmostEqual(result, 0.0, delta=1e-4)

        # Constant data
        dataset = np.ones((10, 5))
        result = entropy(dataset, axis=1)
        expected = np.log(5) * 10
        self.assertAlmostEqual(result, expected, delta=1e-4)

        dataset = np.ones((10, 5))
        result = entropy(dataset, axis=0)
        expected = np.log(10) * 5
        self.assertAlmostEqual(result, expected, delta=1e-4)        

    def test_hypersphere(self):
        """Testing hypersphere calculations."""
        # Testing random angles -> cartesian -> angles
        rng = np.random.default_rng(34321)
        random_angles = rng.uniform(0, np.pi, size=9)
        cartesian = hypersphere.hyperspherical_vector(random_angles)
        reconstructed_angles = hypersphere.hyperspherical_angles(cartesian)

        np.testing.assert_allclose(random_angles, reconstructed_angles)

        # Testing cartesian -> angles -> cartesian
        cartesian = rng.uniform(-1, 1, 10)
        cartesian /= np.linalg.norm(cartesian)

        angles = hypersphere.hyperspherical_angles(cartesian)
        reconstructed_cartesian = hypersphere.hyperspherical_vector(angles)

        np.testing.assert_allclose(cartesian, reconstructed_cartesian)

    def test_procustes(self):
        """Testing procustes rotation."""
        rng = np.random.default_rng(84913)

        dataset = rng.standard_normal(size=(30, 4))
        rotation_matrix = rng.uniform(-2, 2, size=(40, 4))
        rotation_matrix = rotation_matrix.T @ rotation_matrix
        rotation_matrix, _, _ = np.linalg.svd(rotation_matrix)
        
        rotated_dataset = dataset @ rotation_matrix.T

        recovered_rotation = procrustes_rotation(dataset, 
                                                 rotated_dataset)
        np.testing.assert_allclose(rotation_matrix, 
                                   recovered_rotation)

    def test_reverse_score(self):
        """Testing reverse score function."""

        dataset = np.array([
            [1, 2, 3, 4, 5, INVALID_RESPONSE]
        ])

        result = reverse_score(dataset, np.array([True]), 5)

        np.testing.assert_equal(result,[[5, 4, 3, 2, 1, INVALID_RESPONSE]])

        result = reverse_score(dataset, np.array([False]), 5)

        np.testing.assert_equal(result, dataset)

    def test_cronbach_alpha(self):
        """Testing cronbach alpha."""
        rng = np.random.default_rng(435616365426513465612613263218)

        dataset = rng.integers(1, 5, (10, 1000))
        
        # Regression Tests
        result = cronbach_alpha(dataset)
        self.assertAlmostEqual(result, .0901, places=4)

        mask = rng.uniform(0, 1, (10, 1000)) < .05
        dataset[mask] = INVALID_RESPONSE
        
        result = cronbach_alpha(dataset)
        self.assertAlmostEqual(result, .1230, places=4)

    def test_cronbach_alpha2(self):
        """Testing cronbach alpha second test."""
        rng = np.random.default_rng(63352413128574135234)

        dataset = rng.integers(1, 5, (9, 1250))
        
        # Regression Tests
        result = cronbach_alpha(dataset)
        self.assertAlmostEqual(result, .0526, places=4)

        mask = rng.uniform(0, 1, (9, 1250)) < .05
        dataset[mask] = INVALID_RESPONSE
        
        result = cronbach_alpha(dataset)
        self.assertAlmostEqual(result, .1184, places=4)        

if __name__ == "__main__":
    unittest.main()
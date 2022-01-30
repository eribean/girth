import unittest

import numpy as np

from girth.factoranalysis import sparsify_loadings


class TestRotation(unittest.TestCase):
    """Test fixture for testing Rotation."""

    def test_rotation_orthogonal(self):
        """Testing recovery of orthogonal rotation."""
        rotation_angle = np.radians(37.4)
        cos_theta, sin_theta = np.cos(rotation_angle), np.sin(rotation_angle)
        rotation = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

        # Uncorrelated Loadings
        real_loadings = np.array([1, 0] * 5 + [0, 1]* 5).reshape(10, 2)
        rotated_loadings = real_loadings @ rotation

        loadings, bases = sparsify_loadings(rotated_loadings, seed=11354684,
                                            orthogonal=True)

        np.testing.assert_allclose(loadings, real_loadings, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(bases, rotation, rtol=1e-4, atol=1e-4)

    
    def test_rotation_oblique(self):
        """Testing recovery of oblique rotation."""
        real_loadings = np.array([1, 0] * 5 + [0, 1]* 5).reshape(10, 2)

        rotation_angle = np.radians(75.3)
        transformation = np.array([[1, 0], [np.cos(rotation_angle), np.sin(rotation_angle)]])

        transformed_loadings = real_loadings @ transformation
        
        # np.random.seed(262929) # For the Basin Hopping Routine
        loadings, bases = sparsify_loadings(transformed_loadings, seed=1954327,
                                            orthogonal=False, alpha=0.0)
        
        np.testing.assert_allclose(loadings, real_loadings, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(bases, transformation, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
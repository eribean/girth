import unittest

import numpy as np

from girth import create_synthetic_irt_dichotomous


class TestSynthetic(unittest.TestCase):

    def test_synthetic_irt_creation(self):
        """Testing the creation of synthetic data."""
        seed = 31

        # Regression test
        expected = np.array([[False, False, False, False,  True,  True],
                             [False, False,  True,  True,  True,  True],
                             [False, False, False,  True,  True,  True]])

        value = create_synthetic_irt_dichotomous(np.array([1.2, -0.2, 1.3]),
                                                 1.31, np.linspace(-6, 6, 6),
                                                 seed)

        np.testing.assert_array_equal(expected, value)


if __name__ == '__main__':
    unittest.main()

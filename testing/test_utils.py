import unittest

import numpy as np

from girth import irt_evaluation

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



if __name__ == '__main__':
    unittest.main()

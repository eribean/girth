from multiprocessing import shared_memory
from multiprocessing.managers import SharedMemoryManager
import unittest

import numpy as np

from scipy.stats import mvn

from girth.common import contingency_table, validate_contingency_table
from girth.common import polychoric_correlation, polychoric_correlation_serial
from girth.common.polychoric import _polychoric_correlation_value, _polychoric_engine


def create_data_from_contingency_table(table):
    """Creates two vectors that match the supplied table."""
    out1 = list()
    out2 = list()

    for ndx1 in range(table.shape[0]):
        for ndx2 in range(table.shape[1]):
            out1.append([ndx1,] * table[ndx1, ndx2])
            out2.append([ndx2,] * table[ndx1, ndx2])
    return np.concatenate(out1), np.concatenate(out2)
    

class TestContingencyTable(unittest.TestCase):
    """Test fixture for contingency table."""

    def test_contingency_table(self):
        """Testing the contingency table creation."""
        rng = np.random.default_rng(9756324)
        expected_table = rng.integers(4, 10, size=(3, 3))

        dataset = create_data_from_contingency_table(expected_table)

        result = contingency_table(*dataset, start_val=0, stop_val=2)

        np.testing.assert_equal(result, expected_table)

    def test_contingency_table_oddsize(self):
        """Testing the contingency table creation."""
        rng = np.random.default_rng(13247)
        expected_table = rng.integers(4, 10, size=(4, 3))

        dataset = create_data_from_contingency_table(expected_table)

        result = contingency_table(*dataset, start_val=0, stop_val=3)

        np.testing.assert_equal(result[:, :3], expected_table)
        self.assertEqual(result[:, -1].sum(), 0)

    def test_contingency_table_offset(self):
        """Testing the contingency table creation."""
        rng = np.random.default_rng(8543748)
        expected_table = rng.integers(4, 10, size=(4, 4))

        dataset = create_data_from_contingency_table(expected_table)

        result1 = contingency_table(dataset[0] + 2, dataset[1] + 2, 
                                    start_val=2, stop_val=5)

        result2 = contingency_table(dataset[0] + 4, dataset[1] + 4, 
                                    start_val=4, stop_val=7)   

        np.testing.assert_equal(result1, expected_table)
        np.testing.assert_equal(result2, expected_table)

    def test_validate_contingency_table_vertical(self):
        """Testing table validation."""
        rng = np.random.default_rng(8543748)
        expected_table = rng.integers(4, 10, size=(4, 4))       
        expected_table[2] = 0

        dataset = create_data_from_contingency_table(expected_table)

        result = contingency_table(*dataset, start_val=0, stop_val=3)
        new_result = validate_contingency_table(result)

        np.testing.assert_equal(new_result, np.delete(expected_table, 2, axis=0))

    def test_validate_contingency_table_horizontal(self):
        """Testing table validation."""
        rng = np.random.default_rng(5443574)
        expected_table = rng.integers(4, 10, size=(4, 4))       
        expected_table[:, 2] = 0

        dataset = create_data_from_contingency_table(expected_table)

        result = contingency_table(*dataset, start_val=0, stop_val=3)
        new_result = validate_contingency_table(result)

        np.testing.assert_equal(new_result, np.delete(expected_table, 2, axis=1))

    def test_validate_contingency_table_passthrough(self):
        """Testing table validation."""
        rng = np.random.default_rng(5443574)
        expected_table = rng.integers(4, 10, size=(4, 4))       
        dataset = create_data_from_contingency_table(expected_table)
        result = contingency_table(*dataset, start_val=0, stop_val=3)
        new_result = validate_contingency_table(result)

        np.testing.assert_equal(new_result, result)
        np.testing.assert_equal(new_result, expected_table)


class TestPolychoricCorrelations(unittest.TestCase):
    """Test Fixture for polychoric correlations."""

    def test_single_polychoric_value_positive(self):
        """Testing a single polycorrelation value."""

        # Create fake data
        thresh1 = [-23, -.3, .1, 1.2, 23]
        thresh2 = [-23, -.7, -.1, 0.8, 23]

        mean = [0, 0]
        expected_table = np.zeros((4, 4))
        rho = 0.432
        corr = np.array([[1, rho], [rho, 1]])
        
        for ndx1 in range(4):
            for ndx2 in range(4):
                lower = (thresh2[ndx1], thresh1[ndx2])
                upper = (thresh2[ndx1+1], thresh1[ndx2+1])

                expected_table[ndx1, ndx2] = mvn.mvnun(lower, upper, mean, corr)[0]

        table = (10000 * expected_table).round(0).astype('int')
        dataset = create_data_from_contingency_table(table)

        rho_found = _polychoric_correlation_value(dataset[0], dataset[1], 0, 3)

        self.assertAlmostEqual(rho, rho_found, delta=0.001)

    def test_single_polychoric_value_negative(self):
        """Testing a single polycorrelation value."""

        # Create fake data
        thresh1 = [-23, -.3, .1, 1.2, 23]
        thresh2 = [-23, -.7, -.1, 0.8, 23]

        mean = [0, 0]
        expected_table = np.zeros((4, 4))
        rho = -0.767
        corr = np.array([[1, rho], [rho, 1]])
        
        for ndx1 in range(4):
            for ndx2 in range(4):
                lower = (thresh2[ndx1], thresh1[ndx2])
                upper = (thresh2[ndx1+1], thresh1[ndx2+1])

                expected_table[ndx1, ndx2] = mvn.mvnun(lower, upper, mean, corr)[0]

        table = (10000 * expected_table).round(0).astype('int')
        dataset = create_data_from_contingency_table(table)

        rho_found = _polychoric_correlation_value(dataset[0], dataset[1], 0, 3)
        self.assertAlmostEqual(rho, rho_found, delta=0.001)

    def test_single_polychoric_value_zero(self):
        """Testing a single polycorrelation value."""
        # Create fake data
        thresh1 = [-23, -.3, .1, 1.2, 23]
        thresh2 = [-23, -.7, -.1, 0.8, 23]

        mean = [0, 0]
        expected_table = np.zeros((4, 4))
        rho = 0.0
        corr = np.array([[1, rho], [rho, 1]])
        
        for ndx1 in range(4):
            for ndx2 in range(4):
                lower = (thresh2[ndx1], thresh1[ndx2])
                upper = (thresh2[ndx1+1], thresh1[ndx2+1])

                expected_table[ndx1, ndx2] = mvn.mvnun(lower, upper, mean, corr)[0]

        table = (10000 * expected_table).round(0).astype('int')
        dataset = create_data_from_contingency_table(table)

        rho_found = _polychoric_correlation_value(dataset[0], dataset[1], 0, 3)
        self.assertAlmostEqual(rho, rho_found, delta=0.001)
    
    def test_polychoric_matrix_creation(self):
        """Testing the creation of a polychoric matrix."""

        # Since the heavy lifting is done in _polychoric_correlation_value
        # This just amounts to a smoke test for serial and parallel methods
        rng = np.random.default_rng(9843)
        expected_table = rng.integers(4, 10, size=(4, 4))       
        dataset1 = create_data_from_contingency_table(expected_table)

        dataset2 = create_data_from_contingency_table(rng.permutation(expected_table))

        dataset = np.c_[dataset1[0], dataset1[1], dataset2[0], dataset2[1]].T
        corr_mat_serial = polychoric_correlation_serial(dataset, 0, 3)
        corr_mat_parallel = polychoric_correlation(dataset, 0, 3, num_processors=2)
        corr_mat_serial2 = polychoric_correlation(dataset, 0, 3, num_processors=1)

        np.testing.assert_equal(corr_mat_serial, corr_mat_parallel)
        np.testing.assert_equal(corr_mat_serial, corr_mat_serial2)

    def test_polychoric_engine(self):
        """Testing shared memory process."""
        rng = np.random.default_rng(345714)
        expected_table = rng.integers(4, 10, size=(4, 4))       
        dataset1 = create_data_from_contingency_table(expected_table)
        dataset2 = create_data_from_contingency_table(rng.permutation(expected_table))
        dataset = np.c_[dataset1[0], dataset1[1], dataset2[0], dataset2[1]].T
        subset = np.array(list(zip(*np.tril_indices(4, -1))))

        with SharedMemoryManager() as smm:
            shm = smm.SharedMemory(size=dataset.nbytes)
            shared_buff = np.ndarray(dataset.shape, 
                                     dtype=dataset.dtype, buffer=shm.buf) 
            shared_buff[:] = dataset[:]

            corr = _polychoric_engine(shm.name, 0, 3, dataset.shape, 
                                      dataset.dtype, subset)

        # Regression tests
        expected = np.array([0.03818822,  0.99899599,  0.15021597,  0.06534598,  
                             0.66277946, -0.05768902])
        np.testing.assert_allclose(corr, expected, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
from multiprocessing import shared_memory
from multiprocessing.managers import SharedMemoryManager
from concurrent import futures
from itertools import product, repeat

import numpy as np
from scipy.optimize import optimize
from scipy.special import erfinv
from scipy.stats import mvn


__all__ = ["validate_contingency_table", "contingency_table", 
           "polychoric_correlation_serial", "polychoric_correlation"]


def validate_contingency_table(the_table):
    """Checks tables for columns or rows with all zeros and deletes.

    Args:
        the_table: input contingency table

    Returns:
        updated table with columns/rows corresponding to all zeros
    """
    mask = list()
    for ndx in range(2):
        sums = the_table.sum(axis=ndx)
        mask.append(sums != 0)
    return the_table[mask[1]][:, mask[0]]


def contingency_table(vals1, vals2, start_val=1, stop_val=5):
    """Creates a contingency table for two ordinal values.

    Args:
        vals1: First vector of ordinal data
        vals2: Second Vector of ordinal_data
        start_val: first value of record
        stop_val:  last value of record

    Returns:
        array of counts for each pair in (vals1, vals2)
    """
    n_vals = stop_val - start_val + 1
    cont_table = np.zeros((n_vals, n_vals))

    linear_ndx = vals1 * (stop_val+1) + vals2
    for ndx1, ndx2 in product(range(start_val, n_vals+start_val), 
                              range(start_val, n_vals+start_val)):
        cont_table[ndx1-start_val, ndx2-start_val] = np.count_nonzero(ndx1*(stop_val+1)+ndx2
                                                                      == linear_ndx)
    return cont_table


def _polychoric_correlation_value(vals1, vals2, start_val=1, stop_val=5):
    """Computes the polychoric correlation coefficient.
    
    Finds the correlation by fitting a bivariatate normal to the
    contingency table.

    Args:
        vals1:  First Vector of data
        vals2:  Second Vector of data
        start_val: starting value of ordinal in vals1 and vals2
        stop_val: stopping value of ordinal in vals1 and vals2

    Returns:
        rho_polychoric: polychoric correlation coefficient
        
    Note:
        If a column or row is empty, it collapses the table.
    """
    the_table = contingency_table(vals1, vals2, start_val, stop_val)
    the_table = validate_contingency_table(the_table)
    the_probabilities = np.zeros_like(the_table)

    # Determine the threshold values in each direction
    thresholds = list()
    for ndx in range(2):
        norm_vals = the_table.sum(axis=ndx).cumsum()
        norm_vals /= norm_vals[-1]
        threshold = np.sqrt(2) * erfinv(2 * norm_vals[:-1] - 1)
        thresholds.append(np.concatenate(([-23.0], threshold, [23.0])))

    corr = np.eye(2)
    mu = np.array([0, 0])
    
    # Minimization function, assumes (x, y) not (row, column)
    def min_func(rho):
        corr[0, 1] = corr[1, 0] = rho
        for ndx1, ndx2 in product(range(thresholds[1].size-1), range(thresholds[0].size-1)):
            # Integration Bounds
            low = (thresholds[1][ndx1], thresholds[0][ndx2])
            upp = (thresholds[1][ndx1+1], thresholds[0][ndx2+1])
            
            # Integral
            integral = mvn.mvnun(low, upp, mu, corr, abseps=1e-12)[0]
            the_probabilities[ndx1, ndx2] = np.log(max(1e-46, integral))
            
        return -1 * np.sum(the_table * the_probabilities)

    return optimize.fminbound(min_func, -0.999, .999)


def polychoric_correlation_serial(ordinal_data, start_val=1, stop_val=5):
    """Creates a correlation matrix of polychoric correlations.

    Analagous to a correlation coefficient matrix, except polychloric
    correlations are used in lieu of pearsons coefficient.

    Args:
        ordinal_data: [items x observations] matrix of ordinal data
        start_val: starting value of ordinal data
        stop_val: stopping value of ordinal data

    Returns:
        correlation_matrix: [items x items] polychoric correlation matrix

    See Also:
        To run on a multi-core cpu, use "polychoric_correlation"
    """
    corr_matrix = np.eye((ordinal_data.shape[0]))
    row_index, column_index = np.tril_indices(corr_matrix.shape[0], -1)

    for row_ndx, col_ndx in zip(row_index, column_index):
        corr_matrix[row_ndx, col_ndx] = _polychoric_correlation_value(ordinal_data[row_ndx],
                                                                      ordinal_data[col_ndx],
                                                                      start_val, stop_val)
        corr_matrix[col_ndx, row_ndx] = corr_matrix[row_ndx, col_ndx]
    
    return corr_matrix


def polychoric_correlation(ordinal_data, start_val=1, stop_val=5, num_processors=2):
    """Creates a correlation matrix of polychoric correlations.

    Analagous to a correlation coefficient matrix, except polychloric
    correlations are used in lieu of pearsons coefficient.

    Args:
        ordinal_data: [items x observations] matrix of ordinal data
        start_val: starting value of ordinal data
        stop_val: stopping value of ordinal data
        num_processors: number of processors on multi-core cpu

    Returns:
        correlation_matrix: [items x items] polychoric correlation matrix
    """
    if num_processors == 1:
        return polychoric_correlation_serial(ordinal_data, start_val, stop_val)
    
    indices = np.tril_indices(ordinal_data.shape[0], -1)
    chunk_indices = np.array_split(list(zip(*indices)), num_processors)

    # Do the parallel calculation
    with SharedMemoryManager() as smm:
        shm = smm.SharedMemory(size=ordinal_data.nbytes)
        shared_buff = np.ndarray(ordinal_data.shape, 
                                 dtype=ordinal_data.dtype, buffer=shm.buf)
        shared_buff[:] = ordinal_data[:]

        with futures.ProcessPoolExecutor(max_workers=num_processors) as pool:
            results = pool.map(_polychoric_engine, repeat(shm.name), repeat(start_val),
                               repeat(stop_val), repeat(ordinal_data.shape),
                               repeat(ordinal_data.dtype), chunk_indices)
            
    # Create the correlation matrix
    corr_matrix = np.eye(ordinal_data.shape[0])
    
    for ndx, result in enumerate(results):
        row_ndx = chunk_indices[ndx][:, 0]
        col_ndx = chunk_indices[ndx][:, 1]
        corr_matrix[row_ndx, col_ndx] = result
        corr_matrix[col_ndx, row_ndx] = result
        
    return corr_matrix


def _polychoric_engine(name, start_val, stop_val, shape, dtype, subset):
    """Function to run parallel polychoric correlations."""
    correlation_results = np.zeros((subset.shape[0]))

    # Read the shared memory buffer
    existing_shm = shared_memory.SharedMemory(name=name)    
    ordinal_data = np.ndarray(shape, dtype=dtype, 
                              buffer=existing_shm.buf)

    for ndx, (row_ndx, col_ndx) in enumerate(subset):
        correlation_results[ndx] = _polychoric_correlation_value(ordinal_data[row_ndx],
                                                                 ordinal_data[col_ndx],
                                                                 start_val=start_val, 
                                                                 stop_val=stop_val)

    return correlation_results
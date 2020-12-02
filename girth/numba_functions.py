import numpy as np
import numba as nb

# Default to maximum 2 threads
if nb.config.NUMBA_DEFAULT_NUM_THREADS > 1:  
    nb.set_num_threads(2)


@nb.njit()
def numba_expit(array):
    """Computes the logistic function.

    Logistic is defined as:
        1 / (1 + exp(x))

    Args:
        array: Input array 'x' values
    
    Returns:
        array: Output array of logistic values
    """
    return 1.0 / (1.0 + np.exp(array))


@nb.njit(parallel=True)
def _compute_partial_integral(theta, difficulty, discrimination, 
                              the_sign, the_output):
    """ Computes the partial integral for an IRT item

    The return array is 2D with dimensions
        (person x theta)

    Args:
        theta: (1d array) Quadrature locations
        difficulty: (float) Item difficulty parameter
        discrimination: (float) Item discrimination parameter
        the_sign: (1d array) Sign of logistic function (-1, 0, 1)
        the_output: (2d array) result of partial integral

    Returns:
        the_output: (2d array) is modified in-place

    Notes:
        DO NOT USE THIS WITHOUT NUMBA!!!
        This will be slow without the compiler!
    """ 
    # Parallelize over people
    for ndx1 in nb.prange(the_output.shape[0]):
        local_sign = the_sign[ndx1] * discrimination

        for ndx2 in range(the_output.shape[1]):
            kernel = local_sign * (theta[ndx2] - difficulty)
            the_output[ndx1, ndx2] = 1.0 / (1.0 + np.exp(kernel))
    
    return the_output


@nb.njit()
def _array_LUT(alpha, beta, theta, weight, output):
    """Computes the look up table values used to speed
       up parameter estimation
    """
    for ndx1 in range(alpha.shape[0]):
        temp1 = alpha[ndx1] * theta 
        for ndx2 in range(beta.shape[0]):
            temp2 = 1.0 + np.exp(alpha[ndx1] * beta[ndx2] - temp1)
            output[ndx1, ndx2] = np.sum(weight / temp2)
import numpy as np
import numba as nb


@nb.njit()
def _array_LUT(alpha, beta, theta, weight, output): #pragma: no cover
    """Computes the look up table values used to speed
       up parameter estimation
    """
    for ndx1 in range(alpha.shape[0]):
        temp1 = alpha[ndx1] * theta 
        for ndx2 in range(beta.shape[0]):
            temp2 = 1.0 + np.exp(alpha[ndx1] * beta[ndx2] - temp1)
            output[ndx1, ndx2] = np.sum(weight / temp2)
import numpy as np
from scipy import integrate
from scipy.optimize import fminbound

from girth.numba_functions import numba_expit


def condition_polytomous_response(dataset, trim_ends=True, _reference=1.0):
    """ Recodes polytomous responses into linear indices.

    Takes an input array of ordinal values and converts it into
    an array of linear indices to access difficulty parameters through 
    fancy indexing. 

    Args:
        dataset:  [n_items x n_takers] 2d array of ordinal responses
        trim_ends:  (boolean) trims responses that are either all no or all yes

    Returns:
        dataset: (2d array) ordinal values converted to linear indices
        beta_length: (1d array) number of unique values per item
    """
    # Remove all no / yes endorsements
    min_value, max_value = dataset.min(), dataset.max()
    n_items = dataset.shape[0]

    if trim_ends:
        raw_score_sums = dataset.sum(0)
        mask = ((raw_score_sums == (n_items * min_value)) | 
                (raw_score_sums == (n_items * max_value)))
        dataset = dataset[:, ~mask]
    
    betas_length = np.zeros((n_items,), dtype='int')
    the_output = dataset.copy()
    the_output -= min_value
    
    # Loop over rows, determine the number of unique
    # responses, and replace with linear indexing
    cnt = 0
    for ndx, item in enumerate(the_output):
        values, indices = np.unique(item, return_inverse=True)
        betas_length[ndx] = values.size

        # Recode from zero to N-1
        values = np.arange(0, betas_length[ndx]) + cnt * _reference
        the_output[ndx] = values[indices]

        # Update linear index
        cnt += betas_length[ndx]

    return the_output, betas_length


def _solve_for_constants(item_responses):
    """Computes the ratios needed for grm difficulty estimates."""
    _, counts = np.unique(item_responses, return_counts=True)
    diagonal = counts[:-1] + counts[1:]
    A_matrix = (np.diag(diagonal) + np.diag(-counts[2:], -1) + 
                np.diag(-counts[:-2], 1))
    
    return np.linalg.inv(A_matrix)[:, 0] * counts[1]
    

def _graded_partial_integral(theta, betas, betas_roll,
                             discrimination, responses):
    """Computes the partial integral for the graded response."""
    temp1 = (betas[:, None] - theta) * discrimination[:, None]
    temp2 = (betas_roll[:, None] - theta) * discrimination[:, None]
    graded_prob = numba_expit(temp1) 
    graded_prob -= numba_expit(temp2)

    return graded_prob[responses, :]


def _solve_integral_equations_LUT(discrimination, ratio, _, __, interpolate_function):
    """Solve single sigmoid integral for difficulty
    parameter using Look Up Table.
    """
    return interpolate_function(discrimination, ratio)


def _solve_integral_equations(discrimination, ratio, distribution, theta, _):
    """Solve single sigmoid integral for difficulty parameter."""
    difficulty = np.zeros_like(ratio)
    temp1 = np.exp(-1 * discrimination * theta)
    
    for ratio_ndx, value in enumerate(ratio):
        def _min_func_local(estimate):
            kernel = 1. / (1 + np.exp(estimate*discrimination) * temp1)
            integral = np.sum(kernel * distribution)
            return np.square(value - integral)
                        
        difficulty[ratio_ndx] = fminbound(_min_func_local, -6, 6, xtol=1e-4)
    return difficulty


def _credit_partial_integral(theta, betas, discrimination, 
                             response_set):
    """Computes the partial integral for the partial credit model."""
    # Creates a 2d array [beta x thetas]
    kernel = theta[None, :] - betas[:, None]
    kernel *= discrimination

    # This can be removed since its a scalar
    # in the optimization
    kernel[0] = 0
    
    # PCM is additive in log space
    np.cumsum(kernel, axis=0, out=kernel)
    np.exp(kernel, out=kernel)

    # Normalize probability to equal one
    kernel /= np.nansum(kernel, axis=0)[None, :]

    return kernel[response_set, :]


def _unfold_partial_integral(theta, delta, betas, 
                             discrimination, fold_span,
                             response_set):
    """Computes the partial integral for the _GGUM model."""
    # Unfolding_Model
    thresholds = np.exp(-discrimination * np.cumsum(betas))

    kernel = np.outer(fold_span, discrimination * (theta - delta))

    np.cosh(kernel, out=kernel)
    kernel[1:, :] *= thresholds[:, None]
    kernel /= np.nansum(kernel, axis=0)

    return kernel[response_set, :]
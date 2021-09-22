import numpy as np
from scipy import integrate
from scipy.optimize import fminbound
from scipy.special import expit

from girth import INVALID_RESPONSE


__all__ = ["condition_polytomous_response"]


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
        valid_response_mask: (2d_array) boolean mask of valid responses
                             that is the same size as dataset
    """
    # Remove all no / yes endorsements
    valid_response_mask = dataset != INVALID_RESPONSE
    min_value = np.min(dataset[valid_response_mask])
    max_value = np.max(dataset[valid_response_mask])
    n_items = dataset.shape[0]

    if trim_ends:
        raw_score_sums = np.sum(dataset, where=valid_response_mask, axis=0)
        valid_score_sums = np.count_nonzero(valid_response_mask, axis=0)
        ratio_score_sums = raw_score_sums / valid_score_sums

        mask = ((ratio_score_sums == min_value) | 
                (ratio_score_sums == max_value))
        dataset = dataset[:, ~mask]
        valid_response_mask = dataset != INVALID_RESPONSE
    
    betas_length = np.zeros((n_items,), dtype='int')
    the_output = dataset.copy()
    the_output -= min_value
    
    # Loop over rows, determine the number of unique
    # responses, and replace with linear indexing
    cnt = 0
    for ndx, item in enumerate(the_output):
        values, indices = np.unique(item[valid_response_mask[ndx]], 
                                    return_inverse=True)
        betas_length[ndx] = max(values.size, 2)

        # Recode from zero to N-1
        recode = np.arange(0, betas_length[ndx])
        if values.size == 1:
            recode = np.array([values[0] != 0], dtype='int')
        values = recode + cnt * _reference
        the_output[ndx, valid_response_mask[ndx]] = values[indices]

        # Update linear index
        cnt += betas_length[ndx]
    
    # Reset the invalid responses
    the_output[~valid_response_mask] = INVALID_RESPONSE

    return the_output, betas_length, valid_response_mask


def _solve_for_constants(item_responses):
    """Computes the ratios needed for grm difficulty estimates."""
    value, counts = np.unique(item_responses, return_counts=True)

    if counts.shape[0] > 1:
        diagonal = counts[:-1] + counts[1:]
        A_matrix = (np.diag(diagonal) + np.diag(-counts[2:], -1) + 
                    np.diag(-counts[:-2], 1))
        constants = np.linalg.inv(A_matrix)[:, 0] * counts[1]
    
    else:
        constants = np.array([0]) if value == 0 else np.array([1])
    
    return constants
    

def _graded_partial_integral(theta, betas, betas_roll,
                             discrimination, response_set,
                             invalid_response_mask):
    """Computes the partial integral for the graded response."""
    temp1 = (theta - betas[:, None]) * discrimination[:, None]
    temp2 = (theta - betas_roll[:, None]) * discrimination[:, None]
    graded_prob = expit(temp1) 
    graded_prob -= expit(temp2)

    # Set all the responses and fix afterward
    temp_output = graded_prob[response_set, :]
    temp_output[invalid_response_mask] = 1.0

    return temp_output


def _graded_partial_integral_md(theta, betas, betas_roll,
                                discrimination, response_set,
                                invalid_response_mask):
    """Computes the partial integral for the multidimensional graded response."""
    temp1 = discrimination @ theta + betas[:, None]
    temp2 = discrimination @ theta + betas_roll[:, None]
    graded_prob = expit(temp1) 
    graded_prob -= expit(temp2)

    # Set all the responses and fix afterward
    temp_output = graded_prob[response_set, :]
    temp_output[invalid_response_mask] = 1.0

    return temp_output


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
                             response_set, invalid_response_mask):
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

    # Set all the responses and fix afterward
    temp_output = kernel[response_set, :]
    temp_output[invalid_response_mask] = 1.0

    return temp_output


def _unfold_partial_integral(theta, delta, betas, 
                             discrimination, fold_span,
                             response_set, invalid_response_mask):
    """Computes the partial integral for the _GGUM model."""
    # Unfolding_Model
    thresholds = np.exp(-discrimination * np.cumsum(betas))

    kernel = np.outer(fold_span, discrimination * (theta - delta))

    np.cosh(kernel, out=kernel)
    kernel[1:, :] *= thresholds[:, None]
    kernel /= np.nansum(kernel, axis=0)

    # Set all the responses and fix afterward
    temp_output = kernel[response_set, :]
    temp_output[invalid_response_mask] = 1.0

    return temp_output
import numpy as np
from scipy import integrate
from scipy.optimize import fminbound

from girth import irt_evaluation


def condition_polytomous_response(dataset, trim_ends=True, _reference=1.0):
    """
    Transforms item responses for easier use during parameter
    estimation

    This takes an input array of ordinal values and converts it into
    an array of linear indices to access difficulty parameters through 
    fancy indexing. 

    Args:
        dataset:  [n_items x n_takers] 2d array of ordinal responses
        trim_ends:  (boolean) trims responses that are either all no or all yes

    Returns:
        updated dataset array adjusted for linear indexing, 
        vector of lengths associated with each item
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
                             discrimination, responses,
                             distribution):
    """Computes the partial integral for the graded response."""
    graded_prob = (irt_evaluation(betas, discrimination, theta) - 
                   irt_evaluation(betas_roll, discrimination, theta))

    #TODO: Potential chunking for memory limited systems    
    return distribution[None, :] * graded_prob[responses, :].prod(axis=0)


def _solve_integral_equations(discrimination, ratio, distribution, theta):
    """Solve single sigmoid integral for difficulty parameter."""
    difficulty = np.zeros_like(ratio)
    
    for ratio_ndx, value in enumerate(ratio):

        def _min_func_local(estimate):
            kernel = (distribution / 
                     (1 + np.exp(discrimination * (estimate - theta))))
            integral = integrate.fixed_quad(lambda x: kernel, -5, 5, n=61)[0]
            return np.square(value - integral)
                        
        difficulty[ratio_ndx] = fminbound(_min_func_local, -6, 6)
    
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
    
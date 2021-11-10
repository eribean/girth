import numpy as np
from scipy.special import expit


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
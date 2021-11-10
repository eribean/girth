from scipy.special import expit


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
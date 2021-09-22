import numpy as np


def _compute_partial_integral_3pl(theta, difficulty, discrimination, guessing, the_sign):
    """
    Computes the partial integral for a set of item parameters

    Args:
        theta: (array) evaluation points
        difficulty: (array) set of difficulty parameters
        discrimination: (array | number) set of discrimination parameters
        guessing: (array) set of guessing parameters
        the_sign:  (array) positive or negative sign
                            associated with response vector

    Returns:
        partial_integral: (2d array) 
            integration of items defined by "sign" parameters
            axis 0: individual persons
            axis 1: evaluation points (at theta)

    Notes:
        Implicitly multiplies the data by the gaussian distribution
    """
    # This represents a 3-dimensional array
    # [Response Set, Person, Theta]
    # The integration happens over response set and the result is an
    # array of [Person, Theta]
    kernel = the_sign[:, :, None] * np.ones((1, 1, theta.size))
    kernel *= discrimination[:, None, None]
    kernel *= (theta[None, None, :] - difficulty[:, None, None])
    
    otpt = (1.0 / (1.0 + np.exp(kernel))) * (1 - guessing[:, None, None])
    otpt += 0.5 * (1 - the_sign[:, :, None]) * guessing[:, None, None]

    return otpt.prod(axis=0).squeeze()
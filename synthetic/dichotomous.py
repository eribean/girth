import numpy as np
from scipy.special import expit


__all__ = ["create_synthetic_irt_dichotomous"]


def create_synthetic_irt_dichotomous(difficulty, discrimination, thetas,
                                     guessing=0, seed=None):
    """ Creates dichotomous unidimensional synthetic IRT data.

    Creates synthetic IRT data to test parameters estimation functions.  
    Only for use with dichotomous outputs

    Assumes the model for univariate:
        P(theta) = 1.0 / (1 + exp(-discrimination * (theta - difficulty)))

    and for multivariate
        P(theta) = 1.0 / (1 + exp(-1 * (discrimination @ theta + intercept)))

    Args:
        difficulty: [array] of difficulty parameters
        discrimination:  [array | number] of discrimination parameters
        thetas: [array] of person abilities
        guessing: [array | number] of guessing parameters associated with items
        seed: Optional setting to reproduce results

    Returns:
        synthetic_data: (2d array) realization of possible response given parameters

    """
    rng = np.random.default_rng(seed)

    if np.size(discrimination) == 1:
        discrimination = np.full_like(difficulty, discrimination)

    if np.size(guessing) == 1:
        guessing = np.full_like(difficulty, guessing)

    if np.ndim(discrimination) == 1:
        kernel = thetas[None, :] - difficulty[:, None]
        kernel *= discrimination[:, None]

    elif np.ndim(discrimination) == 2:
        kernel = discrimination @ thetas
        kernel += difficulty[:, None]
    
    else:
        raise AssertionError("Discrimination dimensions are not 1 or 2, "
            f"got {np.ndim(discrimination)}")

    continuous_output = expit(kernel)

    # Add guessing parameters
    continuous_output *= (1.0 - guessing[:, None])
    continuous_output += guessing[:, None]

    # Convert to binary based on probability, this is faster than Bernoulli
    random_compare = rng.uniform(size=continuous_output.shape)

    return (random_compare <= continuous_output).astype('int')
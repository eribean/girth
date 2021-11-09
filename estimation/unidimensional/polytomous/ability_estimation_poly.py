import numpy as np


def _ability_eap_abstract(partial_int, weight, theta):
    """Generic function to compute abilities

    Estimates the ability parameters (theta) for models via
    expected a posterior likelihood estimation.

    Args:
        partial_int: (2d array) partial integrations over items
        weight: weighting to apply before summation
        theta: quadrature evaluation locations
    
    Returns:
        abilities: the estimated latent abilities
    """
    local_int = partial_int * weight

    # Compute the denominator
    denominator = np.sum(local_int, axis=1)

    # compute the numerator
    local_int *= theta
    numerator = np.sum(local_int, axis=1)

    return numerator / denominator
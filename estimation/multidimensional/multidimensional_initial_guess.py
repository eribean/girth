import numpy as np

from girth import validate_estimation_options

from girth.utils import INVALID_RESPONSE
from girth.common import polychoric_correlation
from girth.factoranalysis import maximum_likelihood_factor_analysis as mlfa


__all__ = ['initial_guess_md']


def _constrained_rotation(discrimination):
    """Applies constraints to discrimination parameters.
    
    The bottom lower right diagonal is zeros with positive
    values on the diagonal.
    """
    n_factors = discrimination.shape[1]
    rotation_matrix, _ = np.linalg.qr(discrimination[-n_factors:][::-1].T)

    rotated_discrimination = discrimination @ rotation_matrix

    # Make diagonal be positive
    diagonal_values = np.diag_indices(n_factors)
    diagonal_values = (discrimination.shape[0] - 1 - diagonal_values[0],
                       diagonal_values[1])
    
    update_sign = np.sign(rotated_discrimination[diagonal_values])
    rotated_discrimination *= update_sign.reshape(1, -1)

    return rotated_discrimination


def initial_guess_md(dataset, n_factors, options=None):
    """Determine an inital guess for multidimensional IRT models.

    Currently only valid for 2PL and GRM multidimensional models.

    Args:
        dataset: [n_items x n_observations] collected inputs
        n_factors: [int] number of factors to extract
        options: dictionary with updates to default options

    Returns:
        initial_guess: numpy array of initial discrimination parameters
    
    Options:
        * num_processors: [int] number of cores to use for polychoric correlation
    """
    options = validate_estimation_options(options)
    valid_mask = dataset != INVALID_RESPONSE

    unique_values = np.unique(dataset[valid_mask])
    start_value, stop_value = unique_values[0], unique_values[-1]

    correlation = polychoric_correlation(dataset, start_value,
                                         stop_value, options['num_processors'])

    factors, eigenvalues, unique_variance = mlfa(correlation, n_factors)

    # Estimated Discrimination parameters
    estimated_discrimination = 1.7 * factors / np.sqrt(unique_variance).reshape(-1, 1)

    # Rotate to conform to constraints
    if n_factors > 1:
        estimated_discrimination = _constrained_rotation(estimated_discrimination)

    return estimated_discrimination

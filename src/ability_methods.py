import numpy as np

from scipy import integrate
from scipy.stats import uniform
from scipy.stats import norm as gaussian
from scipy.optimize import fminbound
from girth import convert_responses_to_kernel_sign, validate_estimation_options
from girth.utils import (INVALID_RESPONSE, _get_quadrature_points, 
                         _compute_partial_integral)


__all__ = ["ability_mle", "ability_map", "ability_eap"]


def ability_mle(dataset, difficulty, discrimination, no_estimate=np.nan):
    """Estimates the abilities for dichotomous models.

    Estimates the ability parameters (theta) for dichotomous models via
    maximum likelihood estimation.  Response sets with no variance are trimmed
    from evaluation

    Args:
        dataset: [n_items, n_participants] (2d Array) of measured responses
        difficulty: (1d Array) of difficulty parameters for each item
        discrimination: (1d Array) of disrimination parameters for each item
        no_estimate: value to use for response sets that cannot be estimated
                     defaults to numpy.nan, if a number is used then
                     -no_estimate -> 0 and no_estimate -> 1

    Returns:
        abilities: (1d array) estimated abilities
    """
    # Find any missing data
    bad_mask = dataset == INVALID_RESPONSE

    # Locations where endorsement isn't constant
    mask = ~(np.ma.masked_array(dataset, bad_mask).var(axis=0) == 0)

    # Use only appropriate data
    valid_dataset = dataset[:, mask]

    # Call MAP with uniform distribution
    trimmed_theta = ability_map(valid_dataset, difficulty, discrimination,
                                {'distribution': uniform(-7, 14).pdf})

    # Replace no_estimate values
    thetas = np.full((dataset.shape[1],), np.abs(no_estimate), dtype='float')
    thetas[mask] = trimmed_theta

    # Convert all zeros to negative estimate
    mask2 = ~mask & (dataset.min(axis=0) == 0)
    thetas[mask2] *= -1

    return thetas


def ability_map(dataset, difficulty, discrimination, options=None):
    """Estimates the abilities for dichotomous models.

    Estimates the ability parameters (theta) for dichotomous models via
    maximum a posterior likelihood estimation.

    Args:
        dataset: [n_items, n_participants] (2d Array) of measured responses
        difficulty: (1d Array) of difficulty parameters for each item
        discrimination: (1d Array) of disrimination parameters for each item
        options: dictionary with updates to default options

    Returns:
        abilities: (1d array) estimated abilities

    Options:
        distribution: 

    Notes:
        If distribution is uniform, please use ability_mle instead. A large set 
        of probability distributions can be found in scipy.stats
        https://docs.scipy.org/doc/scipy/reference/stats.html
    """
    options = validate_estimation_options(options)
    distribution = options['distribution']

    if np.atleast_1d(discrimination).size == 1:
        discrimination = np.full(dataset.shape[0], discrimination,
                                 dtype="float")

    n_takers = dataset.shape[1]
    the_sign = convert_responses_to_kernel_sign(dataset)
    thetas = np.zeros((n_takers,))

    for ndx in range(n_takers):
        # pylint: disable=cell-var-from-loop
        scalar = the_sign[:, ndx] * discrimination

        def _theta_min(theta):
            otpt = 1.0 / (1.0 + np.exp(scalar * (theta - difficulty)))

            return -(np.log(otpt).sum() + np.log(distribution(theta)))

        # Solves for the ability for each person
        thetas[ndx] = fminbound(_theta_min, -6, 6)

    return thetas


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


def ability_eap(dataset, difficulty, discrimination, options=None):
    """Estimates the abilities for dichotomous models.

    Estimates the ability parameters (theta) for dichotomous models via
    expected a posterior likelihood estimation.

    Args:
        dataset: [n_items, n_participants] (2d Array) of measured responses
        difficulty: (1d Array) of difficulty parameters for each item
        discrimination: (1d Array) of disrimination parameters for each item
        options: dictionary with updates to default options

    Returns:
        abilities: (1d array) estimated abilities

    Options:
        * distribution: callable
        * quadrature_bounds: (float, float)
        * quadrature_n: int

    """
    options = validate_estimation_options(options)
    quad_start, quad_stop = options['quadrature_bounds']
    quad_n = options['quadrature_n']

    if np.atleast_1d(discrimination).size == 1:
        discrimination = np.full(dataset.shape[0], discrimination,
                                 dtype='float')

    invalid_response_mask = dataset == INVALID_RESPONSE
    unique_sets = dataset.copy()
    unique_sets[invalid_response_mask] = 0 # For Indexing, fixed later

    theta, weights = _get_quadrature_points(quad_n, quad_start, quad_stop)
    partial_int = np.ones((dataset.shape[1], quad_n))
    for ndx in range(dataset.shape[0]):
        partial_int *= _compute_partial_integral(theta, difficulty[ndx], 
                                                 discrimination[ndx],
                                                 unique_sets[ndx],
                                                 invalid_response_mask[ndx])
    distribution_x_weights = options['distribution'](theta) * weights

    return _ability_eap_abstract(partial_int, distribution_x_weights,
                                 theta)

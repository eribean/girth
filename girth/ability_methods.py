import numpy as np

from scipy import integrate
from scipy.stats import uniform
from scipy.stats import norm as gaussian
from scipy.optimize import fminbound
from girth import convert_responses_to_kernel_sign, validate_estimation_options
from girth.numba_functions import _compute_partial_integral
from girth.utils import _get_quadrature_points


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
    # Locations where endorsement isn't constant
    mask = np.nanvar(dataset, axis=0) > 0

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


def ability_eap(dataset, difficulty, discrimination, options=None):
    """Estimates the abilities for dichotomous models.

    Estimates the ability parameters (theta) for dichotomous models via
    expaected a posterior likelihood estimation.

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

    the_sign = convert_responses_to_kernel_sign(dataset)
    the_output = np.zeros((the_sign.shape[1], quad_n), dtype='float64')

    theta, weights = _get_quadrature_points(quad_n, quad_start, quad_stop)
    partial_int = np.ones_like(the_output)
    for ndx in range(the_sign.shape[0]):
        partial_int *= _compute_partial_integral(theta, difficulty[ndx], 
                                                 discrimination[ndx], the_sign[ndx],
                                                 the_output)

    # Weight by the input ability distribution
    partial_int *= (options['distribution'](theta) * weights)

    # Compute the denominator
    denominator = np.sum(partial_int, axis=1)

    # compute the numerator
    partial_int *= theta
    numerator = np.sum(partial_int, axis=1)

    return numerator / denominator

import numpy as np

from scipy import integrate
from scipy.stats import uniform
from scipy.stats import norm as gaussian
from scipy.optimize import fminbound
from girth import convert_responses_to_kernel_sign, validate_estimation_options
from girth.utils import _get_quadrature_points
from girth.three_pl.three_pl_utils import _compute_partial_integral_3pl


def ability_3pl_mle(dataset, difficulty, discrimination,
                    guessing, no_estimate=np.nan):
    """Estimates the abilities for dichotomous models.

    Estimates the ability parameters (theta) for dichotomous models via
    maximum likelihood estimation.  Response sets with no variance are trimmed
    from evaluation

    Args:
        dataset: [n_items, n_participants] (2d Array) of measured responses
        difficulty: (1d Array) of difficulty parameters for each item
        discrimination: (1d Array) of disrimination parameters for each item
        guessing: (1d Array) of guessing parameters for each item
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
    trimmed_theta = ability_3pl_map(valid_dataset, difficulty, discrimination,
                                    guessing,
                                    {'distribution': uniform(-7, 14).pdf})

    # Replace no_estimate values
    thetas = np.full((dataset.shape[1],), np.abs(no_estimate), dtype='float')
    thetas[mask] = trimmed_theta

    # Convert all zeros to negative estimate
    mask2 = ~mask & (dataset.min(axis=0) == 0)
    thetas[mask2] *= -1

    return thetas


def ability_3pl_map(dataset, difficulty, discrimination,
                    guessing, options=None):
    """Estimates the abilities for dichotomous models.

    Estimates the ability parameters (theta) for dichotomous models via
    maximum a posterior likelihood estimation.

    Args:
        dataset: [n_items, n_participants] (2d Array) of measured responses
        difficulty: (1d Array) of difficulty parameters for each item
        discrimination: (1d Array) of disrimination parameters for each item
        guessing: (1d Array) of guessing parameters for each item        
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

    n_takers = dataset.shape[1]
    the_sign = convert_responses_to_kernel_sign(dataset)
    thetas = np.zeros((n_takers,))

    # Pre-Compute guessing offset
    multiplier = 1.0 - guessing
    additive = guessing[:, None] * (the_sign == -1).astype('float')

    for ndx in range(n_takers):
        # pylint: disable=cell-var-from-loop
        scalar = the_sign[:, ndx] * discrimination
        adder = additive[:, ndx]

        def _theta_min(theta):
            otpt = multiplier / (1.0 + np.exp(scalar * (theta - difficulty)))
            otpt += adder

            return -(np.log(otpt).sum() + np.log(distribution(theta)))

        # Solves for the ability for each person
        thetas[ndx] = fminbound(_theta_min, -6, 6)

    return thetas


def ability_3pl_eap(dataset, difficulty, discrimination,
                guessing=None, options=None):
    """Estimates the abilities for dichotomous models.

    Estimates the ability parameters (theta) for dichotomous models via
    expected a posterior likelihood estimation.

    Args:
        dataset: [n_items, n_participants] (2d Array) of measured responses
        difficulty: (1d Array) of difficulty parameters for each item
        discrimination: (1d Array) of disrimination parameters for each item
        guessing: (1d Array) of guessing parameters for each item        
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

    the_sign = convert_responses_to_kernel_sign(dataset)

    theta, _ = _get_quadrature_points(quad_n, quad_start, quad_stop)
    partial_int = _compute_partial_integral_3pl(
        theta, difficulty, discrimination, guessing, the_sign)

    # Weight by the input ability distribution
    partial_int *= options['distribution'](theta)

    # Compute the denominator
    denominator = integrate.fixed_quad(
        lambda x: partial_int, quad_start, quad_stop, n=quad_n)[0]

    # compute the numerator
    partial_int *= theta

    numerator = integrate.fixed_quad(
        lambda x: partial_int, quad_start, quad_stop, n=quad_n)[0]

    return numerator / denominator

import numpy as np

from scipy import integrate
from scipy.stats import uniform
from scipy.stats import norm as gaussian
from scipy.optimize import fminbound
from girth import convert_responses_to_kernel_sign
from girth.utils import _compute_partial_integral, _get_quadrature_points


def ability_map(dataset, difficulty, discrimination, distribution=None):
    """Estimates the abilities for dichotomous models.

    Estimates the ability parameters (theta) for dichotomous models via
    maximum a posterior likelihood estimation.

    Args:
        dataset: [n_items, n_participants] (2d Array) of measured responses
        difficulty: (1d Array) of difficulty parameters for each item
        discrimination: (1d Array) of disrimination parameters for each item
        distribution: function handle to PDF of ability distribution, p = f(theta)
                      the default is gaussian (i.e: scipy.stats.norm(0, 1).pdf)

    Returns
        1d array of estimated abilities

    Notes:
        If distribution is uniform, MAP is equivalent to MLE. A large set of
        probability distributions can be found in scipy.stats
        https://docs.scipy.org/doc/scipy/reference/stats.html
    """
    if distribution is None:
        distribution = gaussian(0, 1).pdf
    
    discrimination = np.atleast_1d(discrimination)
    if discrimination.size == 1:
        discrimination = np.full(dataset.shape[0], discrimination)

    n_takers = dataset.shape[1]
    the_sign = convert_responses_to_kernel_sign(dataset)
    thetas = np.zeros((n_takers,))

    for ndx in range(n_takers):
        # pylint: disable=cell-var-from-loop
        scalar = the_sign[:, ndx] * discrimination
        def _theta_min(theta):
            otpt = 1.0  / (1.0 + np.exp(scalar * (theta - difficulty)))

            return -(np.log(otpt).sum() + np.log(distribution(theta)))

        # Solves for the ability for each person
        thetas[ndx] = fminbound(_theta_min, -6, 6)

    return thetas


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

    Returns
        1d array of estimated abilities
    """
    # Locations where endorsement isn't constant
    mask = np.nanvar(dataset, axis=0) > 0 

    # Use only appropriate data
    valid_dataset = dataset[:, mask]

    # Call MAP with uniform distribution
    trimmed_theta = ability_map(valid_dataset, difficulty, discrimination,
                                uniform(-7, 14).pdf)
    
    # Replace no_estimate values
    thetas = np.full((dataset.shape[1],), np.abs(no_estimate), dtype='float')
    thetas[mask] = trimmed_theta

    # Convert all zeros to negative estimate
    mask2 = ~mask & (dataset.min(axis=0) == 0)
    thetas[mask2] *= -1    

    return thetas


def ability_eap(dataset, difficulty, discrimination, distribution=None):
    """Estimates the abilities for dichotomous models.

    Estimates the ability parameters (theta) for dichotomous models via
    expaected a posterior likelihood estimation.

    Args:
        dataset: [n_items, n_participants] (2d Array) of measured responses
        difficulty: (1d Array) of difficulty parameters for each item
        discrimination: (1d Array) of disrimination parameters for each item
        distribution: function handle to PDF of ability distribution, p = f(theta)
                      the default is gaussian (i.e: scipy.stats.norm(0, 1).pdf)

    Returns
        1d array of estimated abilities
    """
    if distribution is None:
        distribution = gaussian(0, 1).pdf
    
    discrimination = np.atleast_1d(discrimination)
    if discrimination.size == 1:
        discrimination = np.full(dataset.shape[0], discrimination)

    the_sign = convert_responses_to_kernel_sign(dataset)

    theta = _get_quadrature_points(61, -5, 5)
    partial_int = _compute_partial_integral(theta, difficulty, discrimination, the_sign)

    # Need to remove guassian distribution that is appended in the partial integration
    remove_disribution = np.sqrt(2 * np.pi) / np.exp(-np.square(theta) / 2)
    new_distribution = distribution(theta) * remove_disribution
    partial_int *= new_distribution

    # Compute the denominator
    denominator = integrate.fixed_quad(lambda x: partial_int, -5, 5, n=61)[0]

    # compute the numerator
    partial_int *= theta

    numerator = integrate.fixed_quad(lambda x: partial_int, -5, 5, n=61)[0]

    return numerator / denominator
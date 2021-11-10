import numpy as np
from scipy.stats import norm as gaussian
from scipy.special import roots_legendre


# There is no int nan, so use this as a placeholder
INVALID_RESPONSE = -99999


__all__ = ["mml_approx", "trim_response_set_and_counts", "tag_missing_data",
            "convert_responses_to_kernel_sign", "get_true_false_counts", 
            "tag_missing_data", "INVALID_RESPONSE"]


def tag_missing_data(dataset, valid_responses):
    """Checks the data for valid responses.
    
    Args:
        dataset: (array) array to validate
        valid_responses: (array-like) list of valid responses
        
    Returns:
        updated_dataset: (array) data that holds only valid_responses and
                         invalid_fill
    """
    mask = np.isin(dataset, valid_responses)
    output = dataset.copy()
    output[~mask] = INVALID_RESPONSE
    
    return output


def get_true_false_counts(responses):
    """ Returns the number of true and false for each item.

    Takes in a responses array and returns counts associated with
    true / false.  True is a value in the dataset which equals '1'
    and false is a value which equals '0'.  All other values are
    ignored

    Args:
        responses: [n_items x n_participants] array of response values

    Returns:
        n_false: (1d array) "false" counts per item
        n_true: (1d array) "true" counts per item
    """
    n_false = np.count_nonzero(responses == 0, axis=1)
    n_true = np.count_nonzero(responses == 1, axis=1)

    return n_false, n_true


def mml_approx(dataset, discrimination=1, scalar=None):
    """ Difficulty parameter estimates of IRT model.
    
    Analytic estimates of the difficulty parameters 
    in an IRT model assuming a normal distribution .

    Args:
        dataset: [items x participants] matrix of True/False Values
        discrimination: scalar of discrimination used in model (default to 1)
        scalar: (1d array) logarithm of "false counts" to "true counts" (log(n_no / n_yes))

    Returns:
        difficulty: (1d array) difficulty estimates
    """
    if scalar is None:
        n_no, n_yes = get_true_false_counts(dataset)
        scalar = np.log(n_no / n_yes)

    return (np.sqrt(1 + discrimination**2 / 3) *
            scalar / discrimination)


def convert_responses_to_kernel_sign(responses):
    """Converts dichotomous responses to the appropriate kernel sign.

    Takes in an array of responses coded as either [True/False] or [1/0]
    and converts it into [+1 / -1] to be used during parameter estimation.

    Values that are not 0 or 1 are converted into a zero which means these
    values do not contribute to parameter estimates.  This can be used to 
    account for missing values.

    Args:
        responses: [n_items x n_participants] array of response values

    Returns:
        the_sign: (2d array) sign values associated with input responses
    """
    # The default value is now 0
    the_sign = np.zeros_like(responses, dtype='float')

    # 1 -> -1
    mask = responses == 1
    the_sign[mask] = -1

    # 0 -> 1
    mask = responses == 0
    the_sign[mask] = 1

    return the_sign


def trim_response_set_and_counts(response_sets, counts):
    """ Trims all true or all false responses from the response set/counts.

    Args:
        response_set:  (2D array) response set by persons obtained by running
                        numpy.unique
        counts:  counts associated with response set

    Returns:
        response_set: updated response set with removal of undesired response patterns
        counts: updated counts to account for removal
    """
    # Find any missing data
    bad_mask = response_sets == INVALID_RESPONSE

    # Remove response sets where output is all true/false
    mask = ~(np.ma.masked_array(response_sets, bad_mask).var(axis=0) == 0)
    response_sets = response_sets[:, mask]
    counts = counts[mask]

    return response_sets, counts


def _get_quadrature_points(n, a, b):
    """ Quadrature points needed for gauss-legendre integration.

    Utility function to get the legendre points,
    shifted from [-1, 1] to [a, b]

    Args:
        n: number of quadrature_points
        a: lower bound of integration
        b: upper bound of integration

    Returns:
        quadrature_points: (1d array) quadrature_points for 
                           numerical integration
        weights: (1d array) quadrature weights

    Notes:
        A local function of the based fixed_quad found in scipy, this is
        done for processing optimization
    """
    x, w = roots_legendre(n)
    x = np.real(x)

    # Legendre domain is [-1, 1], convert to [a, b]
    scalar = (b - a) * 0.5
    return scalar * (x + 1) + a, scalar * w

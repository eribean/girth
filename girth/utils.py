import numpy as np
from scipy import interpolate
from scipy.stats import norm as gaussian
from scipy.special import roots_legendre

from girth import _array_LUT


def default_options():
    """ Dictionary of options used in Girth.

    Args:
        max_iteration: [int] maximum number of iterations
            allowed during processing. (Default = 25)
        distribution: [callable] function that returns a pdf
            evaluated at quadrature points, p = f(theta).
            (Default = scipy.stats.norm(0, 1).pdf)
        quadrature_bounds: (lower, upper) bounds to limit
            numerical integration. Default = (-5, 5)
        quadrature_n: [int] number of quadrature points to use
                        Default = 61
        use_LUT: [boolean] use a look up table in mml functions
        estimate_distribution: [boolean] estimate the latent distribution
                               using cubic splines
        number_of_samples: [int] number of samples to use when
                           estimating distribuion, must be > 5
    """
    return {"max_iteration": 25,
            "distribution": gaussian(0, 1).pdf,
            "quadrature_bounds": (-4.5, 4.5),
            "quadrature_n": 41,
            "use_LUT": True,
            "estimate_distribution": False,
            "number_of_samples": 9
            }


def validate_estimation_options(options_dict=None):
    """ Validates an options dictionary.

    Args:
        options_dict: Dictionary with updates to default_values

    Returns:
        options_dict: Updated dictionary

    """
    validate = {'max_iteration':
                    lambda x: isinstance(x, int) and x > 0,
                'distribution':
                    callable,
                'quadrature_bounds':
                    lambda x: isinstance(x, (tuple, list)) and (x[1] > x[0]),
                'quadrature_n':
                    lambda x: isinstance(x, int) and x > 7,
                'use_LUT':
                    lambda x: isinstance(x, bool),
                'estimate_distribution':
                    lambda x: isinstance(x, bool),
                "number_of_samples": 
                    lambda x: isinstance(x, int) and x >= 5,
                }
    
    # A complete options dictionary
    full_options = default_options()
    
    if options_dict:
        if not isinstance(options_dict, dict):
            raise AssertionError("Options must be a dictionary got: "
                                f"{type(options_dict)}.")

        for key, value in options_dict.items():
            if not validate[key](value):
                raise AssertionError("Unexpected key-value pair: "
                                     f"{key}: {value}. Please see "
                                     "documentation for expected inputs.")

        full_options.update(options_dict)

    return full_options


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

    Takes in an array of responses coded as either [True/False] or [0/1]
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
    # Remove response sets where output is all true/false
    mask = ~(np.nanvar(response_sets, axis=0) == 0)
    response_sets = response_sets[:, mask]
    counts = counts[mask]

    return response_sets, counts


def irt_evaluation(difficulty, discrimination, thetas):
    """ Evaluation of unidimensional IRT model.

    Evaluates an IRT model and returns the exact values.  This function
    supports only unidimemsional models

    Assumes the model
        P(theta) = 1.0 / (1 + exp(discrimination * (theta - difficulty)))

    Args:
        difficulty: (1d array) item difficulty parameters
        discrimination:  (1d array | number) item discrimination parameters
        thetas: (1d array) person abilities

    Returns:
        probabilities: (2d array) evaluation of sigmoid for given inputs
    """
    # If discrimination is a scalar, make it an array
    if np.atleast_1d(discrimination).size == 1:
        discrimination = np.full_like(difficulty, discrimination,
                                      dtype='float')

    kernel = difficulty[:, None] - thetas
    kernel *= discrimination[:, None]
    return 1.0 / (1 + np.exp(kernel))


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


def create_beta_LUT(alpha, beta, options=None):
    """Creates a Look Up Table to speed up conversion.
    
    Args:
        alpha: (array-like) [alpha_start, alpha_stop, alpha_n]
        beta: (array-like) [beta_start, beta_stop, beta_n]
        options: dictionary with updates to default options
        
    Returns:
        func: function that linear interpolates for 
              beta given (alpha, p-value)
        
    Options:
        * distribution: callable
        * quadrature_bounds: (float, float)
        * quadrature_n: int
    """
    options = validate_estimation_options(options)
    quad_start, quad_stop = options['quadrature_bounds']
    quad_n = options['quadrature_n']
    
    theta, weight = _get_quadrature_points(quad_n, quad_start, quad_stop)
    distribution = options['distribution'](theta)
    distribution_x_weight = distribution * weight
    
    alpha = np.linspace(*alpha)
    beta = np.linspace(*beta)
    
    # Get the index into the array
    interp_a = interpolate.interp1d(alpha, 
                                    np.arange(alpha.size, dtype='float'),
                                    kind='linear')
    
    the_output = np.zeros((alpha.size, beta.size))
    _array_LUT(alpha, beta, theta, distribution_x_weight, the_output)
    
    func_list = list()
    for values in the_output:
        func_list.append(interpolate.interp1d(values, beta, kind='linear', 
                                                 fill_value=(beta[0], beta[-1]), 
                                                 bounds_error=False))
    
    # return function that returns beta value
    def interpolate_function(alpha_value, p_value):
        tmp = interp_a(alpha_value)
        tmpL = int(tmp)
        tmpH = int(tmp + 1)
        dx = tmp - tmpL
        return ((1 - dx) * func_list[tmpL](p_value) + 
                dx * func_list[tmpH](p_value))        
    
    return interpolate_function

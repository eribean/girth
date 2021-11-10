import numpy as np
from scipy.optimize import fminbound
from scipy.special import expit

from girth.utilities import (validate_estimation_options,
    get_true_false_counts, create_beta_LUT, INVALID_RESPONSE)
from girth.utilities.utils import _get_quadrature_points    
from girth.unidimensional.dichotomous.partial_integrals import _compute_partial_integral


__all__ = ["rasch_mml", "onepl_mml"]


def _mml_abstract(difficulty, scalar, discrimination,
                  theta, distribution):
    """ Abstraction of base functionality in separable
        mml estimation methods.

        Assumes calling function has vetted arguments
    """
    for item_ndx in range(difficulty.shape[0]):
        # pylint: disable=cell-var-from-loop
        def min_zero_local(estimate):
            temp = discrimination[item_ndx] * (theta - estimate)
            kernel = expit(temp)
            integral = kernel.dot(distribution)
            
            return np.square(integral - scalar[item_ndx])

        difficulty[item_ndx] = fminbound(min_zero_local, -6, 6, xtol=1e-4)

    return difficulty   


def rasch_mml(dataset, discrimination=1, options=None):
    """ Estimates parameters in a Rasch IRT model

    Args:
        dataset: [items x participants] matrix of True/False Values
        discrimination: scalar of discrimination used in model (default to 1)
        options: dictionary with updates to default options

    Returns:
        difficulty: (1d array) estimates of item difficulties

    Options:
        * distribution: callable
        * quadrature_bounds: (float, float)
        * quadrature_n: int
    """
    return onepl_mml(dataset, alpha=discrimination, options=options)


def onepl_mml(dataset, alpha=None, options=None):
    """ Estimates parameters in an 1PL IRT Model.

    Args:
        dataset: [items x participants] matrix of True/False Values
        alpha: [int] discrimination constraint
        options: dictionary with updates to default options

    Returns:
        discrimination: (float) estimate of test discrimination
        difficulty: (1d array) estimates of item diffiulties

    Options:
        * distribution: callable
        * quadrature_bounds: (float, float)
        * quadrature_n: int
    """
    options = validate_estimation_options(options)
    quad_start, quad_stop = options['quadrature_bounds']
    quad_n = options['quadrature_n']

    # Difficulty Estimation parameters
    n_items = dataset.shape[0]
    n_no, n_yes = get_true_false_counts(dataset)
    scalar = n_yes / (n_yes + n_no)

    unique_sets, counts = np.unique(dataset, axis=1, return_counts=True)
    invalid_response_mask = unique_sets == INVALID_RESPONSE
    unique_sets[invalid_response_mask] = 0 # For Indexing, fixed later

    discrimination = np.ones((n_items,))
    difficulty = np.zeros((n_items,))

    # Quadrature Locations
    theta, weights = _get_quadrature_points(quad_n, quad_start, quad_stop)
    distribution = options['distribution'](theta)
    distribution_x_weights = distribution * weights

    # Inline definition of cost function to minimize
    def min_func(estimate):
        discrimination[:] = estimate
        _mml_abstract(difficulty, scalar, discrimination,
                      theta, distribution_x_weights)

        partial_int = np.ones((unique_sets.shape[1], theta.size))
        for ndx in range(n_items):
            partial_int *= _compute_partial_integral(theta, difficulty[ndx], 
                                                      discrimination[ndx], 
                                                      unique_sets[ndx],
                                                      invalid_response_mask[ndx])
        partial_int *= distribution_x_weights

        # compute_integral
        otpt = np.sum(partial_int, axis=1)
        return -np.log(otpt).dot(counts)

    # Perform the minimization
    if alpha is None:  # OnePL Method
        alpha = fminbound(min_func, 0.25, 10)
    else:  # Rasch Method
        min_func(alpha)

    return {"Discrimination": alpha, 
            "Difficulty": difficulty}

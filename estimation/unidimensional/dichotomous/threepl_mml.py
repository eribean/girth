import numpy as np
from scipy.optimize import fminbound, fmin_slsqp
from scipy import integrate

from girth import (validate_estimation_options, get_true_false_counts, 
                   convert_responses_to_kernel_sign)
from girth.mml_methods import _mml_abstract
from girth.utils import _get_quadrature_points
from girth.three_pl.three_pl_utils import _compute_partial_integral_3pl


def threepl_mml(dataset, options=None):
    """ Estimates parameters in a 3PL IRT model.

    Args:
        dataset: [items x participants] matrix of True/False Values
        options: dictionary with updates to default options

    Returns:
        discrimination: (1d array) estimate of item discriminations
        difficulty: (1d array) estimates of item diffiulties
        guessing: (1d array) estimates of item guessing
    
    Options:
        * max_iteration: int
        * distribution: callable
        * quadrature_bounds: (float, float)
        * quadrature_n: int
    """
    options = validate_estimation_options(options)
    quad_start, quad_stop = options['quadrature_bounds']
    quad_n = options['quadrature_n']

    n_items = dataset.shape[0]
    n_no, n_yes = get_true_false_counts(dataset)
    scalar = n_yes / (n_yes + n_no)
    
    unique_sets, counts = np.unique(dataset, axis=1, return_counts=True)
    the_sign = convert_responses_to_kernel_sign(unique_sets)

    theta, weights = _get_quadrature_points(quad_n, quad_start, quad_stop)
    distribution = options['distribution'](theta)
    distribution_x_weights = distribution * weights

    # Perform the minimization
    discrimination = np.ones((n_items,))
    difficulty = np.zeros((n_items,))
    guessing = np.zeros((n_items,))
    
    local_scalar = np.zeros((1, 1))

    for iteration in range(options['max_iteration']):
        previous_discrimination = discrimination.copy()

        # Quadrature evaluation for values that do not change
        # This is done during the outer loop to address rounding errors
        partial_int = _compute_partial_integral_3pl(theta, difficulty,
                                                discrimination, guessing, the_sign)
        partial_int *= distribution

        for ndx in range(n_items):
            # pylint: disable=cell-var-from-loop

            # remove contribution from current item
            local_int = _compute_partial_integral_3pl(theta, difficulty[ndx, None],
                                                  discrimination[ndx, None], 
                                                  guessing[ndx, None],
                                                  the_sign[ndx, None])

            partial_int /= local_int

            def min_func_local(estimate):
                discrimination[ndx] = estimate[0]
                guessing[ndx] = estimate[1]
                
                local_scalar[0, 0] = (scalar[ndx] - guessing[ndx]) / (1. - guessing[ndx])
                _mml_abstract(difficulty[ndx, None], local_scalar,
                              discrimination[ndx, None], theta, distribution_x_weights)
                estimate_int = _compute_partial_integral_3pl(theta, difficulty[ndx, None],
                                                         discrimination[ndx, None],
                                                         guessing[ndx, None],
                                                         the_sign[ndx, None])

                estimate_int *= partial_int
                otpt = integrate.fixed_quad(
                    lambda x: estimate_int, quad_start, quad_stop, n=quad_n)[0]
                
                return -np.log(otpt).dot(counts)

            # Solve for the discrimination parameters
            initial_guess = [discrimination[ndx], guessing[ndx]]
            fmin_slsqp(min_func_local, initial_guess, 
                       bounds=([0.25, 4], [0, .33]), iprint=False)

            # Update the partial integral based on the new found values
            estimate_int = _compute_partial_integral_3pl(theta, difficulty[ndx, None],
                                                     discrimination[ndx, None],
                                                     guessing[ndx, None], 
                                                     the_sign[ndx, None])
            # update partial integral
            partial_int *= estimate_int

        if np.abs(discrimination - previous_discrimination).max() < 1e-3:
            break

    return {'Discrimination': discrimination, 
            'Difficulty': difficulty, 
            'Guessing': guessing}
import numpy as np
from scipy.optimize import fmin_slsqp
from scipy import integrate

from girth import (validate_estimation_options,
                   convert_responses_to_kernel_sign)
from girth.utils import _get_quadrature_points
from girth.three_pl.three_pl_utils import _compute_partial_integral_3pl


def threepl_full(dataset, options=None):
    """ Estimates parameters in a 2PL IRT model.

    Please use twopl_mml instead.

    Args:
        dataset: [items x participants] matrix of True/False Values
        options: dictionary with updates to default options

    Returns:
        discrimination: (1d array) estimates of item discrimination
        difficulty: (1d array) estimates of item difficulties
        guessing: (1d array) estimates of guessing parameters

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
    unique_sets, counts = np.unique(dataset, axis=1, return_counts=True)
    the_sign = convert_responses_to_kernel_sign(unique_sets)

    theta, _ = _get_quadrature_points(quad_n, quad_start, quad_stop)
    distribution = options['distribution'](theta)

    discrimination = np.ones((n_items,))
    difficulty = np.zeros((n_items,))
    guessing = np.zeros((n_items,))

    for iteration in range(options['max_iteration']):
        previous_discrimination = discrimination.copy()

        # Quadrature evaluation for values that do not change
        partial_int = _compute_partial_integral_3pl(theta, difficulty,
                                                    discrimination, guessing,
                                                    the_sign)
        partial_int *= distribution

        for item_ndx in range(n_items):
            # pylint: disable=cell-var-from-loop
            local_int = _compute_partial_integral_3pl(theta, difficulty[item_ndx, None],
                                                      discrimination[item_ndx, None],
                                                      guessing[item_ndx, None],
                                                      the_sign[item_ndx, None])

            partial_int /= local_int

            def min_func_local(estimate):
                discrimination[item_ndx] = estimate[0]
                difficulty[item_ndx] = estimate[1]
                guessing[item_ndx] = estimate[2]

                estimate_int = _compute_partial_integral_3pl(theta,
                                                             difficulty[item_ndx, None],
                                                             discrimination[item_ndx, None],
                                                             guessing[item_ndx, None],
                                                             the_sign[item_ndx, None])

                estimate_int *= partial_int
                otpt = integrate.fixed_quad(
                    lambda x: estimate_int, quad_start, quad_stop, n=quad_n)[0]

                return -np.log(otpt).dot(counts)

            # Two parameter solver that doesn't need derivatives
            initial_guess = np.concatenate((discrimination[item_ndx, None],
                                            difficulty[item_ndx, None],
                                            guessing[item_ndx, None]))

            fmin_slsqp(min_func_local, initial_guess, disp=False,
                       bounds=[(0.25, 4), (-6, 6), (0., 0.33)])

            # Update the partial integral based on the new found values
            estimate_int = _compute_partial_integral_3pl(theta, difficulty[item_ndx, None],
                                                         discrimination[item_ndx, None],
                                                         guessing[item_ndx, None],
                                                         the_sign[item_ndx, None])
            # update partial integral
            partial_int *= estimate_int

        if(np.abs(discrimination - previous_discrimination).max() < 1e-3):
            break

    return {'Discrimination': discrimination, 
            'Difficulty': difficulty, 
            'Guessing': guessing}

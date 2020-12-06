import numpy as np
from scipy import integrate, stats
from scipy.optimize import fminbound, fmin_powell, fmin_slsqp

from girth import (irt_evaluation, convert_responses_to_kernel_sign,
                   validate_estimation_options, mml_approx)
from girth.utils import _get_quadrature_points
from girth.latent_ability_distribution import LatentPDF
from girth.numba_functions import _compute_partial_integral
from girth.polytomous_utils import (condition_polytomous_response,
                                    _credit_partial_integral,
                                    _unfold_partial_integral)


def rasch_full(dataset, discrimination=1, options=None):
    """ Estimates difficulty parameters in Rash IRT model.

    Args:
        dataset: [items x participants] matrix of True/False Values
        discrimination: scalar of discrimination used in model (default to 1)
        options: dictionary with updates to default options

    Returns:
        difficulty: (1d array) difficulty estimates

    Options:
        * max_iteration: int
        * distribution: callable
        * quadrature_bounds: (float, float)
        * quadrature_n: int
    """
    return onepl_full(dataset, alpha=discrimination, options=options)


def onepl_full(dataset, alpha=None, options=None):
    """ Estimates parameters in an 1PL IRT Model.

    This function is slow, please use onepl_mml

    Args:
        dataset: [items x participants] matrix of True/False Values
        alpha: scalar of discrimination used in model (default to 1)
        options: dictionary with updates to default options

    Returns:
        discrimination: (float) estimate of test discrimination
        difficulty: (1d array) estimates of item diffiulties

    Options:
        * max_iteration: int
        * distribution: callable
        * quadrature_bounds: (float, float)
        * quadrature_n: int

    Notes:
        If alpha is supplied then this solves a Rasch model
    """
    options = validate_estimation_options(options)
    quad_start, quad_stop = options['quadrature_bounds']
    quad_n = options['quadrature_n']

    n_items = dataset.shape[0]
    unique_sets, counts = np.unique(dataset, axis=1, return_counts=True)
    the_sign = convert_responses_to_kernel_sign(unique_sets)

    theta, weights = _get_quadrature_points(quad_n, quad_start, quad_stop)
    distribution = options['distribution'](theta)
    distribution_x_weights = distribution * weights

    discrimination = np.ones((n_items,))
    difficulty = np.zeros((n_items,))
    the_output = np.zeros((the_sign.shape[1], theta.size), dtype='float64')

    def alpha_min_func(alpha_estimate):
        discrimination[:] = alpha_estimate

        for iteration in range(options['max_iteration']):
            previous_difficulty = difficulty.copy()

            # Quadrature evaluation for values that do not change
            partial_int = np.ones_like(the_output)
            for ndx in range(n_items):
                partial_int *= _compute_partial_integral(theta, difficulty[ndx], 
                                                         discrimination[ndx], the_sign[ndx],
                                                         the_output)
            partial_int *= distribution_x_weights

            for item_ndx in range(n_items):
                # pylint: disable=cell-var-from-loop

                # remove contribution from current item
                local_int = _compute_partial_integral(theta, difficulty[item_ndx],
                                                      discrimination[item_ndx],
                                                      the_sign[item_ndx], the_output)

                partial_int /= local_int

                def min_local_func(beta_estimate):
                    difficulty[item_ndx] = beta_estimate

                    estimate_int = _compute_partial_integral(theta, difficulty[item_ndx],
                                                             discrimination[item_ndx],
                                                             the_sign[item_ndx], the_output)

                    estimate_int *= partial_int
                    otpt = np.sum(estimate_int, axis=1)
                    return -np.log(otpt).dot(counts)

                fminbound(min_local_func, -4, 4)

                # Update the partial integral based on the new found values
                estimate_int = _compute_partial_integral(theta, difficulty[item_ndx],
                                                         discrimination[item_ndx],
                                                         the_sign[item_ndx], the_output)
                # update partial integral
                partial_int *= estimate_int

            if(np.abs(previous_difficulty - difficulty).max() < 1e-3):
                break

        cost = np.sum(partial_int, axis=1)
        return -np.log(cost).dot(counts)

    if alpha is None:  # OnePl Solver
        alpha = fminbound(alpha_min_func, 0.1, 4)
    else:  # Rasch Solver
        alpha_min_func(alpha)

    return {'Discrimination': alpha, 
            'Difficulty': difficulty}


def twopl_full(dataset, options=None):
    """ Estimates parameters in a 2PL IRT model.

    Please use twopl_mml instead.

    Args:
        dataset: [items x participants] matrix of True/False Values
        options: dictionary with updates to default options

    Returns:
        discrimination: (1d array) estimates of item discrimination
        difficulty: (1d array) estimates of item difficulties

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

    theta, weights = _get_quadrature_points(quad_n, quad_start, quad_stop)
    distribution = options['distribution'](theta)
    distribution_x_weights = distribution * weights

    discrimination = np.ones((n_items,))
    difficulty = np.zeros((n_items,))
    the_output = np.zeros((the_sign.shape[1], theta.size), dtype='float64')

    for iteration in range(options['max_iteration']):
        previous_discrimination = discrimination.copy()

        # Quadrature evaluation for values that do not change
        partial_int = np.ones_like(the_output)
        for ndx in range(n_items):
            partial_int *= _compute_partial_integral(theta, difficulty[ndx], 
                                                     discrimination[ndx], the_sign[ndx],
                                                     the_output)
        partial_int *= distribution_x_weights

        for item_ndx in range(n_items):
            # pylint: disable=cell-var-from-loop
            local_int = _compute_partial_integral(theta, difficulty[item_ndx],
                                                  discrimination[item_ndx],
                                                  the_sign[item_ndx], the_output)

            partial_int /= local_int

            def min_func_local(estimate):
                discrimination[item_ndx] = estimate[0]
                difficulty[item_ndx] = estimate[1]

                estimate_int = _compute_partial_integral(theta,
                                                         difficulty[item_ndx],
                                                         discrimination[item_ndx],
                                                         the_sign[item_ndx], the_output)

                estimate_int *= partial_int
                otpt = np.sum(estimate_int, axis=1)

                return -np.log(otpt).dot(counts)

            # Two parameter solver that doesn't need derivatives
            initial_guess = np.concatenate((discrimination[item_ndx, None],
                                            difficulty[item_ndx, None]))
            fmin_slsqp(min_func_local, initial_guess, disp=False,
                       bounds=[(0.25, 4), (-4, 4)])

            # Update the partial integral based on the new found values
            estimate_int = _compute_partial_integral(theta, difficulty[item_ndx],
                                                     discrimination[item_ndx],
                                                     the_sign[item_ndx], the_output)
            # update partial integral
            partial_int *= estimate_int

        if(np.abs(discrimination - previous_discrimination).max() < 1e-3):
            break

    return {'Discrimination': discrimination, 
            'Difficulty': difficulty}


def pcm_mml(dataset, options=None):
    """Estimate parameters for partial credit model.

    Estimate the discrimination and difficulty parameters for
    the partial credit model using marginal maximum likelihood.

    Args:
        dataset: [n_items, n_participants] 2d array of measured responses
        options: dictionary with updates to default options

    Returns:
        discrimination: (1d array) estimates of item discrimination
        difficulty: (2d array) estimates of item difficulties x item thresholds

    Options:
        * estimate_distribution: Boolean    
        * number_of_samples: int >= 5       
        * max_iteration: int
        * distribution: callable
        * quadrature_bounds: (float, float)
        * quadrature_n: int
    """
    options = validate_estimation_options(options)

    responses, item_counts = condition_polytomous_response(dataset, trim_ends=False,
                                                           _reference=0.0)
    n_items = responses.shape[0]

    # Quadrature Locations
    latent_pdf = LatentPDF(options)
    theta = latent_pdf.quadrature_locations

    # Initialize difficulty parameters for estimation
    betas = np.full((n_items, item_counts.max()), np.nan)
    discrimination = np.ones((n_items,))
    partial_int = np.ones((responses.shape[1], theta.size))

    # Not all items need to have the same
    # number of response categories
    betas[:, 0] = 0
    for ndx in range(n_items):
        betas[ndx, 1:item_counts[ndx]] = np.linspace(-1, 1, item_counts[ndx]-1)

    #############
    # 1. Start the iteration loop
    # 2. Estimate Dicriminatin/Difficulty Jointly
    # 3. Integrate of theta
    # 4. minimize and repeat
    #############
    for iteration in range(options['max_iteration']):
        previous_discrimination = discrimination.copy()
        previous_betas = betas.copy()

        # Quadrature evaluation for values that do not change
        # This is done during the outer loop to address rounding errors
        # and for speed

        partial_int = np.ones((responses.shape[1], theta.size))
        for item_ndx in range(n_items):
            partial_int *= _credit_partial_integral(theta, betas[item_ndx],
                                                    discrimination[item_ndx],
                                                    responses[item_ndx])
        # Estimate the distribution if requested
        distribution_x_weight = latent_pdf(partial_int, iteration)
        partial_int *= distribution_x_weight        

        # Loop over each item and solve for the alpha / beta parameters
        for item_ndx in range(n_items):
            # pylint: disable=cell-var-from-loop
            item_length = item_counts[item_ndx]
            new_betas = np.zeros((item_length))

            # Remove the previous output
            old_values = _credit_partial_integral(theta, previous_betas[item_ndx],
                                                  previous_discrimination[item_ndx],
                                                  responses[item_ndx])
            partial_int /= old_values

            def _local_min_func(estimate):
                new_betas[1:] = estimate[1:]
                new_values = _credit_partial_integral(theta, new_betas,
                                                      estimate[0],
                                                      responses[item_ndx])
                new_values *= partial_int
                otpt = np.sum(new_values, axis=1)
                return -np.log(otpt).sum()

            # Initial Guess of Item Parameters
            initial_guess = np.concatenate(([discrimination[item_ndx]],
                                            betas[item_ndx, 1:item_length]))

            otpt = fmin_slsqp(_local_min_func, initial_guess,
                              disp=False,
                              bounds=[(.25, 4)] + [(-6, 6)] * (item_length - 1))

            discrimination[item_ndx] = otpt[0]
            betas[item_ndx, 1:item_length] = otpt[1:]

            new_values = _credit_partial_integral(theta, betas[item_ndx],
                                                  discrimination[item_ndx],
                                                  responses[item_ndx])

            partial_int *= new_values

        if np.abs(previous_discrimination - discrimination).max() < 1e-3:
            break

    # TODO:  look where missing values are and place NAN there instead
    # of appending them to the end
    # Compute statistics for final iteration
    partial_int /= distribution_x_weight
    null_metrics = latent_pdf.compute_metrics(partial_int, latent_pdf.null_distribution * 
                                             latent_pdf.weights, 0)
    full_metrics = latent_pdf.compute_metrics(partial_int, distribution_x_weight,
                                             latent_pdf.n_points-3)

    return {'Discrimination': discrimination,
            'Difficulty': betas[:, 1:],
            'LatentPDf': latent_pdf,
            'AIC': {'final': full_metrics[0],
                    'null': null_metrics[0],
                    'delta': null_metrics[0] - full_metrics[0]},
            'BIC': {'final': full_metrics[1],
                    'null': null_metrics[1],
                    'delta': null_metrics[1] - full_metrics[1]}}


def gum_mml(dataset, options=None):
    """Estimate parameters for graded unfolding model.

    Estimate the discrimination, delta and threshold parameters for
    the graded unfolding model using marginal maximum likelihood.

    Args:
        dataset: [n_items, n_participants] 2d array of measured responses
        options: dictionary with updates to default options

    Returns:
        discrimination: (1d array) estimates of item discrimination
        delta: (1d array) estimates of item folding values
        difficulty: (2d array) estimates of item thresholds

    Options:
        * estimate_distribution: Boolean    
        * number_of_samples: int >= 5       
        * max_iteration: int
        * distribution: callable
        * quadrature_bounds: (float, float)
        * quadrature_n: int
    """
    options = validate_estimation_options(options)

    responses, item_counts = condition_polytomous_response(dataset, trim_ends=False,
                                                           _reference=0.0)
    n_items = responses.shape[0]

    # Interpolation Locations
    # Quadrature Locations
    latent_pdf = LatentPDF(options)
    theta = latent_pdf.quadrature_locations

    # Initialize item parameters for iterations
    discrimination = np.ones((n_items,))
    betas = np.full((n_items, item_counts.max() - 1), np.nan)
    delta = np.zeros((n_items,))
    partial_int = np.ones((responses.shape[1], theta.size))

    # Set initial estimates to evenly spaced
    for ndx in range(n_items):
        item_length = item_counts[ndx] - 1
        betas[ndx, :item_length] = np.linspace(-1, 1, item_length)

    # This is the index associated with "folding" about the center
    fold_span = ((item_counts[:, None] - 0.5) -
                 np.arange(betas.shape[1] + 1)[None, :])

    #############
    # 1. Start the iteration loop
    # 2. Estimate Dicriminatin/Difficulty Jointly
    # 3. Integrate of theta
    # 4. minimize and repeat
    #############
    for iteration in range(options['max_iteration']):
        previous_discrimination = discrimination.copy()
        previous_betas = betas.copy()
        previous_delta = delta.copy()

        # Quadrature evaluation for values that do not change
        # This is done during the outer loop to address rounding errors
        # and for speed
        partial_int = np.ones((responses.shape[1], theta.size))
        for item_ndx in range(n_items):
            partial_int *= _unfold_partial_integral(theta, delta[item_ndx],
                                                    betas[item_ndx],
                                                    discrimination[item_ndx],
                                                    fold_span[item_ndx],
                                                    responses[item_ndx])
        # Estimate the distribution if requested
        distribution_x_weight = latent_pdf(partial_int, iteration)
        partial_int *= distribution_x_weight

        # Loop over each item and solve for the alpha / beta parameters
        for item_ndx in range(n_items):
            # pylint: disable=cell-var-from-loop
            item_length = item_counts[item_ndx] - 1

            # Remove the previous output
            old_values = _unfold_partial_integral(theta, previous_delta[item_ndx],
                                                  previous_betas[item_ndx],
                                                  previous_discrimination[item_ndx],
                                                  fold_span[item_ndx],
                                                  responses[item_ndx])
            partial_int /= old_values

            def _local_min_func(estimate):
                new_betas = estimate[2:]
                new_values = _unfold_partial_integral(theta, estimate[1],
                                                      new_betas,
                                                      estimate[0], fold_span[item_ndx],
                                                      responses[item_ndx])

                new_values *= partial_int
                otpt = np.sum(new_values, axis=1)
                return -np.log(otpt).sum()

            # Initial Guess of Item Parameters
            initial_guess = np.concatenate(([discrimination[item_ndx]],
                                            [delta[item_ndx]],
                                            betas[item_ndx]))

            otpt = fmin_slsqp(_local_min_func, initial_guess,
                              disp=False,
                              bounds=[(.25, 4)] + [(-2, 2)] + [(-6, 6)] * item_length)

            discrimination[item_ndx] = otpt[0]
            delta[item_ndx] = otpt[1]
            betas[item_ndx, :] = otpt[2:]

            new_values = _unfold_partial_integral(theta, delta[item_ndx],
                                                  betas[item_ndx],
                                                  discrimination[item_ndx],
                                                  fold_span[item_ndx],
                                                  responses[item_ndx])

            partial_int *= new_values

        if np.abs(previous_discrimination - discrimination).max() < 1e-3:
            break
    
    # Compute statistics for final iteration
    partial_int /= distribution_x_weight
    null_metrics = latent_pdf.compute_metrics(partial_int, latent_pdf.null_distribution * 
                                             latent_pdf.weights, 0)
    full_metrics = latent_pdf.compute_metrics(partial_int, distribution_x_weight,
                                             latent_pdf.n_points-3)
    
    return {'Discrimination': discrimination, 
            'Difficulties': np.c_[betas, np.zeros((delta.size,)), 
                                  -betas[:, ::-1]] + delta[:, None],
            'Delta': delta,
            'Tau': betas,
            'LatentPDf': latent_pdf,
            'AIC': {'final': full_metrics[0],
                    'null': null_metrics[0],
                    'delta': null_metrics[0] - full_metrics[0]},
            'BIC': {'final': full_metrics[1],
                    'null': null_metrics[1],
                    'delta': null_metrics[1] - full_metrics[1]}}            

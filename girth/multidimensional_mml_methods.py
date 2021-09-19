from itertools import repeat, product

import numpy as np
from scipy.optimize import fmin_slsqp, fminbound

from girth import (condition_polytomous_response,
                   validate_estimation_options)

from girth.utils import create_beta_LUT, INVALID_RESPONSE
from girth.latent_ability_distribution import LatentPDF
from girth.polytomous_utils import (_graded_partial_integral_md, _solve_for_constants,
                                    _solve_integral_equations, 
                                    _solve_integral_equations_LUT)


__all__ = ["multidimensional_twopl_mml", "multidimensional_grm_mml"]


def _build_einsum_string(n_factors):
    """Builds a string for computing a tensor product."""
    if n_factors > 10:
        raise ValueError("Number of factors must be less than 10.")

    values = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'][:n_factors]

    return ", ".join(values) + " -> " + "".join(values)


def multidimensional_twopl_mml(dataset, n_factors, options=None):
    """Estimate parameters for multidimensional twopl model.

    Estimate the discrimination and difficulty parameters for
    a two parameter logistic model using marginal maximum likelihood.

    Args:
        dataset: [n_items, n_participants] 2d array of measured responses
        n_factors: (int) number of factors to extract
        options: dictionary with updates to default options

    Returns:
        results_dictionary:
        * Discrimination: (1d array) estimate of item discriminations
        * Difficulty: (2d array) estimates of item diffiulties by item thresholds
        * LatentPDF: (object) contains information about the pdf
        * AIC: (dictionary) null model and final model AIC value
        * BIC: (dictionary) null model and final model BIC value

    Options:
        * use_LUT: boolean
        * max_iteration: int
        * quadrature_bounds: (float, float)
        * quadrature_n: int
    """    
    results = multidimensional_grm_mml(dataset, n_factors, options)
    results['Difficulty'] = results['Difficulty'].squeeze()
    return results    


def multidimensional_grm_mml(dataset, n_factors, options=None):
    """Estimate parameters for multidimensional graded response model.

    Estimate the discrimination and difficulty parameters for
    a graded response model using marginal maximum likelihood.

    Args:
        dataset: [n_items, n_participants] 2d array of measured responses
        n_factors: (int) number of factors to extract
        options: dictionary with updates to default options

    Returns:
        results_dictionary:
        * Discrimination: (2d array) estimate of item discriminations
        * Difficulty: (2d array) estimates of item diffiulties by item thresholds
        * LatentPDF: (object) contains information about the pdf
        * AIC: (dictionary) null model and final model AIC value
        * BIC: (dictionary) null model and final model BIC value

    Options:
        * use_LUT: boolean
        * max_iteration: int
        * quadrature_bounds: (float, float)
        * quadrature_n: int
    """
    if n_factors < 2:
        raise AssertionError("Number of factors specified must be greater than 1.")

    options = validate_estimation_options(options)

    cpr_result = condition_polytomous_response(dataset, trim_ends=False)
    responses, item_counts, valid_response_mask = cpr_result
    invalid_response_mask = ~valid_response_mask
    
    n_items = responses.shape[0]
    
    # Should we use the LUT
    _integral_func = _solve_integral_equations
    _interp_func = None
    if options['use_LUT']:
        _integral_func = _solve_integral_equations_LUT
        _interp_func = create_beta_LUT((.15, 6, 1000), (-6, 6, 500), options)
    
    # Multi-dimensional Quadrature Locations
    latent_pdf = LatentPDF(options)
    theta = np.asarray(list(product(*repeat(latent_pdf.quadrature_locations, 
                                            n_factors)))).T

    dist_x_weight = latent_pdf(None, 0)
    einsum_string = _build_einsum_string(n_factors)
    distribution_x_weight = np.einsum(einsum_string, 
                                      *repeat(dist_x_weight, n_factors)).flatten()

    # Compute the values needed for integral equations
    integral_counts = list()
    for ndx in range(n_items):
        temp_output = _solve_for_constants(responses[ndx, valid_response_mask[ndx]])
        integral_counts.append(temp_output)

    # Initialize difficulty parameters for estimation
    betas = np.full((item_counts.sum(),), 10000.0)
    cumulative_item_counts = item_counts.cumsum()
    start_indices = np.roll(cumulative_item_counts, 1)
    start_indices[0] = 0

    for ndx in range(n_items):
        end_ndx = cumulative_item_counts[ndx]
        start_ndx = start_indices[ndx] + 1
        betas[start_ndx:end_ndx] = np.linspace(-1, 1,
                                               item_counts[ndx] - 1)
    betas_roll = np.roll(betas, -1)
    betas_roll[cumulative_item_counts-1] = -10000
    
    # Multi-dimensional discrimination
    discrimination = np.zeros((betas.shape[0], n_factors))
    discrimination[:, 0] = 1.0
    
    # Set invalid index to zero, this allows minimal
    # changes for invalid data and it is corrected
    # during integration
    responses[invalid_response_mask] = 0

    # Set the boundaries for searching
    bounds = [[(-3, 3) for _ in range(n_factors)] 
              for __ in range(n_items)]
    for ndx1 in range(n_factors-1):
        ndx2 = n_items - ndx1 - 1
        bounds[ndx2][ndx1] = (.15, 4)
        bounds[ndx2][ndx1+1:] = ()

    #############
    # 1. Start the iteration loop
    # 2. estimate discrimination
    # 3. solve for difficulties
    # 4. minimize and repeat
    #############
    for iteration in range(options['max_iteration']):
        previous_discrimination = discrimination.copy()
        previous_betas = betas.copy()
        previous_betas_roll = betas_roll.copy()

        # Quadrature evaluation for values that do not change
        # This is done during the outer loop to address rounding errors
        partial_int = np.ones((responses.shape[1], theta.shape[1]))
        for item_ndx in range(n_items):
            partial_int *= _graded_partial_integral_md(theta, betas, betas_roll,
                                                       discrimination,
                                                       responses[item_ndx],
                                                       invalid_response_mask[item_ndx])
        
        # Add the weighting
        partial_int *= distribution_x_weight

        for item_ndx in range(n_items):
            # pylint: disable=cell-var-from-loop

            # Indices into linearized difficulty parameters
            start_ndx = start_indices[item_ndx]
            end_ndx = cumulative_item_counts[item_ndx]
            discrimination_length = len(bounds[item_ndx])

            old_values = _graded_partial_integral_md(theta, previous_betas,
                                                     previous_betas_roll,
                                                     previous_discrimination,
                                                     responses[item_ndx],
                                                     invalid_response_mask[item_ndx])
            partial_int /= (old_values + 1e-23)

            def _local_min_func(estimate):
                # Solve integrals for diffiulty estimates
                univariate_estimate = np.sqrt(np.square(estimate).sum()).clip(.15, 6)
                new_betas = _integral_func(univariate_estimate, integral_counts[item_ndx],
                                           dist_x_weight, latent_pdf.quadrature_locations, 
                                           _interp_func) * -univariate_estimate
                    
                betas[start_ndx+1:end_ndx] = new_betas
                betas_roll[start_ndx:end_ndx-1] = new_betas
                discrimination[start_ndx:end_ndx, :discrimination_length] = estimate

                new_values = _graded_partial_integral_md(theta, betas, betas_roll,
                                                         discrimination,
                                                         responses[item_ndx],
                                                         invalid_response_mask[item_ndx])

                new_values *= partial_int
                otpt = np.sum(new_values, axis=1)

                return -np.log(otpt).sum()

            # Univariate minimization for discrimination parameter
            if item_ndx == (n_items-1):
                fminbound(_local_min_func, 0.2, 5.0)
            else:
                initial_guess = discrimination[start_ndx, :][:discrimination_length]
                fmin_slsqp(_local_min_func, initial_guess,
                           disp=False,
                           bounds=bounds[item_ndx])

            new_values = _graded_partial_integral_md(theta, betas, betas_roll,
                                                     discrimination,
                                                     responses[item_ndx],
                                                     invalid_response_mask[item_ndx])

            partial_int *= new_values

        if np.abs(previous_discrimination - discrimination).max() < 1e-3:
            break

    # Recompute partial int for later calculations
    partial_int = np.ones((responses.shape[1], theta.shape[1]))
    for item_ndx in range(n_items):
        partial_int *= _graded_partial_integral_md(theta, betas, betas_roll,
                                                   discrimination,
                                                   responses[item_ndx],
                                                   invalid_response_mask[item_ndx])

    # Trim difficulties to conform to standard output
    # TODO:  look where missing values are and place NAN there instead
    # of appending them to the end
    output_betas = np.full((n_items, item_counts.max()-1), np.nan)
    for ndx, (start_ndx, end_ndx) in enumerate(zip(start_indices, cumulative_item_counts)):
        output_betas[ndx, :end_ndx-start_ndx-1] = betas[start_ndx+1:end_ndx]
    
    # Compute statistics for final iteration
    full_metrics = latent_pdf.compute_metrics(partial_int, distribution_x_weight,
                                             latent_pdf.n_points-3)
    otpt = np.sum(partial_int * distribution_x_weight, axis=1)

    # Ability estimates
    # eap_abilities = _ability_eap_abstract(partial_int, distribution_x_weight, theta)

    return {'Discrimination': discrimination[start_indices, :],
            'Difficulty': output_betas,
            'Ability': None,
            'LatentPDF': latent_pdf,            
            'AIC': full_metrics[0],
            'BIC': full_metrics[1],
            'LL': np.log(otpt).sum()
           }
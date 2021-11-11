import numpy as np
from scipy.optimize import fminbound
from scipy.special import expit

from girth import (condition_polytomous_response, validate_estimation_options,
                   create_beta_LUT)
from girth.utilities import INVALID_RESPONSE
from girth.utilities.latent_ability_distribution import LatentPDF
from girth.utilities.polytomous_utils import (
    INVALID_RESPONSE, _solve_for_constants,
    _solve_integral_equations, _solve_integral_equations_LUT)
from girth.unidimensional.polytomous.partial_integrals_poly import _graded_partial_integral
from girth.unidimensional.polytomous.ability_estimation_poly import _ability_eap_abstract


__all__ = ["grm_mml"]


def grm_mml(dataset, options=None):
    """Estimate parameters for graded response model.

    Estimate the discrimination and difficulty parameters for
    a graded response model using marginal maximum likelihood.

    Args:
        dataset: [n_items, n_participants] 2d array of measured responses
        options: dictionary with updates to default options

    Returns:
        results_dictionary:
        * Discrimination: (1d array) estimate of item discriminations
        * Difficulty: (2d array) estimates of item diffiulties by item thresholds
        * LatentPDF: (object) contains information about the pdf
        * AIC: (dictionary) null model and final model AIC value
        * BIC: (dictionary) null model and final model BIC value

    Options:
        * estimate_distribution: Boolean    
        * number_of_samples: int >= 5    
        * use_LUT: boolean
        * max_iteration: int
        * distribution: callable
        * quadrature_bounds: (float, float)
        * quadrature_n: int
    """
    options = validate_estimation_options(options)

    cpr_result = condition_polytomous_response(dataset, trim_ends=False)
    responses, item_counts, valid_response_mask = cpr_result
    invalid_response_mask = ~valid_response_mask
    n_items = responses.shape[0]
    
    # Initialize difficulty parameters for estimation
    betas = np.full((item_counts.sum(),), -10000.0)
    discrimination = np.ones_like(betas)
    cumulative_item_counts = item_counts.cumsum()
    start_indices = np.roll(cumulative_item_counts, 1)
    start_indices[0] = 0

    for ndx in range(n_items):
        end_ndx = cumulative_item_counts[ndx]
        start_ndx = start_indices[ndx] + 1
        betas[start_ndx:end_ndx] = np.linspace(-1, 1,
                                               item_counts[ndx] - 1)
    betas_roll = np.roll(betas, -1)
    betas_roll[cumulative_item_counts-1] = 10000

    # Should we use the LUT
    _integral_func = _solve_integral_equations
    _interp_func = None
    if options['use_LUT']:
        _integral_func = _solve_integral_equations_LUT
        _interp_func = create_beta_LUT((.15, 5.05, 500), (-6, 6, 500), options)
    
    # Quadrature Locations
    latent_pdf = LatentPDF(options)
    theta = latent_pdf.quadrature_locations

    # Compute the values needed for integral equations
    integral_counts = list()
    for ndx in range(n_items):
        temp_output = _solve_for_constants(responses[ndx, valid_response_mask[ndx]])
        integral_counts.append(temp_output)

    # Set invalid index to zero, this allows minimal
    # changes for invalid data and it is corrected
    # during integration
    responses[invalid_response_mask] = 0
    
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
        partial_int = np.ones((responses.shape[1], theta.size))
        for item_ndx in range(n_items):
            partial_int *= _graded_partial_integral(theta, betas, betas_roll,
                                                    discrimination,
                                                    responses[item_ndx],
                                                    invalid_response_mask[item_ndx])
        
        # Estimate the distribution if requested
        distribution_x_weight = latent_pdf(partial_int, iteration)
        partial_int *= distribution_x_weight
        
        # Update the lookup table if necessary
        if (options['use_LUT'] and options['estimate_distribution'] and
            iteration > 0):
            new_options = dict(options)
            new_options.update({'distribution': latent_pdf.cubic_splines[-1]})

            _interp_func = create_beta_LUT((.15, 5.05, 500), 
                                           (-6, 6, 500), 
                                           new_options)

        for item_ndx in range(n_items):
            # pylint: disable=cell-var-from-loop

            # Indices into linearized difficulty parameters
            start_ndx = start_indices[item_ndx]
            end_ndx = cumulative_item_counts[item_ndx]

            old_values = _graded_partial_integral(theta, previous_betas,
                                                  previous_betas_roll,
                                                  previous_discrimination,
                                                  responses[item_ndx],
                                                  invalid_response_mask[item_ndx])
            partial_int /= old_values

            def _local_min_func(estimate):
                # Solve integrals for diffiulty estimates
                new_betas = _integral_func(estimate, integral_counts[item_ndx],
                                           distribution_x_weight, theta, 
                                           _interp_func)
                    
                betas[start_ndx+1:end_ndx] = new_betas
                betas_roll[start_ndx:end_ndx-1] = new_betas
                discrimination[start_ndx:end_ndx] = estimate

                new_values = _graded_partial_integral(theta, betas, betas_roll,
                                                      discrimination,
                                                      responses[item_ndx],
                                                      invalid_response_mask[item_ndx])

                new_values *= partial_int
                otpt = np.sum(new_values, axis=1)

                return -np.log(otpt).sum()

            # Univariate minimization for discrimination parameter
            fminbound(_local_min_func, 0.2, 5.0)

            new_values = _graded_partial_integral(theta, betas, betas_roll,
                                                  discrimination,
                                                  responses[item_ndx],
                                                  invalid_response_mask[item_ndx])

            partial_int *= new_values

        if np.abs(previous_discrimination - discrimination).max() < 1e-3:
            break
            
    # Recompute partial int for later calculations
    partial_int = np.ones((responses.shape[1], theta.size))
    for item_ndx in range(n_items):
        partial_int *= _graded_partial_integral(theta, betas, betas_roll,
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
    null_metrics = latent_pdf.compute_metrics(partial_int, latent_pdf.null_distribution * 
                                             latent_pdf.weights, 0)
    full_metrics = latent_pdf.compute_metrics(partial_int, distribution_x_weight,
                                             latent_pdf.n_points-3)

    # Ability estimates
    eap_abilities = _ability_eap_abstract(partial_int, distribution_x_weight, theta)

    return {'Discrimination': discrimination[start_indices],
            'Difficulty': output_betas,
            'Ability': eap_abilities,
            'LatentPDF': latent_pdf,
            'AIC': {'final': full_metrics[0],
                    'null': null_metrics[0],
                    'delta': null_metrics[0] - full_metrics[0]},
            'BIC': {'final': full_metrics[1],
                    'null': null_metrics[1],
                    'delta': null_metrics[1] - full_metrics[1]}}

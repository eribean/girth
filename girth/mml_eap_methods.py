import numpy as np
from scipy import stats

from girth import (condition_polytomous_response, validate_estimation_options,
                   convert_responses_to_kernel_sign)
from girth.utils import create_beta_LUT
from girth.latent_ability_distribution import LatentPDF
from girth.polytomous_utils import (_graded_partial_integral, _solve_for_constants,
                                    _solve_integral_equations_LUT)
from girth.ability_methods import _ability_eap_abstract


__all__ = ["twopl_mml_eap", "grm_mml_eap"]


def twopl_mml_eap(dataset, options=None):
    """Estimate parameters for a two parameter logistic model.

    Estimate the discrimination and difficulty parameters for
    a two parameter logistic model using a mixed Bayesian / 
    Marginal Maximum Likelihood algorithm, good for small 
    sample sizes

    Args:
        dataset: [n_items, n_participants] 2d array of measured responses
        options: dictionary with updates to default options

    Returns:
        results_dictionary:
        * Discrimination: (1d array) estimate of item discriminations
        * Difficulty: (1d array) estimates of item difficulties
        * LatentPDF: (object) contains information about the pdf
        * Rayleigh_Scale: (int) Rayleigh scale value of the discrimination prior
        * AIC: (dictionary) null model and final model AIC value
        * BIC: (dictionary) null model and final model BIC value

    Options:
        * estimate_distribution: Boolean    
        * number_of_samples: int >= 5    
        * max_iteration: int
        * distribution: callable
        * quadrature_bounds: (float, float)
        * quadrature_n: int
        * hyper_quadrature_n: int
    """
    result = grm_mml_eap(dataset.astype('int'), options)
    result['Difficulty'] = result['Difficulty'].squeeze()
    return result


def grm_mml_eap(dataset, options=None):
    """Estimate parameters for graded response model.

    Estimate the discrimination and difficulty parameters for
    a graded response model using a mixed Bayesian / Marginal Maximum
    Likelihood algorithm, good for small sample sizes

    Args:
        dataset: [n_items, n_participants] 2d array of measured responses
        options: dictionary with updates to default options

    Returns:
        results_dictionary:
        * Discrimination: (1d array) estimate of item discriminations
        * Difficulty: (2d array) estimates of item difficulties by item thresholds
        * LatentPDF: (object) contains information about the pdf
        * Rayleigh_Scale: (int) Rayleigh scale value of the discrimination prior
        * AIC: (dictionary) null model and final model AIC value
        * BIC: (dictionary) null model and final model BIC value

    Options:
        * estimate_distribution: Boolean    
        * number_of_samples: int >= 5    
        * max_iteration: int
        * distribution: callable
        * quadrature_bounds: (float, float)
        * quadrature_n: int
        * hyper_quadrature_n: int
    """
    options = validate_estimation_options(options)

    cpr_result = condition_polytomous_response(dataset, trim_ends=False)
    responses, item_counts, valid_response_mask = cpr_result
    invalid_response_mask = ~valid_response_mask

    n_items = responses.shape[0]
    
    # Only use LUT
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
    
    # Set invalid index to zero, this allows minimal
    # changes for invalid data and it is corrected
    # during integration
    responses[invalid_response_mask] = 0    

    # Prior Parameters
    ray_scale = 1.0
    eap_options = {'distribution': stats.rayleigh(loc=.25, scale=ray_scale).pdf,
                   'quadrature_n': options['hyper_quadrature_n'], 
                   'quadrature_bounds': (0.25, 5)}
    prior_pdf = LatentPDF(eap_options) 
    alpha_evaluation = np.zeros((eap_options['quadrature_n'],))

    # Meta-Prior Parameter
    hyper_options = {'distribution': stats.lognorm(loc=0, s=0.25).pdf,
                     'quadrature_n': options['hyper_quadrature_n'], 
                     'quadrature_bounds': (0.1, 5)}
    hyper_pdf = LatentPDF(hyper_options) 
    hyper_evaluation = np.zeros((hyper_options['quadrature_n'],))
    base_hyper = (hyper_pdf.weights * hyper_pdf.null_distribution).astype('float128')
    linear_hyper = base_hyper * hyper_pdf.quadrature_locations

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
        if (options['estimate_distribution'] and iteration > 0):
            new_options = dict(options)
            new_options.update({'distribution': latent_pdf.cubic_splines[-1]})

            _interp_func = create_beta_LUT((.15, 5.05, 500), 
                                           (-6, 6, 500), 
                                           new_options)
        
        # EAP Discrimination Parameter
        discrimination_pdf = stats.rayleigh(loc=0.25, scale=ray_scale).pdf
        base_alpha = (prior_pdf.weights * 
                      discrimination_pdf(prior_pdf.quadrature_locations)).astype('float128')
        linear_alpha = (base_alpha * prior_pdf.quadrature_locations).astype('float128')

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
                
                return np.log(otpt.clip(1e-313, np.inf)).sum()

            # Mean Discrimination Value
            for ndx, disc_location in enumerate(prior_pdf.quadrature_locations):
                alpha_evaluation[ndx] = _local_min_func(disc_location)

            alpha_evaluation -= alpha_evaluation.max()
            total_probability = np.exp(alpha_evaluation.astype('float128'))
            numerator = np.sum(total_probability * linear_alpha)
            denominator = np.sum(total_probability * base_alpha)
            
            alpha_eap = numerator / denominator

            # Reset the Value the updated discrimination estimation
            _local_min_func(alpha_eap.astype('float64'))
     
            new_values = _graded_partial_integral(theta, betas, betas_roll,
                                                  discrimination,
                                                  responses[item_ndx],
                                                  invalid_response_mask[item_ndx])

            partial_int *= new_values
            
        # Compute the Hyper prior mean value
        for ndx, scale_value in enumerate(hyper_pdf.quadrature_locations):
            temp_distribution = stats.rayleigh(loc=0.25, scale=scale_value).pdf
            hyper_evaluation[ndx] = np.log(temp_distribution(discrimination) + 
                                           1e-313).sum()
            
        hyper_evaluation -= hyper_evaluation.max()
        hyper_evaluation = np.exp(hyper_evaluation.astype('float128'))
        ray_scale = (np.sum(hyper_evaluation * linear_hyper) / 
                     np.sum(hyper_evaluation * base_hyper)).astype('float64')

        # Check Termination Criterion                     
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
            'Rayleigh_Scale': ray_scale,
            'AIC': {'final': full_metrics[0],
                    'null': null_metrics[0],
                    'delta': null_metrics[0] - full_metrics[0]},
            'BIC': {'final': full_metrics[1],
                    'null': null_metrics[1],
                    'delta': null_metrics[1] - full_metrics[1]}}
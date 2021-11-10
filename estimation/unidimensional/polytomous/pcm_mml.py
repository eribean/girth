import numpy as np
from scipy.optimize import fmin_slsqp

from girth.utilities import (condition_polytomous_response,
    INVALID_RESPONSE, validate_estimation_options)
from girth.utilities.latent_ability_distribution import LatentPDF
from girth.unidimensional.polytomous.partial_integrals_poly import _credit_partial_integral
from girth.unidimensional.polytomous.ability_estimation_poly import _ability_eap_abstract


__all__ = ["pcm_mml"]


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

    cpr_result =  condition_polytomous_response(dataset, trim_ends=False, _reference=0.0)
    responses, item_counts, valid_response_mask = cpr_result
    invalid_response_mask = ~valid_response_mask

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

    # Set invalid index to zero, this allows minimal
    # changes for invalid data and it is corrected
    # during integration
    responses[invalid_response_mask] = 0

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
                                                    responses[item_ndx],
                                                    invalid_response_mask[item_ndx])
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
                                                  responses[item_ndx],
                                                  invalid_response_mask[item_ndx])
            partial_int /= old_values

            def _local_min_func(estimate):
                new_betas[1:] = estimate[1:]
                new_values = _credit_partial_integral(theta, new_betas,
                                                      estimate[0],
                                                      responses[item_ndx],
                                                      invalid_response_mask[item_ndx])
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
                                                  responses[item_ndx],
                                                  invalid_response_mask[item_ndx])

            partial_int *= new_values

        if np.abs(previous_discrimination - discrimination).max() < 1e-3:
            break

    # Recompute partial int for later calculations
    partial_int = np.ones((responses.shape[1], theta.size))
    for item_ndx in range(n_items):
        partial_int *= _credit_partial_integral(theta, betas[item_ndx],
                                                discrimination[item_ndx],
                                                responses[item_ndx],
                                                invalid_response_mask[item_ndx])

    # TODO:  look where missing values are and place NAN there instead
    # of appending them to the end
    # Compute statistics for final iteration
    null_metrics = latent_pdf.compute_metrics(partial_int, latent_pdf.null_distribution * 
                                              latent_pdf.weights, 0)
    full_metrics = latent_pdf.compute_metrics(partial_int, distribution_x_weight,
                                              latent_pdf.n_points-3)

    # Ability estimates
    eap_abilities = _ability_eap_abstract(partial_int, distribution_x_weight, theta)

    return {'Discrimination': discrimination,
            'Difficulty': betas[:, 1:],
            'Ability': eap_abilities,
            'LatentPDF': latent_pdf,
            'AIC': {'final': full_metrics[0],
                    'null': null_metrics[0],
                    'delta': null_metrics[0] - full_metrics[0]},
            'BIC': {'final': full_metrics[1],
                    'null': null_metrics[1],
                    'delta': null_metrics[1] - full_metrics[1]}}
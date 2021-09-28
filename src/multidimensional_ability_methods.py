from itertools import repeat, product

import numpy as np
from scipy.optimize import fmin_slsqp
from scipy.special import expit

from girth.latent_ability_distribution import LatentPDF
from girth import (condition_polytomous_response,
                   validate_estimation_options)
from girth.polytomous_utils import _graded_partial_integral_md, _build_einsum_string


__all__ = ["multidimensional_ability_map", "multidimensional_ability_eap"]


def multidimensional_ability_map(dataset, difficulty, discrimination, options=None):
    """Estimates the abilities for dichotomous models.

    Estimates the ability parameters (theta) for dichotomous models via
    maximum a posterior likelihood estimation.

    Args:
        dataset: [n_items, n_participants] (2d Array) of measured responses
        difficulty: (1d Array) of difficulty parameters for each item
        discrimination: (1d Array) of disrimination parameters for each item
        options: dictionary with updates to default options

    Returns:
        abilities: (2d array) estimated abilities

    Options:
        * distribution: callable
    """
    n_factors = discrimination.shape[1]

    if n_factors < 2:
        raise AssertionError("Number of factors specified must be greater than 1.")

    options = validate_estimation_options(options)

    cpr_result = condition_polytomous_response(dataset, trim_ends=False)
    responses, item_counts, valid_response_mask = cpr_result
    invalid_response_mask = ~valid_response_mask
    
    n_items = responses.shape[0]
    difficulty = difficulty.reshape(n_items, -1)

    abilities = np.zeros((n_factors, dataset.shape[1]))
       
    # Initialize difficulty parameter storage
    betas = np.full((item_counts.sum(),), 10000.0)
    cumulative_item_counts = item_counts.cumsum()
    start_indices = np.roll(cumulative_item_counts, 1)
    start_indices[0] = 0
    
    # Initialize discrimination parameters storage
    local_discrimination = np.zeros((betas.size, n_factors))

    for ndx in range(n_items):
        end_ndx = cumulative_item_counts[ndx]
        start_ndx = start_indices[ndx] 
        betas[(start_ndx + 1):end_ndx] = difficulty[ndx][:item_counts[ndx] - 1]
        local_discrimination[start_ndx:end_ndx, :] = discrimination[ndx]

    betas_roll = np.roll(betas, -1)
    betas_roll[cumulative_item_counts-1] = -10000.0
    
    # Set invalid index to zero, this allows minimal
    # changes for invalid data and it is corrected
    # during integration
    responses[invalid_response_mask] = 0
    
    distribution = options['distribution']
    
    for person_ndx in range(abilities.shape[1]):
        person_response = responses[:, person_ndx]
        invalid_person_mask = invalid_response_mask[:, person_ndx]
        
        def _person_function(estimates):
            kernel = (local_discrimination * estimates).sum(1)
            temp1 = kernel + betas
            temp2 = kernel + betas_roll
            graded_prob = expit(temp1) 
            graded_prob -= expit(temp2)

            # Set all the responses and fix afterward
            temp_output = graded_prob[person_response]
            temp_output[invalid_person_mask] = 1.0    
            
            return -(np.log(temp_output).sum() + np.log(distribution(estimates)).sum())
        
        abilities[:, person_ndx] = fmin_slsqp(_person_function, np.zeros((n_factors)), 
                                              bounds=[(-4, 4),] * n_factors,
                                              disp=False)
   
    return abilities


def multidimensional_ability_eap(dataset, difficulty, discrimination, options=None):
    """Estimates the abilities for dichotomous models.

    Estimates the ability parameters (theta) for dichotomous models via
    expected a posterior likelihood estimation.

    Args:
        dataset: [n_items, n_participants] (2d Array) of measured responses
        difficulty: (1d Array) of difficulty parameters for each item
        discrimination: (1d Array) of disrimination parameters for each item
        options: dictionary with updates to default options

    Returns:
        abilities: (2d array) estimated abilities

    Options:
        * distribution: callable
        * quadrature_bounds: (float, float)
        * quadrature_n: int

    """
    n_factors = discrimination.shape[1]

    if n_factors < 2:
        raise AssertionError("Number of factors specified must be greater than 1.")

    options = validate_estimation_options(options)

    cpr_result = condition_polytomous_response(dataset, trim_ends=False)
    responses, item_counts, valid_response_mask = cpr_result
    invalid_response_mask = ~valid_response_mask
    
    n_items = responses.shape[0]
    difficulty = difficulty.reshape(n_items, -1)
    
    # Multi-dimensional Quadrature Locations
    latent_pdf = LatentPDF(options)
    theta = np.asarray(list(product(*repeat(latent_pdf.quadrature_locations, 
                                            n_factors)))).T

    dist_x_weight = latent_pdf(None, 0)
    einsum_string = _build_einsum_string(n_factors)
    distribution_x_weight = np.einsum(einsum_string, 
                                      *repeat(dist_x_weight, n_factors)).flatten()
    
    # Initialize difficulty parameter storage
    betas = np.full((item_counts.sum(),), 10000.0)
    cumulative_item_counts = item_counts.cumsum()
    start_indices = np.roll(cumulative_item_counts, 1)
    start_indices[0] = 0
    
    # Initialize discrimination parameters storage
    local_discrimination = np.zeros((betas.size, n_factors))

    for ndx in range(n_items):
        end_ndx = cumulative_item_counts[ndx]
        start_ndx = start_indices[ndx] 
        betas[(start_ndx + 1):end_ndx] = difficulty[ndx][:item_counts[ndx] - 1]
        local_discrimination[start_ndx:end_ndx, :] = discrimination[ndx]

    betas_roll = np.roll(betas, -1)
    betas_roll[cumulative_item_counts-1] = -10000.0
    
    # Set invalid index to zero, this allows minimal
    # changes for invalid data and it is corrected
    # during integration
    responses[invalid_response_mask] = 0

    # Compute partial int for calculations
    partial_int = np.ones((responses.shape[1], theta.shape[1]))
    for item_ndx in range(n_items):
        partial_int *= _graded_partial_integral_md(theta, betas, betas_roll,
                                                   local_discrimination,
                                                   responses[item_ndx],
                                                   invalid_response_mask[item_ndx])

    # Loop over each dimension and compute 
    abilities = np.zeros((n_factors, dataset.shape[1]))
     
    adjustment = ([latent_pdf.quadrature_locations] 
                  + [np.ones_like(latent_pdf.quadrature_locations),]
                  * (n_factors-1))
    
    denominator = (partial_int * distribution_x_weight)
    denominator_sum = denominator.sum(1)

    for factor_ndx in range(n_factors):
        linear_adjustment = np.einsum(einsum_string, *adjustment).flatten()

        numerator = denominator * linear_adjustment
        
        abilities[factor_ndx] = numerator.sum(1) / denominator_sum
        adjustment.insert(0, adjustment.pop())
   
    return abilities

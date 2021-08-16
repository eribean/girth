import numpy as np
from scipy import integrate, stats
from scipy.optimize import fminbound
from scipy.special import expit

from girth import (condition_polytomous_response, validate_estimation_options,
                   get_true_false_counts)
from girth.utils import (_get_quadrature_points, create_beta_LUT,
                         _compute_partial_integral, INVALID_RESPONSE)
from girth.latent_ability_distribution import LatentPDF
from girth.polytomous_utils import (_graded_partial_integral, _solve_for_constants,
                                    _solve_integral_equations, 
                                    _solve_integral_equations_LUT)
from girth.ability_methods import _ability_eap_abstract


__all__ = ["rasch_mml", "onepl_mml", "twopl_mml", "grm_mml"]


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


def twopl_mml(dataset, options=None):
    """ Estimates parameters in a 2PL IRT model.

    Args:
        dataset: [items x participants] matrix of True/False Values
        options: dictionary with updates to default options

    Returns:
        discrimination: (1d array) estimate of item discriminations
        difficulty: (1d array) estimates of item diffiulties
    
    Options:
        * max_iteration: int
        * distribution: callable
        * quadrature_bounds: (float, float)
        * quadrature_n: int
        * estimate_distribution: Boolean    
        * number_of_samples: int >= 5    
        * use_LUT: boolean    
    """
    results = grm_mml(dataset, options)
    results['Difficulty'] = results['Difficulty'].squeeze()
    return results


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

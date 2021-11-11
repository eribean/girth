import numpy as np
from scipy.optimize import fmin_slsqp, fminbound
from scipy.special import expit

from girth.utilities import (
    condition_polytomous_response, validate_estimation_options)


__all__ = ["grm_jml"]


def _jml_inequality(test):
    """Inequality constraints for graded jml minimization."""
    # First position is discrimination, next are difficulties
    return np.concatenate(([1, 1], np.diff(test)[1:]))


def grm_jml(dataset, options=None):
    """Estimate parameters for graded response model.

    Estimate the discrimination and difficulty parameters for
    a graded response model using joint maximum likelihood.

    Args:
        dataset: [n_items, n_participants] 2d array of measured responses
        options: dictionary with updates to default options

    Returns:
        discrimination: (1d array) estimate of item discriminations
        difficulty: (2d array) estimates of item diffiulties by item thresholds

    Options:
        * max_iteration: int
    """
    options = validate_estimation_options(options)

    cpr_result = condition_polytomous_response(dataset)
    responses, item_counts, valid_response_mask = cpr_result
    invalid_response_mask = ~valid_response_mask    
    n_items, n_takers = responses.shape

    # Initialize difficulty parameters for iterations
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

    # Set initial parameter estimates to default
    betas_roll = np.roll(betas, -1)
    betas_roll[cumulative_item_counts-1] = 10000
    thetas = np.zeros((n_takers,))
    
    # Set invalid index to zero, this allows minimal
    # changes for invalid data and it is corrected
    # during integration
    responses[invalid_response_mask] = 0
    
    for iteration in range(options['max_iteration']):
        previous_betas = betas.copy()

        #####################
        # STEP 1
        # Estimate theta, given betas / alpha
        # Loops over all persons
        #####################
        for ndx in range(n_takers):
            def _theta_min(theta):
                # Solves for ability parameters (theta) 
                graded_prob = (expit(discrimination * (theta - betas)) 
                               - expit(discrimination * (theta - betas_roll)))

                values = graded_prob[responses[:, ndx]]
                return -np.log(values[valid_response_mask[:, ndx]] + 1e-313).sum()

            thetas[ndx] = fminbound(_theta_min, -6, 6)

        # Recenter theta to identify model
        thetas -= thetas.mean()
        thetas /= thetas.std(ddof=1)
        #####################
        # STEP 2
        # Estimate Betas / alpha, given Theta
        # Loops over all items
        #####################
        for ndx in range(n_items):
            # pylint: disable=cell-var-from-loop
            # Compute ML for static items
            start_ndx = start_indices[ndx]
            end_ndx = cumulative_item_counts[ndx]

            def _alpha_beta_min(estimates):
                # Set the estimates int
                discrimination[start_ndx:end_ndx] = estimates[0]
                betas[start_ndx+1:end_ndx] = estimates[1:]
                betas_roll[start_ndx:end_ndx-1] = estimates[1:]

                graded_prob = (expit(discrimination[:, None] * (thetas[None, :] - betas[:, None]))
                               - expit(discrimination[:, None] * (thetas[None, :] - betas_roll[:, None])))

                values = np.take_along_axis(
                    graded_prob, responses[None, ndx], axis=0).squeeze()
                np.clip(values, 1e-23, np.inf, out=values)
                return -np.log(values[valid_response_mask[ndx]]).sum()

            # Solves jointly for parameters using numerical derivatives
            initial_guess = np.concatenate(([discrimination[start_ndx]],
                                            betas[start_ndx+1:end_ndx]))
            otpt = fmin_slsqp(_alpha_beta_min, initial_guess,
                              disp=False, f_ieqcons=_jml_inequality,
                              bounds=[(.25, 4)] + [(-6, 6)] * (item_counts[ndx]-1))

            discrimination[start_ndx:end_ndx] = otpt[0]
            betas[start_ndx+1:end_ndx] = otpt[1:]
            betas_roll[start_ndx:end_ndx-1] = otpt[1:]

        # Check termination criterion
        if(np.abs(previous_betas - betas).max() < 1e-3):
            break

    # Trim difficulties to conform to standard output
    # TODO:  look where missing values are and place NAN there instead
    # of appending them to the end
    output_betas = np.full((n_items, item_counts.max()-1), np.nan)
    for ndx, (start_ndx, end_ndx) in enumerate(zip(start_indices, cumulative_item_counts)):
        output_betas[ndx, :end_ndx-start_ndx-1] = betas[start_ndx+1:end_ndx]

    return {'Discrimination': discrimination[start_indices], 
            'Difficulty': output_betas}

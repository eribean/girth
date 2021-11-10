import numpy as np
from scipy.optimize import fmin_slsqp, fminbound
from scipy.special import expit

from girth.utilities import (
    condition_polytomous_response, validate_estimation_options)


__all__ = ["pcm_jml"]


def pcm_jml(dataset, options=None):
    """Estimate parameters for partial credit model.

    Estimate the discrimination and difficulty parameters for
    the partial credit model using joint maximum likelihood.

    Args:
        dataset: [n_items, n_participants] 2d array of measured responses
        options: dictionary with updates to default options

    Returns:
        discrimination: (1d array) estimates of item discrimination
        difficulty: (2d array) estimates of item difficulties x item thresholds

    Options:
        * max_iteration: int
    """
    options = validate_estimation_options(options)

    cpr_result = condition_polytomous_response(dataset, _reference=0.0)
    responses, item_counts, valid_response_mask = cpr_result
    invalid_response_mask = ~valid_response_mask    
    n_items, n_takers = responses.shape

    # Set initial parameter estimates to default
    thetas = np.zeros((n_takers,))

    # Initialize item parameters for iterations
    discrimination = np.ones((n_items,))
    betas = np.full((n_items, item_counts.max() - 1), np.nan)
    scratch = np.zeros((n_items, betas.shape[1] + 1))

    for ndx in range(n_items):
        item_length = item_counts[ndx] - 1
        betas[ndx, :item_length] = np.linspace(-1, 1, item_length)

    # Set invalid index to zero, this allows minimal
    # changes for invalid data and it is corrected
    # during integration
    responses[invalid_response_mask] = 0        

    for iteration in range(options['max_iteration']):
        previous_discrimination = discrimination.copy()

        #####################
        # STEP 1
        # Estimate theta, given betas / alpha
        # Loops over all persons
        #####################
        for ndx in range(n_takers):
            # pylint: disable=cell-var-from-loop
            response_set = responses[:, ndx]

            def _theta_min(theta, scratch):
                # Solves for ability parameters (theta)

                # Graded PCM Model
                scratch *= 0.
                scratch[:, 1:] = theta - betas
                scratch *= discrimination[:, None]
                np.cumsum(scratch, axis=1, out=scratch)
                np.exp(scratch, out=scratch)
                scratch /= np.nansum(scratch, axis=1)[:, None]

                # Probability associated with response
                values = np.take_along_axis(
                    scratch, response_set[:, None], axis=1).squeeze()
                return -np.log(values[valid_response_mask[:, ndx]] + 1e-313).sum()

            thetas[ndx] = fminbound(_theta_min, -6, 6, args=(scratch,))

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
            response_set = responses[ndx]

            def _alpha_beta_min(estimates):
                # PCM_Model
                kernel = thetas[:, None] - estimates[None, :]
                kernel *= estimates[0]
                kernel[:, 0] = 0
                np.cumsum(kernel, axis=1, out=kernel)
                np.exp(kernel, out=kernel)
                kernel /= np.nansum(kernel, axis=1)[:, None]
                # Probability associated with response
                values = np.take_along_axis(
                    kernel, response_set[:, None], axis=1).squeeze()
                return -np.log(values[valid_response_mask[ndx]]).sum()

            # Solves jointly for parameters using numerical derivatives
            initial_guess = np.concatenate(([discrimination[ndx]],
                                            betas[ndx, :item_counts[ndx]-1]))
            otpt = fmin_slsqp(_alpha_beta_min, initial_guess,
                              disp=False,
                              bounds=[(.25, 4)] + [(-6, 6)] * (item_counts[ndx]-1))

            discrimination[ndx] = otpt[0]
            betas[ndx, :item_counts[ndx]-1] = otpt[1:]

        # Check termination criterion
        if(np.abs(previous_discrimination - discrimination).max() < 1e-3):
            break

    return {'Discrimination': discrimination, 
            'Difficulty': betas}

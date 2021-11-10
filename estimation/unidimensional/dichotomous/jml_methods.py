import numpy as np
from scipy.optimize import fmin_slsqp, fminbound
from scipy.special import expit

from girth.utilities import (convert_responses_to_kernel_sign,
    mml_approx, trim_response_set_and_counts, validate_estimation_options)


__all__ = ["rasch_jml", "onepl_jml", "twopl_jml"]


def _jml_abstract(dataset, _item_min_func,
                  discrimination=1, max_iter=25):
    """ Defines common framework for joint maximum likelihood
        estimation in dichotomous models."""
    unique_sets, counts = np.unique(dataset, axis=1, return_counts=True)
    n_items, _ = unique_sets.shape

    # Use easy model to seed guess
    alphas = np.full((n_items,), discrimination,
                     dtype='float')  # discrimination
    betas = mml_approx(dataset, alphas)  # difficulty

    # Remove the zero and full count values
    unique_sets, counts = trim_response_set_and_counts(unique_sets, counts)

    n_takers = unique_sets.shape[1]
    the_sign = convert_responses_to_kernel_sign(unique_sets)
    thetas = np.zeros((n_takers,))

    for iteration in range(max_iter):
        previous_betas = betas.copy()

        #####################
        # STEP 1
        # Estimate theta, given betas
        # Loops over all persons
        #####################
        for ndx in range(n_takers):
            # pylint: disable=cell-var-from-loop
            scalar = the_sign[:, ndx] * alphas

            def _theta_min(theta):
                otpt = np.exp(scalar * (theta - betas))

                return np.log1p(otpt).sum()

            # Solves for the ability for each person
            thetas[ndx] = fminbound(_theta_min, -6, 6)

        # Recenter theta to identify model
        thetas -= thetas.mean()
        thetas /= thetas.std(ddof=1)

        #####################
        # STEP 2
        # Estimate Item Parameters
        # given Theta,
        #####################
        alphas, betas = _item_min_func(n_items, alphas, thetas,
                                       betas, the_sign, counts)

        if(np.abs(previous_betas - betas).max() < 1e-3):
            break

    return {'Discrimination': alphas, 
            'Difficulty': betas}


def rasch_jml(dataset, discrimination=1, options=None):
    """ Estimates difficulty parameters in an IRT model

    Args:
        dataset: [items x participants] matrix of True/False Values
        discrimination: scalar of discrimination used in model (default to 1)
        options: dictionary with updates to default options

    Returns:
        difficulty: (1d array) estimates of item difficulties

    Options:
        * max_iteration: int
    """
    options = validate_estimation_options(options)

    # Defines item parameter update function
    def _item_min_func(n_items, alphas, thetas,
                       betas, the_sign, counts):
        # pylint: disable=cell-var-from-loop

        for ndx in range(n_items):
            scalar = alphas[0] * the_sign[ndx, :]

            def _beta_min(beta):
                otpt = np.exp(scalar * (thetas - beta))
                return np.log1p(otpt).dot(counts)

            # Solves for the beta parameters
            betas[ndx] = fminbound(_beta_min, -6, 6)

        return alphas, betas

    result = _jml_abstract(dataset, _item_min_func,
                           discrimination, options['max_iteration'])

    return result


def onepl_jml(dataset, options=None):
    """ Estimates parameters in an 1PL IRT Model.

    Args:
        dataset: [items x participants] matrix of True/False Values
        options: dictionary with updates to default options

    Returns:
        discrimination: (float) estimate of test discrimination
        difficulty: (1d array) estimates of item diffiulties

    Options:
        * max_iteration: int
"""
    options = validate_estimation_options(options)

    # Defines item parameter update function
    def _item_min_func(n_items, alphas, thetas,
                       betas, the_sign, counts):
        # pylint: disable=cell-var-from-loop

        def _alpha_min(estimate):
            # Initialize cost evaluation to zero
            cost = 0
            for ndx in range(n_items):
                # pylint: disable=cell-var-from-loop
                scalar = the_sign[ndx, :] * estimate

                def _beta_min(beta):
                    otpt = np.exp(scalar * (thetas - beta))
                    return np.log1p(otpt).dot(counts)

                # Solves for the difficulty parameter for a given item at
                # a specific discrimination parameter
                betas[ndx] = fminbound(_beta_min, -6, 6)
                cost += _beta_min(betas[ndx])

            return cost

        min_alpha = fminbound(_alpha_min, 0.25, 5)
        alphas[:] = min_alpha

        return alphas, betas

    result = _jml_abstract(dataset, _item_min_func, discrimination=1,
                           max_iter=options['max_iteration'])
    result['Discrimination'] = result['Discrimination'][0]
    
    return result


def twopl_jml(dataset, options=None):
    """ Estimates parameters in a 2PL IRT model.

    Args:
        dataset: [items x participants] matrix of True/False Values
        options: dictionary with updates to default options

    Returns:
        discrimination: (1d array) estimates of item discrimination
        difficulty: (1d array) estimates of item difficulties

    Options:
        * max_iteration: int
    """
    options = validate_estimation_options(options)

    # Defines item parameter update function
    def _item_min_func(n_items, alphas, thetas,
                       betas, the_sign, counts):
        # pylint: disable=cell-var-from-loop
        for ndx in range(n_items):
            def _alpha_beta_min(estimates):
                otpt = np.exp((thetas - estimates[1]) *
                              the_sign[ndx, :] * estimates[0])
                return np.log1p(otpt).dot(counts)

            # Solves jointly for parameters using numerical derivatives
            otpt = fmin_slsqp(_alpha_beta_min, (alphas[ndx], betas[ndx]),
                              bounds=[(0.25, 4), (-6, 6)], disp=False)
            alphas[ndx], betas[ndx] = otpt

        return alphas, betas

    return _jml_abstract(dataset, _item_min_func, discrimination=1,
                         max_iter=options['max_iteration'])

import numpy as np
from scipy.optimize import fmin_slsqp, fminbound

from girth import (condition_polytomous_response,
                   convert_responses_to_kernel_sign, irt_evaluation,
                   mml_approx, trim_response_set_and_counts)


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
                otpt = 1.0 / (1.0 + np.exp(scalar *
                                           (theta - betas)))

                return -np.log(otpt).sum()

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

    return alphas, betas


def rasch_jml(dataset, discrimination=1, max_iter=25):
    """
        Estimates difficulty parameters in an IRT model

        Args:
            dataset: [items x participants] matrix of True/False Values
            discrimination: scalar of discrimination used in model (default to 1)
            max_iter: maximum number of iterations to run

        Returns:
            array of difficulty estimates
    """
    # Defines item parameter update function
    def _item_min_func(n_items, alphas, thetas,
                       betas, the_sign, counts):
        # pylint: disable=cell-var-from-loop

        for ndx in range(n_items):
            scalar = alphas[0] * the_sign[ndx, :]

            def _beta_min(beta):
                otpt = 1.0 / (1.0 + np.exp(scalar *
                                           (thetas - beta)))
                return -np.log(otpt).dot(counts)

            # Solves for the beta parameters
            betas[ndx] = fminbound(_beta_min, -6, 6)

        return alphas, betas

    result = _jml_abstract(dataset, _item_min_func,
                           discrimination, max_iter)

    return result[1]


def onepl_jml(dataset, max_iter=25):
    """
        Estimates difficulty and discrimination paramters in 1PL IRT model

        Args:
            dataset: [items x participants] matrix of True/False Values
            max_iter: maximum number of iterations to run

        Returns:
            discrimination, array of difficulty estimates
    """
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
                    otpt = 1.0 / (1.0 + np.exp(scalar *
                                               (thetas - beta)))
                    return -np.log(otpt).dot(counts)

                # Solves for the difficulty parameter for a given item at
                # a specific discrimination parameter
                betas[ndx] = fminbound(_beta_min, -6, 6)
                cost += _beta_min(betas[ndx])

            return cost

        min_alpha = fminbound(_alpha_min, 0.25, 5)
        alphas[:] = min_alpha

        return alphas, betas

    result = _jml_abstract(dataset, _item_min_func, discrimination=1,
                           max_iter=max_iter)

    return result[0][0], result[1]


def twopl_jml(dataset, max_iter=25):
    """
        Estimates difficulty and discrimination paramters in 2PL IRT model

        Args:
            dataset: [items x participants] matrix of True/False Values
            max_iter: maximum number of iterations to run

        Returns:
            array of discriminations, array of difficulty estimates
    """
    # Defines item parameter update function
    def _item_min_func(n_items, alphas, thetas,
                       betas, the_sign, counts):
        # pylint: disable=cell-var-from-loop
        for ndx in range(n_items):
            def _alpha_beta_min(estimates):
                otpt = 1.0 / (1.0 + np.exp((thetas - estimates[1]) *
                                           the_sign[ndx, :] * estimates[0]))
                return -np.log(otpt).dot(counts)

            # Solves jointly for parameters using numerical derivatives
            otpt = fmin_slsqp(_alpha_beta_min, (alphas[ndx], betas[ndx]),
                              bounds=[(0.25, 4), (-6, 6)], disp=False)
            alphas[ndx], betas[ndx] = otpt

        return alphas, betas

    return _jml_abstract(dataset, _item_min_func, discrimination=1,
                         max_iter=max_iter)


def _jml_inequality(test):
    """Inequality constraints for graded jml minimization."""
    # First position is discrimination, next are difficulties
    return np.concatenate(([1, 1], np.diff(test)[1:]))


def grm_jml(dataset, max_iter=25):
    """
        Estimates difficulty and discrimination paramaters
        for a graded response two parameter model

        difficulty parameters are category boundaries and by
        definition must be ordered

        Args:
            dataset: [items x participants] matrix of ordinal values
            max_iter: maximum number of iterations to run

        Returns:
            array of discriminations, 
            array of difficulty estimates (np.nan is a null value)
            array of person ability estimates
    """
    responses, item_counts = condition_polytomous_response(dataset)
    n_items, n_takers = responses.shape

    # Set initial parameter estimates to default
    thetas = np.zeros((n_takers,))

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
    betas_roll = np.roll(betas, -1)
    betas_roll[cumulative_item_counts-1] = 10000

    for iteration in range(max_iter):
        previous_betas = betas.copy()

        #####################
        # STEP 1
        # Estimate theta, given betas / alpha
        # Loops over all persons
        #####################
        for ndx in range(n_takers):
            def _theta_min(theta):
                # Solves for ability parameters (theta)
                graded_prob = (irt_evaluation(betas, discrimination, theta) -
                               irt_evaluation(betas_roll, discrimination, theta))

                values = graded_prob[responses[:, ndx]]
                return -np.log(values).sum()

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

                graded_prob = (irt_evaluation(betas, discrimination, thetas) -
                               irt_evaluation(betas_roll, discrimination, thetas))

                values = np.take_along_axis(
                    graded_prob, responses[None, ndx], axis=0)
                np.clip(values, 1e-23, np.inf, out=values)
                return -np.log(values).sum()

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

    return discrimination[start_indices], output_betas, thetas


def pcm_jml(dataset, max_iter=25):
    """
        Estimates difficulty and discrimination paramaters
        for a partial credit response two parameter model

        difficulty parameters can be disordered

        Args:
            dataset: [items x participants] matrix of ordinal values
            max_iter: maximum number of iterations to run

        Returns:
            array of discriminations, 
            array of difficulty estimates (np.nan is a null value)
            array of person ability estimates
    """
    responses, item_counts = condition_polytomous_response(
        dataset, _reference=0.0)
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

    for iteration in range(max_iter):
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
                    scratch, response_set[:, None], axis=1)
                return -np.log(values + 1e-23).sum()

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
                    kernel, response_set[:, None], axis=1)
                return -np.log(values).sum()

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

    return discrimination, betas

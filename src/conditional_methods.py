import numpy as np

from scipy.optimize import fminbound

from girth import (trim_response_set_and_counts,
                   validate_estimation_options)


__all__ = ["rasch_conditional"]


def _symmetric_functions(betas):
    """Computes the symmetric functions based on the betas

        Indexes by score, left to right

    """
    polynomials = np.c_[np.ones_like(betas), np.exp(-betas)]

    # This is an easy way to compute all the values at once,
    # not necessarily the fastest
    otpt = 1
    for polynomial in polynomials:
        otpt = np.convolve(otpt, polynomial)
    return otpt


def rasch_conditional(dataset, discrimination=1, options=None):
    """ Estimates the difficulty parameters in a Rasch IRT model

    Args:
        dataset: [items x participants] matrix of True/False Values
        discrimination: scalar of discrimination used in model (default to 1)
        options: dictionary with updates to default options

    Returns:
        difficulty: (1d array) estimates of item difficulties

    Options:
        * max_iteration: int

    Notes:
        This function sets the sum of difficulty parameters to 
        zero for identification purposes
    """
    options = validate_estimation_options(options)
    n_items = dataset.shape[0]
    unique_sets, counts = np.unique(dataset, axis=1, return_counts=True)

    # Initialize all the difficulty parameters to zeros
    # Set an identifying_mean to zero
    ##TODO: Add option to specifiy position
    betas = np.zeros((n_items, ))
    identifying_mean = 0.0

    # Remove the zero and full count values
    unique_sets, counts = trim_response_set_and_counts(unique_sets, counts)

    response_set_sums = unique_sets.sum(axis=0)

    for iteration in range(options['max_iteration']):
        previous_betas = betas.copy()

        for ndx in range(n_items):
            partial_conv = _symmetric_functions(np.delete(betas, ndx))

            def min_func(estimate):
                betas[ndx] = estimate
                full_convolution = np.convolve([1, np.exp(-estimate)], partial_conv)

                denominator = full_convolution[response_set_sums]

                return (np.sum(unique_sets * betas[:,None], axis=0).dot(counts) + 
                        np.log(denominator).dot(counts))

            # Solve for the difficulty parameter
            betas[ndx] = fminbound(min_func, -5, 5)

            # recenter
            betas += (identifying_mean - betas.mean())

        # Check termination criterion
        if np.abs(betas - previous_betas).max() < 1e-3:
            break

    return {'Discrimination': discrimination,
            'Difficulty': betas / discrimination}

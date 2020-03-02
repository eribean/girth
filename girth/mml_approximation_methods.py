import numpy as np

from scipy import integrate
from scipy.optimize import fminbound

from girth.utils import _get_quadrature_points, _compute_partial_integral

def rasch_approx(dataset, discrimination=1):
    """
        Estimates the difficulty parameters via the approximation

        Args:
            dataset: [items x participants] matrix of True/False Values
            discrimination: scalar of discrimination used in model (default to 1)

        Returns:
            array of discrimination estimates
    """
    n_no = np.count_nonzero(~dataset, axis=1)
    n_yes = np.count_nonzero(dataset, axis=1)
    return (np.sqrt(1 + discrimination**2 / 3) *
            np.log(n_no / n_yes) / discrimination)


def onepl_approx(dataset):
    """
        Estimates the difficulty parameters via the approximation

        Args:
            dataset: [items x participants] matrix of True/False Values

        Returns:
            array of discrimination, difficulty estimates
    """
    n_no = np.count_nonzero(~dataset, axis=1)
    n_yes = np.count_nonzero(dataset, axis=1)
    scalar = np.log(n_no / n_yes)

    unique_sets, counts = np.unique(dataset, axis=1, return_counts=True)
    the_sign = (-1)**unique_sets

    # Inline definition of quadrature function
    #TODO: Use partial integration methods to speed up processing
    def quadrature_function(theta, difficulty, discrimination, response):
        gauss = 1.0 / np.sqrt(2 * np.pi) * np.exp(-np.square(theta) / 2)
        kernel = the_sign[:, :, None] * np.ones((1, 1, theta.size))
        kernel *= discrimination
        kernel *= (theta[None, None, :] - difficulty[:, None, None])

        return  gauss[None, :] * (1.0 / (1.0 + np.exp(kernel))).prod(axis=0).squeeze()

    # Inline definition of cost function to minimize
    def min_func(estimate):
        difficulty = np.sqrt(1 + estimate**2 / 3) * scalar / estimate
        otpt = integrate.fixed_quad(quadrature_function, -5, 5,
                                    (difficulty, estimate, unique_sets), n=61)[0]
        return -np.log(otpt).dot(counts)

    # Perform the minimization
    discrimination = fminbound(min_func, 0.25, 10)

    return discrimination, np.sqrt(1 + discrimination**2 / 3) * scalar / discrimination


def twopl_approx(dataset, max_iter=25):
    """ Estimates the difficulty/discrimination parameters

        Uses the approximation of the one dimensional marginal likelihood
        to separate difficulty from discrimination estimation

        Args:
            dataset: [items x participants] matrix of True/False Values
            max_iter:  maximum number of iterations to run

        Returns:
            array of discrimination, difficulty estimates
    """
    n_items = dataset.shape[0]
    unique_sets, counts = np.unique(dataset, axis=1, return_counts=True)
    the_sign = (-1)**unique_sets

    theta = _get_quadrature_points(61, -5, 5)

    # Inline definition of quadrature function
    def quadrature_function(theta, discrimination, old_discrimination,
                            difficulty, old_difficulty,
                            partial_int, the_sign):
        kernel1 = the_sign[:, None] * (theta[None, :] - difficulty)
        kernel1 *= discrimination

        kernel2 = the_sign[:, None] * (theta[None, :] - old_difficulty)
        kernel2 *= old_discrimination

        return partial_int * (1 + np.exp(kernel2)) / (1 + np.exp(kernel1))


    # Inline definition of cost function to minimize
    def min_func(estimate, dataset, old_estimate, old_difficulty,
                 partial_int, the_sign):
        new_difficulty = rasch_approx(dataset, estimate)
        otpt = integrate.fixed_quad(quadrature_function, -5, 5,
                                    (estimate, old_estimate,
                                     new_difficulty, old_difficulty,
                                     partial_int, the_sign), n=61)[0]
        return -np.log(otpt).dot(counts)

    # Perform the minimization
    initial_guess = np.ones((dataset.shape[0],))
    difficulties = rasch_approx(dataset)

    for iteration in range(max_iter):
        previous_guess = initial_guess.copy()
        previous_difficulty = difficulties.copy()

        #Quadrature evaluation for values that do not change
        partial_int = _compute_partial_integral(theta, difficulties,
                          initial_guess, the_sign)

        for ndx in range(n_items):
            def min_func_local(estimate):
                return min_func(estimate, dataset[ndx].reshape(1, -1),
                                previous_guess[ndx],
                                previous_difficulty[ndx],
                                partial_int, the_sign[ndx])

            initial_guess[ndx] = fminbound(min_func_local, 0.25, 6, xtol=1e-3)
            difficulties[ndx] = rasch_approx(dataset[ndx].reshape(1, -1),
                                               initial_guess[ndx])

            partial_int = quadrature_function(theta, initial_guess[ndx],
                                              previous_guess[ndx], difficulties[ndx],
                                              previous_difficulty[ndx],
                                              partial_int, the_sign[ndx])

        if np.abs(initial_guess - previous_guess).max() < 1e-3:
            break

    return initial_guess, difficulties

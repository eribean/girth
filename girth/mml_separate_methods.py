import numpy as np
from scipy import integrate
from scipy.optimize import fminbound, brentq

from girth import irt_evaluation, rasch_approx
from girth.utils import _get_quadrature_points, _compute_partial_integral


def rasch_separate(dataset, discrimination=1, max_iter=25):
    """
        Estimates parameters in an IRT model with full
        gaussian quadrature

        Args:
            dataset: [items x participants] matrix of True/False Values
            discrimination: scalar of discrimination used in model (default to 1)
            max_iter: maximum number of iterations to run

        Returns:
            array of discrimination estimates
    """
    n_items = dataset.shape[0]
    n_no = np.count_nonzero(~dataset, axis=1)
    n_yes = np.count_nonzero(dataset, axis=1)
    scalar = n_yes / (n_yes + n_no)

    if np.ndim(discrimination) < 1:
        discrimination = np.full(n_items, discrimination)

    # Inline definition of quadrature function
    def quadrature_function(theta, difficulty, discrimination):
        gauss = 1.0 / np.sqrt(2 * np.pi) * np.exp(-np.square(theta) / 2)
        return irt_evaluation(np.array([difficulty]),
                              np.array([discrimination]), theta) * gauss

    # Initialize the discrimination parameters
    the_parameters = np.zeros((n_items,))

    # Perform the minimization
    for ndx in range(n_items):

        # Minimize each item separately
        def min_zero_local(estimate):
            return (scalar[ndx] -
                    integrate.fixed_quad(quadrature_function, -10, 10,
                    (estimate, discrimination[ndx]), n=101)[0])

        the_parameters[ndx] = brentq(min_zero_local, -6, 6)

    return the_parameters


def onepl_separate(dataset):
    """
        Estimates the difficulty and single discrimination parameter

        Separates the difficulty estimation from the discrimination
        parameters

        Args:
            dataset: [items x participants] matrix of True/False Values

        Returns:
            array of discrimination, difficulty estimates
    """
    unique_sets, counts = np.unique(dataset, axis=1, return_counts=True)
    the_sign = (-1)**unique_sets
    
    # Inline definition of cost function to minimize
    def min_func(estimate):
        difficulty = rasch_separate(dataset, estimate)
        otpt = integrate.fixed_quad(_compute_partial_integral, -5, 5,
                                    (difficulty, estimate, the_sign), n=61)[0]

        return -np.log(otpt).dot(counts)

    # Perform the minimization
    discrimination = fminbound(min_func, 0.25, 10)

    return discrimination, rasch_separate(dataset, discrimination)


def twopl_separate(dataset, max_iter=25):
    """
        Estimates the difficulty and discrimination parameters

        Separates the difficulty estimation from the discrimination
        parameters

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
        new_difficulty = rasch_separate(dataset, estimate)
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

        # Quadrature evaluation for values that do not change
        # This is done during the outer loop to address rounding errors
        partial_int = _compute_partial_integral(theta, difficulties,
                          initial_guess, the_sign)

        for ndx in range(n_items):
            def min_func_local(estimate):
                return min_func(estimate, dataset[ndx].reshape(1, -1),
                                previous_guess[ndx],
                                previous_difficulty[ndx],
                                partial_int, the_sign[ndx])

            # Solve for the discrimination parameters
            initial_guess[ndx] = fminbound(min_func_local, 0.25, 6, xtol=1e-3)
            difficulties[ndx] = rasch_separate(dataset[ndx].reshape(1, -1),
                                                   initial_guess[ndx])

            # Update the partial integral based on the new found values
            partial_int = quadrature_function(theta, initial_guess[ndx],
                                              previous_guess[ndx], difficulties[ndx],
                                              previous_difficulty[ndx],
                                              partial_int, the_sign[ndx])

        if np.abs(initial_guess - previous_guess).max() < 1e-3:
            break

    return initial_guess, difficulties

import numpy as np
from scipy import integrate
from scipy.optimize import fminbound, brentq, fmin_powell

from girth import irt_evaluation, convert_responses_to_kernel_sign
from girth.utils import _get_quadrature_points, _compute_partial_integral
from girth import rasch_approx, onepl_approx


def _rasch_full_abstract(dataset, discrimination=1, max_iter=25):
    """
        Method used by several functions during the estimation process
    """
    n_items = dataset.shape[0]
    unique_sets, counts = np.unique(dataset, axis=1, return_counts=True)
    the_sign = convert_responses_to_kernel_sign(unique_sets)

    theta = _get_quadrature_points(61, -5, 5)

    # Inline definition of quadrature function
    def quadrature_function(theta, difficulty, old_difficulty, partial_int, the_sign):
        kernel1 = the_sign[:, None] * (theta[None, :] - difficulty)
        kernel1 *= discrimination

        kernel2 = the_sign[:, None] * (theta[None, :] - old_difficulty)
        kernel2 *= discrimination

        return partial_int * (1 + np.exp(kernel2)) / (1 + np.exp(kernel1))

    # Inline definition of cost function to minimize
    def min_func(difficulty, old_difficulty, partial_int, the_sign):
        otpt = integrate.fixed_quad(quadrature_function, -5, 5,
                (difficulty, old_difficulty, partial_int, the_sign), n=61)[0] + 1e-23
        return -np.log(otpt).dot(counts)

    # Get approximate guess to begin with
    initial_guess = rasch_approx(dataset, discrimination=discrimination)

    for iteration in range(max_iter):
        previous_guess = initial_guess.copy()

        #Quadrature evaluation for values that do not change
        partial_int = _compute_partial_integral(theta, initial_guess,
                          discrimination, the_sign)

        for ndx in range(n_items):
            # pylint: disable=cell-var-from-loop
            # Minimize each one separately
            value = initial_guess[ndx] * 1.0

            def min_func_local(estimate):
                return min_func(estimate, previous_guess[ndx],
                                partial_int, the_sign[ndx])

            # Given an estimate of discrimination parameter, compute the
            # discrimination parameters
            initial_guess[ndx] = fminbound(min_func_local,
                                           value-0.75,
                                           value+0.75)

            # Update the integral with the new found vaolues
            partial_int = quadrature_function(theta, initial_guess[ndx],
                                              previous_guess[ndx], partial_int, the_sign[ndx])

        if(np.abs(initial_guess - previous_guess).max() < 0.001):
            break

    # Get the value of the cost function
    cost = integrate.fixed_quad(lambda x: partial_int, -5, 5, n=61)[0]

    return initial_guess, -np.log(cost).dot(counts)


def rasch_full(dataset, discrimination=1, max_iter=25):
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
    return _rasch_full_abstract(dataset, discrimination, max_iter)[0]


def onepl_full(dataset, max_iter=25):
    """
        Estimates difficulty and discrimination parameters in a one pl IRT model

        Args:
            dataset: [items x participants] matrix of True/False Values

        Returns:
            array of discrimination, difficulty estimates
    """
    # Use the rasch model and minimize over the singel discrimination paramter
    def min_func_local(estimate):
        _, cost = _rasch_full_abstract(dataset, estimate, max_iter)
        return cost

    # Solve for the discrimination parameter
    discrimination = fminbound(min_func_local, 0.5, 4)

    return discrimination, rasch_full(dataset, discrimination)


def twopl_full(dataset, max_iter=25):
    """
        Estimates parameters in a 2PL IRT model with marginal likelihood

        Args:
            dataset: [items x participants] matrix of True/False Values

        Returns:
            array of discrimination, difficulty estimates
    """
    n_items = dataset.shape[0]
    unique_sets, counts = np.unique(dataset, axis=1, return_counts=True)
    the_sign = convert_responses_to_kernel_sign(unique_sets)

    theta = _get_quadrature_points(61, -5, 5)

    # Inline definition of quadrature function
    def quadrature_function(theta, estimates, old_estimates, partial_int, the_sign):
        kernel1 = the_sign[:, None] * (theta[None, :] - estimates[1])
        kernel1 *= estimates[0]

        kernel2 = the_sign[:, None] * (theta[None, :] - old_estimates[1])
        kernel2 *= old_estimates[0]

        return partial_int * (1 + np.exp(kernel2)) / (1 + np.exp(kernel1))

    # Inline definition of cost function to minimize
    def min_func(estimates, old_estimates, partial_int, the_sign):
        otpt = integrate.fixed_quad(quadrature_function, -5, 5,
                (estimates, old_estimates, partial_int, the_sign), n=61)[0] + 1e-23
        return -np.log(otpt).dot(counts)

    # Get approximate guess to begin with rasch model
    a1, b1 = onepl_approx(dataset)
    initial_guess = np.c_[np.full_like(b1, a1), b1]

    for iteration in range(max_iter):
        previous_guess = initial_guess.copy()

        #Quadrature evaluation for values that do not change
        partial_int = _compute_partial_integral(theta, initial_guess[:, 1],
                          initial_guess[:, 0], the_sign)

        for ndx in range(n_items):
            # pylint: disable=cell-var-from-loop
            # Minimize each one separately
            value = initial_guess[ndx] * 1.0

            def min_func_local(estimate):
                return min_func(estimate, previous_guess[ndx],
                                partial_int, the_sign[ndx])

            # Two parameter solver that doesn't need derivatives
            initial_guess[ndx] = fmin_powell(min_func_local, value, xtol=1e-3, disp=0)

            # Update the integral for new found values
            partial_int = quadrature_function(theta, initial_guess[ndx],
                                              previous_guess[ndx], partial_int, the_sign[ndx])

        if(np.abs(initial_guess - previous_guess).max() < 0.001):
            break

    return initial_guess[:, 0], initial_guess[:, 1]

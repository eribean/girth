import numpy as np

from scipy.optimize import fixed_point

from girth.factoranalysis import principal_components_analysis as pca


__all__ = ['principal_axis_factor']


def _paf_fixed_point_iterate(unique_variance, input_matrix, n_factors):
    """Principal axis factor iteration function."""
    adjusted_matrix = input_matrix - np.diag(unique_variance)
    facts, _, _ = pca(adjusted_matrix, n_factors)
    return np.diag(input_matrix - facts @ facts.T)


def principal_axis_factor(input_matrix, n_factors, initial_guess=None,
                          max_iterations=2000, tolerance=0.001):
    """Performs Principal Axis Factor Analysis on a symmetric matrix.

    Args:
        input_matrix: input correlation or covariance matrix
        n_factors:  number of factors to extract
        initial_guess:  Initial guess to seed the iteration (Default: all zeros)
        max_iterations:  Maximum number of iterations to run (Default: 2000)
        tolerance:  tolerance to terminate iteration (Default: 0.001)

    Returns:
        loadings: extracted factor loadings
        eigenvalues: extracted eigenvalues
        unique_variance: estimated unique variance
    """
    if initial_guess is None:
        initial_guess = np.zeros((input_matrix.shape[0]))

    args = (input_matrix, n_factors)
    unique_variance = fixed_point(_paf_fixed_point_iterate, initial_guess,
                                  args=args, xtol=tolerance, maxiter=max_iterations)

    loadings, eigenvalues, _ = pca(input_matrix - np.diag(unique_variance), 
                                   n_factors)
    
    return loadings, eigenvalues, unique_variance


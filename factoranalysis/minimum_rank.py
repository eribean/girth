import numpy as np
from numpy.linalg import svd
from scipy.linalg import ldl
from scipy.optimize import minimize

from girth.factoranalysis import principal_components_analysis as pca
from girth.factoranalysis import principal_axis_factor as paf


__all__ = ['minimum_rank_factor_analysis']


def _mrfa_min_func(inverse_half_variance, correlation_cholesky, n_factors):
    """Min function for minimum rank factor analysis"""
    u, singular_vals, vt = svd(inverse_half_variance 
                               * correlation_cholesky)
    
    cost = (singular_vals[n_factors:]).sum() / singular_vals[-1]
    
    correlation_u = correlation_cholesky.T @ u
    partial_derivative = (vt.T * correlation_u)[:, n_factors:]
    temp = partial_derivative.sum(1)
    
    derivative = ((temp - cost * partial_derivative[:, -1]) 
                  / singular_vals[-1])
    
    return cost, derivative
    

def minimum_rank_factor_analysis(correlation_matrix, n_factors,
                                 initial_guess=None, n_iter=500):
    """Performs minimum rank factor analysis on a correlation matrix.

    This method constrains the search region to force the resulting matrix to
    be semi-positive definite.

    Args:
        correlation_matrix:  input array to decompose (m x m)
        n_factors:  number of factors to keep
        initial_guess: Guess to seed the search algorithm, defaults to
                       the result of principal axis factor
        n_iter: Maximum number of iterations to run (Default: 500)

    Returns:
        loadings: extracted factor loadings
        eigenvalues: extracted eigenvalues
        unique_variance: estimated unique variance
    """
    if initial_guess is None:
        _, _, initial_guess = paf(correlation_matrix, n_factors)
    
    # Protect against semi-positive definite
    correlation_cholesky = ldl(correlation_matrix)
    correlation_cholesky = (np.diag(np.sqrt(np.diag(correlation_cholesky[1]))) 
                            @ correlation_cholesky[0].T)  

    args = (correlation_cholesky, n_factors)
    bounds = [(1, 100),] * (correlation_matrix.shape[0])
    initial_guess = 1 / np.sqrt(initial_guess)
    
    result = minimize(_mrfa_min_func, initial_guess, args, method='SLSQP',
                      bounds=bounds, options={'maxiter': n_iter}, jac=True)

    # Convert result into unique variance
    eigs = svd(result['x'] * correlation_cholesky, compute_uv=False)
    unique_variance = np.square(eigs[-1] / result['x'])
    loadings, eigs, _ = pca(correlation_matrix 
                            - np.diag(unique_variance), n_factors)
    
    return loadings, eigs, unique_variance

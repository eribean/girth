import numpy as np


__all__ = ['principal_components_analysis']


def principal_components_analysis(input_matrix, n_factors=None):
    """Performs principal components analysis.

    Args:
        input_matrix: input correlation or covariance matrix
        n_factors: number of factors to extract

    Returns:
        loadings: factor loadings
        eigenvalues: eigenvalues for the factors
        unique_variance: vector of all zeros
    """
    if n_factors is None:
        n_factors = input_matrix.shape[0]

    eigenvalues, loadings = np.linalg.eigh(input_matrix,)

    # Only return the requested factors
    eigenvalues = eigenvalues[-n_factors:][::-1]
    loadings = loadings[:, -n_factors:][:, ::-1]

    # Scale Loadings
    loadings *= np.sqrt(eigenvalues).reshape(1, -1)

    return loadings, eigenvalues, np.zeros((input_matrix.shape[0]))
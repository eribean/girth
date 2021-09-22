import numpy as np

from girth.factoranalysis import principal_components_analysis as pca


__all__ = ["maximum_likelihood_factor_analysis"]


def maximum_likelihood_factor_analysis(input_matrix, n_factors, tolerance=1e-7,
                                       max_iter=100, initial_guess=None):
    """Performs maximum likelihood factor analysis on a symmetric matrix.

    This method models residuals as gaussian, be aware if your data doesn't 
    meet this criteria.

    Uses a conditionaly maximization algorithm, DOI: 10.1007/s11222-007-9042-y

    Args:
        input_matrix:  input correlation | covariance array
        n_factors:  number of factors to keep
        tolerance: change in unique variance to terminate iteration (Default: 1e-7)
        max_iter: maximum number of iterations (Defaults: 100)
        initial_guess: Guess to seed the search algorithm, defaults to
                       the result of principal axis factor
                       
    Returns:
        loadings: extracted factor loadings
        eigenvalues: extracted eigenvalues
        unique_variance: estimated unique variance
    """
    n_items = input_matrix.shape[0]
    
    # Initial Guess
    loads, _, _ = pca(input_matrix, n_factors)
    
    if initial_guess is None:
        uvars = np.diag(input_matrix - loads @ loads.T).copy()
    else:
        uvars = initial_guess.copy()
    
    identity_items = np.eye(n_items)

    loads_tilde = np.diag(1 / np.sqrt(uvars)) @ loads

    for iteration in range(max_iter):
        previous_unique_variance = uvars.copy()
        
        # Solve for Updated Loadings
        psi_sqrt = 1 / np.sqrt(uvars)
        input_tilde = np.diag(psi_sqrt) @ input_matrix @ np.diag(psi_sqrt)

        s1, u1 = np.linalg.eigh(input_tilde)  
        valid_eigs = s1[-n_factors:]
        valid_vects = u1[:, -n_factors:]

        lower_chol = np.linalg.cholesky(input_tilde).T.copy()
        
        loads_update = valid_vects @ np.diag(np.sqrt(valid_eigs - 1))

        # Solve for Updated Unique Variance
        inv_B = (valid_vects @ np.diag(1 / valid_eigs - 1)
                 @ valid_vects.T + identity_items)

        for ndx in range(n_items):
            scalar = inv_B[0, ndx]
            selected_row = lower_chol @ inv_B[0]
            selected_row = selected_row.dot(selected_row)
            omega = (selected_row - scalar) / scalar**2
            uvars[ndx] = max(1e-6, (omega + 1) * uvars[ndx])
            
            # Recursively update Inverse of B
            selected_column = inv_B[0, ndx+1:] * (-omega / (1 + omega * scalar))
            adjustment = selected_column.reshape(-1, 1) * inv_B[0]
            inv_B = inv_B[1:, :] + adjustment

        loads_tilde = loads_update.copy()

        if np.abs(previous_unique_variance - uvars).max() < tolerance:
            break

    update_loads = loads_tilde * np.sqrt(uvars)[:, None]    
    return update_loads, np.square(update_loads).sum(0), uvars

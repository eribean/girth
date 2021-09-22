import numpy as np

__all__ = ["procrustes_rotation"]


def procrustes_rotation(target_matrix, collected_matrix):
    """Computes the procrustes rotation between two matrices.
    
    Finds the orthogonal rotation matrix that transforms the
    collected_matrix into the target matrix in the 
    least-squares sense

    Equation:
    ----------
    arg min SUM (target - collected @ rotation)**2

    Args:
        target_matrix: Reference matrix to attempt to emulate
        collected_matrix: Matrix to rotate

    Returns:
        rotation: square matrix that rotates the collected_matrix into the
                  target matrix

    Note:
        There may be axis reflections in the rotation matrix, these 
        reflections are not handled in this function.
    """
    left_basis, _, right_basis = np.linalg.svd(collected_matrix.T 
                                               @ target_matrix)

    return left_basis @ right_basis                                               

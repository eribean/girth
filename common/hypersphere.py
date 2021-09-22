import numpy as np

__all__ = ['hyperspherical_vector', 'hyperspherical_angles']


def hyperspherical_vector(thetas):
    """Returns a unit vector in a hypersphere.

    Args:
        theta: array-like iterable of angles in radians

    Returns:
        Hyperspherical unit-vector
    """
    vector = np.ones((np.size(thetas) + 1,))
    vector[1:] = np.cumprod(np.sin(thetas))
    vector[:-1] *= np.cos(thetas)
    return vector


#NOTE: This implementation is convoluted due to vectorizing, see
# https://en.wikipedia.org/wiki/N-sphere for algorithm details
def hyperspherical_angles(vector):
    """Converts a vector into the angles

    Args:
        vector:  Input to convert to angles

    Returns:
        the angles where size(angles) = size(vector) - 1
    """
    the_angles = np.zeros_like(vector)
    denominator = vector * vector
    denominator = np.sqrt(np.cumsum(denominator[::-1])[::-1])

    mask = denominator == 0.0
    the_angles = np.where(mask & (vector >= 0.0), 0.0, np.pi)
    the_angles[~mask] = np.arccos(vector[~mask] / denominator[~mask])

    # Adjust the last sign if necessary
    if vector[-1] < 0:
        the_angles[-2] = 2 * np.pi - the_angles[-2]

    return the_angles[:-1]
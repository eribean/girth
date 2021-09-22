import numpy as np


__all__ = ["entropy"]


def entropy(the_array, axis=1):
    """Computes a modified entropy value for each row

    Args:
        the_array:  input array of factors
        axis: Axis to perform calculation on
                0: Vertical
                1: Horizontal
    Return:
        entropy value
    """
    power = the_array * the_array
    sum_power = power.sum(axis=axis)
    if axis:
        power /= sum_power.reshape(-1, 1)
    else:
        power /= sum_power
    return -(power * np.log(power + 1e-23)).sum()
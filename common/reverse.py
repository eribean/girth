import numpy as np

from girth.utils import INVALID_RESPONSE


__all__ = ['reverse_score']


def reverse_score(dataset, mask, max_val=5):
    """Reverse scores a set of ordinal data.

    Args:
        data: Matrix [items x observations] of measured responses
        mask: Boolean Vector with True indicating a reverse scored item
        max_val:  (int) maximum value in the Likert (-like) scale

    Returns:
        data_reversed: Dataset with reverresponses with the reverse scoring removed
    """
    if(dataset.shape[0] != mask.shape[0]):
        raise AssertionError("First dimension of data and mask must be equal")

    new_data = dataset.copy()

    new_data[mask] = max_val + 1 - new_data[mask]
    
    new_data[dataset == INVALID_RESPONSE] = INVALID_RESPONSE

    return new_data
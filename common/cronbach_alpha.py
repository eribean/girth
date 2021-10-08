import numpy as np

from girth.utils import INVALID_RESPONSE


__all__ = ["cronbach_alpha"]


def cronbach_alpha(data):
    """Computes reliability coefficient based on data

    Args:
        data: Data as a matrix (Questions, Participants)

    Returns:
        cronbachs alpha (tau-equivalent) measure of reliability

    Notes:
        Data needs to be reverse scored

    """
    n_items = data.shape[0]

    valid_mask = data != INVALID_RESPONSE
    
    item_variance = np.var(data, axis=1, ddof=1, where=valid_mask).sum()
    people_variance = (n_items * n_items
                       * np.mean(data, axis=0, where=valid_mask).var(ddof=1))
    
    return (n_items / (n_items - 1) 
            * (1 - item_variance / people_variance))
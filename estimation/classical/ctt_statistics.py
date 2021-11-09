import numpy as np
from scipy.stats import norm

from girth.utils import INVALID_RESPONSE
from girth.common import polyserial_correlation, cronbach_alpha


__all__ = ["classical_test_statistics"]
    

def classical_test_statistics(dataset, start_value=1, stop_value=5):
    """Calculates classical test theory for ordinal inputs.

    Args:
        dataset: [items x observations] 2D Array of ordinal
        start_value: [int] first valid response
        end_value: [int] last valid response

    Returns:
        ctt_dictionary: Classical test theory dictionary that
            computes means, stds, item-score correlation and 
            polyserial correlations, and Cronbach's Alpha if
            item deleted

    Notes:
        If binary inputs then it computes pt-biserial and biserial
        correlations
    """
    valid_mask = dataset != INVALID_RESPONSE

    total_sum_score = (dataset * valid_mask).sum(1)
    n_total = np.count_nonzero(valid_mask, axis=1)
    total_score = total_sum_score / n_total
    total_std = np.array([dataset[ndx][valid_mask[ndx]].std(ddof=1) 
                          for ndx in range(dataset.shape[0])])

    # Initialize Arrays
    item_counts = np.arange(dataset.shape[0])
    item_score_correlations = np.zeros(dataset.shape[0])
    polyserial_correlations = np.zeros(dataset.shape[0])
    cronbach_alpha_if_deleted = np.zeros(dataset.shape[0])
    
    # Compute the Item-Total Score Correlations    
    for item_ndx in range(dataset.shape[0]):
        data_subset = np.delete(item_counts, item_ndx)

        valid_mask_subset = valid_mask[data_subset]
        dataset_subset = dataset[data_subset]
        total_sub_score = dataset_subset.mean(axis=0, 
                                              where=valid_mask_subset)
        
        valid_item_mask = valid_mask[item_ndx]
        
        total_sub_score_valid = total_sub_score[valid_item_mask]
        dataset_valid = dataset[item_ndx][valid_item_mask]
        
        item_corr = np.corrcoef(total_sub_score_valid,
                                dataset_valid)
        item_score_correlations[item_ndx] = item_corr[0, 1]
        
        polyserial_correlations[item_ndx] = polyserial_correlation(total_sub_score_valid,
                                                                   dataset_valid)
        cronbach_alpha_if_deleted[item_ndx] = cronbach_alpha(dataset_subset)
    
    return {
        "Item Counts": n_total,
        "Mean": total_score,
        "Std":  total_std,
        'Cronbach Alpha': cronbach_alpha_if_deleted,        
        "Item-Score Correlation": item_score_correlations,
        "Polyserial Correlation": polyserial_correlations}
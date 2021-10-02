import numpy as np
from scipy.stats import norm


__all__ = ["classical_test_theory_dichotomous"]


def classical_test_theory_dichotomous(dataset):
    """Calculates classical test theory for dichotomous inputs.

    Args:
        dataset: [items x observations] 2D Array of dichotomous values

    Returns:
        ctt_dictionary: Classical test theory dictionary that
            computes means, stds, item-score correlation and 
            biserial correlations
    """
    # Get the number of correct / incorrect
    mask_correct = dataset == 1
    mask_incorrect = dataset == 0

    n_correct = np.count_nonzero(mask_correct, axis=1)
    n_wrong = np.count_nonzero(mask_incorrect, axis=1)
    n_total = n_correct + n_wrong

    # Mean and standard deviation
    total_score = n_correct / n_total
    total_std = np.sqrt(total_score * (1 - total_score))

    # Compute the Item-Total Score Correlations
    item_counts = np.arange(dataset.shape[0])
    item_score_correlations = np.zeros(dataset.shape[0])
    for item_ndx in range(dataset.shape[0]):
        data_subset = np.delete(item_counts, item_ndx)

        # Sum with item removed
        total_sub_score = np.count_nonzero(mask_correct[data_subset], axis=0)

        # Point-Biserial Correlation
        mean_correct = total_sub_score[mask_correct[item_ndx]].mean()
        mean_incorrect = total_sub_score[mask_incorrect[item_ndx]].mean()

        scalar = np.sqrt(n_correct[item_ndx] * n_wrong[item_ndx] 
                         / n_total[item_ndx] / (n_total[item_ndx] - 1))

        item_score_correlations[item_ndx] = mean_correct - mean_incorrect
        item_score_correlations[item_ndx] *= scalar / total_sub_score.std(ddof=1)

    # Biserial Correlation
    z_score = norm.isf(total_score)
    y_probabililty = norm.pdf(z_score)
    biserial_correlation = (item_score_correlations 
                            * np.sqrt(total_score * (1 - total_score))
                            / y_probabililty)

    return {
        "Item Counts": n_total,
        "Mean": total_score,
        "Std":  total_std,
        "Item-Score Correlation": item_score_correlations,
        "Biserial Correlation": biserial_correlation
    }
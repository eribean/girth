import numpy as np
from scipy.stats import norm
from scipy.optimize import fminbound


__all__ = ["polyserial_correlation"]


def polyserial_correlation(continuous, ordinal):
    """Computes the polyserial correlation.
    
    Estimates the correlation value based on a bivariate
    normal distribution.
    
    Args:
        continuous: Continuous Measurement
        ordinal: Ordinal Measurement
        
    Returns:
        polyserial_correlation: converged value
        
    Notes:
        User must handle missing data
    """
    # Get the number of ordinal values
    values, counts = np.unique(ordinal, return_counts=True)
    
    # Compute the thresholds (tau's)
    thresholds = norm.isf(1 - counts.cumsum() / counts.sum())[:-1]
    
    # Standardize the continuous variable
    standardized_continuous = ((continuous - continuous.mean())
                               / continuous.std(ddof=1))

    def _min_func(correlation):
        denominator = np.sqrt(1 - correlation * correlation)
        k = standardized_continuous * correlation
        log_likelihood = 0
        
        for ndx, value in enumerate(values):
            mask = ordinal == value
            
            if ndx == 0:
                numerator = thresholds[ndx] - k[mask]
                probabilty = norm.cdf(numerator / denominator)
                
            elif ndx == (values.size -1):
                numerator = thresholds[ndx-1] - k[mask]
                probabilty = (1 - norm.cdf(numerator / denominator))
                
            else:
                numerator1 = thresholds[ndx] - k[mask]
                numerator2 = thresholds[ndx-1] - k[mask]
                probabilty = (norm.cdf(numerator1 / denominator)
                              - norm.cdf(numerator2 / denominator))
        
            log_likelihood -= np.log(probabilty).sum()
        
        return log_likelihood
        
    rho = fminbound(_min_func, -.99, .99)
    
    return rho
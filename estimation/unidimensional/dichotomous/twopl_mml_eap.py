import numpy as np
from scipy import stats

from girth.unidimensional.polytomous import grm_mml_eap


__all__ = ["twopl_mml_eap"]


def twopl_mml_eap(dataset, options=None):
    """Estimate parameters for a two parameter logistic model.

    Estimate the discrimination and difficulty parameters for
    a two parameter logistic model using a mixed Bayesian / 
    Marginal Maximum Likelihood algorithm, good for small 
    sample sizes

    Args:
        dataset: [n_items, n_participants] 2d array of measured responses
        options: dictionary with updates to default options

    Returns:
        results_dictionary:
        * Discrimination: (1d array) estimate of item discriminations
        * Difficulty: (1d array) estimates of item difficulties
        * LatentPDF: (object) contains information about the pdf
        * Rayleigh_Scale: (int) Rayleigh scale value of the discrimination prior
        * AIC: (dictionary) null model and final model AIC value
        * BIC: (dictionary) null model and final model BIC value

    Options:
        * estimate_distribution: Boolean    
        * number_of_samples: int >= 5    
        * max_iteration: int
        * distribution: callable
        * quadrature_bounds: (float, float)
        * quadrature_n: int
        * hyper_quadrature_n: int
    """
    result = grm_mml_eap(dataset.astype('int'), options)
    result['Difficulty'] = result['Difficulty'].squeeze()
    return result

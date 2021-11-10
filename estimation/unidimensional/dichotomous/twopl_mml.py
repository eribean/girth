from girth.unidimensional.polytomous import grm_mml


def twopl_mml(dataset, options=None):
    """ Estimates parameters in a 2PL IRT model.

    Args:
        dataset: [items x participants] matrix of True/False Values
        options: dictionary with updates to default options

    Returns:
        discrimination: (1d array) estimate of item discriminations
        difficulty: (1d array) estimates of item diffiulties
    
    Options:
        * max_iteration: int
        * distribution: callable
        * quadrature_bounds: (float, float)
        * quadrature_n: int
        * estimate_distribution: Boolean    
        * number_of_samples: int >= 5    
        * use_LUT: boolean    
    """
    results = grm_mml(dataset, options)
    results['Difficulty'] = results['Difficulty'].squeeze()
    return results
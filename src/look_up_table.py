import numpy as np
from scipy import interpolate
from scipy.special import expit

from girth.utils import validate_estimation_options, _get_quadrature_points


__all__ = ["create_beta_LUT"]


def _array_LUT(alpha, beta, theta, weight):
    """Computes the look up table values used to speed
       up parameter estimation
    """
    # 3D Array
    temp1 = theta[:, None] - beta[None, : ]
    temp2 = np.einsum('kj, i -> ijk', temp1, alpha)
    expit(temp2, out=temp2)

    # The Integral over theta is returned
    return np.einsum('ijk, k -> ij',temp2, weight)


def create_beta_LUT(alpha, beta, options=None):
    """Creates a Look Up Table to speed up conversion.
    
    Args:
        alpha: (array-like) [alpha_start, alpha_stop, alpha_n]
        beta: (array-like) [beta_start, beta_stop, beta_n]
        options: dictionary with updates to default options
        
    Returns:
        func: function that linear interpolates for 
              beta given (alpha, p-value)
        
    Options:
        * distribution: callable
        * quadrature_bounds: (float, float)
        * quadrature_n: int
    """
    options = validate_estimation_options(options)
    quad_start, quad_stop = options['quadrature_bounds']
    quad_n = options['quadrature_n']
    
    theta, weight = _get_quadrature_points(quad_n, quad_start, quad_stop)
    distribution = options['distribution'](theta)
    distribution_x_weight = distribution * weight
    
    alpha = np.linspace(*alpha)
    beta = np.linspace(*beta)
    
    # Get the index into the array
    interp_a = interpolate.interp1d(alpha, 
                                    np.arange(alpha.size, dtype='float'),
                                    kind='linear')
    
    the_output = _array_LUT(alpha, beta, theta, distribution_x_weight)
    
    func_list = list()
    for values in the_output:
        func_list.append(interpolate.interp1d(values, beta, kind='linear', 
                                                 fill_value=(beta[0], beta[-1]), 
                                                 bounds_error=False))
    
    # return function that returns beta value
    def interpolate_function(alpha_value, p_value):
        tmp = interp_a(alpha_value)
        tmpL = int(tmp)
        tmpH = int(tmp + 1)
        dx = tmp - tmpL
        return ((1 - dx) * func_list[tmpL](p_value) + 
                dx * func_list[tmpH](p_value))        
    
    return interpolate_function
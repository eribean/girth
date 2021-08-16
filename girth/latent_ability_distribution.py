import numpy as np
from scipy.optimize import fmin_slsqp

from girth import validate_estimation_options
from girth.utils import _get_quadrature_points


__all__ = ["LatentPDF", "CubicSplinePDF"]


def _parameter_constraints(current_parameters, sample_space):
    """Constraints placed on parameters for model indentification."""
    constraints = np.zeros((3,))

    # Integral Must = 1
    constraints[0] = current_parameters.sum() - 1

    # Mean Must = 0
    constraints[1] = current_parameters.dot(sample_space)

    # Variance Must = 1
    delta_sample = sample_space[1] - sample_space[0]
    constraints[2] = (current_parameters.dot(sample_space**2) - 
                      (3 - delta_sample**2) / 3)

    return constraints


def resample(cubic_spline_pdf, new_number_of_samples):
    """Resamples a cubic spline pdf to a new number of points.

    Args:
        cubic_spline_pdf: The current cubic spline object
        new_number_of_samples: (int) number of new samples

    Returns:
        cubic_spline_pdf_new: new CubicSplinePDF object with
                              coefficients set to correct values
    """
    if cubic_spline_pdf.number_of_samples == new_number_of_samples:
        raise AssertionError("The new number of samples must not "
                             'be equal to the current. '
                             f'Got {cubic_spline_pdf.number_of_samples} '
                             f'expected {new_number_of_samples}')
        
    # Create the new object
    options = {'number_of_samples': new_number_of_samples,
               'quadrature_bounds': cubic_spline_pdf.quad_bounds}
    new_spline_pdf = CubicSplinePDF(options)
    
    # Get the values at the new locations
    new_values = (cubic_spline_pdf(new_spline_pdf.sample_space) * 
                  new_spline_pdf.delta_sample)
    
    filter_matrix = new_spline_pdf.filter_matrix(new_spline_pdf.sample_space)
    filter_matrix = filter_matrix[:, 2:-2]
    
    def _local_min(estimates, _):
        return np.square(new_values - filter_matrix @ estimates).sum()
    
    initial_guess = np.linalg.pinv(filter_matrix) @ new_values

    coffs = fmin_slsqp(_local_min, initial_guess,
                       bounds=[(0, np.inf),] * new_number_of_samples, 
                       f_eqcons=_parameter_constraints,
                       args=(new_spline_pdf.sample_space[2:-2], ),
                       iprint=False)
    
    new_spline_pdf.update_coefficients(coffs)
    return new_spline_pdf


class CubicSplinePDF(object):
    """Implements a cubic spline pdf for ability estimation.
    
    Parameters:
        options: dictionary with updates to default options

    Options:
        * number_of_samples: int
        * quadrature_bounds: (float, float)
    """
    def __init__(self, options):
        """Constructor for cubic spline class."""
        # Create the parameters needed to
        # generate the cubic spline
        self.number_of_samples = options['number_of_samples']
        self.quad_bounds = options['quadrature_bounds']
        
        quad_stop = self.quad_bounds[1]
        bounds = (quad_stop * (self.number_of_samples - 1) / 
                  (self.number_of_samples + 3))

        self.delta_sample = 2 * bounds / (self.number_of_samples - 1)
        self.sample_space = (-(bounds + 2 * self.delta_sample) + 
                             np.arange(self.number_of_samples + 4) * 
                             self.delta_sample)
        self.sample_space_squared = np.square(self.sample_space)
        
        # Used to define cubic spline
        self.coefficients = np.zeros_like(self.sample_space)
   
    @staticmethod
    def cubic_spline(x_position):
        """Evaluates a cubic spline at input position 

        Args:
            x_position: location to sample the cubic spline
            
        Returns:
            y_value: result of cubic spline evaluationn
        """
        abs_x = np.abs(x_position)
        y_value = np.zeros_like(abs_x)
        y1 = 2. / 3. - np.power(abs_x, 2) + np.power(abs_x, 3) * 0.5
        y2 = np.power(2 - abs_x, 3) / 6
        y_value[abs_x < 2] = y2[abs_x < 2]
        y_value[abs_x < 1] = y1[abs_x < 1]
        
        return y_value
    
    def update_coefficients(self, new_coefficients):
        """Updates the cubic b-spline coefficients.
        
        Args:
            new_coefficients: (array) new values of the b-spline coefficients
        """
        if np.size(new_coefficients) != self.number_of_samples:
            raise AssertionError('New Coefficients must have size ' 
                                 f'= {self.number_of_samples}')
            
        self.coefficients[2:-2] = np.atleast_1d(new_coefficients)
        
    def continuous_pdf(self, bounds=(-5, 5), n_points=2001):
        """Returns a finely sampled cubic-spline pdf
        
        Args:
            bound: (tuple) start/stop locations of pdf
            n_points: number of points to use
            
        Returns:
            x_locations: (array) evalutation points
            continuous_pdf: (array) pdf normalized to 1
        """
        x_locations = np.linspace(*bounds, n_points) / self.delta_sample
        y_values = np.zeros_like(x_locations)

        for weight, offset in zip(self.coefficients, self.sample_space):
            y_values += self.cubic_spline(x_locations - 
                                          offset / self.delta_sample) * weight
        
        # Normalize to 1
        x_locations *= self.delta_sample
        y_values /= self.delta_sample
        return x_locations, y_values
    
    def filter_matrix(self, evaluation_locations):
        """Matrix to evaluate spline at input locations.
        
        Args:
            evaluation_locations: (tuple-like) computes the values
                                 at the supplied location
        Returns:
            filter_matrix: (2d array) Matrix needed to multiply
                           spline coefficients to get f(x)
        """
        x_positions = (np.atleast_1d(evaluation_locations)[:, None] - 
                       self.sample_space[None, :])
        x_positions /= self.delta_sample
        return self.cubic_spline(x_positions) 
       
    def __call__(self, evaluation_locations):
        """Evaluate the cubic spline at the input locations.
        
        Args:
            evaluation_locations: (tuple-like) computes the values
                                  at the supplied location
                                  
        Returns:
            y_values: result of evaluation
            
        Notes:
            Only to be evaluated at a few locations, to return
            the continuous pdf, use class method "continuous_pdf"
        """
        # If the number of evaluation locations is large
        # then look into implementing a filter-bank
        filter_matrix = self.filter_matrix(evaluation_locations)
                
        return np.sum(filter_matrix * self.coefficients, axis=1) / self.delta_sample 


class LatentPDF(object):
    """Controls the latent ability distribution definition.
    
    Parameters:
        options: dictionary with updates to default options
        
    Options:
        * estimate_distribution: Boolean    
        * number_of_samples: int >= 5 
        * distribution: callable
        * quadrature_bounds: (float, float)
        * quadrature_n: int
        
    Notes:
        The distribution in options is used as the
        null-hypothesis or if latent estimation
        is not desired
    """
    def __init__(self, options=None):
        """Constructor for latent estimation class"""
        options = validate_estimation_options(options)   
        
        # Quadrature Parameters
        quad_start, quad_stop = options['quadrature_bounds']
        quad_n = options['quadrature_n']
        theta, weights = _get_quadrature_points(quad_n, quad_start, quad_stop)
        self.quad_bounds = (quad_start, quad_stop)
        
        # The locations and weight to use by default
        self.quadrature_locations = theta
        self.weights = weights
        self.null_distribution = options['distribution'](theta)
        
        # Triggers to run the estimation or use default
        self.estimate_distribution = options['estimate_distribution']  
        self.n_points = options['number_of_samples'] if self.estimate_distribution else 3  
        
        # Initialize the first cubic-spline class
        # and set the distibution be an inverted U-shape
        cubic_spline = self._init_cubic_spline()
        cubic_spline.coefficients[self.n_points // 2 + 2] = 1
        self.cubic_splines = [cubic_spline]
    
    def _init_cubic_spline(self):
        """Initializes a cubic spline class."""
        options = {'number_of_samples': self.n_points,
                   'quadrature_bounds': self.quad_bounds}
        return CubicSplinePDF(options)
    
    def optimize_distribution(self, unweighted_integration):
        """Optimizes the distribution for the current parameters."""
        local_spline = self.cubic_splines[-1]
        
        filter_matrix = local_spline.filter_matrix(self.quadrature_locations)
        coefficients = np.zeros_like(local_spline.sample_space)
        
        def _local_minimizer(estimates, _=None):
            coefficients[2:-2] = estimates
            
            # Compute the distribution at the interpolation points
            distribution = np.sum(filter_matrix * coefficients, axis=1)
            distribution *= self.weights
            
            # Perform the numerical integration 
            otpt = np.sum(unweighted_integration * distribution, axis=1)
            
            return -np.log(otpt).sum()

        # run the minimization
        fmin_slsqp(_local_minimizer, 
                   local_spline.coefficients[2:-2], 
                   bounds=[(0, np.inf),] * local_spline.number_of_samples, 
                   f_eqcons=_parameter_constraints,
                   args=(local_spline.sample_space[2:-2],),
                   iprint=False)
        
        # Update the estimates for the cubic spline
        local_spline.update_coefficients(coefficients[2:-2])
        
        # return the distribution
        return local_spline(self.quadrature_locations)
    
    def compute_metrics(self, unweighted_integration, 
                        distribution_x_weights, k_params):
        """Computes the AIC and BIC for a function."""
        # Numericial Integration
        otpt = np.sum(unweighted_integration * 
                      distribution_x_weights, axis=1)
        log_likelihood = np.log(otpt).sum()
        aic = 2 * (k_params - log_likelihood)
        bic = (k_params * np.log(unweighted_integration.shape[0]) - 
               2 * log_likelihood)
        
        return aic, bic
        
    def __call__(self, unweighted_integration, iteration):
        """Runs a latent ability estimation iteration.
        
        Returns:
            distribution_x_weights: weighting function for integration
        """
        # Return null distribution if estimation not requested
        # or the first iteration in an estimation
        if (not self.estimate_distribution) or (iteration == 0):
            dist_x_weights = self.null_distribution * self.weights
        
        else:
            dist_x_weights = self.optimize_distribution(unweighted_integration)
            dist_x_weights *= self.weights
                    
        return dist_x_weights


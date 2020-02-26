import numpy as np
from scipy.special import roots_legendre


def _get_quadrature_points(n, a, b):
    """
        Utility function to get the legendre points,
        shifted from [-1, 1] to [a, b]

        Args:
            n: number of quadrature_points
            a: lower bound of integration
            b: upper bound of integration

        Returns:
            Array of quadrature_points for numerical integration

        Notes:
            A local function of the based fixed_quad found in scipy, this is
            done for processing optimization
    """
    x, w = roots_legendre(n)
    x = np.real(x)

    # Legendre domain is [-1, 1], convert to [a, b]
    return (b - a) * (x + 1) * 0.5 + a


def _compute_partial_integral(theta, difficulty, discrimination, the_sign):
    """
        Computes the partial integral for a set of item parameters

        Args:
            theta: (array) evaluation points
            difficulty: (array) set of difficulty parameters
            discrimination: (array | number) set of discrimination parameters
            the_sign:  (array) positive or negative sign
                               associated with response vector

        Returns:
            2d array of integration of items defined by "sign" parameters
                axis 0: individual persons
                axis 1: evaluation points (at theta)

        Notes:
            Implicitly multiplies the data by the gaussian distribution

        TODO:
            add address handle to vary the types of ability distributions
    """
    # Size single discrimination into full array
    if np.ndim(discrimination) < 1:
        discrimination = np.full(the_sign.shape[0], discrimination)

    # This represents a 3-dimensional array
    # [Response Set, Person, Theta]
    # The integration happens over response set and the result is an
    # array of [Person, Theta]
    kernel = the_sign[:, :, None] * np.ones((1, 1, theta.size))
    kernel *= discrimination[:, None, None]
    kernel *= (theta[None, None, :] - difficulty[:, None, None])

    # Distribution
    gauss = 1.0 / np.sqrt(2 * np.pi) * np.exp(-np.square(theta) / 2)

    return  gauss[None, :] * (1.0 / (1.0 + np.exp(kernel))).prod(axis=0).squeeze()


def irt_evaluation(difficulty, discrimination, thetas):
    """
        Evaluates an IRT model and returns the exact values.  This function
        supports only unidimemsional models

        Args:
            difficulty: [array] of difficulty parameters
            discrimination:  [array | number] of discrimination parameters
            thetas: [array] of person abilities

        Returns:
            dichotomous matrix of [difficulty.size x thetas.size] representing
            synthetic data
    """
    # If discrimination is a scalar, make it an array
    if not np.ndim(discrimination):
        discrimination = np.ones_like(difficulty) * discrimination

    kernel = difficulty[:, None] - thetas
    kernel *= discrimination[:, None]
    return 1.0 / (1 + np.exp(kernel))

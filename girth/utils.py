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
    x, _ = roots_legendre(n)
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

    # Distribution assumption
    # TODO: Make it a function pointer to allow various types
    gauss = 1.0 / np.sqrt(2 * np.pi) * np.exp(-np.square(theta) / 2)

    return  gauss[None, :] * (1.0 / (1.0 + np.exp(kernel))).prod(axis=0).squeeze()


def trim_response_set_and_counts(response_sets, counts):
    """
        Trims all true or all false responses from the response set/counts.

        Requires np.unique to have already been run so that the first and
        last response correspond to 0 / N

        Args:
            response_set:  (2D array) response set by persons obtained by running
                            numpy.unique
            counts:  counts associated with response set

        Returns
            response_set, counts updated to reflect removal of response patterns
    """

    # Remove the zero and full count values
    if(response_sets[:, 0].sum() == 0):
        response_sets = np.delete(response_sets, 0, axis=1)
        counts = np.delete(counts, 0)

    if(response_sets[:, -1].sum() == response_sets.shape[0]):
        response_sets = np.delete(response_sets, -1, axis=1)
        counts = np.delete(counts, -1)

    return response_sets, counts


def condition_polytomous_response(dataset, trim_ends=True):
    """
    Transforms item responses for easier use during parameter
    estimation

    This takes an input array of ordinal values and converts it into
    an array of linear indices to access difficulty parameters through 
    fancy indexing. 

    Args:
        dataset:  [n_items x n_takers] 2d array of ordinal responses
        trim_ends:  (boolean) trims responses that are either all no or all yes

    Returns:
        updated dataset array adjusted for linear indexing, 
        vector of lengths associated with each item
    """
    # Remove all no / yes endorsements
    min_value, max_value = dataset.min(), dataset.max()
    n_items = dataset.shape[0]

    if trim_ends:
        raw_score_sums = dataset.sum(0)
        mask = ((raw_score_sums == (n_items * min_value)) | 
                (raw_score_sums == (n_items * max_value)))
        dataset = dataset[:, ~mask]
    
    betas_length = np.zeros((n_items,), dtype='int')
    the_output = dataset.copy()
    the_output -= min_value
    
    # Loop over rows, determine the number of unique
    # responses, and replace with linear indexing
    cnt = 0
    for ndx, item in enumerate(the_output):
        values, indices = np.unique(item, return_inverse=True)
        betas_length[ndx] = values.size

        # Recode from zero to N-1
        values = np.arange(0, betas_length[ndx]) + cnt
        the_output[ndx] = values[indices]

        # Update linear index
        cnt += betas_length[ndx]

    return the_output, betas_length


def irt_evaluation(difficulty, discrimination, thetas):
    """
        Evaluates an IRT model and returns the exact values.  This function
        supports only unidimemsional models

        Assumes the model
            P(theta) = 1.0 / (1 + exp(discrimination * (theta - difficulty)))

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


def graded_probability_from_parameters(difficulty, discrimination, 
                                       max_response, response):
    """
        Computes the probability for a polytomous graded response model
        given a set of responses

        Args:
            difficulty: (2D array) [n_items, n_levels] of ordered difficulties
            discrimination: (1D Array) [n_items] of discrimination parameters
            max_response: (1D Array) [n_items] maximum number of levels for each item
            response: Response
    """
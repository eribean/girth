import numpy as np

from girth import irt_evaluation


def create_correlated_abilities(correlation_matrix, n_participants):
    """ Creates correlated ability parameters based on an input correlation matrix.

        This is a helper function for use in synthesizing multi-dimensional data
    
        Args:
            correlation_matrix: [2D Array, NxN] Symmetric matrix defining
                                the correlation between the abilities
            n_participants: number of participants to synthesize

        Returns:
            [N, n_participants] array of ability parameters
    """
    lower = np.linalg.cholesky(correlation_matrix)

    return lower @ np.random.randn(correlation_matrix.shape[0], n_participants)


def create_synthetic_irt_dichotomous(difficulty, discrimination, thetas,
                                     seed=None):
    """
        Creates synthetic IRT data to test parameters estimation
        functions.  Only for use with dichotomous outputs

        Assumes the model
            P(theta) = 1.0 / (1 + exp(discrimination * (theta - difficulty)))

        Args:
            difficulty: [array] of difficulty parameters
            discrimination:  [array | number] of discrimination parameters
            thetas: [array] of person abilities
            seed: Optional setting to reproduce results

        Returns:
            dichotomous matrix of [difficulty.size x thetas.size] representing
            synthetic data
    """
    if seed:
        np.random.seed(seed)

    continuous_output = irt_evaluation(difficulty, discrimination, thetas)

    # convert to binary based on probability
    random_compare = np.random.rand(*continuous_output.shape)

    return random_compare <= continuous_output


def create_synthetic_mirt_dichotomous(difficulty, discrimination, thetas,
                                     seed=None):
    """
        Creates synthetic multidimensional IRT data to test 
        parameters estimation functions.  Only for use with 
        dichotomous outputs

        Assumes the model
            P(theta) = 1.0 / (1 + exp(-1 * (dot(discrimination,theta) + difficulty)))

        Args:
            difficulty: [array, M] of difficulty parameters
            discrimination:  [2-D array, MxN] of discrimination parameters
            thetas: [2-D array, NxP] of person abilities
            seed: Optional setting to reproduce results

        Returns:
            dichotomous matrix of [difficulty.size x thetas.size] representing
            synthetic data

        Example:
            n_factors = 3
            n_items = 15
            n_people = 500
            difficulty = np.linspace(-2.5, 2.5, n_items)
            discrimination = np.random.randn(n_items, n_factors)
            thetas = np.random.randn(n_factors, n_people)

            synthetic_data = create_synthetic_mirt_dichotomous(difficulty, discrimination, thetas)
    """
    if seed:
        np.random.seed(seed)

    # If the input is just a vector of discriminations
    if (np.ndim(discrimination) == 1) or (discrimination.shape[0] == 1):
        discrimination = np.vstack((discrimination,) * difficulty.shape[0])

    # Inline computation of the logistic kernel
    kernel_terms = discrimination @ thetas
    kernel_terms += difficulty[:, None]
    continuous_output = 1.0 / (1.0 + np.exp(-kernel_terms))

    # convert to binary based on probability
    random_compare = np.random.rand(*continuous_output.shape)

    return random_compare <= continuous_output

import numpy as np

from irt_source.utils import irt_evaluation

def create_synthetic_irt_dichotomous(difficulty, discrimination, thetas,
                                     seed=None):
    """
        Creates synthetic IRT data to test parameters estimation
        functions.  Only for use with dichotomous outputs

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

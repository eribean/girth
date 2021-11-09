import numpy as np
from scipy.special import expit


__all__ = ["create_synthetic_irt_polytomous"]


# Private functions for polytomous outputs
def _my_digitize(the_input):
    """
    Private function to compute polytomous levels.
    The input has been concatenated to use the
    vectorize functions (value, thresholds)       
    """
    return np.searchsorted(the_input[1:], the_input[0])


def _graded_func(difficulty, discrimination, thetas, output):
    """
    Private function to compute the probabilities for
    the graded response model.  This is done in place
    and does not return anything
    """
    # This model is based on the difference of standard
    # logistic functions.
    temp = discrimination * thetas
    intercept = discrimination * difficulty

    # Do first level
    output[0] = 1.0 - expit(temp - intercept[0])

    for level_ndx in range(1, output.shape[0]-1):
        right = expit(temp - intercept[level_ndx])
        left = expit(temp - intercept[level_ndx-1])
        output[level_ndx] = left - right

    # Do last level
    output[-1] = expit(temp - intercept[-1])


def _graded_func_md(difficulty, discrimination, thetas, output):
    """
    Private function to compute the probabilities for
    the multidimensional graded response model.  
    """
    temp = discrimination @ thetas
    
    # Do first level
    output[0] = 1.0 - expit(temp + difficulty[0])

    for level_ndx in range(1, output.shape[0]-1):
        right = expit(temp + difficulty[level_ndx])
        left = expit(temp + difficulty[level_ndx-1])
        output[level_ndx] = left - right

    # Do last level
    output[-1] = expit(temp + difficulty[-1])


def _credit_func(difficulty, discrimination, thetas, output):
    """
    Private function to compute the probabilities for
    the partial credit model.  This is done in place
    and does not return anything
    """
    # This model is based on exponentials and normalized to
    # make sure the expected probablity is equal to one
    output *= 0.0  # clear any previous values
    output[1:, :] += thetas
    output[1:, :] -= difficulty[:, None]
    output *= discrimination
    np.cumsum(output, axis=0, out=output)
    np.exp(output, out=output)

    normalizing_term = 1.0 / np.sum(output, axis=0)
    output *= normalizing_term


def _unfold_func(difficulty, discrimination, thetas, output, src=0.):
    """
    Private function to compute the probabilities for
    the graded unfolding model.  This is done in place
    and does not return anything
    """
    # Call partial credit model on difficulties
    _credit_func(difficulty, discrimination, thetas, output)

    # Add the probabilities together, should always be even
    for ndx in range(output.shape[0] // 2):
        output[ndx] += (output[-(ndx+1)] * (1. - src))
        output[-(ndx+1)] *= src


def _check_difficulty_parameters(difficulty, model):
    """
    Given a model type, check the difficulty parameters
    for validity
    """
    max_value = difficulty.shape[1] + 1

    if model in ["grm"]:
        # Check that all the arguments are sorted
        if not np.all(difficulty[:, :-1] < difficulty[:, 1:]):
            raise AssertionError("Difficulty Parameters must be "
                                 "in ascending order")

    elif model in ['gum']:
        # Parameters must be odd
        if max_value % 2:
            raise AssertionError("There must be an odd number of "
                                 "difficulty parameters")

        # Parameters must be skew-symmetric about the center point
        middle_index = (difficulty.shape[1] - 1) // 2
        adjusted_difficulty = (difficulty -
                               difficulty[:, middle_index][:, None])

        if not np.all(np.abs(adjusted_difficulty.sum(axis=1)) < 1e-7):
            raise AssertionError("Difficulty Parameters must be "
                                 "symmetric about offset")

        max_value = middle_index + 1

    return max_value


def create_synthetic_irt_polytomous(difficulty, discrimination, thetas,
                                    model='grm', seed=None):
    """ Creates polytomous unidimensional synthetic IRT data.

    Creates polytomous output with specified number of levels from [1, levels]

    Args:
        difficulty: [2D array (items x n_levels-1)] of difficulty parameters
        discrimination:  [array | number] of discrimination parameters
        thetas: [array] of person abilities
        model: ["grm", "pcm", "gum", 'grm_md] string specifying which polytomous model to use
                'grm': Graded Response Model
                'pcm': Generalized Partial Credit Model
                'gum': Generalized Graded Unfolding Model
                'grm_md': Multidimensional Graded Response Model
        seed: Optional setting to reproduce results

    Returns:
        synthetic_data: (2d array) realization of possible response given parameters
    """
    difficulty = np.atleast_2d(difficulty)
    n_items, n_levels = difficulty.shape

    if n_levels == 1:
        raise AssertionError("Polytomous items must have more than 1 threshold")

    rng = np.random.default_rng(seed)

    # Check for single input of discrimination
    if np.atleast_1d(discrimination).size == 1:
        discrimination = np.full((n_items,), discrimination)

    theta_length = thetas.shape[-1]

    # Get the model to use, will throw error if not supported
    probability_func = {'grm': _graded_func,
                        'grm_md': _graded_func_md,
                        'pcm': _credit_func,
                        'gum': _unfold_func}[model.lower()]

    # Check difficulty parameters for validity
    clip_high = _check_difficulty_parameters(difficulty, model.lower())

    # Initialize output for memory concerns
    level_scratch = np.zeros((n_levels + 2, theta_length))
    output = np.zeros((n_items, theta_length), dtype='int')

    # Loop over items and compute probability estimates
    # for each of the levels and assign level based on
    # those probabilities
    for item_ndx in range(n_items):
        # Obtain the probabilities for the data (in-place)
        probability_func(difficulty[item_ndx], discrimination[item_ndx],
                         thetas, level_scratch[1:, :])

        # Get the thresholds of the levels
        np.cumsum(level_scratch[1:, :], axis=0, out=level_scratch[1:, :])
        level_scratch[0] = rng.uniform(size=theta_length)

        # Discritize the outputs based on the thresholds
        output[item_ndx] = np.apply_along_axis(
            _my_digitize, axis=0, arr=level_scratch)

    # Add 1 to return [1, n_levels]
    output += 1
    np.clip(output, 1, clip_high, out=output)
    return output

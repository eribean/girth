import numpy as np


def rasch_model(abilities, difficulty):
    """Computes the rasch_model sigmoid function.

    Args:
        abilities: numpy array vector of participant abilities
        difficulty: numpy array vector of item difficulties

    Returns:
        array of probabilities computed over abilities and difficulty
        [items, participants]
    """
    return one_parameter_model(abilities, difficulty, 1.0)


def one_parameter_model(abilities, difficulty, scale=1.7):
    """Computes the one parameter sigmoid function.

    Args:
        abilities: numpy array vector of participant abilities
        difficulty: numpy array vector of item difficulties
        scale: scalar value adjusting the discrimination (default=1.7)

    Returns:
        array of probabilities
        [items, participants]

    Note:
        If scale is set to 1, then use rasch_model instead
    """
    if np.ndim(scale) != 0:
        raise AssertionError("Scale value can only be scalar.")
    kernel = abilities[None, :] - difficulty[:, None]
    kernel *= -1 * scale

    return 1.0 / (1.0 + np.exp(kernel))


def two_parameter_model(abilities, difficulty, scale):
    """Computes the two parameter sigmoid function.

    Args:
        abilities: numpy array vector of participant abilities
        difficulty: numpy array vector of item difficulties
        scale: numpy array, vector of item discrimination parameters

    Returns:
        array of probabilities
        [items, participants]

    Note:
        difficulty length and discrimination length must be equal
    """
    if scale.size != difficulty.size:
        raise AssertionError("Scale sizes must be the same length as difficulty.")
    kernel = abilities[None, :] - difficulty[:, None]
    kernel *= scale[:, None] * -1

    return 1.0 / (1.0 + np.exp(-1 * kernel))


def three_parameter_model(abilites, difficulty, scale, guessing):
    """Computes the two parameter sigmoid function.

    Args:
        abilities: numpy array vector of participant abilities
        difficulty: numpy array vector of item difficulties
        scale: numpy array, vector of item discrimination parameters
        guessing: numpy array, vector of item guessing parameters

    Returns:
        array of probabilities computed over abilities and difficulty
        [items, participants]

    Note:
        difficulty length and discrimination length must be equal
    """
    if scale.size != guessing.size:
        raise AssertionError("Scale value can only be scalar.")

    temp = two_parameter_model(abilites, difficulty, scale)
    temp *= (1.0 - guessing[:, None])
    temp += guessing[:, None]

    return temp

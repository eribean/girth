import numpy as np


def rasch_model(irtArray):
    """Computes the rasch_model sigmoid function.

    Args:
        irtArray:  IRT Array with the following properties
            Abilities: array of participant abilities
            Difficulty: array of item difficulties

    Returns:
        array of probabilities computed over abilities and difficulty
        [items, participants]
    """
    kernel = irtArray['Difficulty'][:, None] - irtArray['Abilities'][None, :]

    return 1.0 / (1.0 + np.exp(kernel))

def one_parameter_model(irtArray):
    """Computes the one parameter sigmoid function.

    Args:
        irtArray:  IRT Array with the following properties
            Abilities: array of participant abilities
            Difficulty: array of item difficulties
            Discrimination: array of length 1 of item Discrimination

    Returns:
        array of probabilities
        [items, participants]

    Note:
        If scale is set to 1, then use rasch_model instead
    """
    if irtArray.shapes['Discrimination'] != 1:
        raise AssertionError("Discrimination value can only be scalar.")
    kernel = irtArray['Difficulty'][:, None] - irtArray['Abilities'][None, :]
    kernel *= irtArray['Discrimination']

    return 1.0 / (1.0 + np.exp(kernel))


def two_parameter_model(irtArray):
    """Computes the two parameter sigmoid function.

    Args:
        irtArray:  IRT Array with the following properties
            Abilities: array of participant abilities
            Difficulty: array of item difficulties
            Discrimination: array of item discriminations

    Returns:
        array of probabilities
        [items, participants]

    Note:
        difficulty length and discrimination length must be equal
    """
    if irtArray.shapes['Discrimination'] != irtArray.shapes['Difficulty']:
        raise AssertionError("Discrimination sizes must be the same length as difficulty.")
    kernel = irtArray['Difficulty'][:, None] - irtArray['Abilities'][None, :]
    kernel *= irtArray['Discrimination'][:, None]

    return 1.0 / (1.0 + np.exp(kernel))


def three_parameter_model(irtArray):
    """Computes the two parameter sigmoid function.

    Args:
        irtArray:  IRT Array with the following properties
            Abilities: array of participant abilities
            Difficulty: array of item difficulties
            Discrimination: array of item discriminations
            Guessing: array of item guessing parameters

    Returns:
        array of probabilities computed over abilities and difficulty
        [items, participants]

    Note:
        difficulty length and discrimination length must be equal
    """
    if irtArray.shapes['Guessing'] != irtArray.shapes['Difficulty']:
        raise AssertionError("Guessing sizes must be same length as difficulty.")

    guess = irtArray['Guessing'][:, None]

    temp = two_parameter_model(irtArray)
    temp *= (1.0 - guess)
    temp += guess

    return temp

"""Contains the ReLU function."""

import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    """
    Rectified Linear Unit.

    A piecewise-linear function that returns the input unless the value is less
    than 0, at which point it returns 0.

    https://www.bitlog.com/knowledge-base/machine-learning/activation-function/#relu

    Parameters
    ----------
    x : np.ndarray
        The input

    """
    result = x.copy()
    result[result < 0] = 0
    return result


def relu_prime(x: np.ndarray) -> np.ndarray:
    """
    Compute the derivative of the rectified linear unit.

    A piecewise-linear function that returns 0 less than 0 and 1 when >= 0.

    Parameters
    ----------
    x : float
        The input

    """
    result = x.copy()
    result.fill(1)
    result[x < 0] = 0
    return result

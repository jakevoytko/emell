"""Contains the quadratic cost function."""

from typing import cast

import numpy as np


def quadratic_loss(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Calculate the vectorized quadratic loss for a single example.

    Parameters
    ----------
    x : np.ndarray
        A vector of the predicted values.
    y : np.ndarray
        A vector of the expected values. Must match x's dimensions.

    """
    if x.ndim != 1:
        raise ValueError("Can only compute the quadratic cost for vectors")

    if x.shape != y.shape:
        raise ValueError(
            "Can only compute the quadratic cost of two vectors of equal shape"
        )

    result = 0.5 * ((y - x) ** 2)
    return cast(np.ndarray, result)


def delta_quadratic_loss(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Calculate the vectorization of the derivative of quadratic loss.

    Partial derivative taken with respect to x.
    """
    if x.ndim != 1:
        raise ValueError("Can only compute the quadratic cost for vectors")

    if x.shape != y.shape:
        raise ValueError(
            "Can only compute the quadratic cost of two vectors of equal shape"
        )

    result = x - y
    return cast(np.ndarray, result)

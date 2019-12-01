"""Contains the quadratic cost function."""

import numpy as np


def quadratic_cost(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate the mean square error of the input and output values.

    https://www.bitlog.com/knowledge-base/machine-learning/cost-function/#quadratic-cost

    Parameters
    ----------
    x : np.ndarray
        A vector of the estimates. Must be a vector.
    y : np.ndarray
        A vector of the actual values. Must match x's dimensions.

    """
    if x.ndim != 1:
        raise ValueError("Can only compute the quadratic cost for vectors")

    if x.shape != y.shape:
        raise ValueError(
            "Can only compute the quadratic cost of two vectors of equal shape"
        )

    num_samples = x.shape[0]

    return float(np.sum((x - y) ** 2)) / num_samples

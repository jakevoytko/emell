"""Contains the ReLU function."""


def relu(x: float) -> float:
    """
    Rectified Linear Unit.

    A piecewise-linear function that returns the input unless the value is less
    than 0, at which point it returns 0.

    https://www.bitlog.com/knowledge-base/machine-learning/activation-function/#relu

    Parameters
    ----------
    x : float
        The input

    """
    if x < 0:
        return 0
    return x

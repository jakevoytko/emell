"""Contains the L2 loss function."""


def l2(x: float, y: float) -> float:
    """
    Compute the L2 loss function.

    Return the square difference between the inputs.

    https://www.bitlog.com/knowledge-base/machine-learning/loss-function/#l2-error

    Parameters
    ----------
    x : float
        The estimate
    y : float
        The actual value

    """
    return (x - y) ** 2

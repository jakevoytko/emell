"""Contains the L1 loss function."""


def l1_loss(x: float, y: float) -> float:
    """
    Compute the L1 loss function.

    This is the absolute value of the difference of the inputs.

    https://www.bitlog.com/knowledge-base/machine-learning/loss-function/#l1-error

    Parameters
    ----------
    x : float
        The estimate
    y : float
        The actual value

    """
    return abs(x - y)

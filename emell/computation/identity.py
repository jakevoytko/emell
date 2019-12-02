"""Contains an identity function."""

from typing import TypeVar

T = TypeVar("T")


def identity(x: T) -> T:
    """
    Return the input.

    Parameters
    ----------
    x : Any
        The value to be returned.

    """
    return x

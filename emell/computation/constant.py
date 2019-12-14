"""Contains a constant function helper."""

from typing import Callable, TypeVar

T = TypeVar("T")


def constant(x: T) -> Callable[..., T]:
    """
    Return a function that always returns the same value.

    Parameters
    ----------
    x : any
        The value that will be returned in the function.

    """
    return lambda *args: x

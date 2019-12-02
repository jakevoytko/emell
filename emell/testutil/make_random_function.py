"""Contains a utility for making a random() substitute."""

from typing import Callable, List


def make_random_function(returns: List[float]) -> Callable[[], float]:
    """
    Return a function that returns the input values in sequence.

    Used to simulate a random function for tests.

    Parameters
    ----------
    returns : list
        The values to return in sequence.

    """
    returns_copy = returns.copy()

    def next_random() -> float:
        nonlocal returns_copy

        if len(returns_copy) == 0:
            raise IndexError("Needed more random numbers than were provided")
        value_to_return = returns_copy[0]
        returns_copy = returns_copy[1:]
        return value_to_return

    return next_random

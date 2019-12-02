"""Tests for relu.py"""

import unittest

import numpy as np

from emell.computation import relu, relu_prime


class TestRelu(unittest.TestCase):
    """Tests the relu function."""

    def test_relu(self) -> None:
        """Tests some basic inputs and outputs of the relu function."""
        np.testing.assert_array_equal(
            np.array([0, 0, 0.5, 1]), relu(np.array([-1, 0, 0.5, 1])),
        )

    def test_relu_prime(self) -> None:
        """Tests some basic inputs and outputs of the relu derivative function."""
        np.testing.assert_array_equal(
            np.array([0, 1, 1, 1]), relu_prime(np.array([-1, 0, 0.5, 1])),
        )


if __name__ == "__main__":
    unittest.main()

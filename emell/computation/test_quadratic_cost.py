"""Contains tests for test_quadratic_cost.py"""

import unittest

import numpy as np

from emell.computation import quadratic_cost

DELTA = 0.00001


class TestQuadraticCost(unittest.TestCase):
    """Contains tests for quadratic_cost"""

    def test_quadratic_cost(self) -> None:
        """Test quadratic_cost with basic values"""
        self.assertAlmostEqual(
            0.0, quadratic_cost(np.zeros((1)), np.zeros((1))), delta=DELTA
        )
        self.assertAlmostEqual(
            1.0, quadratic_cost(np.zeros((1)), np.ones((1))), delta=DELTA
        )
        self.assertAlmostEqual(
            1.0, quadratic_cost(np.zeros((100)), np.ones((100))), delta=DELTA
        )
        self.assertAlmostEqual(
            4.0, quadratic_cost(np.array([0]), np.array([2]),), delta=DELTA
        )
        self.assertAlmostEqual(
            4.0, quadratic_cost(np.array([0, 1]), np.array([2, 3]),), delta=DELTA
        )

    def test_quadratic_cost_errors(self) -> None:
        """Test quadratic_cost with input values that raise exceptions."""
        with self.assertRaises(ValueError):
            quadratic_cost(np.ndarray([]), np.ndarray([]))

        with self.assertRaises(ValueError):
            quadratic_cost(np.ones((1)), np.ones((2)))

        with self.assertRaises(ValueError):
            quadratic_cost(np.ones((2, 2)), np.ones((2, 2)))


if __name__ == "__main__":
    unittest.main()

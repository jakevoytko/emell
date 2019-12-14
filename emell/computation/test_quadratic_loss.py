"""Tests for quadratic_loss.py."""

import unittest

import numpy as np

from emell.computation import delta_quadratic_loss, quadratic_loss


class TestQuadraticLoss(unittest.TestCase):
    """Tests the quadratic loss function and its derivative."""

    def test_quadratic_loss(self) -> None:
        """Test cases for the vectorized quadratic loss function for one element."""
        np.testing.assert_array_equal(
            np.array([0]), quadratic_loss(np.array([0]), np.array([0]))
        )
        np.testing.assert_array_equal(
            np.array([0]), quadratic_loss(np.array([1]), np.array([1]))
        )
        np.testing.assert_array_equal(
            np.array([0]), quadratic_loss(np.array([-1]), np.array([-1]))
        )
        np.testing.assert_array_equal(
            np.array([2]), quadratic_loss(np.array([1]), np.array([-1]))
        )
        np.testing.assert_array_equal(
            np.array([2]), quadratic_loss(np.array([-1]), np.array([1]))
        )
        np.testing.assert_allclose(
            np.array([0.03125]), quadratic_loss(np.array([0.5]), np.array([0.25]))
        )

    def test_quadratic_loss_1_dimension(self) -> None:
        """
        Test cases for the vectorized quadratic loss function.

        This contains all the test cases in the 0d example.
        """
        np.testing.assert_allclose(
            np.array([0, 0, 0, 2, 2, 0.03125]),
            quadratic_loss(
                np.array([0, 1, -1, 1, -1, 0.5]), np.array([0, 1, -1, -1, 1, 0.25])
            ),
        )

    def test_delta_quadratic_loss(self) -> None:
        """Test cases for the derivative of the vectorized quadratic loss function."""
        np.testing.assert_array_equal(
            np.array([0]), delta_quadratic_loss(np.array([0]), np.array([0]))
        )
        np.testing.assert_array_equal(
            np.array([0]), delta_quadratic_loss(np.array([1]), np.array([1]))
        )
        np.testing.assert_array_equal(
            np.array([0]), delta_quadratic_loss(np.array([-1]), np.array([-1]))
        )
        np.testing.assert_array_equal(
            np.array([2]), delta_quadratic_loss(np.array([1]), np.array([-1]))
        )
        np.testing.assert_array_equal(
            np.array([-2]), delta_quadratic_loss(np.array([-1]), np.array([1]))
        )
        np.testing.assert_allclose(
            np.array([0.25]), delta_quadratic_loss(np.array([0.5]), np.array([0.25]))
        )

    def test_quadratic_delta_loss_1_dimension(self) -> None:
        """
        Test cases for the derivative of the vectorized quadratic loss function.

        This contains all the test cases in the 0d example.
        """
        np.testing.assert_allclose(
            np.array([0, 0, 0, 2, -2, 0.25]),
            delta_quadratic_loss(
                np.array([0, 1, -1, 1, -1, 0.5]), np.array([0, 1, -1, -1, 1, 0.25])
            ),
        )


if __name__ == "__main__":
    unittest.main()

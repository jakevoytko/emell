"""Tests for l1_loss.py."""

import unittest

from emell.computation import l1_loss


class TestL1(unittest.TestCase):
    """Tests the l1 loss function."""

    def test_l1(self) -> None:
        """Test cases for the l1 loss function."""
        self.assertEqual(0, l1_loss(0, 0))
        self.assertEqual(0, l1_loss(1, 1))
        self.assertEqual(0, l1_loss(-1, -1))
        self.assertEqual(2, l1_loss(1, -1))
        self.assertEqual(2, l1_loss(-1, 1))
        self.assertAlmostEqual(0.25, l1_loss(0.5, 0.25), delta=0.000001)


if __name__ == "__main__":
    unittest.main()

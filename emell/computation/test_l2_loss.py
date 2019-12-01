"""Tests for l2_loss.py."""

import unittest

from emell.computation import l2_loss


class TestL2(unittest.TestCase):
    """Tests the l2 loss function."""

    def test_l2(self) -> None:
        """Test cases for the l2 loss function."""
        self.assertEqual(0, l2_loss(0, 0))
        self.assertEqual(0, l2_loss(1, 1))
        self.assertEqual(0, l2_loss(-1, -1))
        self.assertEqual(4, l2_loss(1, -1))
        self.assertEqual(4, l2_loss(-1, 1))
        self.assertAlmostEqual(0.0625, l2_loss(0.5, 0.25), delta=0.000001)


if __name__ == "__main__":
    unittest.main()

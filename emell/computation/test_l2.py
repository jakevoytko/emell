"""Tests for l2.py."""

import unittest

from emell.computation.l2 import l2


class TestL2(unittest.TestCase):
    """Tests the l2 loss function."""

    def test_l2(self) -> None:
        """Test cases for the l2 loss function."""
        self.assertEqual(0, l2(0, 0))
        self.assertEqual(0, l2(1, 1))
        self.assertEqual(0, l2(-1, -1))
        self.assertEqual(4, l2(1, -1))
        self.assertEqual(4, l2(-1, 1))
        self.assertAlmostEqual(0.0625, l2(0.5, 0.25), delta=0.000001)


if __name__ == "__main__":
    unittest.main()

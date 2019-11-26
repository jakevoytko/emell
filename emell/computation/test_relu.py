"""Tests for relu.py"""
import unittest

from emell.computation.relu import relu


class TestRelu(unittest.TestCase):
    """Tests the relu function."""

    def test_relu(self) -> None:
        """Tests some basic inputs and outputs of the relu function."""
        self.assertEqual(0, relu(-1))
        self.assertEqual(0, relu(0))
        self.assertEqual(1, relu(1))
        self.assertEqual(0.5, relu(0.5))


if __name__ == "__main__":
    unittest.main()

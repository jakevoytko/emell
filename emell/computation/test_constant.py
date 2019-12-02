"""Contains tests for constant.py"""

import unittest

from emell.computation import constant


class TestConstant(unittest.TestCase):
    """Contains tests for the constant second-order function."""

    def test_constant(self) -> None:
        """Tests constant() against different inputs."""
        self.assertEqual("a", constant("a")())
        self.assertEqual(5, constant(5)())
        f = constant("a")
        self.assertEqual("a", f())
        self.assertEqual("a", f())
        self.assertEqual("a", f())


if __name__ == "__main__":
    unittest.main()

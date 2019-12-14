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

    def test_constant_multipleargs(self) -> None:
        """
        Tests that constant() can receive multiple args.

        This is mostly a test of the linter.
        """
        self.assertEqual("a", constant("a")(1, 2, 3, "goodbye world"))


if __name__ == "__main__":
    unittest.main()

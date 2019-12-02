"""Contains tests for identity.py"""

import unittest

from emell.computation import identity


class TestIdentity(unittest.TestCase):
    """Tests the identity function."""

    def test_identity(self) -> None:
        """Runs simple tests for identity()."""
        self.assertEqual("a", identity("a"))
        self.assertEqual(5, identity(5))
        self.assertEqual([3], identity([3]))


if __name__ == "__main__":
    unittest.main()

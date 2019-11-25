import unittest
from emell.computation.l1 import l1

class TestL1(unittest.TestCase):
    def testL1(self):
        self.assertEqual(0, l1(0, 0))
        self.assertEqual(0, l1(1, 1))
        self.assertEqual(0, l1(-1, -1))
        self.assertEqual(2, l1(1, -1))
        self.assertEqual(2, l1(-1, 1))
        self.assertAlmostEqual(.25, l1(.5, .25), delta=.000001)

if __name__ == '__main__':
    unittest.main()

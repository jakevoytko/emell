import unittest
from emell.computation.l2 import l2

class TestL2(unittest.TestCase):
    def testL1(self):
        self.assertEqual(0, l2(0, 0))
        self.assertEqual(0, l2(1, 1))
        self.assertEqual(0, l2(-1, -1))
        self.assertEqual(4, l2(1, -1))
        self.assertEqual(4, l2(-1, 1))
        self.assertAlmostEqual(.0625, l2(.5, .25), delta=.000001)

if __name__ == '__main__':
    unittest.main()

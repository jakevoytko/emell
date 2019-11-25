import unittest
from emell.computation.relu import relu

class TestRelu(unittest.TestCase):
    def testRelu(self):
        self.assertEqual(0, relu(-1))
        self.assertEqual(0, relu(0))
        self.assertEqual(1, relu(1))
        self.assertEqual(.5, relu(.5))

if __name__ == '__main__':
    unittest.main()

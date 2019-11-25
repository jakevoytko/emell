import emell.neuralnetwork.neuron as neuron
import numpy as np
import unittest
from emell.computation.relu import relu
from typing import Callable, List

def make_random_function(returns: List[float]) -> Callable[[],float]:
    returns_copy = returns.copy()

    def next_random() -> float:
        nonlocal returns_copy

        if len(returns_copy) == 0:
            raise IndexError('Needed more random numbers than were provided')
        value_to_return = returns_copy[0]
        returns_copy = returns_copy[1:]
        return value_to_return

    return next_random

delta = .00001

class TestNeuron(unittest.TestCase):
    def testOneWeight(self):
        n = neuron.Neuron(1, relu, make_random_function([.5, .5]))
        self.assertAlmostEqual(.5 + .5, n.compute(np.array([1])), delta = delta)
        self.assertAlmostEqual(.5 + .125, n.compute(np.array([.25])), delta = delta)

    def testTwoWeights(self):
        n = neuron.Neuron(2, relu, make_random_function([.5, .25, .1]))
        self.assertAlmostEqual(.5 + .25 + .1, n.compute(np.array([1, 1])))
        self.assertAlmostEqual(.5 + .1, n.compute(np.array([1, 0])))
        self.assertAlmostEqual(.25 + .1, n.compute(np.array([0, 1])))

    def testActivationFunctionUsed(self):
        n = neuron.Neuron(1, relu, make_random_function([1, 0]))
        self.assertAlmostEqual(.1, n.compute(np.array([.1])))
        self.assertAlmostEqual(0, n.compute(np.array([0])))
        # Check that the ReLU is clipping the value at 0.
        self.assertAlmostEqual(0, n.compute(np.array([-.1])))

    def testNoNeuronsThrows(self):
        with self.assertRaises(ValueError) as context:
            neuron.Neuron(0, relu, make_random_function([]))

if __name__ == '__main__':
    unittest.main()

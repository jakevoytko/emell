"""Tests for neuron.py."""
import unittest

import numpy as np

import emell.neuralnetwork.neuron as neuron
from emell.testutil import make_random_function

DELTA = 0.00001


def relu(x: float) -> float:
    """A testing implementation of relu."""
    if x < 0:
        return 0
    return x


class TestNeuron(unittest.TestCase):
    """Tests for the Neuron class."""

    def test_one_weight(self) -> None:
        """Verify the result of using one weight."""
        n = neuron.Neuron(1, relu, make_random_function([0.5, 0.5]))
        self.assertAlmostEqual(0.5 + 0.5, n.compute(np.array([1])), delta=DELTA)
        self.assertAlmostEqual(0.5 + 0.125, n.compute(np.array([0.25])), delta=DELTA)

    def test_two_weights(self) -> None:
        """Verify the result of using two weights."""
        n = neuron.Neuron(2, relu, make_random_function([0.5, 0.25, 0.1]))
        self.assertAlmostEqual(0.5 + 0.25 + 0.1, n.compute(np.array([1, 1])))
        self.assertAlmostEqual(0.5 + 0.1, n.compute(np.array([1, 0])))
        self.assertAlmostEqual(0.25 + 0.1, n.compute(np.array([0, 1])))

    def test_activation_function_used(self) -> None:
        """Verify that the activation function is used."""
        n = neuron.Neuron(1, relu, make_random_function([1, 0]))
        self.assertAlmostEqual(0.1, n.compute(np.array([0.1])))
        self.assertAlmostEqual(0, n.compute(np.array([0])))
        # Check that the ReLU is clipping the value at 0.
        self.assertAlmostEqual(0, n.compute(np.array([-0.1])))

    def test_no_neurons_throws(self) -> None:
        """Verify that a neuron cannot be initialized without a non-bias term."""
        with self.assertRaises(ValueError):
            neuron.Neuron(0, relu, make_random_function([]))


if __name__ == "__main__":
    unittest.main()

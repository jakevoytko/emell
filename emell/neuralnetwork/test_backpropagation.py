"""Contains tests for backpropagation.py"""

import unittest

import numpy as np

from emell.computation import delta_quadratic_loss, quadratic_loss, relu, relu_prime
from emell.neuralnetwork import Backpropagation, DenseLayer, Network
from emell.testutil import make_random_function


class BackpropagationTest(unittest.TestCase):
    """Tests for the Backpropagation class."""

    def test_backpropagation(self) -> None:
        """
        End-to-end test of backpropagation with 2 non-input layers.

        This is just an expected value test. At minimum, it's nice to have
        known output values when doing refactorings of the code.
        """
        network = Network(2)

        random_function = make_random_function(
            [
                # Weights for the first neuron of the first layer.
                0.1,
                0.2,
                # Weights for the second neuron of the first layer.
                0.3,
                0.4,
                # Weights for the third neuron of the first layer.
                0.5,
                0.6,
                # Weights for the only neuron of the second layer.
                0.7,
                0.8,
                0.9,
            ]
        )

        layer1 = DenseLayer(3, relu, relu_prime, random_function=random_function)
        layer2 = DenseLayer(1, relu, relu_prime, random_function=random_function)

        network.add_layer(layer1)
        network.add_layer(layer2)

        network_input = np.array([10, 100])
        expected = np.array([110])

        # This is basically a snapshot test - it asserts values based off their
        # previous values to ensure that refactorings do not break the algorithm.
        backpropagation = Backpropagation(
            network, quadratic_loss, delta_quadratic_loss, 0.001
        )
        np.testing.assert_allclose(
            backpropagation.train(network_input, expected), np.array([6048.816458])
        )
        np.testing.assert_allclose(
            backpropagation.train(network_input, expected), np.array([6036.160353])
        )
        np.testing.assert_allclose(
            backpropagation.train(network_input, expected), np.array([6023.077231])
        )


if __name__ == "__main__":
    unittest.main()

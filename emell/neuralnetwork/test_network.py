"""Contains tests for network.py."""

import unittest

import numpy as np

from emell.computation import constant, identity
from emell.neuralnetwork import DenseLayer, Network
from emell.testutil import make_random_function


class TestNetwork(unittest.TestCase):
    """Contains tests for the Network class."""

    def test_add(self) -> None:
        """
        Contains an end-to-end test just to verify that the network is working.

        Tests that an addition function can be written.
        """
        network = Network(2)
        random_function = make_random_function([0, 0])
        layer = DenseLayer(
            neuron_count=1,
            activation=identity,
            activation_prime=constant(np.array([0])),
            random_function=random_function,
        )
        network.add_layer(layer)

        layer.update_bias(np.array([0]))
        layer.update_weights(np.array([[1, 1]]))

        self.check_add(network, 0, 1, -1)
        self.check_add(network, 2, 1, 1)
        self.check_add(network, 12, 3, 9)
        self.check_add(network, -2, -1, -1)

    def test_add_bias(self) -> None:
        """
        Contains an end-to-end test just to verify that the network is working.

        Tests that an addition function can be written that also has a bias.
        """
        network = Network(2)
        random_function = make_random_function([0, 0])
        layer = DenseLayer(
            neuron_count=1,
            activation=identity,
            activation_prime=constant(np.array([0])),
            random_function=random_function,
        )
        network.add_layer(layer)

        layer.update_bias(np.array([1]))
        layer.update_weights(np.array([[1, 1]]))

        self.check_add(network, 1, 1, -1)
        self.check_add(network, 3, 1, 1)
        self.check_add(network, 13, 3, 9)
        self.check_add(network, -1, -1, -1)

    def check_add(self, network: Network, expected: float, x: float, y: float) -> None:
        """
        Attempts to add two numbers on the network and checks the output.
        """
        result = network.compute(np.array([x, y]))
        np.testing.assert_array_equal(np.array([expected]), result.output)

    def test_intensive(self) -> None:
        """
        Contains an end-to-end test with more layers.
        """
        network = Network(3)

        random_function = make_random_function(
            [
                # Three weights for layer 1 neuron 1.
                0.3,
                0.4,
                0.5,
                # Three weights for layer 1 neuron 2.
                0.6,
                0.7,
                0.8,
                # Two weights for layer 2.
                1.0,
                1.1,
            ]
        )

        layer1 = DenseLayer(
            neuron_count=2,
            activation=identity,
            activation_prime=constant(np.array([0])),
            random_function=random_function,
        )
        layer1.update_bias(np.array((0.1, 0.2)))

        layer2 = DenseLayer(
            neuron_count=1,
            activation=identity,
            activation_prime=constant(np.array([0])),
            random_function=random_function,
        )
        layer2.update_bias(np.array((0.9)))

        network.add_layer(layer1)
        network.add_layer(layer2)

        result = network.compute(np.array([10, 20, 30]))

        input_layer_expected = np.array([10, 20, 30])
        second_layer_expected = np.array(
            [
                0.01 * (10 * 0.3 + 20 * 0.4 + 30 * 0.5) + 0.1,
                0.01 * (10 * 0.6 + 20 * 0.7 + 30 * 0.8) + 0.2,
            ]
        )
        third_layer_expected = np.array(
            [
                0.01 * (second_layer_expected[0] * 1.0 + second_layer_expected[1] * 1.1)
                + 0.9,
            ]
        )

        np.testing.assert_allclose(
            input_layer_expected, result.results[0].weighted_output
        )
        np.testing.assert_allclose(input_layer_expected, result.results[0].output)
        np.testing.assert_allclose(
            second_layer_expected, result.results[1].weighted_output
        )
        np.testing.assert_allclose(second_layer_expected, result.results[1].output)
        np.testing.assert_allclose(
            third_layer_expected, result.results[2].weighted_output
        )
        np.testing.assert_allclose(third_layer_expected, result.results[2].output)

        np.testing.assert_allclose(result.output, third_layer_expected)


if __name__ == "__main__":
    unittest.main()

"""Contains tests for dense_layer.py."""

import unittest

import numpy as np

from emell.computation import relu, relu_prime
from emell.neuralnetwork import DenseLayer
from emell.testutil import make_random_function


class DenseLayerTest(unittest.TestCase):
    """Contains tests for the DenseLayer class."""

    def test_init(self) -> None:
        """Tests initialization of the dense layer."""
        random_function = make_random_function([])
        dense_layer = DenseLayer(
            neuron_count=2,
            activation=relu,
            activation_prime=relu_prime,
            random_function=random_function,
        )

        np.testing.assert_array_equal(np.array([0.0, 0.0]), dense_layer.bias)
        self.assertIsNone(dense_layer.weights)
        # Cannot get uninitialized weights.
        with self.assertRaises(RuntimeError):
            dense_layer.get_weights()
        self.assertEqual(2, dense_layer.neuron_count)

    def test_add_weights(self) -> None:
        """Tests setting the weights."""
        random_function = make_random_function(
            [
                # Weights for the first neuron.
                0.3,
                0.4,
                # Weights for the second neuron.
                0.5,
                0.6,
            ]
        )
        dense_layer = DenseLayer(
            neuron_count=2,
            activation=relu,
            activation_prime=relu_prime,
            random_function=random_function,
        )
        dense_layer.add_weights(2)
        np.testing.assert_array_equal(np.zeros([2]), dense_layer.bias)
        dense_layer.update_bias(np.array([0.1, 0.2]))
        np.testing.assert_array_equal(np.array([0.1, 0.2]), dense_layer.bias)
        self.assertEqual(2, dense_layer.neuron_count)
        np.testing.assert_allclose(
            0.01 * np.array([[0.3, 0.4], [0.5, 0.6],]), dense_layer.get_weights()
        )

    def test_update_weights_and_biases(self) -> None:
        """Tests updating the biases and weights of the dense layer."""
        random_function = make_random_function(
            [
                # Weights for the first neuron.
                0.0,
                0.0,
                # Weights for the second neuron.
                0.0,
                0.0,
            ]
        )
        dense_layer = DenseLayer(
            neuron_count=2,
            activation=relu,
            activation_prime=relu_prime,
            random_function=random_function,
        )
        dense_layer.add_weights(2)
        np.testing.assert_array_equal(np.zeros([2]), dense_layer.bias)
        np.testing.assert_array_equal(np.zeros([2, 2]), dense_layer.get_weights())

        dense_layer.update_bias(np.array([0.1, 0.2]))
        dense_layer.update_weights(np.array([[0.3, 0.4], [0.5, 0.6]]))

        np.testing.assert_array_equal(np.array([0.1, 0.2]), dense_layer.bias)
        np.testing.assert_allclose(
            np.array([[0.3, 0.4], [0.5, 0.6],]), dense_layer.get_weights()
        )

    def test_compute_simple(self) -> None:
        """Tests computing with the dense layer."""
        random_function = make_random_function(
            [
                # Weights for the first neuron.
                0.3,
                0.4,
                # Weights for the second neuron.
                0.5,
                0.6,
            ]
        )
        dense_layer = DenseLayer(
            neuron_count=2,
            activation=relu,
            activation_prime=relu_prime,
            random_function=random_function,
        )
        dense_layer.add_weights(2)
        dense_layer.update_bias(np.array([0.1, 0.2]))

        result = dense_layer.compute(np.array([1, 1]))
        expected_weighted_output = np.array(
            [0.01 * (0.3 + 0.4) + 0.1, 0.01 * (0.5 + 0.6) + 0.2]
        )
        self.assertEqual(dense_layer, result.layer)
        np.testing.assert_allclose(
            expected_weighted_output, result.weighted_output,
        )
        np.testing.assert_allclose(
            expected_weighted_output, result.output,
        )

    def test_compute_neuron_and_input_count_differ(self) -> None:
        """Tests computing with the dense layer."""
        random_function = make_random_function(
            [
                # Weights for the first neuron.
                0.4,
                0.5,
                # Weights for the second neuron.
                0.6,
                0.7,
                # Weights for the third neuron.
                0.8,
                0.9,
            ]
        )
        dense_layer = DenseLayer(
            neuron_count=3,
            activation=relu,
            activation_prime=relu_prime,
            random_function=random_function,
        )
        dense_layer.add_weights(2)
        dense_layer.update_bias(np.array([0.1, 0.2, 0.3]))

        result = dense_layer.compute(np.array([4, 5]))
        expected_weighted_output = np.array(
            [
                0.01 * (4 * 0.4 + 5 * 0.5) + 0.1,
                0.01 * (4 * 0.6 + 5 * 0.7) + 0.2,
                0.01 * (4 * 0.8 + 5 * 0.9) + 0.3,
            ]
        )
        self.assertEqual(dense_layer, result.layer)
        np.testing.assert_allclose(
            expected_weighted_output, result.weighted_output,
        )
        np.testing.assert_allclose(
            expected_weighted_output, result.output,
        )

    def test_compute_activation_function(self) -> None:
        """Tests that the activation function is used."""
        random_function = make_random_function(
            [
                # Weights for the first neuron.
                0.3,
                0.4,
                # Weights for the second neuron.
                0.5,
                0.6,
            ]
        )
        dense_layer = DenseLayer(
            neuron_count=2,
            activation=relu,
            activation_prime=relu_prime,
            random_function=random_function,
        )
        dense_layer.add_weights(2)
        dense_layer.update_bias(np.array([-10, -10]))
        result = dense_layer.compute(np.array([1, 1]))
        expected_weighted_output = np.array(
            [0.01 * (0.3 + 0.4) - 10, 0.01 * (0.5 + 0.6) - 10]
        )
        expected_output = np.zeros(2)

        np.testing.assert_allclose(
            expected_weighted_output, result.weighted_output,
        )
        np.testing.assert_allclose(
            expected_output, result.output,
        )


if __name__ == "__main__":
    unittest.main()

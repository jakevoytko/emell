"""Contains tests for input_layer.py."""

import unittest

import numpy as np

from emell.neuralnetwork import InputLayer, Layer


class InputLayerTest(unittest.TestCase):
    """Contains tests for the InputLayer class."""

    def test_init(self) -> None:
        """Verifies the properties are set properly."""
        layer1 = InputLayer(input_count=3)
        self.assertEqual(3, layer1.neuron_count)

        layer2 = InputLayer(input_count=1)
        self.assertEqual(1, layer2.neuron_count)

    def test_add_weights(self) -> None:
        """Verifies that calling add_weights errors."""
        layer = InputLayer(input_count=3)
        with self.assertRaises(NotImplementedError):
            layer.add_weights(2)

    def test_update_weights(self) -> None:
        """Verify that calling update_weights errors."""
        layer = InputLayer(input_count=3)
        with self.assertRaises(NotImplementedError):
            layer.update_weights(np.array([1, 2, 3]))

    def test_update_bias(self) -> None:
        """
        Verify that calling update bias does not error.

        Also checks that it has no impact on output.

        This is done during backpropagation to avoid special-casing the input
        layer.
        """
        layer = InputLayer(input_count=3)
        layer.update_bias(np.array([1, 1, 1]))
        layer_input = np.array([1, 2, 3])
        self.assertEqual(
            Layer.Result(layer, layer_input, layer_input), layer.compute(layer_input)
        )

    def test_get_weights(self) -> None:
        """Verify that getting the weights returns ones."""
        layer = InputLayer(input_count=3)
        np.testing.assert_array_equal(layer.get_weights(), np.ones([3]))

    def test_compute(self) -> None:
        """Verify that the input layer returns its inputs."""
        layer = InputLayer(input_count=3)
        layer_input = np.array([1, 2, 3])
        self.assertEqual(
            Layer.Result(layer, layer_input, layer_input), layer.compute(layer_input)
        )

    def test_compute_wrong_shape(self) -> None:
        """Verify that a mismatched shape raises a ValueError."""
        layer = InputLayer(input_count=4)
        layer_input = np.array([[1, 2], [3, 4]])
        with self.assertRaises(ValueError):
            layer.compute(layer_input)

    def test_compute_length_mismatch(self) -> None:
        """Verify that a vector length mismatch raises a ValueError."""
        layer = InputLayer(input_count=3)
        layer_input = np.array([1, 2])
        with self.assertRaises(ValueError):
            layer.compute(layer_input)


if __name__ == "__main__":
    unittest.main()

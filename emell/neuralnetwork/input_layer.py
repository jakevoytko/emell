"""Contains the definition of an input layer for a neural network."""

import numpy as np

from emell.computation import constant, identity
from emell.neuralnetwork.layer import Layer


class InputLayer(Layer):
    """
    The input layer to the network.

    Used so that some operations inside the neural network can be implemented
    uniformly. For instance, adding weights to the layer being added to the
    network.
    """

    def __init__(self, input_count: int):
        """
        Initialize the input layer.

        Parameters
        ----------
        input_count : int
            The number of parameters that are added to the network.

        """
        super().__init__(
            neuron_count=input_count,
            activation=identity,
            activation_prime=constant(np.zeros(input_count)),
        )

    def add_weights(self, weights_count: int) -> None:
        """Raise an error, as there are no input weights to the network."""
        raise NotImplementedError("An input layer cannot be given weights")

    def compute(self, layer_input: np.ndarray) -> "Layer.Result":
        """
        Compute the output of the input layer.

        Parmaeters
        ----------
        layer_input : np.ndarray
            The input to the layer. Also the input to the network.

        """
        if layer_input.ndim != 1:
            raise ValueError("Can only work on vectors")

        if layer_input.shape[0] != self.neuron_count:
            raise ValueError("The layer input does not match the declared input size")

        return Layer.Result(
            layer=self, weighted_output=layer_input, output=layer_input,
        )

"""Contains the abstract definition of a neural network layer."""

from typing import Callable, NamedTuple

import numpy as np


class Layer:
    """
    A base representation of a layer.

    This is intended to be a protocol in Python 3.8.
    """

    def __init__(
        self,
        neuron_count: int,
        activation: Callable[[np.ndarray], np.ndarray],
        activation_prime: Callable[[np.ndarray], np.ndarray],
    ):
        """
        Initialize a Layer.

        Parameters
        ----------
        neuron_count : int
            The number of outputs.
        activation : function(np.ndarray) -> np.ndarray
            The activation function for the layer. Must operate element-wise on
            the input vector.
        activation_prime : function(np.ndarray) -> np.ndarray
            The derivative of the activation function. Must also operate
            element-wise on the input vector.

        """
        super().__init__()
        self.neuron_count = neuron_count
        self.activation = activation
        self.activation_prime = activation_prime

    def add_weights(self, weights_count: int) -> None:
        """
        Add weights.

        Intended to be used by Network to match the output size of the previous
        layer with the weights for the added layer.

        Parameters
        ----------
        weights_count : int
            The number of weights to add.

        """
        raise NotImplementedError("The layer protocol is not usable.")

    def get_weights(self) -> np.ndarray:
        """
        Get the weights for the layer.

        Returns
        -------
        The weight matrix for the layer.

        """
        raise NotImplementedError("The layer protocol is not usable.")

    def update_weights(self, delta: np.ndarray) -> None:
        """
        Update weights.

        Intended to be used in backpropagation to modify the layer weights.
        Adds `delta` to the weights.

        Parameters
        ----------
        delta : np.ndarray
            The amount to add to the weights.

        """
        raise NotImplementedError("The layer protocol is not usable.")

    def update_bias(self, delta: np.ndarray) -> None:
        """
        Update bias.

        Intended to be used in backpropagation to modify the layer bias. Adds
        `delta` to the bias.

        Parameters
        ----------
        delta : np.ndarray
            The amount to add to the bias.

        """
        raise NotImplementedError("The layer protocol is not usable.")

    def compute(self, layer_input: np.ndarray) -> "Layer.Result":
        """
        Compute the output for the layer.

        Parameters
        ----------
        layer_input : np.ndarray
            The input to the layer.

        """
        raise NotImplementedError("The layer protocol type is not usable")

    class Result(NamedTuple):
        """
        Represents all outputs of a neural network computation.

        Useful for backpropagation.
        """

        layer: "Layer"
        weighted_output: np.ndarray
        output: np.ndarray

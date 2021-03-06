"""Contains the implementation of a densely-connected neural network layer."""

from random import random
from typing import Callable, Optional

import numpy as np

from emell.neuralnetwork.layer import Layer


class DenseLayer(Layer):
    """
    A dense layer for a feedforward neural network.

    More information can be found in my knowledge base:
    https://www.bitlog.com/knowledge-base/machine-learning/feedforward-neural-network/
    """

    def __init__(
        self,
        neuron_count: int,
        activation: Callable[[np.ndarray], np.ndarray],
        activation_prime: Callable[[np.ndarray], np.ndarray],
        random_function: Callable[[], float] = random,
    ):
        """
        Initialize the dense layer.

        Parameters
        ----------
        neuron_count : int
            The number of neurons in the layer
        activation : function(np.ndarray) -> np.ndarray
            The activation function for the layer. Must operate element-wise on
            the input vector.
        activation_prime : function(np.ndarray) -> np.ndarray
            The derivative of the activation function. Must also operate
            element-wise on the input vector.
        random_function : function() -> float
            A function to produce random numbers.

        """
        super().__init__(neuron_count, activation, activation_prime)
        self.neuron_count = neuron_count
        self.random = random_function
        self.bias = np.zeros((neuron_count))
        self.weights: Optional[np.ndarray] = None

    def add_weights(self, weights_count: int) -> None:
        """
        Add weights to the dense layer.

        Parameters
        ----------
        weights_count : int
            The number of input weights for each neuron.

        """
        self.weights = np.array(
            [
                [0.01 * self.random() for _ in range(weights_count)]
                for x in range(self.neuron_count)
            ]
        )

    def get_weights(self) -> np.ndarray:
        """
        Get the weights for the layer.

        Returns
        -------
        The weight matrix for the layer. A 2D matrix that is MxN, where M is
        the number of neurons and N is the number of inputs.

        """
        if self.weights is None:
            raise RuntimeError("Cannot get weights for an unimplemented layer.")

        return self.weights

    def update_weights(self, delta: np.ndarray) -> None:
        """
        Update the weights.

        Parameters
        ----------
        delta -> np.ndarray
            The amount to add to the weights.

        """
        self.weights += delta

    def update_bias(self, delta: np.ndarray) -> None:
        """
        Update the bias.

        Parameters
        ----------
        delta -> np.ndarray
            The amount to add to the bias.

        """
        self.bias += delta

    def compute(self, layer_input: np.ndarray) -> Layer.Result:
        """
        Compute the output of the dense layer.

        Parameters
        ----------
        layer_input : np.ndarray
            The input to the layer.

        """
        weighted_output = np.dot(self.weights, layer_input) + self.bias
        output = self.activation(weighted_output)
        return Layer.Result(layer=self, weighted_output=weighted_output, output=output,)

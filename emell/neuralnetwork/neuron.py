"""Contains the Neuron class for building neural network architectures."""

from random import random
from typing import Callable

import numpy as np


class Neuron:
    """
    Represents a single neuron in a neural network.

    Uses simple random initialization for the weights.

    https://www.bitlog.com/knowledge-base/machine-learning/neuron/
    """

    def __init__(
        self,
        number_of_inputs: int,
        activation_function: Callable[[float], float],
        random_function: Callable[[], float] = random,
    ):
        """
        Construct a neuron.

        Parameters
        ----------
        number_of_inputs : int
            The number of input values to the network.
        activation_function : function(float) -> float
            Applied to the sum of the weights and bias.
        random_function : function() -> float
            Injected function to provide values in the range [0.0 and 1.0]

        """
        super().__init__()
        if number_of_inputs < 1:
            raise ValueError("Number of NN inputs must be at least one")

        self.number_of_inputs = number_of_inputs
        self.activation_function = activation_function
        self.weights = np.array([random_function() for i in range(number_of_inputs)])
        self.bias = random_function()

    def compute(self, inputs: np.ndarray) -> float:
        """
        Generate an output value from the inputs.

        Parameters
        ----------
        inputs : list
            The number of inputs to the network. Must match number_of_inputs at
            construction.

        """
        if len(inputs) != self.number_of_inputs:
            raise ValueError("Incorrect input size %s" % len(inputs))

        # Compute the non-bias terms.
        summed_weights = np.dot(inputs, self.weights)

        # Compute the bias.
        bias = 1 * self.bias

        # Return the total sum passed through the activation function
        return self.activation_function(summed_weights + bias)

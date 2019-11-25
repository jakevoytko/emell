from random import random
from typing import Callable, List
import numpy as np

class Neuron(object):
    def __init__(
        self,
        number_of_inputs: int,
        activation_function: Callable[[float], float],
        random_function: Callable[[], float] = random
    ):
        super().__init__()
        if (number_of_inputs < 1):
            raise ValueError('Number of NN inputs must be at least one')

        self.number_of_inputs = number_of_inputs
        self.activation_function = activation_function
        self.weights = np.array([random_function() for i in range(number_of_inputs)])
        self.bias = random_function()
    
    def compute(self, inputs: List[float]) -> float:
        if len(inputs) != self.number_of_inputs:
            raise ValueError('Incorrect input size %s' % len(inputs))

        # Compute the non-bias terms.
        summed_weights = np.dot(inputs, self.weights)

        # Compute the bias.
        bias = 1 * self.bias

        # Return the total sum passed through the activation function
        return self.activation_function(summed_weights + bias)

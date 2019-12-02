"""Contains the definition of a feedforward neural network."""

from typing import List, NamedTuple

import numpy as np

from emell.neuralnetwork.input_layer import InputLayer
from emell.neuralnetwork.layer import Layer


class Network:
    """Represents a feedforward neural network."""

    def __init__(self, input_count: int):
        """
        Initialize the network.

        Parameters
        ----------
        input_count : int
            The size of the input.

        """
        super().__init__()
        self.layers: List[Layer] = [InputLayer(input_count)]

    def add_layer(self, layer: Layer) -> None:
        """
        Add a layer to the network by appending it to be the last layer.

        This has the side effect of initializing the input weights to the layer.

        Parameters
        ----------
        layer : Layer
            The layer being appended to the network.

        """
        layer.add_weights(self.layers[-1].neuron_count)
        self.layers.append(layer)

    def compute(self, network_input: np.ndarray) -> "Network.Result":
        """
        Compute the output for the network.

        Parameters
        ----------
        network_input : np.ndarray
            The input to the network.

        """
        intermediate = network_input
        results = []
        for layer in self.layers:
            result = layer.compute(intermediate)
            intermediate = result.output
            results.append(result)
        return Network.Result(results, intermediate)

    class Result(NamedTuple):
        """Represents the result of computation over the entire network."""

        results: List[Layer.Result]
        output: np.ndarray

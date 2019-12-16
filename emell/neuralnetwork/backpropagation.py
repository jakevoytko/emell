"""Contains the backpropagation algorithm."""

from typing import Callable

import numpy as np

from emell.neuralnetwork.network import Network


class Backpropagation:
    """
    A class that performs the backpropagation algorithm.

    More information can be found in my knowledge base:
    https://www.bitlog.com/knowledge-base/machine-learning/backpropagation/
    """

    def __init__(
        self,
        network: Network,
        loss_function: Callable[[np.ndarray, np.ndarray], np.ndarray],
        loss_delta_function: Callable[[np.ndarray, np.ndarray], np.ndarray],
        alpha: float,
    ):
        """
        Initialize the environment for the backpropagation algorithm.

        Parameters
        ----------
        network -> Network
            The network to train.

        loss_function -> callable
            The loss function.

        loss_delta -> callable
            The vectorized derivative of the loss function with respect to each
            output.

        alpha -> float
            The training rate.
        """
        super().__init__()
        self.network = network
        self.loss_function: Callable[
            [np.ndarray, np.ndarray], np.ndarray
        ] = loss_function
        self.loss_delta_function = loss_delta_function
        self.alpha = alpha

    def train(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Perform backpropagation on the input.

        Parameters
        ----------
        x -> np.ndarray
            The input training example.
        y -> np.ndarray
            The expected output.

        Returns
        -------
        The loss for the given training example.

        """
        result = self.network.compute(x)
        layer_results = result.results

        # Compute the rate of change of the error, i.e. the rate of change of
        # the loss function times the rate of change of the activation
        # function.
        #
        # Bootstraps the loop by calculating the "next layer" values needed
        # for the final layer.
        delta_loss = self.loss_delta_function(y, result.output)
        delta_next_layer = delta_loss * layer_results[-1].layer.activation_prime(
            layer_results[-1].weighted_output
        )
        previous_layer_result = None

        for layer_result in reversed(result.results):
            layer = layer_result.layer
            weighted_output = layer_result.weighted_output
            output = layer_result.output

            # Calculate the rate of change of the error at the current layer.
            delta_layer = delta_next_layer * layer.activation_prime(weighted_output)

            # The rate of change w.r.t. the bias is the error at the layer.
            layer.update_bias(self.alpha * delta_layer)

            # Calculate and cache updates needed to get results for the next
            # layer.
            if previous_layer_result is not None:
                # The vectorized rate of change of the weights of the next
                # layer is the activation of this layer times the error in
                # the next layer.
                weight_update = self.alpha * (output @ delta_next_layer)
                previous_layer_result.layer.update_weights(weight_update)

            # Precompute for the next iteration.
            delta_next_layer = np.transpose(layer.get_weights()) @ delta_layer
            previous_layer_result = layer_result

        # The loss isn't used in backpropagation. It's just interesting to know.
        return self.loss_function(y, result.output)

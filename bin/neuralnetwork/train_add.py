"""A script to train a neural network to add two numbers."""

from emell.computation import identity, constant, quadratic_loss, delta_quadratic_loss
from emell.neuralnetwork import Network, DenseLayer, Backpropagation

import numpy as np
from random import randint, random


def add():
    """
    Naively trains a neural network that adds two numbers.
    
    This is meant to be the simplest possible example to demonstrate feedforward
    neural networks with backpropagation.
    """
    print("Learning x1 + x2 = y")
    alpha = .0001

    network = Network(2)
    network.add_layer(DenseLayer(1, identity, constant(1)))

    backpropagation = Backpropagation(network, quadratic_loss, delta_quadratic_loss, alpha)
    
    for i in range(100000):
        x0 = randint(-50, 50)
        x1 = randint(-50, 50)
        y = x0 + x1
        error = backpropagation.train(np.array([x0, x1]), np.array([y]))

        # Print the error every 1000 training examples.
        if (i % 10000) == 0:
            print('%s%% done. Error: %s' % (100.0*i/100000.0, error))

    # Print the weights and bias at the end.
    print('Final network configuration for straight addition')
    print('This is expected to converge to roughly the correct weights')
    print(network.layers[1].weights, network.layers[1].bias)


def formula():
    """
    Trains a neural network that does the formula .5 * x1 + x2 + 1 = 3.
    
    This is meant to be the simplest possible example to demonstrate feedforward
    neural networks with backpropagation.
    """
    print("Learning .5 * x1 + x2 + 1 = y")
    alpha = .0001

    network = Network(2)
    network.add_layer(DenseLayer(1, identity, constant(1)))

    backpropagation = Backpropagation(network, quadratic_loss, delta_quadratic_loss, alpha)
    
    for i in range(100000):
        x0 = randint(-50, 50)
        x1 = randint(-50, 50)
        y = .5 * x0 + x1 + 1
        error = backpropagation.train(np.array([x0, x1]), np.array([y]))

        # Print the error every 1000 training examples.
        if (i % 10000) == 0:
            print('%s%% done. Error at this step: %s' % (100.0*i/100000.0, error))

    # Print the weights and bias at the end.
    print('Final network configuration for the formula addition')
    print('This is not expected to converge to the correct weights')
    print(network.layers[1].weights, network.layers[1].bias)


def main():
    """
    A demo that learns different configurations for adding numbers.
    """
    add()
    formula()

if __name__ == '__main__':
    main()

# August Leander Hedman
# augu1789@edu.nextkbh.dk
# NEXT Sukkertoppen, S3n

import numpy as np
from typing import Callable, List, Tuple

class Layer:
    def __init__(self):
        self.weights = None
        self.biases = None
        self.value = None

    def initalize_layer(self, inputNodes : int, outputNodes : int) -> None:
        self.weights = np.random.randn(outputNodes, inputNodes) * np.sqrt(2 / inputNodes)
        self.biases = np.random.randn(outputNodes,)

class Input(Layer):
    def __init__(self, inputShape: Tuple[int]):
        self.inputShape = inputShape
        self.numNodes = np.prod(inputShape)

    def feedforward(self, a: np.ndarray) -> np.ndarray:
        return a

class Dense(Layer):
    def __init__(self, numNodes : int, activation : Callable[[np.ndarray], np.ndarray]):
        super().__init__()
        self.numNodes = numNodes
        self.activation = activation

    def feedforward(self, activations : np.ndarray) -> np.ndarray:
        # Beregn z = W * a + b 
        z = np.dot(self.weights, activations) + self.biases
        # Beregn a = f(z) (f er aktiveringsfunktionen)
        return self.activation(z)
    
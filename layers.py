# August Leander Hedman
# augu1789@edu.nextkbh.dk
# NEXT Sukkertoppen, S3n

import numpy as np
from typing import Callable, List, Tuple, Union
import activation

class Layer:
    def __init__(self):
        self.weights = None
        self.biases = None
        self.value = None
        self.a = None
        self.z = None
        self.activation_function = None

    def initalize_layer(self, inputNodes : int, outputNodes : int) -> None:
        self.weights = np.random.randn(outputNodes, inputNodes) * np.sqrt(2 / inputNodes)
        self.biases = np.random.randn(outputNodes,)

    def store_layer_values(self, data_point):
        self.z = np.dot(self.weights, data_point) + self.biases
        self.a = self.activation_function(self.z)
        return self.a


class Input(Layer):
    def __init__(self, inputShape: Tuple[int]):
        self.inputShape = inputShape
        self.numNodes = np.prod(inputShape)
        self.a = None

    def feedForward(self, a: np.ndarray) -> np.ndarray:
        return a
    
    def store_layer_values(self, data_point):
        self.a = data_point
        return self.a

class Dense(Layer):
    def __init__(self, numNodes : int, activation : Callable[[np.ndarray], np.ndarray]):
        super().__init__()
        self.numNodes = numNodes
        self.activation_function = activation

    def feedForward(self, activations : np.ndarray, type="forward") -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        # Beregn z = W * a + b 
        z = np.dot(self.weights, activations) + self.biases
        # Beregn a = f(z) (f er aktiveringsfunktionen)
        #if type=="backpropagation": return (z, self.activation_function(z))
        return self.activation_function(z)
    
    def outputLayer_gradients(self, y_actual, prev_a, cost):
        # \frac{\partial a}{\partial z} \cdot \frac{\partial C}{\partial a}
        nodeValue = cost.cost_derivative(y_true=y_actual, y_pred=self.a) * self.activation_function(activations=self.z, derivative=True)

        # nabla_w[index] = a_{l-1} \cdot output_nodeValue
        nabla_w = np.outer(nodeValue, prev_a)

        #nabla_b[index] = 1 \cdot output_nodeValue
        nabla_b = nodeValue

        return nabla_w, nabla_b
    
    def hiddenLayer_gradients(self, next_w, next_nodeValue):
        # \frac{\partial a}{\partial z} \cdot \frac{\partial C}{\partial a}
        nodeValue = np.dot(next_w.T, next_nodeValue) * self.activation_function(activations=self.z, derivative=True)

        # nabla_w[index] = a_{l-1} \cdot output_nodeValue
        nabla_w = np.outer(nodeValue, self.a)

        #nabla_b[index] = 1 \cdot output_nodeValue
        nabla_b = nodeValue

        return nabla_w, nabla_b
    
    
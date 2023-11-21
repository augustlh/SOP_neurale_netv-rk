# August Leander Hedman
# augu1789@edu.nextkbh.dk
# NEXT Sukkertoppen, S3n

import numpy as np
from typing import Callable, List, Tuple
from layers import Input, Layer

Label = int
Activation = np.ndarray

class Network:
    def __init__(self) -> None:
        self.layers: List[Layer] = []
        self.cost = MeanSquaredError

    def add(self, layer : Layer):
        if len(self.layers) == 0:
            if not isinstance(layer, Input):
                raise TypeError("Det første lag skal være et Input lag")
        else:
            if isinstance(layer, Input):
                raise TypeError("Input laget skal være det første lag")
            
        self.layers.append(layer)

    def build(self) -> None:
        if len(self.layers) == 0: raise ValueError("Netværket skal have mindst et lag")
        for i in range(1, len(self.layers)):
            self.layers[i].initalize_layer(self.layers[i-1].numNodes, self.layers[i].numNodes)

    def feedForward(self, a : np.ndarray) -> np.ndarray:
        for layer in self.layers:
            a = layer.feedForward(a)
        return a
    
    def compute_gradients(self, data):
        # Formatering af data for et givent træningseksempel. Label onehot-encodes for at kunne udføre vektor operationer.
        X, y_actual = data[0], np.eye(10)[data[1]]
        nabla_w = [None] * len(self.layers)
        nabla_b = [None] * len(self.layers)

        # Find og gem alle z, a værdier for alle lag
        self.calculate_outputs(X)

        #Beregn fejlen for det sidste lag
        index = len(self.layers) - 1
        output_layer = self.layers[index]
        
        # --- Beregn gradienterne for output laget ---
        # \frac{\partial C}{\partial W^index} = \frac{\partial z^index}{\partial w^index} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial C}{\partial a}
        #                                     = a^(index - 1) \cdot y'(z^index) \cdot 2(a^index - y)

        # \frac{\partial a}{\partial z} \cdot \frac{\partial C}{\partial a}
        error = self.cost.cost_derivative(y_true=y_actual, y_pred=output_layer.a) * output_layer.activation_function(activations=output_layer.z, derivative=True)

        nabla_w[-1] = np.outer(error, self.layers[index-1].a)
        nabla_b[-1] = error

        return nabla_w, nabla_b
        

    def calculate_outputs(self, data_point):
        for layer in self.layers:
            data_point = layer.store_layer_values(data_point)

    
class MeanSquaredError:
    @staticmethod
    def cost(y_true : np.ndarray, y_pred : np.ndarray) -> float:
        return np.sum(np.square(y_true - y_pred)) 
    
    @staticmethod
    def cost_derivative(y_true : np.ndarray, y_pred : np.ndarray) -> np.ndarray:
        return 2 * (y_pred - y_true)
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
    
    def backProp(self, data : Tuple[Activation, Label]) -> Tuple[np.ndarray, np.ndarray]:
        X, y = data
        nabla_w = [np.zeros(layer.weights.shape) for layer in self.layers[1:]]

        # Beregning af alle aktiveringsværdier og z-værdier i netværket
        z = []
        a = [data[0]]
        for layer in self.layers[1:]:
            print("l")


    
    def updateParameters(self, gradients : List[Tuple[np.ndarray, np.ndarray]]) -> None:
        pass
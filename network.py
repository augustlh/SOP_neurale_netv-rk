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
        # Kald backprop for  et layer[i-1] med et argument værdier[i-1]
        # Definer et array som gemmer alle aktiveringsværdierne for hvert lag for et givent træningseksempel. Find således disse værdier
        #o utput error er givet ved a[:-1] - one_hot(Label)
        # # Vi ønsker at finde alle aktiveringsværdier og alle z-værdier gennem hele netværket for et givent træningseksempel
        X, y = data
        nabla_w, nabla_b = [], []

        for i in range(1, len(self.layers)):
            n_w, n_b = self.layers[i].backProp(X)
            X = self.layers[i].feedForward(X)
            nabla_w.append(n_w)
            nabla_b.append(n_b)

        return (nabla_w, nabla_b)
    
    def updateParameters(self, gradients : List[Tuple[np.ndarray, np.ndarray]]) -> None:
        pass
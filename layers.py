# August Leander Hedman
# augu1789@edu.nextkbh.dk
# NEXT Sukkertoppen, S3n


# Importer biblioteker og moduler, som er relevante for programmet
import numpy as np
from typing import Callable, List, Tuple, Union
import activation

# Layer klassen
class Layer:
    def __init__(self):
        # Tideling af værdi til diverse attributter
        self.weights = None
        self.biases = None
        self.value = None
        self.a = None
        self.z = None
        self.activation_function = None

    # Metode til at initalisere et lag
    def initalize_layer(self, inputNodes : int, outputNodes : int) -> None:
        # Der initaliseres vægt og bias matricer med henholdsvis (outputNodes, inputNodes) og (outputNodes)
        # Hvis et lag har 10 neuroner og det forrige lag har 784 aktiveringer, så vil vægt matricen skulle have shapen (10, 784)
        self.weights = np.random.randn(outputNodes, inputNodes) * np.sqrt(2 / inputNodes)
        # Hvis et lag har 10 neuroner vil hver neuron have sin egen bias, hvilket repræsenteres i en kollone-vektor af formen (10,)
        self.biases = np.random.randn(outputNodes,)

    # Metode til at gemme beregnede værdier ved forwardprop (anvendes ved backprop)
    def store_layer_values(self, data_point):
        self.z = np.dot(self.weights, data_point) + self.biases
        self.a = self.activation_function(self.z)
        return self.a

# Nedarver fra Layer klassen og er et lag af typen input
class Input(Layer):
    def __init__(self, inputShape: Tuple[int]):
        self.inputShape = inputShape
        # Beregner antallet af input-værdier på inputShapen
        # Et gray scale billede 28x1 vil have 28x28 = 784 input
        self.numNodes = np.prod(inputShape)
        # Tildeler self.a værdien none
        self.a = None

    # Metode til videregivning af information til et andet lag i netværket.
    def feedForward(self, a: np.ndarray) -> np.ndarray:
        return a
    # Metode til at gemme beregnede værdier ved forwardprop (anvendes ved backprop)
    def store_layer_values(self, data_point):
        self.a = data_point
        return self.a

# Nedarver fra Layer klassen og er et lag af typen Dense
class Dense(Layer):
    def __init__(self, numNodes : int, activation : Callable[[np.ndarray], np.ndarray]):
        # Hopper tilbage i koden og kører superklassens constructor og definerer yderligere attributer relevante for denne type Lag
        super().__init__()
        self.numNodes = numNodes
        self.activation_function = activation

    def feedForward(self, activations : np.ndarray, type="forward") -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        # Beregn z = W * a + b 
        z = np.dot(self.weights, activations) + self.biases
        # Beregn a = f(z) (f er aktiveringsfunktionen)
        return self.activation_function(z)
    
    
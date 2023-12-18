# August Leander Hedman
# augu1789@edu.nextkbh.dk
# NEXT Sukkertoppen, S3n

# Importer biblioteker og moduler, som er relevante for programmet
import numpy as np
from typing import Callable, List, Tuple
from layers import Input, Layer
import os
import matplotlib.pyplot as plt
import visualizer
import random

#Definer typer, som anvedes
Label = int
Activation = np.ndarray
Data = List[Tuple[Activation, Label]]

class Network:
    def __init__(self) -> None:
        # Et netværk består af lag, som gemmes i listen layers
        self.layers: List[Layer] = []
        self.cost = MeanSquaredError

    # Metode til at tilføje et lag til objektet
    def add(self, layer : Layer):
        if len(self.layers) == 0:
            if not isinstance(layer, Input):
                raise TypeError("Det første lag skal være et Input lag")
        else:
            if isinstance(layer, Input):
                raise TypeError("Input laget skal være det første lag")
            
        self.layers.append(layer)

    # Metoder der "bygger" netværket ved at initalizere lagene
    def build(self) -> None:
        if len(self.layers) == 0: raise ValueError("Netværket skal have mindst et lag")
        for i in range(1, len(self.layers)):
            self.layers[i].initalize_layer(self.layers[i-1].numNodes, self.layers[i].numNodes)

    # Metode der tager et input og giver netværkets output
    def feedForward(self, a : Activation) -> Activation:
        for layer in self.layers:
            a = layer.feedForward(a)
        return a

    # Metode til at beregne nabla_C = [partial_C/partial_W_1, partial_C/partial_W_1, ..., partial_C/partial_W_n], partial_C/partial_B_n], som derefter bruges til at opdatere vægtene og bias    
    def compute_gradients(self, data : Data) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        # Formatering af data for et givent træningseksempel. Label onehot-encodes for at kunne udføre vektor operationer. F.eks. hvis label = 1, så er y_actual = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        X, y_actual = data[0], np.eye(10)[data[1]]
        nabla_w = [None] * len(self.layers)
        nabla_b = [None] * len(self.layers)

        # Find og gem alle z, a værdier for alle lag
        self.calculate_outputs(X)

        #Beregn fejlen for det sidste lag
        index = len(self.layers) - 1
        output_layer = self.layers[index]
        
        # --- Bestem nabla_C for output laget ---
        # \frac{\partial C}{\partial W^index} = \frac{\partial z^index}{\partial w^index} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial C}{\partial a}
        #                                     = a^(index - 1) \cdot y'(z^index) \cdot 2(a^index - y)

        # \frac{\partial a}{\partial z} \cdot \frac{\partial C}{\partial a}
        error = self.cost.cost_derivative(y_true=y_actual, y_pred=output_layer.a) * output_layer.activation_function(activations=output_layer.z, derivative=True)

        nabla_w[-1] = np.outer(error, self.layers[index-1].a)
        nabla_b[-1] = error

        # --- Bestem nalba_C for de skjulte lag ---
        for index in range(len(self.layers)-2, 0, -1):
            hidden_layer = self.layers[index]
            next_layer = self.layers[index+1]

            # \frac{\partial C}{\partial W^index} = \frac{\partial z^index}{\partial w^index} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial C}{\partial a}
            #                                     = a^(index - 1) \cdot y'(z^index) \cdot 2(a^index - y)

            # \frac{\partial a}{\partial z} \cdot \frac{\partial C}{\partial a}
            error = np.dot(next_layer.weights.T, error) * hidden_layer.activation_function(activations=hidden_layer.z, derivative=True)

            #  \frac{\partial z^index}{\partial w^index} \cdot the rest
            nabla_w[index] = np.outer(error, self.layers[index-1].a)
            nabla_b[index] = error

        return nabla_w, nabla_b

    # Metode til at få alle lag til at gemme de værdier (z & a), når der udføres forward prop (skal bruges til backprop)
    def calculate_outputs(self, data_point : Activation) -> None:
        for layer in self.layers:
            data_point = layer.store_layer_values(data_point)

    # Metode der træner netværket (opdaterer lagenes vægte og bias) samt kan generere grafer
    def train(self, training_data : Data, epochs : int, test_data : Data, learning_rate : float = 0.1, generate_graphs : bool = False, graph_name : str = "Default") -> None:
        epoch_loss_training, epoch_loss_validation, epoch_test_accuracy, epoch_training_accuracy = [], [], [], []
        #initial_learning_rate = learning_rate
        # Training loop
        for i in range(1, epochs + 1):
            #learning_rate = initial_learning_rate / (1 + i * 0.1)
            random.shuffle(training_data)
            for data in training_data:
                #Beregn nabla_C_data
                nabla_w, nabla_b = self.compute_gradients(data)

                # Opdater vægte og bias for hvert lag
                for index in range(1, len(self.layers)):
                    self.layers[index].weights -= learning_rate * nabla_w[index]
                    self.layers[index].biases -= learning_rate * nabla_b[index]

            test_accuracy = self.evaluate(test_data=test_data)
            test_loss = self.cost.total_cost(data_set=test_data, model = self)
            print("Epoch:", i, "Loss: ", test_loss, "Accuracy: ", test_accuracy)

            if generate_graphs == True:
                # Beregn værdier for nuværende epoch til grafer
                epoch_loss_training.append((i, self.cost.total_cost(data_set=training_data, model=self)))
                epoch_loss_validation.append((i, test_loss))
                epoch_test_accuracy.append((i, test_accuracy))
                epoch_training_accuracy.append((i, self.evaluate(test_data=training_data)))

        # Generer grafer
        if generate_graphs:
            # Epoch loss graph
            visualizer.generate_graph(graph_name, epochs, ("Epoch", "Loss"), ("Test loss", "Trænings loss"), epoch_loss_validation, epoch_loss_training)
            # Epoch accuracy graph
            visualizer.generate_graph(graph_name, epochs, ("Epoch", "Accuracy"), ("Test nøjagtighed", "Trænings nøjagtighed"), epoch_test_accuracy, epoch_training_accuracy)  
   
    def evaluate(self, test_data : Data) -> float:
        correct = 0
        for data in test_data:
            if np.argmax(self.feedForward(data[0])) == data[1]:
                correct += 1

        return (correct/len(test_data))

    # Metode til at gemme weights og bias til et netværk
    def save(self, name) -> None:
        path = os.path.join("model", name)
        if not os.path.exists(path): os.mkdir(path)

        for i, layer in enumerate(self.layers[1:]):
            np.save(os.path.join(path, f'weights_layer_{i}.npy'), layer.weights)
            np.save(os.path.join(path, f'biases_layer_{i}.npy'), layer.biases)

    # Metode til at load weights og bias til netværket fra en fil
    def load(self, name) -> None:
        path = os.path.join("model", name)

        for i, layer in enumerate(self.layers[1:]):
            layer.weights = np.load(os.path.join(path, f'weights_layer_{i}.npy'))
            layer.biases = np.load(os.path.join(path, f'biases_layer_{i}.npy'))

    
class MeanSquaredError:
    # Metode til at beregne MSE for et givent eksempel
    @staticmethod
    def cost(y_true : np.ndarray, y_pred : np.ndarray) -> float:
        return np.sum(np.square(y_true - y_pred)) / len(y_true)
    
    # f'(x) for et givent eksempel
    @staticmethod
    def cost_derivative(y_true : np.ndarray, y_pred : np.ndarray) -> np.ndarray:
        return 2 * (y_pred - y_true)
    
    # Metode der Beregner den "totale_cost" gennem et helt data_set
    @staticmethod
    def total_cost(data_set : Data, model : Network) -> float:
        total_loss = 0
        for data in data_set:
            total_loss += model.cost.cost(y_true=np.eye(10)[data[1]], y_pred=model.feedForward(data[0]))

        return total_loss / len(data_set)
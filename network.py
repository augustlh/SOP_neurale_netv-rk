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
    
    def beregn_gradienter(self, data):
        X, y_actual = data[0], np.eye(10)[data[1]]

        self.calculate_outputs(X)

        #(30,) * (10,) * (10,)
        l = (self.layers[-1]).activation_function(self.layers[-1].z, derivative=True) * (self.cost).cost_derivative(y_true=y_actual, y_pred=self.feedForward(X))
        newOutputWeight = np.outer(l, self.layers[-2].a)
        newOutputBias = l

        l = np.dot(self.layers[-1].weights.T, l) * self.layers[-2].activation_function(self.layers[-2].z,derivative=True)
        newHiddenWeight = np.outer(l, self.layers[0].a)
        newHiddenBias = l
        
        #print("Output weight (shape): ", newOutputWeight.shape, "Output weight (sum): ", np.sum(newOutputWeight))
        #print("Output bias (shape): ", newOutputBias.shape, "Output bias (sum): ", np.sum(newOutputBias))

        #print("Hidden weight (shape): ", newHiddenWeight.shape, "Hidden weight (sum): ", np.sum(newHiddenWeight))
        #print("Hidden bias (shape): ", newHiddenBias.shape, "Hidden bias (sum): ", np.sum(newHiddenBias))
    
       
        (self.layers[1]).weights -= 0.1 * newHiddenWeight
        (self.layers[1]).biases -= 0.1 * newHiddenBias

        (self.layers[2]).weights -= 0.1 * newOutputWeight
        (self.layers[2]).biases -= 0.1 * newOutputBias

        #hidden_layer.weights += hidden_layer_output.T.dot(output_delta) * 0.1
        #hidden_layer.biases += np.sum(output_delta, axis=0, keepdims=True) * 0.1

        #hidden_layer.weights += X.T.dot(hidden_layer_delta) * 0.1
        #hidden_layer.biases += np.sum(hidden_layer_delta, axis=0, keepdims=True) * 0.1

        loss = self.cost.cost(y_true=y_actual, y_pred=self.feedForward(X))
        return loss

    def train(self, training_data, epochs, test_data):
        for i in range(epochs):
            total_loss = 0
            for data in training_data:
                loss = self.beregn_gradienter(data)
                total_loss += loss

            print("Epoch:", i, "Loss: ", total_loss/len(training_data), "Accuracy: ", self.evaluate(test_data))

    def train2(self, training_data, epochs, test_data):
        for i in range(epochs):
            total_loss = 0
            for data in training_data:
                nabla_w, nabla_b = self.compute_gradients(data)

                for index in range(1, len(self.layers)):
                    self.layers[index].weights -= 0.1 * nabla_w[index]
                    self.layers[index].biases -= 0.1 * nabla_b[index]

                loss = self.cost.cost(y_true=np.eye(10)[data[1]], y_pred=self.feedForward(data[0]))
                total_loss += loss

            print("Epoch:", i, "Loss: ", total_loss/len(training_data), "Accuracy: ", self.evaluate(test_data))
   
   
    def evaluate(self, test_data):
        correct = 0
        for data in test_data:
            if np.argmax(self.feedForward(data[0])) == data[1]:
                correct += 1
        return (correct/len(test_data))













    
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

        # --- Beregn gradienterne for de skjulte lag ---
        for index in range(len(self.layers)-2, 0, -1):
            hidden_layer = self.layers[index]
            next_layer = self.layers[index+1]

            # \frac{\partial C}{\partial W^index} = \frac{\partial z^index}{\partial w^index} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial C}{\partial a}
            #                                     = a^(index - 1) \cdot y'(z^index) \cdot 2(a^index - y)

            # \frac{\partial a}{\partial z} \cdot \frac{\partial C}{\partial a}
            error = np.dot(next_layer.weights.T, error) * hidden_layer.activation_function(activations=hidden_layer.z, derivative=True)

            nabla_w[index] = np.outer(error, self.layers[index-1].a)
            nabla_b[index] = error

        return nabla_w, nabla_b
        

    def calculate_outputs(self, data_point):
        for layer in self.layers:
            data_point = layer.store_layer_values(data_point)

    
class MeanSquaredError:
    @staticmethod
    def cost(y_true : np.ndarray, y_pred : np.ndarray) -> float:
        return np.sum(np.square(y_true - y_pred)) / len(y_true)
    @staticmethod
    def cost_derivative(y_true : np.ndarray, y_pred : np.ndarray) -> np.ndarray:
        return 2 * (y_pred - y_true)
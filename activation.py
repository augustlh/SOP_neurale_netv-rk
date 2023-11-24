import numpy as np

def relu(activations : np.ndarray, derivative=False) -> np.ndarray:
    if derivative:
        return np.where(activations > 0, 1.0, 0.0)
    else:
        return np.maximum(activations, 0, activations)

def softmax(activations, derivative=False) -> np.ndarray:
    if derivative:
        #return softmax(activations) * (1 - softmax(activations))
        s = softmax(activations)
        return s * (1 - s)
    else:
        ex = np.exp(activations - np.max(activations))
        return ex / ex.sum(axis=0)

def sigmoid(activations : np.ndarray, derivative=False) -> np.ndarray:
    if derivative:
        return sigmoid(activations) * (1 - sigmoid(activations))
    else:
        return 1.0 / (1.0 + np.exp(-activations))
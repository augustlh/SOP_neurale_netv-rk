# August Leander Hedman
# augu1789@edu.nextkbh.dk
# NEXT Sukkertoppen, S3n

# Importer biblioteker og moduler, som er relevante for programmet
import numpy as np
from layers import Input, Dense
from network import Network
from activation import relu, softmax, sigmoid
from mnist import mnist, filenames

MNIST = mnist("data")

# Hent data fra MNIST-datasættet
training_data, test_data = MNIST.give_data()

# Lav en instants af et netværk
model = Network()

# Definer netværkets struktur (784, 30, 10)
model.add(Input((784,)))
model.add(Dense(300, sigmoid))
model.add(Dense(100, sigmoid))
model.add(Dense(10, sigmoid))

# Byg netværket (initialiserer vægte og bias for lagene)
model.build()

#Træn netværket
model.train(training_data=training_data, epochs=10, test_data=test_data, learning_rate=0.1)
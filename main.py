# August Leander Hedman
# augu1789@edu.nextkbh.dk
# NEXT Sukkertoppen, S3n

# Importer biblioteker og moduler, som er relevante for programmet
import numpy as np
from layers import Input, Dense
from network import Network
from activation import relu, softmax, sigmoid
from mnist import format_data


# Hent data fra MNIST-datasættet
#MNIST = mnist("data")
#training_data, test_data = MNIST.give_data()

training_data, test_data = format_data()

# Lav en instants af klassen Network (et netværk)
model = Network()

# Definer netværkets struktur (784, 30, 10)
model.add(Input((784,)))
model.add(Dense(30, sigmoid))
#model.add(Dense(100, sigmoid))
model.add(Dense(10, sigmoid))

# Byg netværket (initialiserer vægte og bias for lagene)
model.build()

#Træn netværket (0.005)
model.train(training_data=training_data, epochs=50, test_data=test_data, learning_rate=0.01, generate_graphs=True, graph_name="HU30 shuffling0.01")
#model.save("800hu test again")

# Prætrænet model
print("----------------")
model.load("800HU")

#Test netværket
print("Nøjagtighed for prætrænet model:", model.evaluate(test_data))
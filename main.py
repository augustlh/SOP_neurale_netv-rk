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

# Definer netværkets struktur
model.add(Input((784,)))
model.add(Dense(30, relu))
model.add(Dense(10, softmax))

# Byg netværket (initialiserer vægte og bias for lagene)
model.build()

#model.train(training_data[:10000], epochs=10, learning_rate=0.1)
w,b = model.compute_gradients(training_data[0])

print(w, b)
#print(ow)
#print("---------------------------------------------------")
#print(hw)


#model.train(training_data, epochs=10, learning_rate=0.1,test_data=test_data[:1000])
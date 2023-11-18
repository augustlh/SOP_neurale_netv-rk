# August Leander Hedman
# augu1789@edu.nextkbh.dk
# NEXT Sukkertoppen, S3n

# Importer biblioteker og moduler, som er relevante for programmet
import numpy as np
from layers import Input, Dense
from network import Network
from activation import relu, softmax

# Instantier et netværk
model = Network()

# Skab netværks strukturen
model.add(Input((10,)))
model.add(Dense(30, relu))
model.add(Dense(10, softmax))

# Byg netværket (initialiserer vægte og bias for lagene)
model.build()

trainingData = [(np.random.randn(10,), 1), (np.random.randn(10,), 2)]

a, z = model.compute_gradients(trainingData[0])

print(a[1])
print(a[2])
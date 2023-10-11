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

#print(model.layers[1].weights.shape)
#model.layers[1].backProp(np.random.randn(10,))

model.backProp((np.random.randn(10,),1))
#output = model.feedForward(np.random.randn(10,))
#print("Netværkets output:", output, sep="\n")

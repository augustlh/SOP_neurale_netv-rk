# August Leander Hedman
# augu1789@edu.nextkbh.dk
# NEXT Sukkertoppen, S3n

# Importer biblioteker og moduler, som er relevante for programmet
import os, gzip
import numpy as np


# Kilde til filer: https://yann.lecun.com/exdb/mnist/
filenames = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz"
}

arraylengths = {
    "train_images": 60000,
    "train_labels": 60000,
    "test_images": 10000,
    "test_labels": 10000,
}

#metadata = [16,8]

# Variabler med relevante værdier til formatering af data
PATH = "data"
IMAGE_SIZE = 28
IMAGE_OFF_SET = 16
LABEL_OFF_SET = 8

# Funktion til at formatere dataet ved at returnere en tuple bestående af trænings_data og test_data
def format_data():
    training_images = load_images("train_images")
    training_labels = load_labels("train_labels")
    training_data = list(zip(training_images, training_labels))

    test_images = load_images("test_images")
    test_labels = load_labels("test_labels")
    test_data = list(zip(test_images, test_labels))

    return training_data, test_data

# Nedenstående kode er inspireret af https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/image_classification/mnist.py

# Funktion til at load og formatere billededata fra en fil
def load_images(filename):
    with gzip.open(os.path.join(PATH, filenames[filename]), 'rb') as f:
        n = arraylengths[filename]

        # Offset er 16. Hvorefter der er unsigned bytes (8-bit uint)
        f.seek(IMAGE_OFF_SET)

        # Læs bufferen
        buffer = f.read()

        # Betragt bufferen som et 1-dimensionelt np array 
        raw_data = np.frombuffer(buffer=buffer, dtype=np.ubyte, count=IMAGE_SIZE * IMAGE_SIZE * n)

        # Datasættet består af 28x28 billeder med 1 color channel. Således ønskes det at formatere det ovenstående np,array til en matrix af n rækker og 784 kolonner
        # altså (n, 784). På den måde repræsenterer en række et billede / eksempel
        formatted_data = raw_data.reshape(n, IMAGE_SIZE*IMAGE_SIZE,) 

        # Hver pixel-værdi er mellem 0 til 255. Det ønskes at normalisere pixel-værdierne til værdier mellem 0 og 1
        formatted_data = formatted_data / 255.0

        return formatted_data

# Funktion til at load og formatere label data fra en fil    
def load_labels(filename):
    with gzip.open(os.path.join(PATH, filenames[filename]), 'rb') as f:
        # Offset er 8. Hvorefter der er unsigned bytes
        f.seek(LABEL_OFF_SET)

        # Læs bufferen
        buffer = f.read()

        # Betragt bufferen som et 1-dimensionelt nparray
        labels = np.frombuffer(buffer=buffer, dtype=np.ubyte)

        return labels
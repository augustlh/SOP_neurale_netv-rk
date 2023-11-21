# August Leander Hedman
# augu1789@edu.nextkbh.dk
# NEXT Sukkertoppen, S3n

import os, gzip
import numpy as np

filenames = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz"
}

#metadata = [16,8]

class mnist():
    def __init__(self, path):
        self.path = path
        self.IMAGE_SIZE = 28
        self.NUM_CLASSES = 10

    def load_images(self, file):
        with gzip.open(os.path.join(self.path, file), 'rb') as f:
            return np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(-1,self.IMAGE_SIZE * self.IMAGE_SIZE,) / 255.0

    def load_labels(self, file):
        with gzip.open(os.path.join(self.path, file), 'rb') as f:
            return np.frombuffer(f.read(), 'B', offset=8)
    def give_data(self):
        training_images = self.load_images(filenames["train_images"])
        training_labels = self.load_labels(filenames["train_labels"])
        training_data = list(zip(training_images, training_labels))

        test_images = self.load_images(filenames["test_images"])
        test_labels = self.load_labels(filenames["test_labels"])
        test_data = list(zip(test_images, test_labels))

        return training_data, test_data
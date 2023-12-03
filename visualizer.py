# August Leander Hedman
# augu1789@edu.nextkbh.dk
# NEXT Sukkertoppen, S3n

# Importer biblioteker og moduler, som er relevante for programmet
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
import os

# Funktion der generer en "line-graf" med matplotlib
def generate_graph(name : str, epochs : int, axes : Tuple[str, str], labels : Tuple[str, str], data_set1 : List[Tuple[int, float]], data_set2 : List[Tuple[int, float]]):
        plt.figure(figsize=(8, 6))
        plt.plot(*zip(*data_set1), label=labels[0], color="SteelBlue")
        plt.plot(*zip(*data_set2), label = labels[1], color='Crimson')
        plt.xlabel(axes[0])
        plt.ylabel(axes[1])
        plt.legend()
        plt.xticks(np.arange(0, epochs + 1, 5))
        path = os.path.join("graphs", name)
        if not os.path.exists(path): os.mkdir(path)
        plt.savefig(os.path.join(path, f'{axes[0]}_{axes[1]}.png'))
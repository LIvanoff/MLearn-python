from ARC.apriori import Apriori
from clustering.KMeans import KMeans
import matplotlib.pyplot as plt
import time
import numpy as np

start = time.time()
if __name__ == "__main__":
    # arc = Apriori(filename='data.json', min_length=2)
    # arc.apriori(excel=True)

    X = np.array([[1, 3], [3, 3], [4, 3], [5, 3], [1, 2], [4, 2], [1, 1], [2, 1]])
    # X1 = np.random.normal(loc=[0, -10], size=(100, 2))
    # X2 = np.random.normal(loc=[-10, 0], size=(100, 2))
    # X3 = np.random.normal(loc=[0, 0], size=(100, 2))
    # X = np.vstack((X1, X2, X3))

    km = KMeans(2)
    markers = km.train(X)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[1].scatter(X[:, 0], X[:, 1], c=markers)
    axs[0].scatter(X[:, 0], X[:, 1], marker='o')
    plt.show()

end = time.time()
print(" \n", (end - start) * 10 ** 3, "ms")

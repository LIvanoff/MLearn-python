import numpy as np
import math


class KMeans(object):
    def __init__(self, clusters: int,
                 metric: str = 'euclid_dist',
                 max_iter: int = 300,
                 stop_criteria: bool = True):

        self.X = None
        self.clusters_ = clusters
        self.metric_ = metric
        self.max_iter_ = max_iter
        self.labels = np.array([])

    def train(self, X):
        self.X = X
        centroids_index = np.arange(self.clusters_)
        print(centroids_index)
        centroids_index = np.random.choice(centroids_index, self.clusters_)
        print(centroids_index)

        '''здесь будет switch для выбранной метрики'''
        metric = self.euclid_dist

        for x in X:
            self.labels = np.append(self.labels, np.argmin(metric(x, centroids_index)))

        return

    def predict(self):
        return self.labels

    def euclid_dist(self, x, centroids_index):
        dist = np.array([])
        for y in self.X[centroids_index]:
            dist = np.append(dist, math.sqrt(((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)))
        return dist

    def manhattan_geom(self, x, centroids_index):
        dist = np.array([])
        for y in self.X[centroids_index]:
            dist = np.append(dist, abs((x[0] - y[0]) + (x[1] - y[1])))
        return dist

    def chebyshev_dist(self):
        return

    def square_euclid_dist(self, x, centroids_index):
        dist = np.array([])
        for y in self.X[centroids_index]:
            dist = np.append(dist, (x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)
        return dist

    def pow_dist(self):
        return

    def stop_criteria(self):
        return

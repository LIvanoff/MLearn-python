import numpy as np
import math


class KMeans(object):
    def __init__(self, clusters: int,
                 metric: str = 'euclid_dist',
                 max_iter: int = 300,
                 stop_criteria: bool = True):

        self.clusters_ = clusters
        self.metric_ = metric
        self.max_iter_ = max_iter
        self.labels = np.array([])

    def train(self, X):
        centroids_index = np.arange(self.clusters_)
        centroids_index = np.random.choice(centroids_index, self.clusters_)

        '''здесь будет switch для выбранной метрики'''
        metric = self.euclid_dist

        for x in X:
            self.labels = np.append(self.labels, np.argmin(metric(x, centroids_index, X)))

        return

    def predict(self):
        return self.labels

    def euclid_dist(self, x, centroids_index, X):
        dist = np.array([])
        for y in centroids_index:
            dist = np.append(dist, math.sqrt(pow(x[0] - X[], 2) * pow(x[1] - y, 2)))
        return dist

    def manhattan_geom(self, X):
        dist = np.array([])
        for x, y in zip(X[:, 0], X[:, 1]):
            dist = np.append(dist, abs(x - y))
        return dist

    def chebyshev_dist(self):
        return

    def square_euclid_dist(self, X):
        dist = np.array([])
        for x, y in zip(X[:, 0], X[:, 1]):
            dist = np.append(dist, pow(x - y, 2))
        return dist

    def pow_dist(self):
        return

    def stop_criteria(self):
        return

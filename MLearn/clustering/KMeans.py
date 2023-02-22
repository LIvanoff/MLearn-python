import numpy as np


class KMeans(object):
    def __init__(self, clusters: int,
                 metric: str = 'euclid_dist',
                 max_iter: int = None,
                 stop_criteria_bool: bool = True,
                 criteria_type_str: str = 'MSE'):

        self.X = None
        self.clusters_ = clusters
        self.metric_ = metric
        self.max_iter_ = max_iter
        self.labels = np.array([])
        self.stop_criteria_bool_ = stop_criteria_bool
        self.criteria_type_str_ = criteria_type_str
        self.centroids = np.array([])

    def train(self, X):
        self.X = X
        seeds = np.random.choice(np.arange(len(X)), self.clusters_, replace=False)

        if self.metric_ == 'euclid_dist':
            metric = self.euclid_dist
        else:
            metric = self.select_metrics()

        for x in X:
            self.labels = np.append(self.labels, np.argmin(metric(x, seeds)))

        for centroid in seeds:
            indexes = np.where(self.labels == self.labels[centroid])
            self.centroids = np.append(self.centroids, np.mean(X[indexes], axis=0))

        print(self.labels[seeds])
        print(self.centroids)

        return self.labels

    def predict(self):
        return self.labels

    def euclid_dist(self, x, centroids_index):
        dist = np.array([])
        for y in self.X[centroids_index]:
            dist = np.append(dist, np.sqrt(np.sum((x - y) ** 2)))
        return dist

    def manhattan_geom(self, x, centroids_index):
        dist = np.array([])
        for y in self.X[centroids_index]:
            dist = np.append(dist, np.abs(np.sum(x - y)))
        return dist

    def chebyshev_dist(self):
        return

    def square_euclid_dist(self, x, centroids_index):
        dist = np.array([])
        for y in self.X[centroids_index]:
            dist = np.append(dist, np.sum(x - y))
        return dist

    def pow_dist(self):
        return

    def stop_criteria(self):
        return

    def select_metrics(self):
        if self.metric_ == 'manhattan_geom':
            return self.manhattan_geom
        elif self.metric_ == 'chebyshev_dist':
            return self.chebyshev_dist
        elif self.metric_ == 'square_euclid_dist':
            return self.square_euclid_dist
        elif self.metric_ == 'pow_dist':
            return self.pow_dist

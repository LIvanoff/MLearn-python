import numpy as np


class KMeans(object):
    centroids_ = None
    labels = None
    X = None

    def __init__(self, clusters: int,
                 metric: str = 'euclid_dist',
                 max_iter: int = None,
                 stop_criteria: bool = True):

        self.clusters_ = clusters
        self.metric_ = metric
        self.max_iter_ = max_iter
        self.stop_criteria_ = stop_criteria

    def train(self, X):
        self.X = X
        init_centers = np.random.choice(np.arange(len(X)), self.clusters_, replace=False)

        centroids = np.empty((0, 2))
        for center in init_centers:
            centroids = np.vstack([centroids, X[center]])

        if self.metric_ == 'euclid_dist':
            metric = self.euclid_dist
        else:
            metric = self.select_metrics()

        self.labels = np.zeros(len(X))
        # changed = True
        # while changed:
        #     changed = False
        for j in range(1):
            for i in range(len(X)):
                self.labels[i] = np.argmin(metric(X[i], init_centers))

            for i in range(len(init_centers)):
                indexes = np.where(self.labels == self.labels[init_centers[i]])
                centroids[i] = np.mean(X[indexes]) # np.vstack([centroids, np.mean(X[indexes], axis=0)])
                # centroids = np.delete(centroids, 0, axis=0)

        print(init_centers)
        print(centroids)

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

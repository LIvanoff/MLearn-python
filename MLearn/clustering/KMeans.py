import numpy as np
import matplotlib.pyplot as plt
import time


class KMeans(object):
    '''
    Класс метода кластеризации К-средних

    clusters_ : Количество кластеров
    metric_ : Метрика расчёта дистанции
    max_iter_ : Максимальное количество итераций
    centroids_ : Координаты центродов кластеров
    labels : Метки принадлежности точек к классам
    X : Координаты точек
    '''

    clusters_: int
    centroids_: dict = {}
    labels = np.ndarray
    X: np.ndarray

    def __init__(self, clusters: int,
                 metric: str = 'euclid_dist',
                 max_iter: int = None,
                 stop_criteria: bool = True):

        self.clusters_ = clusters
        self.metric_ = metric
        self.max_iter_ = max_iter
        self.stop_criteria_ = stop_criteria

    def fit_predict(self, X):
        self.X = X
        init_centers = np.random.choice(np.arange(len(X)), self.clusters_, replace=False)

        centroids = {}
        for center, i in zip(init_centers, range(len(init_centers))):
            centroids[i] = X[center]

        if self.metric_ == 'euclid_dist':
            metric = self.euclid_dist
        else:
            metric = self.select_metrics()

        self.labels = np.zeros(len(X))
        loss = [0, 0]

        # plt.ion()
        changed = True
        while changed:
            changed = False
            for i in range(len(X)):
                self.labels[i] = np.argmin(metric(self.X[i], centroids))

            loss.pop(0)
            loss.append(self.wcss(centroids))
            delta = abs(loss[0] - loss[1])

            if delta != 0:
                changed = True

            for key in centroids.keys():
                indexes = np.where(self.labels == key)
                mean = np.mean(X[indexes], axis=0)
                centroids[key] = mean

        #     self.print_clusters(centroids)

        # plt.ioff()
        # plt.show()
        self.centroids_ = centroids
        return self.labels

    def print_clusters(self, centroids):
        centroids_values = list(centroids.values())
        row0 = list([row[0] for row in centroids_values])
        row1 = list([row[1] for row in centroids_values])
        print(centroids)
        plt.clf()
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.labels)
        plt.plot(row0, row1, 'ro')
        plt.draw()
        plt.gcf().canvas.flush_events()
        time.sleep(3)

    def wcss(self, centroids):
        dist_sum = 0
        for key in centroids.keys():
            indexes = np.where(self.labels == key)
            dist_sum += np.sqrt(
                np.sum((self.X[indexes] - centroids[key]) ** 2))
        return dist_sum

    def predict(self):
        return self.labels

    def euclid_dist(self, x, centroids):
        dist = np.array([])
        for key in centroids.keys():
            dist = np.append(dist, np.sqrt(np.sum((x - centroids[key]) ** 2)))
        return dist

    def manhattan_geom(self, x, centroids):
        dist = np.array([])
        for key in centroids.keys():
            dist = np.append(dist, np.abs(np.sum(x - centroids[key])))
        return dist

    def chebyshev_dist(self):
        return

    def square_euclid_dist(self, x, centroids):
        dist = np.array([])
        for key in centroids.keys():
            dist = np.append(dist, np.sum(x - centroids[key]))
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

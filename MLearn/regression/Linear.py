import numpy as np
import matplotlib.pyplot as plt
import time


class Linear(object):
    '''
    Класс метода численной линейной регрессии
    '''
    Y_: np.ndarray
    X_: np.ndarray
    size_: int

    def __init__(self,
                 max_iter: int = None,
                 stop_criteria: bool = True,
                 learning_rate: float = 0.01
                 ):
        self.weight_ = np.random.normal(loc=0.0, scale=0.1)
        self.bias_ = np.random.normal(loc=0.0, scale=0.1)
        self.max_iter_ = max_iter
        self.stop_criteria_ = stop_criteria
        self.learning_rate_ = learning_rate

    def fit(self, X):
        self.X_ = X
        self.size_ = len(self.X_)

        return

    def GD(self):
        weight_deriv = 0
        bias_deriv = 0

        for i in range(self.size_):
            weight_deriv += -2 * self.X_[i] * (self.Y_[i] - (self.weight_ * self.X_[i] + self.bias_)) / self.size_
            bias_deriv += -2 * (self.Y_[i] - (self.weight_ * self.X_[i] + self.bias_)) / self.size_

        self.weight_ -= self.learning_rate_ * weight_deriv
        self.bias_ -= self.learning_rate_ * bias_deriv

    def SGD(self):
        pass

    def print_clusters(self, centroids):
        centroids_values = list(centroids.values())
        row0 = list([row[0] for row in centroids_values])
        row1 = list([row[1] for row in centroids_values])
        print(centroids)
        plt.clf()
        # plt.scatter(self.X[:, 0], self.X[:, 1], c=self.labels)
        plt.plot(row0, row1, 'ro')
        plt.draw()
        plt.gcf().canvas.flush_events()
        time.sleep(3)


    def predict(self):
        return

    def stop_criteria(self):
        return

    # def select_metrics(self):
    #     if self.metric_ == 'manhattan_geom':
    #         return self.manhattan_geom
    #     elif self.metric_ == 'chebyshev_dist':
    #         return self.chebyshev_dist
    #     elif self.metric_ == 'square_euclid_dist':
    #         return self.square_euclid_dist
    #     elif self.metric_ == 'pow_dist':
    #         return self.pow_dist

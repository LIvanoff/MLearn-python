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
    loss_history: np.ndarray
    pred: np.ndarray

    def __init__(self,
                 max_iter: int = 100,
                 stop_criteria: bool = True,
                 learning_rate: float = 0.01,
                 optimizer_name: str = "GD"
                 ):
        self.weight_ = np.random.normal(loc=0.0, scale=0.1)
        self.bias_ = np.random.normal(loc=0.0, scale=0.1)
        self.max_iter_ = max_iter
        self.stop_criteria_ = stop_criteria
        self.learning_rate_ = learning_rate
        self.optimizer_name_ = optimizer_name

    def fit(self, X, Y):
        self.X_ = X
        self.Y_ = Y
        self.size_ = len(self.X_)

        if self.optimizer_name_ == 'GD':
            optimizer = self.GD
        else:
            optimizer = self.select_optimizer()

        plt.ion()
        self.loss_history = np.array([])
        for i in range(self.max_iter_):
            self.predict(self.X_)
            self.plot()
            optimizer()
            # Calculate cost for auditing purposes
            loss = self.MSE()
            self.loss_history = np.append(self.loss_history, loss)
            self.plot()
            # Log Progress
            if i % 10 == 0:
                print("iter: " + str(i) + " loss: " + str(loss))

        plt.ioff()
        plt.show()
        return

    def MSE(self):
        # total_error = 0.0
        # for i in range(compani):
        #     total_error += (sales[i] - (weight * radio[i] + bias)) ** 2
        # return total_error / companies
        return np.mean(np.power((self.Y_ - np.dot(self.X_, self.weight_) - self.bias_), 2))

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

    def Adam(self):
        pass

    def plot(self):
        plt.clf()
        plt.scatter(self.X_, self.Y_, marker='o', alpha=0.8)
        plt.plot(self.X_, self.pred, 'r')
        plt.draw()
        plt.gcf().canvas.flush_events()
        # time.sleep(0.01)

    def predict(self, X):
        self.pred = np.dot(X, self.weight_) + self.bias_
        return self.pred


    def stop_criteria(self):
        return

    def select_optimizer(self):
        if self.optimizer_name_ == 'SGD':
            return self.SGD()
        elif self.optimizer_name_ == 'Adam':
            return self.Adam()

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
    beta1: float
    beta2: float
    EMA1_w: float
    EMA1_b: float
    EMA2_w: float
    EMA2_b: float
    t: int

    def __init__(self,
                 max_iter: int = 100,
                 stop_criteria: bool = True,
                 learning_rate: float = 1 * pow(10, -3),
                 optimizer_name: str = "GD"
                 ):
        self.weight_ = np.random.normal(loc=0.0, scale=0.01)
        self.bias_ = np.random.normal(loc=0.0, scale=0.01)
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
            optimizer()
            loss = self.MSE()
            self.loss_history = np.append(self.loss_history, loss)
            self.plot()

            if i % 10 == 0:
                print("iter: " + str(i) + " loss: " + str(loss))

        plt.ioff()
        plt.show()
        return self

    def MSE(self):
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
        weight_deriv = 0
        bias_deriv = 0
        epsilon = pow(10, -8)

        for i in range(self.size_):
            weight_deriv += -2 * self.X_[i] * (self.Y_[i] - (self.weight_ * self.X_[i] + self.bias_)) / self.size_
            bias_deriv += -2 * (self.Y_[i] - (self.weight_ * self.X_[i] + self.bias_)) / self.size_

        self.EMA1_w = self.beta1 * self.EMA1_w + (1 - self.beta1) * weight_deriv
        self.EMA1_b = self.beta1 * self.EMA1_b + (1 - self.beta1) * bias_deriv
        self.EMA2_w = self.beta2 * self.EMA2_w + (1 - self.beta2) * np.power(weight_deriv, 2)
        self.EMA2_b = self.beta2 * self.EMA2_b + (1 - self.beta2) * np.power(bias_deriv, 2)

        self.EMA1_w = self.EMA1_w / (1 - np.power(self.beta1, self.t))
        self.EMA1_b = self.EMA1_b / (1 - np.power(self.beta1, self.t))
        self.EMA2_w = self.EMA2_w / (1 - np.power(self.beta2, self.t))
        self.EMA2_b = self.EMA2_b / (1 - np.power(self.beta2, self.t))

        self.weight_ -= self.learning_rate_ * self.EMA1_w / ((np.sqrt(self.EMA2_w)) + epsilon)
        self.bias_ -= self.learning_rate_ * self.EMA1_b / ((np.sqrt(self.EMA2_b)) + epsilon)
        self.t += 1

    def plot(self):
        plt.clf()
        plt.scatter(self.X_, self.Y_, marker='o', alpha=0.8)
        plt.plot(self.X_, self.pred, 'r')
        plt.title('y = ' + str(self.weight_) + ' x + ' + str(self.bias_), fontsize=10, color='0.5')
        plt.draw()
        plt.gcf().canvas.flush_events()
        time.sleep(0.01)

    def predict(self, X):
        self.pred = np.dot(X, self.weight_) + self.bias_
        return self.pred

    def stop_criteria(self):
        return

    def select_optimizer(self):
        if self.optimizer_name_ == 'SGD':
            return self.SGD()
        elif self.optimizer_name_ == 'Adam':
            self.beta1 = 0.9  # 0.2
            self.beta2 = 0.9  # 0.8
            self.EMA1_w = 0.0
            self.EMA2_w = 0.0
            self.EMA1_b = 0.0
            self.EMA2_b = 0.0
            self.t = 1
            return self.Adam

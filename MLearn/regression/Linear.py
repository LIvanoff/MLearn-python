import numpy as np
import matplotlib.pyplot as plt
import autograd as ad
import autograd.variable as av
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
    beta1_: float
    beta2_: float
    EMA1_w: float
    EMA1_b: float
    EMA2_w: float
    EMA2_b: float
    t: int
    r2_score: float
    weight_: float
    bias_: float
    gradient: av.Variable

    def __init__(self,
                 max_iter: int = 100,
                 stop_criteria: bool = True,
                 learning_rate: float = 1 * pow(10, -3),
                 optimizer_name: str = 'GD',
                 loss_function: str = 'MSE',
                 beta1: float = 0.9,
                 beta2: float = 0.999
                 ):
        self.weight_ = np.random.normal(loc=0.0, scale=0.01)
        self.bias_ = np.random.normal(loc=0.0, scale=0.01)
        self.max_iter_ = max_iter
        self.stop_criteria_ = stop_criteria
        self.learning_rate_ = learning_rate
        self.optimizer_name_ = optimizer_name
        self.loss_function_ = loss_function
        self.beta1_ = beta1
        self.beta2_ = beta2

    def fit(self, X, Y):
        self.X_ = X
        self.Y_ = Y
        self.size_ = len(self.X_)

        if self.optimizer_name_ == 'GD':
            step = self.GD
        else:
            step = self.select_optimizer()

        if self.loss_function_ == 'MSE':
            loss_function = self.MSE
        else:
            loss_function = self.select_loss_function()

        ad.set_mode('reverse')
        plt.ion()
        self.loss_history = np.array([])

        for epoch in range(self.max_iter_):
            big_variable = av.Variable([self.weight_, self.bias_])
            weight, bias = big_variable[0], big_variable[1]
            self.forward(weight, bias)
            loss = loss_function()
            self.gradient = loss.compute_gradients()
            step()
            self.loss_history = np.append(self.loss_history, loss.data)
            self.plot()
            ad.reset_graph()

            if epoch % 10 == 0:
                print("iter: " + str(epoch) + " loss: " + str(loss.data))

        plt.ioff()
        plt.show()
        return self

    def MSE(self):
        return np.mean(np.power((self.Y_ - self.pred), 2))

    def RMSE(self):
        return np.sqrt(np.mean(np.power((self.Y_ - self.pred), 2)))

    def MAE(self):
        return np.mean(np.abs(self.Y_ - self.pred))

    def GD(self):
        self.weight_ -= self.learning_rate_ * self.gradient[0][0]
        self.bias_ -= self.learning_rate_ * self.gradient[0][1]

    def SGD(self):
        pass

    def RMSprop(self):
        epsilon = pow(10, -8)

        self.EMA1_w = self.beta1_ * self.EMA1_w + (1 - self.beta1_) * np.power(self.gradient[0][0], 2)
        self.EMA1_b = self.beta1_ * self.EMA1_b + (1 - self.beta1_) * np.power(self.gradient[0][1], 2)

        self.weight_ -= self.learning_rate_ * self.gradient[0][0] / np.sqrt(self.EMA1_w + epsilon)
        self.bias_ -= self.learning_rate_ * self.gradient[0][1] / np.sqrt(self.EMA1_b + epsilon)

    def Adam(self):
        epsilon = pow(10, -8)

        self.EMA1_w = self.beta1_ * self.EMA1_w + (1 - self.beta1_) * self.gradient[0][0] / (1 - np.power(self.beta1_, self.t))
        self.EMA1_b = self.beta1_ * self.EMA1_b + (1 - self.beta1_) * self.gradient[0][1] / (1 - np.power(self.beta1_, self.t))
        self.EMA2_w = self.beta2_ * self.EMA2_w + (1 - self.beta2_) * np.power(self.gradient[0][0], 2) / (1 - np.power(self.beta2_, self.t))
        self.EMA2_b = self.beta2_ * self.EMA2_b + (1 - self.beta2_) * np.power(self.gradient[0][1], 2) / (1 - np.power(self.beta2_, self.t))

        self.weight_ -= self.learning_rate_ * self.EMA1_w / (np.sqrt(self.EMA2_w) + epsilon)
        self.bias_ -= self.learning_rate_ * self.EMA1_b / (np.sqrt(self.EMA2_b) + epsilon)
        self.t += 1

    def plot(self):
        self.predict(self.X_)
        plt.clf()
        plt.scatter(self.X_, self.Y_, marker='o', alpha=0.8)
        plt.plot(self.X_, self.pred, 'r')
        plt.title('y = ' + str(self.weight_) + ' x + ' + str(self.bias_) + ' + ' + str(self.loss_history[-1]), fontsize=10, color='0.5')
        plt.draw()
        plt.gcf().canvas.flush_events()
        time.sleep(0.01)

    def predict(self, X):
        self.pred = np.dot(X, self.weight_) + self.bias_
        return self.pred

    def forward(self, weight, bias):
        self.pred = np.dot(self.X_, weight) + bias
        return self.pred

    def stop_criteria(self):
        return

    def select_optimizer(self):
        if self.optimizer_name_ == 'SGD':
            return self.SGD
        elif self.optimizer_name_ == 'Adam':
            self.EMA1_w = 0.0
            self.EMA2_w = 0.0
            self.EMA1_b = 0.0
            self.EMA2_b = 0.0
            self.t = 1
            return self.Adam
        elif self.optimizer_name_ == 'RMSprop':
            self.EMA1_w = 0.0
            self.EMA1_b = 0.0
            return self.RMSprop

    def select_loss_function(self):
        if self.loss_function_ == 'MAE':
            return self.MAE

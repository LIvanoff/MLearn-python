import numpy as np
import matplotlib.pyplot as plt
import autograd as ad
import autograd.variable as av


def cross_entropy(pred, y):
    pred = np.clip(pred, 1e-10, 1 - 1e-10)
    return y * np.log(pred)


class Logistic(object):
    '''
    Класс метода логистической регрессии

    X_           : Вектор объектов-признаков\n
    Y_           : Вектор целевых значение\n
    size_        : Размер выборки\n
    loss_history : История функции потерь\n
    max_iter_    : Максимальное количество итераций/количество эпох\n
    batch_size_  : Рамзер батча\n
    weight       : Веса\n
    pred         : Вектор предсказанных значений\n
    '''
    Y_: np.ndarray
    X_: np.ndarray
    size_: int
    loss_history: np.ndarray
    pred: np.ndarray
    weight: float
    gradient: av.Variable
    batch_size_: int
    max_iter_: int
    optimizer_name_: str
    beta1_: float
    beta2_: float
    EMA1_w: float
    EMA1_b: float
    EMA2_w: float
    EMA2_b: float
    t: int

    def __init__(self, max_iter=300,
                 learning_rate: float = pow(10, -3),
                 optimizer_name: str ='SGD',
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 batch_size: int = None,
                 ):
        self.max_iter_ = max_iter
        self.learning_rate_ = learning_rate
        self.optimizer_name_ = optimizer_name
        self.beta1_ = beta1
        self.beta2_ = beta2
        self.batch_size_ = batch_size
        self.weight = np.random.normal(loc=0.0, scale=0.01)

    def logit(self, X, weight):
        return np.dot(X, weight)

    def sigmoid(self, logits):
        return 1. / (1 + np.exp(-logits))

    def softmax(self, logits):
        return np.exp(logits) / np.mean(np.exp(logits))

    def cross_entropy(self):


    def BCE(self, pred, y):
        pred = np.clip(pred, 1e-10, 1 - 1e-10)
        return np.mean(y * np.log(pred) + (1 - y) * np.log(1 - pred))

    def RMSprop(self):
        epsilon = pow(10, -8)

        self.EMA1_w = self.beta1_ * self.EMA1_w + (1 - self.beta1_) * np.power(self.gradient[0][0], 2)
        self.weight -= self.learning_rate_ * self.gradient[0][0] / np.sqrt(self.EMA1_w + epsilon)

    def Adam(self):
        epsilon = pow(10, -8)

        self.EMA1_w = self.beta1_ * self.EMA1_w + (1 - self.beta1_) * self.gradient[0][0] / (
                    1 - np.power(self.beta1_, self.t))
        self.EMA2_w = self.beta2_ * self.EMA2_w + (1 - self.beta2_) * np.power(self.gradient[0][0], 2) / (
                    1 - np.power(self.beta2_, self.t))

        self.weight -= self.learning_rate_ * self.EMA1_w / (np.sqrt(self.EMA2_w) + epsilon)
        self.t += 1

    def select_optimizer(self):
        if self.optimizer_name_ == 'Adam':
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

    def fit(self, X, Y):
        self.X_ = X
        self.Y_ = Y
        self.size_ = len(self.X_)
        self.loss_history = np.array([])

        if self.optimizer_name_ == 'SGD':
            optimize = self.SGD
        else:
            optimize = self.select_optimizer()

        if self.batch_size_ is None:
            self.batch_size_ = self.size_

        ad.set_mode('reverse')

        for epoch in range(self.max_iter_):
            order = np.random.permutation(len(self.X_))

            for start_index in range(0, len(self.X_), self.batch_size_):
                big_variable = av.Variable([self.weight])
                weight = big_variable[0]

                batch_indexes = order[start_index:start_index + self.batch_size_]

                X_batch = self.X_[batch_indexes]
                y_batch = self.Y_[batch_indexes]

                pred = self.sigmoid(self.logit(X_batch, weight))
                loss = self.__loss(pred=pred, y=y_batch)
                # grad = np.dot(X_batch.T, (pred - y_batch)) / len(y_batch)
                self.gradient = loss.compute_gradients()
                optimize()
                ad.reset_graph()

                if epoch % 10 == 0:
                    print("iter: " + str(epoch) + " loss: " + str(float(loss.data)))

                self.loss_history = np.append(self.loss(pred))

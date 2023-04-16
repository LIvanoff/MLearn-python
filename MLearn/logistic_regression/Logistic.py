import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from matplotlib.colors import ListedColormap


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
    batch_size_: int
    max_iter_: int

    def __init__(self, max_iter=300,
                 learning_rate: float = pow(10, -3),
                 optimizer_name: str = 'SGD',
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

    def logit(self, X, weight):
        return np.dot(X, weight)

    def sigmoid(self, logits):
        return 1. / (1 + np.exp(-logits))

    def BCE(self, pred, y):
        pred = np.clip(pred, 1e-10, 1 - 1e-10)
        return np.mean(-(y * np.log(pred) + (1 - y) * np.log(1 - pred)))

    def softmax(self, logits):
        return np.exp(logits) / np.sum(np.exp(logits))

    def cross_entropy(self, pred, y):
        pred = np.clip(pred, 1e-10, 1 - 1e-10)
        return -y * np.log(pred)

    def predict(self, X, threshold=0.5):
        return self.predict_proba(X) >= threshold

    def predict_proba(self, X):
        n, k = X.shape
        X_ = np.concatenate((np.ones((n, 1)), X), axis=1)
        return self.sigmoid(self.logit(X_, self.weight))

    def gradient_softmax(self, X, pred, y):
        return np.mean(np.dot(X.T, (y - pred))) / len(y)

    def gradient_sigmoid(self, X, pred, y):
        return np.dot(X.T, (pred - y)) / len(y)

    def fit(self, X, Y):
        self.Y_ = Y
        self.loss_history = np.array([])

        n, k = X.shape
        self.X_ = np.concatenate((np.ones((n, 1)), X), axis=1)
        self.size_ = len(self.X_)
        self.weight = np.random.normal(loc=0.0, scale=0.01, size=k + 1)

        if self.batch_size_ is None:
            self.batch_size_ = self.size_

        num_class = len(np.unique(self.Y_))
        if num_class > 2:
            activ = self.softmax
            loss_func = self.cross_entropy
            gradient = self.gradient_softmax
        else:
            activ = self.sigmoid
            loss_func = self.BCE
            gradient = self.gradient_sigmoid

        for epoch in range(self.max_iter_):
            order = np.random.permutation(len(self.X_))

            for start_index in range(0, len(self.X_), self.batch_size_):

                batch_indexes = order[start_index:start_index + self.batch_size_]

                X_batch = self.X_[batch_indexes]
                y_batch = self.Y_[batch_indexes]

                pred = activ(self.logit(X_batch, self.weight))
                loss = loss_func(pred, y_batch)
                grad = gradient(X_batch, pred, y_batch)

                self.weight -= grad * self.learning_rate_
                if epoch % 1000 == 0:
                    print("iter: " + str(epoch) + " loss: " + str(float(loss)))

                self.loss_history = np.append(self.loss_history, loss)


X, y = make_blobs(n_samples=4, centers=[[1, 1], [-1, -1]], cluster_std=1, random_state=42)

colors = ("yellow", "blue")
colored_y = np.zeros(y.size, dtype=str)

for i, cl in enumerate([0, 1]):
    colored_y[y == cl] = str(colors[i])

clf = Logistic(max_iter=10000)

clf.fit(X, y)

plt.figure(figsize=(15, 8))

eps = 0.1
xx, yy = np.meshgrid(np.linspace(np.min(X[:, 0]) - eps, np.max(X[:, 0]) + eps, 500),
                     np.linspace(np.min(X[:, 1]) - eps, np.max(X[:, 1]) + eps, 500))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)

cmap_light = ListedColormap(['#ffff90', '#90ffff'])
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

plt.scatter(X[:, 0], X[:, 1], c=colored_y)
plt.grid(alpha=0.2)
plt.show()

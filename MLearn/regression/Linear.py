import numpy as np
import matplotlib.pyplot as plt
import autograd as ad
import autograd.variable as av
import time

'''
    Пример использования:
    import matplotlib.pyplot as plt
    from regression.Linear import Linear
    
    X = np.array([1, 1.2, 1.6, 1.78, 2, 2.3, 2.4, 3, 3.3, 4, 4.1, 4.12, 4.34, 5, 5.3, 5.6, 6])
    Y = np.array([0.8, 1, 0.9, 1.0, 1.2, 1.1, 1.6, 1.7, 2.0, 2.1, 2.15, 2.22, 2.45, 2.6, 2.12, 2.45, 2.3])

    regression = Linear(learning_rate=0.01,
                        max_iter=100,
                        optimizer_name='RMSprop',
                        batch_size=13)

    regression.fit(X=X, Y=Y)
    fig1, axe = plt.subplots(1, 2, figsize=(15, 6))
    axe[0].scatter(X, Y, marker='o', alpha=0.8)
    axe[0].plot(X, regression.pred, 'r')
    axe[1].plot(x_loss1, regression.loss_history)
    plt.show()
    print(regression.r1_score())
    print(regression.r2_score())
'''


class Linear(object):
    '''
    Класс метода численной линейной регрессии
    size        : Размер выборки\n
    loss_history : История функции потерь\n
    max_iter_    : Максимальное количество итераций/количество эпох\n
    batch_size  : Рамзер батча\n
    weight      : Коэффициент угла наклона регрессии/перваый параметр\n
    bias        : Точка пересечения оси абцисс регрессией/второй параметр\n
    pred         : Вектор предсказанных значений\n
    r1_score     : Коэффициент корреляции\n
    r2_score     : Коэффициент детерминации\n
    '''
    size: int
    loss_history: np.ndarray
    beta1: float
    beta2: float
    EMA1_w: float
    EMA1_b: float
    EMA2_w: float
    EMA2_b: float
    t: int
    r2_score: float
    r1_score: float
    weight: float
    bias: float
    gradient: av.Variable
    batch_size: int
    color: str

    def __init__(self,
                 max_iter: int = 100,
                 stop_criteria: bool = False,
                 learning_rate: float = 1e-3,
                 optimizer_name: str = 'SGD',
                 loss_function: str = 'MSE',
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 batch_size: int = None,
                 ):
        self.weight = np.random.normal(loc=0.0, scale=0.01)
        self.bias = np.random.normal(loc=0.0, scale=0.01)
        self.max_iter = max_iter
        self.stop_criteria = stop_criteria
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name
        self.loss_function = loss_function
        self.beta1 = beta1
        self.beta2 = beta2
        self.batch_size = batch_size

    def fit(self, X, y):
        size = len(X)

        if self.optimizer_name == 'SGD':
            optimize = self.SGD
            self.color = 'r'
        else:
            optimize = self.select_optimizer()

        if self.loss_function == 'MSE':
            loss_function = self.MSE
        else:
            loss_function = self.MAE

        if self.batch_size is None:
            self.batch_size = size

        ad.set_mode('reverse')
        plt.ion()
        self.loss_history = np.array([])

        for epoch in range(self.max_iter):
            order = np.random.permutation(len(X))

            for start_index in range(0, len(X), self.batch_size):
                big_variable = av.Variable([self.weight, self.bias])
                weight, bias = big_variable[0], big_variable[1]

                batch_indexes = order[start_index:start_index + self.batch_size]

                X_batch = X[batch_indexes]
                y_batch = y[batch_indexes]

                pred = self.forward(weight, bias, X_batch)

                loss_value = loss_function(pred, y_batch)
                self.gradient = loss_value.compute_gradients()
                optimize()
                self.loss_history = np.append(self.loss_history, loss_value.data)
                self.plot(X, y, pred)
                ad.reset_graph()

                if epoch % 10 == 0:
                    print("iter: " + str(epoch) + " loss: " + str(float(loss_value.data)))

        plt.ioff()
        plt.show()
        return self

    def MSE(self, pred, y):
        return np.mean(np.power((pred - y), 2))

    def RMSE(self, pred, y):
        return np.sqrt(np.mean(np.power((pred - y), 2)))

    def MAE(self, pred, y):
        return np.mean(np.abs(pred - y))

    def SGD(self):
        self.weight -= self.learning_rate * self.gradient[0][0]
        self.bias -= self.learning_rate * self.gradient[0][1]

    def RMSprop(self):
        epsilon = pow(10, -8)

        self.EMA1_w = self.beta1 * self.EMA1_w + (1 - self.beta1) * np.power(self.gradient[0][0], 2)
        self.EMA1_b = self.beta1 * self.EMA1_b + (1 - self.beta1) * np.power(self.gradient[0][1], 2)

        self.weight -= self.learning_rate * self.gradient[0][0] / np.sqrt(self.EMA1_w + epsilon)
        self.bias -= self.learning_rate * self.gradient[0][1] / np.sqrt(self.EMA1_b + epsilon)

    def Adam(self):
        epsilon = pow(10, -8)

        self.EMA1_w = self.beta1 * self.EMA1_w + (1 - self.beta1) * self.gradient[0][0] / (
                    1 - np.power(self.beta1, self.t))
        self.EMA1_b = self.beta1 * self.EMA1_b + (1 - self.beta1) * self.gradient[0][1] / (
                    1 - np.power(self.beta1, self.t))
        self.EMA2_w = self.beta2 * self.EMA2_w + (1 - self.beta2) * np.power(self.gradient[0][0], 2) / (
                    1 - np.power(self.beta2, self.t))
        self.EMA2_b = self.beta2 * self.EMA2_b + (1 - self.beta2) * np.power(self.gradient[0][1], 2) / (
                    1 - np.power(self.beta2, self.t))

        self.weight -= self.learning_rate * self.EMA1_w / (np.sqrt(self.EMA2_w) + epsilon)
        self.bias -= self.learning_rate * self.EMA1_b / (np.sqrt(self.EMA2_b) + epsilon)
        self.t += 1

    def plot(self, X, y, pred):
        plt.clf()
        plt.scatter(X, y, marker='o', alpha=0.8)
        plt.plot(X, pred, color=self.color, label=str(self.optimizer_name))
        plt.title('y = ' + str(self.weight) + ' x + ' + str(self.bias) + ' + ' + str(self.loss_history[-1]),
                  fontsize=10, color='0.5')
        plt.legend()
        plt.draw()
        plt.gcf().canvas.flush_events()
        time.sleep(0.0001)

    def predict(self, X):
        pred = np.dot(X, self.weight) + self.bias
        return pred

    def forward(self, weight, bias, X):
        pred = np.dot(X, weight) + bias
        return pred

    def stop_criteria(self):
        return

    def select_optimizer(self):
        if self.optimizer_name == 'Adam':
            self.EMA1_w = 0.0
            self.EMA2_w = 0.0
            self.EMA1_b = 0.0
            self.EMA2_b = 0.0
            self.t = 1
            self.color = 'b'
            return self.Adam
        elif self.optimizer_name == 'RMSprop':
            self.EMA1_w = 0.0
            self.EMA1_b = 0.0
            self.color = 'g'
            return self.RMSprop

    def r2_score(self):
        self.r2_score = 1 - np.mean(np.power((self.Y_ - self.pred), 2)) / np.mean(
            np.power((self.Y_ - np.mean(self.Y_)), 2))
        return self.r2_score

    def r1_score(self):
        self.r1_score = np.sqrt(
            1 - np.mean(np.power((self.Y_ - self.pred), 2)) / np.mean(np.power((self.Y_ - np.mean(self.Y_)), 2)))
        return self.r1_score


class Ridge(Linear):
    def __init__(self,
                 l2: float,
                 max_iter: int = 100,
                 stop_criteria: bool = False,
                 learning_rate: float = 1e-3,
                 optimizer_name: str = 'SGD',
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 batch_size: int = None,
                 ):
        super().__init__()
        self.l2 = l2
        self.weight_ = np.random.normal(loc=0.0, scale=0.01)
        self.bias_ = np.random.normal(loc=0.0, scale=0.01)
        self.max_iter_ = max_iter
        self.stop_criteria_ = stop_criteria
        self.learning_rate_ = learning_rate
        self.optimizer_name_ = optimizer_name
        self.beta1_ = beta1
        self.beta2_ = beta2
        self.batch_size_ = batch_size

    def fit(self, X, Y):
        self.X_ = X
        self.Y_ = Y
        self.size_ = len(self.X_)

        if self.optimizer_name_ == 'SGD':
            optimize = self.SGD
            self.color = 'r'
        else:
            optimize = self.select_optimizer()

        loss_function = self.MSE

        if self.batch_size_ is None:
            self.batch_size_ = self.size_

        ad.set_mode('reverse')
        plt.ion()
        self.loss_history = np.array([])

        for epoch in range(self.max_iter_):
            order = np.random.permutation(len(self.X_))

            for start_index in range(0, len(self.X_), self.batch_size_):
                big_variable = av.Variable([self.weight_, self.bias_])
                weight, bias = big_variable[0], big_variable[1]

                batch_indexes = order[start_index:start_index + self.batch_size_]

                X_batch = self.X_[batch_indexes]
                y_batch = self.Y_[batch_indexes]

                self.forward(weight, bias, X_batch)

                loss_value = loss_function(y_batch)
                self.gradient = loss_value.compute_gradients()
                optimize()
                self.loss_history = np.append(self.loss_history, loss_value.data)
                self.plot()
                ad.reset_graph()

                if epoch % 10 == 0:
                    print("iter: " + str(epoch) + " loss: " + str(float(loss_value.data)))

        plt.ioff()
        plt.show()
        return self

    def MSE(self, Y):
        return np.mean(np.power((Y - self.pred), 2)) + self.l2 * np.power((self.weight_ + self.bias_) , 2)
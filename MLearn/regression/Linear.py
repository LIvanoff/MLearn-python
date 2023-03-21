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

    X_           : Вектор объектов-признаков\n
    Y_           : Вектор целевых значение\n
    size_        : Размер выборки\n
    loss_history : История функции потерь\n
    max_iter_    : Максимальное количество итераций/количество эпох\n
    batch_size_  : Рамзер батча\n
    weight_      : Коэффициент угла наклона регрессии/перваый параметр\n
    bias_        : Точка пересечения оси абцисс регрессией\n
    pred         : Вектор предсказанных значений\n
    r1_score     : Коэффициент корреляции\n
    r2_score     : Коэффициент детерминации\n
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
    r1_score: float
    weight_: float
    bias_: float
    gradient: av.Variable
    batch_size_: int
    color: str

    def __init__(self,
                 max_iter: int = 100,
                 stop_criteria: bool = False,
                 learning_rate: float = pow(10, -3),
                 optimizer_name: str = 'SGD',
                 loss_function: str = 'MSE',
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 batch_size: int = None,
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

        if self.loss_function_ == 'MSE':
            loss_function = self.MSE
        else:
            loss_function = self.select_loss_function()

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
        return np.mean(np.power((Y - self.pred), 2))

    def RMSE(self, Y):
        return np.sqrt(np.mean(np.power((Y - self.pred), 2)))

    def MAE(self, Y):
        return np.mean(np.abs(Y - self.pred))

    def SGD(self):
        self.weight_ -= self.learning_rate_ * self.gradient[0][0]
        self.bias_ -= self.learning_rate_ * self.gradient[0][1]

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
        plt.plot(self.X_, self.pred, color=self.color, label=str(self.optimizer_name_))
        plt.title('y = ' + str(self.weight_) + ' x + ' + str(self.bias_) + ' + ' + str(self.loss_history[-1]), fontsize=10, color='0.5')
        plt.legend()
        plt.draw()
        plt.gcf().canvas.flush_events()
        time.sleep(0.0001)

    def predict(self, X):
        self.pred = np.dot(X, self.weight_) + self.bias_
        return self.pred

    def forward(self, weight, bias, X):
        self.pred = np.dot(X, weight) + bias
        return self.pred

    def stop_criteria(self):
        return

    def select_optimizer(self):
        if self.optimizer_name_ == 'Adam':
            self.EMA1_w = 0.0
            self.EMA2_w = 0.0
            self.EMA1_b = 0.0
            self.EMA2_b = 0.0
            self.t = 1
            self.color = 'b'
            return self.Adam
        elif self.optimizer_name_ == 'RMSprop':
            self.EMA1_w = 0.0
            self.EMA1_b = 0.0
            self.color = 'g'
            return self.RMSprop

    def select_loss_function(self):
        if self.loss_function_ == 'MAE':
            return self.MAE
        elif self.loss_function_ == 'RMSE':
            return self.RMSE

    def r2_score(self):
        self.r2_score = 1 - np.mean(np.power((self.Y_ - self.pred), 2)) / np.mean(np.power((self.Y_ - np.mean(self.Y_)), 2))
        return self.r2_score

    def r1_score(self):
        self.r1_score = np.sqrt(1 - np.mean(np.power((self.Y_ - self.pred), 2)) / np.mean(np.power((self.Y_ - np.mean(self.Y_)), 2)))
        return self.r1_score

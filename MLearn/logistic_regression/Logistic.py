import numpy as np
import matplotlib.pyplot as plt
import autograd as ad
import autograd.variable as av


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

    def __init__(self):
        self.weight = np.random.normal(loc=0.0, scale=0.01)

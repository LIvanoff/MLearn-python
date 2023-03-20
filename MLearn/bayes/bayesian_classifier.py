import numpy as np


class Bayes(object):
    '''
    Класс метода наивного байесовского классификатора
    '''
    Y_: np.ndarray
    X_: np.ndarray
    parameters_num_: int
    class_num_: int
    condition_array_: np.ndarray

    def __init__(self, parameters_num: int, class_num):
        self.parameters_num_ = parameters_num
        self.class_num_ = class_num

    def fit(self, X, Y, condition_array):
        self.X_ = X
        self.Y_ = Y
        self.condition_array_ = condition_array



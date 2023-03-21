import numpy as np


class Bayes(object):
    '''
    Класс метода наивного байесовского классификатора
    '''
    Y_: np.ndarray
    X_: np.ndarray
    class_num_: int
    condition_array_: np.ndarray
    class_probability: {}

    def __init__(self, class_num: int):
        self.class_num_ = class_num

    def fit(self, X, Y, condition_array):
        self.X_ = X
        self.Y_ = Y
        self.condition_array_ = condition_array

        for i in range(len(condition_array)):

            for j in range(self.class_num_):
                pass



import numpy as np

'''
    Пример использования:
    from bayes.bayesian_classifier import Bayes
    
    test = pd.read_excel('../bayes_test_decode.xlsx', engine='openpyxl')
    data = test.values
    y = data[:, 3:4]
    x = data[:, :3]
    bayes = Bayes(2)
    condition_array = np.array(['нет', 'да', 'нет'])
    print(bayes.predict(X=x, Y=y, condition_array=condition_array))
'''


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

    def predict(self, X, Y, condition_array):
        self.X_ = X
        self.Y_ = Y
        self.condition_array_ = condition_array
        self.class_probability = dict.fromkeys(np.unique(self.Y_), 1)
        class_frequency = np.array([])

        for key in self.class_probability.keys():
            class_frequency = np.append(class_frequency, np.count_nonzero(self.Y_ == key))

        for i in range(len(condition_array)):
            for key, j in zip(self.class_probability.keys(), range(self.class_num_)):
                Px = np.where(self.X_[:, i] == condition_array[i])[0]
                Py = np.where(self.Y_ == key)[0]
                condition_frequency = np.count_nonzero(np.in1d(Px, Py) == True)
                self.class_probability[key] *= condition_frequency / class_frequency[j]

        for key, j in zip(self.class_probability.keys(), range(self.class_num_)):
            self.class_probability[key] *= class_frequency[j] / len(self.Y_)

        self.normalize()

        return self.class_probability

    def normalize(self):
        probability_sum = sum(self.class_probability.values())
        for key in self.class_probability.keys():
            self.class_probability[key] = self.class_probability[key] / probability_sum

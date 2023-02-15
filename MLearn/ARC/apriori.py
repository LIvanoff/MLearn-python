import pandas as pd
import json
import numpy as np


class Apriori(object):
    f1: np.ndarray
    f2: np.ndarray
    f3: np.ndarray

    transactions_list: np.ndarray
    items_list: np.ndarray

    transactions_count: int
    items_count: int

    transactions_unique: np.ndarray
    items_unique: np.ndarray

    def __init__(self, filename: str, min_length: int):
        self.filename = filename
        self.min_length = min_length
        self.data = self.json_load()
        self.df = pd.DataFrame(self.data)

    def json_load(self):
        with open(self.filename, encoding="utf8") as f:
            return json.load(f)

    def apriori(self, excel: bool = False):
        self.search_f1()
        return

    def search_f1(self):
        self.transactions_count = len(self.df[self.df.columns[0]].unique())
        self.items_count = len(self.df[self.df.columns[1]].unique())

        normal_matrix = np.zeros((self.transactions_count, self.items_count))

        self.transactions_list = self.df[self.df.columns[0]]
        self.items_list = self.df[self.df.columns[1]]

        self.transactions_unique = self.df[self.df.columns[0]].unique()
        self.items_unique = self.df[self.df.columns[1]].unique()

        tmp = ''
        i = 0
        j = 0
        for transaction, item in zip(self.transactions_list, self.items_list):
            if tmp != transaction:
                i = np.where(self.transactions_unique == transaction)
                tmp = transaction
            j = np.where(self.items_unique == item)
            normal_matrix[i][j] = 1
            i = 0
            j = 0
            print(str(transaction) + " " + str(item))
        print(normal_matrix)

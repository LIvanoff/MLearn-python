import pandas as pd
import json
import numpy as np


class Apriori(object):
    def __init__(self, filename: str, min_length: int):
        self.filename = filename
        self.min_length = min_length
        self.data = self.json_load()
        self.df = pd.DataFrame(self.data)
        self.transactions_count = len(self.df[self.df.columns[0]].unique())
        self.items_count = len(self.df[self.df.columns[1]].unique())
        self.normal_matrix = np.zeros((self.transactions_count, self.items_count))
        self.transactions_list = self.df[self.df.columns[0]]
        self.items_list = self.df[self.df.columns[1]]
        self.transactions_unique = self.df[self.df.columns[0]].unique()
        self.items_unique = self.df[self.df.columns[1]].unique()
        self.f1 = np.array([])
        self.f2 = np.array([])
        self.f3 = np.array([])

    def json_load(self):
        with open(self.filename, encoding="utf8") as f:
            return json.load(f)

    def apriori(self, excel: bool = False):
        self.search_f1()
        return

    def search_f1(self):
        tmp = ''
        i: int
        j: int
        for transaction, item in zip(self.transactions_list, self.items_list):
            if tmp != transaction:
                i = list(self.transactions_unique).index(transaction)
                tmp = transaction
            j = list(self.items_unique).index(item)
            self.normal_matrix[i][j] = 1
        print(self.normal_matrix)

        unique_rows = sum(self.normal_matrix[0:])
        print(unique_rows)
        for i in range(len(unique_rows)):
            if unique_rows[i] >= self.min_length:
                self.f1 = np.append(self.f1, self.items_unique[i])
        print(self.f1)





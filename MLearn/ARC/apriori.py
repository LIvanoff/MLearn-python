import pandas as pd
import json
import numpy as np


class Apriori(object):
    f1 = np.array([])
    f2 = np.array([])
    f3 = np.array([])
    transactions_count: int
    items_count: int
    transactions: np.ndarray
    unique_items: np.ndarray

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

        for i in range(self.items_count):
            for j in range(self.transactions_count):
                continue
        print(normal_matrix)

import pandas as pd
import json
import numpy as np


class Apriori(object):
    f1 = np.array([])
    f2 = np.array([])
    f3 = np.array([])
    transactions_count: int
    items_count: int

    def __init__(self, filename: str, min_length: int):
        self.filename = filename
        self.min_length = min_length
        self.data = self.json_load()
        self.df = pd.DataFrame(self.data)
        self.search_f1()

    def json_load(self):
        with open(self.filename, encoding="utf8") as f:
            return json.load(f)

    def apriori(self):
        return

    def search_f1(self):
        self.transactions_count = len(self.df[self.df.columns[0]].unique())
        self.items_count = len(self.df[self.df.columns[1]].unique())
import pandas as pd
import json


class Apriori:
    def __init__(self, filename: str, min_length: int):
        self.filename = filename
        self.min_length = min_length
        self.data = self.json_load()

    def json_load(self):
        with open(self.filename) as f:
            return json.load(f)

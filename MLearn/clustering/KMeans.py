import numpy as np


class KMeans(object):
    def __int__(self,
                clusters: int,
                metric: str = 'euclid_dist',
                max_iter: int = 300,
                stop_criteria: bool = False):

        self.clusters_ = clusters
        self.metric_ = metric
        self.max_iter_ = max_iter

    def train(self):
        return

    def predict(self):
        return

    def euclid_dist(self):
        return

    def manhattan_geom(self):
        return

    def chebyshev_dist(self):
        return

    def sq_euclid_dist(self):
        return

    def pow_dist(self):
        return

    def stop_criteria(self):
        return

# -*- coding: utf-8 -*-

import numpy as np


def gaussian(diff, sig):
    return np.exp(-np.power(diff, 2.) / (2 * sig * sig))


class GaussianBinner:

    def __init__(self, bins=10, w=0.2):
        self.bin_values, self.sigmas = [], []
        self.bins = bins
        self.width = w
        self.eps = 0.000001

    def fit(self, x, features_to_be_binned):
        for index in range(0, features_to_be_binned):

            dimension = x[:, index]
            bin_divisions = np.histogram(dimension, bins=self.bins)[1]

            bin_means = [(bin_divisions[i] + bin_divisions[i+1]) / 2.0
                         for i in range(0, len(bin_divisions) - 1)]

            half_width = abs(bin_divisions[1] - bin_divisions[0]) / 2.0
            bin_means[0:0] = [bin_divisions[0] - half_width]
            bin_means.append(bin_divisions[len(bin_divisions) - 1] + half_width)
            self.bin_values.append(bin_means)

            self.sigmas.append(abs(bin_divisions[1] - bin_divisions[0]) * self.width)
        print(self.sigmas)

    def transform(self, x, features_to_be_binned):
        expanded_features = [x[:, features_to_be_binned:]]
        for index in range(0, features_to_be_binned):

            bin_means = np.array(self.bin_values[index])

            projected_features = gaussian(np.tile(x[:, index], (self.bins + 2, 1)).T - bin_means,
                                              self.sigmas[index])

            sum_f = np.sum(projected_features, axis=1)
            sum_f[sum_f == 0] = self.eps
            projected_features = (projected_features.T / sum_f).T
            expanded_features.append(projected_features)

        return np.concatenate(expanded_features, axis=1)

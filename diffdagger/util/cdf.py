from scipy.interpolate import interp1d
import numpy as np


class CDF:
    def __init__(self, data):
        self.data = data
        self.sorted_data = np.sort(self.data)
        self.cdf = np.arange(1, len(self.sorted_data) + 1) / len(self.sorted_data)
        self.cdf_func = self.create_cdf_function(self.sorted_data, self.cdf)
        self.max_value = max(self.data)

    def create_cdf_function(self, data, cdf):
        return interp1d(
            data, cdf, kind="nearest", fill_value=(0, 1), bounds_error=False
        )

    def __call__(self, value):
        return self.cdf_func(value)

    def get_quantile(self, q):
        """
        Get the quantile value for a given probability q using the sorted data.

        :param q: A float between 0 and 1 representing the desired quantile.
        :return: The value from the original data corresponding to the given quantile.
        """
        if not 0 <= q <= 1:
            raise ValueError("Quantile must be between 0 and 1.")

        index = int(len(self.sorted_data) * q)
        return self.sorted_data[index]

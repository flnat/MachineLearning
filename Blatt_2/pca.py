import numpy as np
import pandas as pd


class PCA:
    def __init__(self, center: bool = True, standardize: bool = False) -> None:
        self._m: int = 0
        self._n: int = 0
        self._center = center
        self._standardize = standardize
        self._u = None
        self._s = None
        self._vh = None

    def fit(self, x: np.ndarray) -> None:

        # check if passed data is a pd.dataframe or series
        # if yes cast to numpy.ndarray
        if isinstance(x, (pd.DataFrame, pd.Series)):
            x = x.values

        # Check if numeric data has been passed
        if not np.issubdtype(x.dtype, np.number):
            raise ValueError("Non numeric data is not supported")

        x = x.copy()
        self._m = x.shape[0]
        self._n = x.shape[1]
        x = self._preprocess(x)

        self._u, self._s, self._vh = np.linalg.svd(x, full_matrices=False)
        self._s = np.diag(self._s)
        if self._n > self._m:
            missing_cols = self._vh.shape[1] - self._s.shape[1]
            self._s = np.hstack((self._s, np.zeros((len(self._s), missing_cols))))
        return

    def get_principal_components(self):
        return self._vh.T

    def get_principal_component_scores(self):
        """
        Calculates the principal component scores / projections on the basis vectors

        :return: np.ndarray of the principal component scores
        """

        return self._u @ self._s
        #return self._u * self._s

    def get_explained_variance(self):
        return (self._s ** 2) / (self._m - 1)

    def _preprocess(self, x: np.ndarray) -> np.ndarray:
        if self._center:
            x -= np.mean(x, axis=0)
        if self._standardize:
            x /= np.std(x, axis=0)

        return x
import numpy as np


class PCA:
    """
    PCA using the Singular-Value Decomposition
    """

    def __init__(self, center: bool = True, scale: bool = True) -> None:
        self.v = None
        self.s = None
        self.u = None
        # m amount of rows
        self._m = None
        # n amount of columns
        self._n = None
        self._center = center
        self._scale = scale

    def fit(self, X: np.ndarray) -> None:
        """
        Performs SVD on the given Design Matrix
        :param X: Design Matrix
        :return:
        """
        X = X.copy()
        X = self._preprocess(X)

        self._m = X.shape[0]
        self._n = X.shape[1]
        self.u, self.s, self.v = np.linalg.svd(X, full_matrices=True)

        self.s = np.diag(self.s)

        # if self._n > self._m:
        missing_cols = self.v.shape[1] - self.s.shape[1]
        missing_rows = self.u.shape[0] - self.s.shape[0]
        self.s = np.hstack((self.s, np.zeros((self.s.shape[0], missing_cols))))
        self.s = np.vstack((self.s, np.zeros((missing_rows, self.s.shape[1]))))

    def get_principal_components(self):
        return self.v

    def get_principal_component_scores(self) -> np.ndarray:
        """
        Return projections of the original variables into the new coordinate system
        :return: Principal Component Scores
        """
        return self.u @ self.s

    def get_explained_variance(self) -> np.ndarray:
        """
        Returns explained variance by the principal components
        :return: Explained Variance
        """
        return (np.diagonal(self.s) ** 2) / np.sqrt(self._m - 1)

    def _preprocess(self, X: np.ndarray) -> np.ndarray:
        """
        Apply Z-standardization as preprocessing
        :param X: Design-Matrix
        :return: Z-standardized Design Matrix
        """
        if self._center:
            X -= np.mean(X, axis=0)
        if self._scale:
            X /= np.std(X, axis=0)
        return X

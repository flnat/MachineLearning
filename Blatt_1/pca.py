import numpy as np


class PCA:
    """
    PCA using the Singular-Value Decomposition
    """

    def __init__(self, center: bool = True, scale: bool = True) -> None:
        self.u, self.s, self.v = None, None, None
        # m amount of rows
        # n amount of columns
        self._m, self._n = None, None
        self.is_fitted: bool = False
        self._center, self._scale = center, scale
        # Means and Standard-Deviations of the training-set
        self._mean, self._sd = None, None

    def fit(self, X: np.ndarray) -> None:
        """
        Performs SVD on the given Design Matrix
        :param X: Design Matrix
        :return:
        """
        X = X.copy()
        self._compute_feature_params(X)
        X = self._preprocess(X)

        self._m = X.shape[0]
        self._n = X.shape[1]
        self.u, self.s, self.v = np.linalg.svd(X, full_matrices=False)

        self.s = np.diag(self.s)

        if self._n > self._m:
            missing_cols = self._n - self._m
            self.s = np.hstack((self.s, np.zeros((len(self.s), missing_cols))))
        self.is_fitted = True

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
            X -= self._mean
        if self._scale:
            X /= self._sd
            return X

    def _compute_feature_params(self, X: np.ndarray) -> None:
        self._mean = np.mean(X, axis=0)
        self._sd = np.std(X, axis=0)

import numpy as np


class PCA():
    """
    PCA using the Singular-Value Decomposition
    """

    def __init__(self, center: bool = True, scale: bool = True) -> None:
        self.v = None
        self.s = None
        self.u = None

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

        self._n = X.shape[0]
        self.u, self.s, self.v = np.linalg.svd(X, full_matrices=False)

    def get_principal_components(self):
        return self.v

    def get_principal_component_scores(self) -> np.ndarray:
        """
        Return projections of the original variables into the new coordinate system
        :return: Principal Component Scores
        """
        return self.u @ np.diag(self.s)

    def get_explained_variance(self) -> np.ndarray:
        """
        Returns explained variance by the principal components
        :return: Explained Variance
        """
        return (self.s ** 2) / np.sqrt(self._n - 1)

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
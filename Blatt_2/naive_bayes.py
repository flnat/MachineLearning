import numpy as np

from typing import Optional


class GaussianNaiveBayes():
    """
    Naive Bayes Classifier under the assumption of gaussian distributed classes
    """

    def __init__(self, priors: Optional[np.ndarray] = None):
        self.fitted: bool = False
        self.priors = None
        self._co
    def fit(self, X: np.ndarray, Y: np.ndarray):
        """
        Fit training data to the Naive Bayes Classifier
        """
        if self.priors is None:
            self.priors = self._get_a_priori_probabilities(Y)

    pass

    def predict(self, X: np.ndarray, probability: bool = False) -> np.ndarray:
        """

        :param X: Data
        :param probability: If True predict returns class membership probabilities if False returns just the labels
        :return: Predictions
        """
        if not self.fitted:
            raise ValueError("Estimator has not yet been fitted!")
        if probability:
            return self._predict_proba(X)
        else:
            return self._predict_scores(X)

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        # TODO Implement
        pass

    def _predict_scores(self, X: np.ndarray) -> np.ndarray:
        # TODO Implement
        pass

    @staticmethod
    def _get_a_priori_probabilities(Y: np.ndarray) -> dict[str, float]:
        classes, counts = np.unique(Y, return_counts=True)
        relative_frequencies = counts / len(Y)
        return {"class": classes, "priori": relative_frequencies}

    def _get_class_seperations(self, X: np.ndarray):
        # TODO Implement
        pass

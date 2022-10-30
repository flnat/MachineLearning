import numpy as np

from typing import Optional


def _get_a_priori_probabilities(Y: np.ndarray):
    classes, counts = np.unique(Y, return_counts=True)
    relative_frequencies = counts / len(Y)
    return {"class": classes, "priori": relative_frequencies}


class GaussianNaiveBayes():
    """
    Naive Bayes Classifier under the assumption of gaussian distributed classes
    """

    def __int__(self, priors: Optional[np.ndarray] = None):
        self.fitted: bool = False
        self.priors = None

    def fit(self, X: np.ndarray, Y: np.ndarray):
        """
        Fit training data to the Naive Bayes Classifier
        """
        if self.priors is None:

    # TODO: Implement Method to estimate a-priori Probability of the Classes
    pass

    def predict(self, X: np.ndarray, probability: bool = False) -> np.ndarray:
        """
        """
        if not self.fitted:
            raise ValueError("Estimator has not yet been fitted!")
        if probability:
            return self._predict_proba(X)
        else:
            return self._predict_scores(X)

    pass

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        # TODO Implement
        pass

    def _predict_scores(self, X: np.ndarray) -> np.ndarray:
        # TODO Implement
        pass

    def _get_class_seperations(self, X: np.ndarray):
        # TODO Implement
        pass

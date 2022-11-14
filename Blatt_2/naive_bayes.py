import numpy as np
from scipy.stats import norm
from typing import Optional


class GaussianNaiveBayes:
    """
    Naive Bayes Classifier under the assumption of gaussian distributed features
    """

    def __init__(self, priors: Optional[np.ndarray] = None) -> None:
        """
        :param priors: Array of the shape (n_classes,) with the prior probabilities of the classes
        """
        # Array of unique Classes in the training Data
        self.classes = None
        # Absolute Frequency of the training classes
        self.class_counts = None
        # Number of distinct classes
        self.n_classes = None
        # Number of distinct features
        self.n_features = None
        # Indicates whether the model has been fitted yet
        self.fitted: bool = False
        # Prior Distribution of the Data --> if not given relative frequencies will be taken
        self.priors = priors
        # Estimated class specific means of the features
        self.means = None
        # Estimated class specific standard deviations of the features
        self.standard_deviations = None

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Fit training data to the Naive Bayes Classifier
        :param X: Training Features
        :param Y: Training Labels
        """
        # Transform labels into 1d Vector
        if Y.ndim != 1:
            Y = Y.flatten()
        self.classes, self.class_counts = np.unique(Y, return_counts=True)
        if self.priors is None:
            self.priors = self.class_counts / len(Y)

        self.n_features, self.n_classes = X.shape[1], len(self.classes)
        # Initialize arrays for the estimated means & standard deviations in the shape (n_features, n_classes)
        self.means = np.ones((self.n_features, self.n_classes))
        self.standard_deviations = np.ones((self.n_features, self.n_classes))

        for idx, clazz in enumerate(self.classes):
            class_samples = X[(Y == clazz), :]
            for feature in range(self.n_features):
                self.means[feature, idx] = np.mean(class_samples[:, feature], axis=0)
                self.standard_deviations[feature, idx] = np.std(class_samples[:, feature], axis=0)

        self.fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the classes of given data
        :param X: Data
        :return: Predicted Classes
        """
        if not self.fitted:
            raise ValueError("Estimator has not yet been fitted!")

        output_size = (len(X), self.n_classes)
        likelihoods = np.ones(output_size)

        for idx, row in enumerate(X):
            for clazz in range(self.n_classes):
                for feature in range(self.n_features):
                    likelihoods[idx, clazz] *= norm.pdf(row[feature], self.means[feature, clazz],
                                                        self.standard_deviations[feature, clazz])
            likelihoods[idx, :] *= self.priors

        predictions = np.zeros(len(X))
        for idx, row in enumerate(likelihoods):
            predictions[idx] = self.classes[np.argmax(row)]

        return predictions

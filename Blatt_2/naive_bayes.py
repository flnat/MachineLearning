import numpy as np
from scipy.stats import norm
from typing import Optional


class GaussianNaiveBayes():
    """
    Naive Bayes Classifier under the assumption of gaussian distributed classes
    """

    def __init__(self, priors: Optional[np.ndarray] = None) -> None:
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
        """
        if self.priors is None:
            self.priors = self._compute_priors(Y)

        # Transform labels into 1d Vector
        if Y.ndim != 1:
            Y = Y.flatten()
        self.classes, self.class_counts = np.unique(Y, return_counts=True)
        self.n_features, self.n_classes = X.shape[1], len(self.classes)
        # Initialize array of the estimated means & standard deviations in the shape (n_features, n_classes)
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
        y_hat = np.ones(output_size)

        for idx, row in enumerate(X):
            for clazz in range(self.n_classes):
                for feature in range(self.n_features):
                    y_hat[idx, clazz] *= norm.pdf(row[feature], self.means[feature, clazz],
                                                  self.standard_deviations[feature, clazz])
            y_hat[idx, :] *= self.priors

        predictions = np.zeros(len(X))
        for idx, row in enumerate(y_hat):
            predictions[idx] = self.classes[np.argmax(row)]

        return predictions

    @staticmethod
    def _compute_priors(Y: np.ndarray) -> np.ndarray[float]:
        classes, counts = np.unique(Y, return_counts=True)
        relative_frequencies = counts / len(Y)
        return relative_frequencies


# if __name__ == "__main__":
#     from face_vectorizer import FaceVectorizer
#     from pca import PCA
#
#     train_images, test_images, train_labels, test_labels = FaceVectorizer(
#         "./data/faces_in_the_wild/lfw_funneled/").get_images()
#     pca = PCA()
#     pca.fit(train_images)
#
#     eigenfaces = pca.v @ pca.s
#     test_projections = test_images @ eigenfaces[0:7, :].T
#     train_projections = train_images @ eigenfaces[0:7, :].T
#
#     train_labels[train_labels != "George_W_Bush"] = -1
#     train_labels[train_labels == "George_W_Bush"] = 1
#
#     test_labels[test_labels != "George_W_Bush"] = -1
#     test_labels[test_labels == "George_W_Bush"] = 1
#
#     nb = GaussianNaiveBayes()
#     nb.fit(train_projections, train_labels)


if __name__ == "__main__":
    from sklearn.datasets import load_iris

    x, y = load_iris(return_X_y=True, as_frame=True)

    nb = GaussianNaiveBayes()
    nb.fit(x.values, y.values.reshape(-1))
    predictions = nb.predict(x.values)
    print(predictions)

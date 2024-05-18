import numpy as np
from operator import itemgetter
import random
import scipy.stats as stats


class LogisticRegressor:
    def __init__(self):
        """
        Initializes the Logistic Regressor model.

        Attributes:
        - weights (np.ndarray): A placeholder for the weights of the model.
                                These will be initialized in the training phase.
        - bias (float): A placeholder for the bias of the model.
                        This will also be initialized in the training phase.
        """
        self.weights = None
        self.bias = None

    def fit(
        self,
        X,
        y,
        learning_rate=0.01,
        num_iterations=1000,
        penalty=None,
        l1_ratio=0.5,
        C=1.0,
        verbose=False,
        print_every=100,
        restart=True,
    ):
        """
        Fits the logistic regression model to the data using gradient descent.

        This method initializes the model's weights and bias, then iteratively updates these parameters by
        moving in the direction of the negative gradient of the loss function (computed using the
        log_likelihood method).

        The regularization terms are added to the gradient of the loss function as follows:

        - No regularization: The standard gradient descent updates are applied without any modification.

        - L1 (Lasso) regularization: Adds a term to the gradient that penalizes the absolute value of
            the weights, encouraging sparsity. The update rule for weight w_j is adjusted as follows:
            dw_j += (C / m) * sign(w_j) - Make sure you understand this!

        - L2 (Ridge) regularization: Adds a term to the gradient that penalizes the square of the weights,
            discouraging large weights. The update rule for weight w_j is:
            dw_j += (C / m) * w_j       - Make sure you understand this!


        - ElasticNet regularization: Combines L1 and L2 penalties.
            The update rule incorporates both the sign and the magnitude of the weights:
            dw_j += l1_ratio * gradient_of_lasso + (1 - l1_ratio) * gradient_of_ridge


        Parameters:
        - X (np.ndarray): The input features, with shape (m, n), where m is the number of examples and n is
                            the number of features.
        - y (np.ndarray): The true labels of the data, with shape (m,).
        - learning_rate (float): The step size at each iteration while moving toward a minimum of the
                            loss function.
        - num_iterations (int): The number of iterations for which the optimization algorithm should run.
        - penalty (str): Type of regularization (None, 'lasso', 'ridge', 'elasticnet'). Default is None.
        - l1_ratio (float): The Elastic Net mixing parameter, with 0 <= l1_ratio <= 1.
                            l1_ratio=0 corresponds to L2 penalty,
                            l1_ratio=1 to L1. Only used if penalty='elasticnet'.
                            Default is 0.5.
        - C (float): Inverse of regularization strength; must be a positive float.
                            Smaller values specify stronger regularization.
        - verbose (bool): Print loss every print_every iterations.
        - print_every (int): Period of number of iterations to show the loss.



        Updates:
        - self.weights: The weights of the model after training.
        - self.bias: The bias of the model after training.
        """
        # TODO: Obtain m (number of examples) and n (number of features)
        m, n = X.shape
        if self.weights is None or restart:
            self.weights = np.zeros(n)
            self.bias = 0
        # TODO: Initialize all parameters to 0

        # TODO: Complete the gradient descent code
        # Tip: You can use the code you had in the previous practice
        # Execute the iterative gradient descent
        for i in range(num_iterations):  # Fill the None here

            # For these two next lines, you will need to implement the respective functions
            # Forward propagation
            y_hat = self.predict_proba(X)
            # Compute loss
            loss = self.log_likelihood(y, y_hat)

            # Logging
            if i % print_every == 0 and verbose:
                print(f"Iteration {i}: Loss {loss}")

            # TODO: Implement the gradient values
            # CAREFUL! You need to calculate the gradient of the loss function (*negative log-likelihood*)
            dz = -(y - y_hat) / m
            dw = X.T @ dz  # Derivative w.r.t. the coefficients
            db = dz.sum()  # Derivative w.r.t. the intercept

            # Regularization:
            # Apply regularization if it is selected.
            # We feed the regularization method the needed values, where "dw" is the derivative for the
            # coefficients, "m" is the number of examples and "C" is the regularization hyperparameter.
            # To do this, you will need to complete each regularization method.
            if penalty == "lasso":
                dw = self.lasso_regularization(dw, m, C)
            elif penalty == "ridge":
                dw = self.ridge_regularization(dw, m, C)
            elif penalty == "elasticnet":
                dw = self.elasticnet_regularization(dw, m, C, l1_ratio)

            # Update parameters
            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db

    def predict_proba(self, X):
        """
        Predicts probability estimates for all classes for each sample X.

        Parameters:
        - X (np.ndarray): The input features, with shape (m, n), where m is the number of samples and
            n is the number of features.

        Returns:
        - A numpy array of shape (m, 1) containing the probability of the positive class for each sample.
        """

        # TODO: z is the value of the logits. Write it here (use self.weights and self.bias):
        z = X @ self.weights + self.bias

        # Return the associated probabilities via the sigmoid trasnformation (symmetric choice)
        return self.sigmoid(z)

    def predict(self, X, threshold=0.5):
        """
        Predicts class labels for samples in X.

        Parameters:
        - X (np.ndarray): The input features, with shape (m, n), where m is the number of samples and n
                            is the number of features.
        - threshold (float): Threshold used to convert probabilities into binary class labels.
                             Defaults to 0.5.

        Returns:
        - A numpy array of shape (m,) containing the class label (0 or 1) for each sample.
        """

        # TODO: Predict the class for each input data given the threshold in the argument
        probabilities = self.predict_proba(X)
        classification_result = np.where(probabilities >= threshold, 1, 0)

        return classification_result

    def lasso_regularization(self, dw, m, C):
        """
        Applies L1 regularization (Lasso) to the gradient during the weight update step in gradient descent.
        L1 regularization encourages sparsity in the model weights, potentially setting some weights to zero,
        which can serve as a form of feature selection.

        The L1 regularization term is added directly to the gradient of the loss function with respect to
        the weights. This term is proportional to the sign of each weight, scaled by the regularization
        strength (C) and inversely proportional to the number of samples (m).

        Parameters:
        - dw (np.ndarray): The gradient of the loss function with respect to the weights, before regularization.
        - m (int): The number of samples in the dataset.
        - C (float): Inverse of regularization strength; must be a positive float.
                    Smaller values specify stronger regularization.

        Returns:
        - np.ndarray: The adjusted gradient of the loss function with respect to the weights,
                      after applying L1 regularization.
        """

        # TODO:
        # ADD THE LASSO CONTRIBUTION TO THE DERIVATIVE OF THE OBJECTIVE FUNCTION

        lasso_gradient = (1 / (C * m)) * np.sign(
            self.weights
        )  # np.where(self.weights >= 0, 1, -1)
        return dw + lasso_gradient

    def ridge_regularization(self, dw, m, C):
        """
        Applies L2 regularization (Ridge) to the gradient during the weight update step in gradient descent.
        L2 regularization penalizes the square of the weights, which discourages large weights and helps to
        prevent overfitting by promoting smaller and more distributed weight values.

        The L2 regularization term is added to the gradient of the loss function with respect to the weights
        as a term proportional to each weight, scaled by the regularization strength (C) and inversely
        proportional to the number of samples (m).

        Parameters:
        - dw (np.ndarray): The gradient of the loss function with respect to the weights, before regularization.
        - m (int): The number of samples in the dataset.
        - C (float): Inverse of regularization strength; must be a positive float.
                     Smaller values specify stronger regularization.

        Returns:
        - np.ndarray: The adjusted gradient of the loss function with respect to the weights,
                        after applying L2 regularization.
        """

        # TODO:
        # ADD THE RIDGE CONTRIBUTION TO THE DERIVATIVE OF THE OBJECTIVE FUNCTION
        ridge_gradient = 1 / (C * m) * self.weights
        return dw + ridge_gradient

    def elasticnet_regularization(self, dw, m, C, l1_ratio):
        """
        Applies Elastic Net regularization to the gradient during the weight update step in gradient descent.
        Elastic Net combines L1 and L2 regularization, incorporating both the sparsity-inducing properties
        of L1 and the weight shrinkage effect of L2. This can lead to a model that is robust to various types
        of data and prevents overfitting.

        The regularization term combines the L1 and L2 terms, scaled by the regularization strength (C) and
        the mix ratio (l1_ratio) between L1 and L2 regularization. The term is inversely proportional to the
        number of samples (m).

        Parameters:
        - dw (np.ndarray): The gradient of the loss function with respect to the weights, before regularization.
        - m (int): The number of samples in the dataset.
        - C (float): Inverse of regularization strength; must be a positive float.
                     Smaller values specify stronger regularization.
        - l1_ratio (float): The Elastic Net mixing parameter, with 0 <= l1_ratio <= 1. l1_ratio=0 corresponds
                            to L2 penalty, l1_ratio=1 to L1. Only used if penalty='elasticnet'.
                            Default is 0.5.

        Returns:
        - np.ndarray: The adjusted gradient of the loss function with respect to the weights,
                      after applying Elastic Net regularization.
        """
        # TODO:
        # ADD THE RIDGE CONTRIBUTION TO THE DERIVATIVE OF THE OBJECTIVE FUNCTION
        # Be careful! You can reuse the previous results and combine them here, but beware how you do this!
        lasso = np.where(self.weights >= 0, 1, -1)
        ridge = self.weights
        elasticnet_gradient = 1 / (C * m) * (l1_ratio * lasso + (1 - l1_ratio) * ridge)
        return dw + elasticnet_gradient

    @staticmethod
    def log_likelihood(y, y_hat):
        """
        Computes the Log-Likelihood loss for logistic regression, which is equivalent to
        computing the cross-entropy loss between the true labels and predicted probabilities.
        This loss function is used to measure how well the model predicts the actual class
        labels. The formula for the loss is:

        L(y, y_hat) = -(1/m) * sum(y * log(y_hat) + (1 - y) * log(1 - y_hat))

        where:
        - L(y, y_hat) is the loss function,
        - m is the number of observations,
        - y is the actual label of the observation,
        - y_hat is the predicted probability that the observation is of the positive class,
        - log is the natural logarithm.

        Parameters:
        - y (np.ndarray): The true labels of the data. Should be a 1D array of binary values (0 or 1).
        - y_hat (np.ndarray): The predicted probabilities of the data belonging to the positive class (1).
                            Should be a 1D array with values between 0 and 1.

        Returns:
        - The computed loss value as a scalar.
        """

        # TODO: Implement the loss function (log-likelihood)
        m = y.shape[0]  # Number of examples
        loss = -(1 / m) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        return loss

    @staticmethod
    def sigmoid(z):
        """
        Computes the sigmoid of z, a scalar or numpy array of any size. The sigmoid function is used as the
        activation function in logistic regression, mapping any real-valued number into the range (0, 1),
        which can be interpreted as a probability. It is defined as 1 / (1 + exp(-z)), where exp(-z)
        is the exponential of the negative of z.

        Parameters:
        - z (float or np.ndarray): Input value or array for which to compute the sigmoid function.

        Returns:
        - The sigmoid of z.
        """

        # TODO: Implement the sigmoid function to convert the logits into probabilities
        sigmoid_value = (1 + np.exp(-z)) ** -1

        return sigmoid_value


class SVM:
    def __init__(self, kernel_func):
        self.kernel_func = kernel_func
        self.Z = None
        self.coeffs = None
        self.X = None

    def create_kernel(self, X):
        self.X = X
        n = X.shape[0]
        self.Z = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                self.Z[i, j] = self.kernel_func(X[i], X[j])  # self.X[i].T @ self.X[j]
                if i != j:
                    self.Z[j, i] = self.Z[i, j]

    def fit(self, X, y, lr=0.01):
        y = np.where(y == 1, 1, -1)
        self.create_kernel(X)
        n = X.shape[0]
        self.coeffs = np.zeros((n))

        for epoch in range(100):
            dw = np.zeros((n))
            for i in range(n):
                gamma = -y[i]
                for j in range(n):
                    gamma += self.coeffs[j] * X[j].T @ X[i]
                dw += gamma * X[i]

            self.coeffs -= lr * dw

    def predict(self, X):
        w = np.sum([self.coeffs[i] * self.X[i] for i in range(self.X.shape[0])], axis=0)
        return np.whree(w @ X > 0, 1, -1)


def minkowski_distance(a, b, p=2):
    """
    Compute the Minkowski distance between two arrays.

    Args:
        a (np.ndarray): First array.
        b (np.ndarray): Second array.
        p (int, optional): The degree of the Minkowski distance. Defaults to 2 (Euclidean distance).

    Returns:
        float: Minkowski distance between arrays a and b.
    """
    return np.sum(abs(a - b) ** p) ** (1 / p)


class knn:
    def __init__(self):
        self.k = None
        self.p = None
        self.x_train = None
        self.y_train = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        k: int = 5,
        p: int = 2,
        job="classification",
    ):
        """
        Fit the model using X as training data and y as target values.

        You should check that all the arguments shall have valid values:
            X and y have the same number of rows.
            k is a positive integer.
            p is a positive integer.

        Args:
            X_train (np.ndarray): Training data.
            y_train (np.ndarray): Target values.
            k (int, optional): Number of neighbors to use. Defaults to 5.
            p (int, optional): The degree of the Minkowski distance. Defaults to 2.
        """
        if len(X_train) != len(y_train):
            raise ValueError(
                "Array of input values and target values must be of same length"
            )
        if k <= 0:
            raise ValueError("K must be greater than 0")
        if p <= 0:
            raise ValueError("p must be greater than 0")
        self.X_train = X_train
        self.y_train = y_train
        self.k = k
        self.p = p
        self.n_tags = len(np.unique(y_train))
        self.job = job

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the class labels for the provided data.

        Args:
            X (np.ndarray): data samples to predict their labels.

        Returns:
            np.ndarray: Predicted class labels.
        """

        if self.job == "classification":
            target_calc = self.most_common_label
        else:
            target_calc = np.mean

        n = len(X)
        Y = np.zeros(n)
        # Y = list()
        for i in range(0, n):
            distances = self.compute_distances(X[i])
            k_nearest_neighbors_indexes = self.get_k_nearest_neighbors(distances)
            # Y.append(self.most_common_label(self.y_train[k_nearest_neighbors_indexes]))
            Y[i] = target_calc(self.y_train[k_nearest_neighbors_indexes])
        return np.array(Y)

    def predict_proba(self, X) -> None:
        """
        Predict the class probabilities for the provided data.

        Each class probability is the amount of each label from the k nearest neighbors
        divided by k.

        Args:
            X (np.ndarray): data samples to predict their labels.

        Returns:
            np.ndarray: Predicted class probabilities.
        """
        n = len(X)
        Y = np.zeros(shape=(n, self.n_tags))
        # Y = np.zeros(shape = (n, 1))

        for i in range(n):
            distances = self.compute_distances(X[i])
            k_nearest_neighbors_indexes = self.get_k_nearest_neighbors(distances)
            neighbors = self.y_train[k_nearest_neighbors_indexes]
            # print(len(neighbors), self.k)
            values = np.zeros(self.n_tags)
            for y_tag in neighbors:
                values[y_tag] += 1
            # vals, values = np.unique(neighbors, return_counts=True)

            Y[i] = np.array(values) / self.k
            # Y[i] = np.sum(neighbors) / self.k
            if Y[i][1] != np.sum(neighbors) / self.k:
                print(Y[i], np.sum(neighbors) / self.k, np.array(values) / self.k)

        return Y

    def compute_distances(self, point: np.ndarray) -> np.ndarray:
        """Compute distance from a point to every point in the training dataset

        Args:
            point (np.ndarray): data sample.

        Returns:
            np.ndarray: distance from point to each point in the training dataset.
        """
        return [
            minkowski_distance(point, neighbor, self.p) for neighbor in self.X_train
        ]

    def get_k_nearest_neighbors(self, distances: np.ndarray) -> np.ndarray:
        """Get the k nearest neighbors indices given the distances matrix from a point.

        Args:
            distances (np.ndarray): distances matrix from a point whose neighbors want to be identified.

        Returns:
            np.ndarray: row indices from the k nearest neighbors.

        Hint:
            You might want to check the np.argsort function.
        """
        indexes = np.argsort(distances)
        nearest_indexes = indexes[: self.k]
        return nearest_indexes

    def most_common_label(self, knn_labels: np.ndarray) -> int:
        """Obtain the most common label from the labels of the k nearest neighbors

        Args:
            knn_labels (np.ndarray): labels from the k nearest neighbors

        Returns:
            int: most common label
        """
        vals, counts = np.unique(knn_labels, return_counts=True)
        index = np.argmax(counts)
        return vals[index]

    def __str__(self):
        """
        String representation of the kNN model.
        """
        return f"kNN model (k={self.k}, p={self.p})"


import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from scipy.spatial import distance
from collections import Counter


class Knn_Neighbors_classifier:
    def _init_(self, n_neighbors=5, p=2) -> None:
        self.k = n_neighbors
        self.p = p
        self.X = None
        self.y = None

    def fit(self, X_train, y_train):
        self.X = X_train
        self.y = y_train

    def predict(self, X_pred):
        y_pred = np.zeros(len(X_pred), dtype=self.y.dtype)
        for i, x in enumerate(X_pred):
            distances = distance.cdist([x], self.X, metric="minkowski", p=self.p)[0]
            k_nearest_neighbours_index = np.argsort(distances)[: self.k]
            k_nearest_neighbours_labels = self.y[k_nearest_neighbours_index].astype(int)
            y_pred[i] = Counter(k_nearest_neighbours_labels).most_common(1)[0][0]
        return y_pred


class LDA:
    def __init__(self):
        self.mean_vectors = None
        self.classes = None

    def fit(self, X, y):
        self.sigma = np.cov(X.T)
        self.classes = np.unique(y)
        means = []
        for unique_class in self.classes:
            class_idx = np.where(y == unique_class)
            means.append(np.mean(X[class_idx], axis=0))
        self.mean_vectors = np.array(means)
        self.sigma_inv = np.linalg.inv(self.sigma)

    def predict(self, X):
        X_sigma = X @ self.sigma_inv

        discriminants = self.mean_vectors @ X_sigma.T

        return np.argmax(discriminants, axis=0)


class QDA:
    def __init__(self):
        self.mean_vectors = None
        self.classes = None
        self.sigmas_inv = None

    def fit(self, X, y):
        self.sigma = np.cov(X.T)
        self.classes = np.unique(y)
        means = []
        sigmas_inv = []
        for unique_class in self.classes:
            class_idx = np.where(y == unique_class)
            means.append(np.mean(X[class_idx], axis=0))
            sigmas_inv.append(np.linalg.inv(np.cov(X[class_idx].T)))
        self.mean_vectors = np.array(means)
        self.sigmas_inv = np.array(sigmas_inv)

    def predict(self, X):
        X_sigma = X @ self.sigmas_inv
        # print((X_sigma @ self.mean_vectors.T)[1, :, 1])

        discriminants = []
        for unique_class in range(len(self.classes)):
            sigma = X_sigma[unique_class]
            discriminants.append(sigma @ self.mean_vectors[unique_class])

        return np.argmax(discriminants, axis=0)


class Naive_Bayes:
    def __init__(self) -> None:
        pass

    def fit(self, X, y, alpha=1, anderson_statistic_threshold=10, use_bins=True):
        self.alpha = alpha
        self.X = X
        self.anderson_statistic_threshold = anderson_statistic_threshold
        self.column_types = [
            (
                "discrete_multinomial"
                if all(X[:, i].astype(int) == X[:, i])
                else (
                    "bins"
                    if use_bins
                    and stats.anderson(X[:, i]).statistic > anderson_statistic_threshold
                    else "gaussian"
                )
            )
            for i in range(X.shape[1])
        ]
        self.y = y
        self.classes = np.unique(y)

        self.dist = {}
        for i in range(self.X.shape[1]):
            self.dist[i] = self.get_distributions(X[:, i], y, self.column_types[i])

    def predict(self, X):

        p_classes = [1 for _ in range(len(self.classes))]

        for i in range(self.X.shape[1]):
            if self.column_types[i] == "gaussian":

                for u_idx, unique_class in enumerate(self.classes):
                    distribution = self.dist[i]
                    mu, sigma2 = (
                        distribution[unique_class]["mean"],
                        distribution[unique_class]["var"],
                    )
                    p_classes[u_idx] *= np.exp(
                        -((X[:, i] - mu) ** 2) / (2 * sigma2)
                    ) / np.sqrt(2 * np.pi * sigma2)

            elif self.column_types[i] == "bins":
                for u_idx, unique_class in enumerate(self.classes):
                    bin_separator = self.dist[i][unique_class]["bin_separator"]

                    x_bins = bin_separator.get_bins(X[:, i])
                    p_classes[u_idx] *= bin_separator.get_fraction(x_bins)

            elif self.column_types[i] == "discrete_multinomial":
                uniques_of_X = np.unique(X[:, i])

                for u_idx, unique_class in enumerate(self.classes):
                    denominator = self.dist[i][unique_class]["denominator"]
                    dict_to_use = self.dist[i][unique_class]["value_counts"]

                    p = len(set(uniques_of_X) | set(dict_to_use.keys()))

                    dict_to_use = {
                        **{unique_of_X: 0 for unique_of_X in uniques_of_X},
                        **dict_to_use,
                    }
                    l = itemgetter(*X[:, i])(dict_to_use)

                    numerator = np.array(l)

                    p_classes[u_idx] *= (numerator + self.alpha) / (
                        denominator + self.alpha * p
                    )

        return np.argmax(p_classes, axis=0)

    @staticmethod
    def get_distributions(X, y, column_type):
        classes = np.unique(y)

        x_by_class = [X[np.where(y == unique_class)] for unique_class in classes]

        if column_type == "gaussian":
            return {
                classes[i]: {
                    "mean": np.mean(x_by_class[i]),
                    "var": np.var(x_by_class[i]),
                }
                for i in range(len(classes))
            }

        elif column_type == "bins":

            values_dict = {}

            for unique_class in classes:
                x_by_class = X[np.where(y == unique_class)]
                bin_separator = BinSeparator(x_by_class, 10)
                bin_separator.set_bins_fractions(x_by_class)

                values_dict[unique_class] = {"bin_separator": bin_separator}

            return values_dict

        elif column_type == "discrete_multinomial":
            return {
                classes[i]: {
                    "denominator": len(x_by_class[i]),
                    "value_counts": {
                        unique_value: counts
                        for unique_value, counts in zip(
                            *np.unique(x_by_class[i], return_counts=True)
                        )
                    },
                }
                for i in range(len(classes))
            }


class BinSeparator:
    def __init__(self, X, nbins):
        min_val, max_val = np.min(X), np.max(X)
        self.bins = np.concatenate((np.linspace(min_val, max_val, nbins), [np.inf]))

    def mapping_function(self, x):
        b = np.where(np.reshape(x, (-1, 1)) - self.bins < 0, 1, 0)
        return np.argmax(b, axis=1)

    def get_bins(self, x):
        return self.mapping_function(x)

    def get_fraction(self, xbins, alpha=1):
        dict_to_use = {
            **{
                bin: {"numerador": 0, "denominador": self.N}
                for bin in range(len(self.bins))
            },
            **self.bin_fractions,
        }
        l = itemgetter(*xbins)(dict_to_use)
        p = len(self.bins)
        return np.array(
            [(bin["numerador"] + alpha) / (bin["denominador"] + p * alpha) for bin in l]
        )

    def set_bins_fractions(self, x):
        x_bins = self.get_bins(x)

        bins, counts = np.unique(x_bins, return_counts=True)

        self.N = len(x)
        self.bin_fractions = {
            bin: {"numerador": count, "denominador": self.N}
            for bin, count in zip(bins, counts)
        }


# UNSUPERVISED


class KMeans:
    def __init__(self, n_clusters, p=2, random_state=None):
        self.n_clusters = n_clusters
        self.p = p
        self.cluster_centers_ = None
        self.random_state = random_state

    def initialize_clusters(self, X):
        np.random.seed(self.random_state)
        self.clusters = np.random.randn(self.n_clusters, X.shape[1])

        furthest_point = None
        for i in range(self.n_clusters):
            if furthest_point is None:
                self.clusters[i] = random.choice(X)
            else:
                self.clusters[i] = furthest_point
            distances = distance.cdist(
                self.clusters[: i + 1], X, metric="minkowski", p=self.p
            )
            distances_sum = np.sum(np.log(distances + 0.01), axis=0)
            # distances_sum = np.sum(distances, axis=0)
            furthest_point = X[np.argmax(distances_sum)]
        self.cluster_centers_ = self.clusters

    def fit(self, X, max_iters=5000) -> None:
        self.initialize_clusters(X)

        for _ in range(max_iters):
            colors = self.find_closest_cluster(X)
            for i in range(self.n_clusters):
                points_assigned = np.where(colors == i)[0]

                if len(points_assigned) > 0:
                    self.clusters[i] = np.mean(X[points_assigned], axis=0)
        self.cluster_centers_ = self.clusters

    def fit_predict(self, X, max_iters=2000):
        self.fit(X, max_iters)
        return self.predict(X)

    def predict(self, X):
        return self.find_closest_cluster(X)

    def find_closest_cluster(self, X):
        distances = distance.cdist(
            X, self.cluster_centers_, metric="minkowski", p=self.p
        )

        colors = np.argmin(distances, axis=1)
        return colors


# KNN Imputer
import pandas as pd
from models.Models import knn
import numpy as np
from sklearn.metrics import mean_absolute_error
from utils.utils import cross_validation


class KNNImputer:
    def __init__(self):
        self.params = {}

    def fit_transform(self, df: pd.DataFrame, columns=None):
        if columns is None:
            columns = df.columns

        df_copy = df.copy().dropna()
        for target_column in columns:
            print("Fitting", target_column)
            inputs = list(set(columns) - set([target_column]))
            X = df_copy[inputs].values
            y = df_copy[target_column].values
            self.params[target_column] = {}
            self.params[target_column]["X_train"] = X
            self.params[target_column]["y_train"] = y
            self.params[target_column]["job"] = "regression"
            self.params[target_column]["p"] = 2

            self.X_train = X
            self.y_train = y

            k_values = [5, 10, 20, 50, 100]
            k_scores, k_stds = [], []
            for k in k_values:
                score, std = cross_validation(
                    knn,
                    X,
                    y,
                    10,
                    params_of_fit={"k": k, "job": "regression"},
                    metric=mean_absolute_error,
                )
                k_scores.append(score)
                k_stds.append(std)

            min_k_idx = np.argmin(k_scores)
            min_k = k_values[min_k_idx]
            self.params[target_column]["k"] = min_k
        print("Finnished fitting")
        return self.transform(df, columns=columns)

    def transform(self, df, columns=None):
        if columns is None:
            columns = self.params.keys()

        df_copy = df.copy()

        for target_column in columns:
            inputs = list(set(columns) - set([target_column]))

            knn_model = knn()
            knn_model.fit(**self.params[target_column])

            na_rows = df[target_column].isna() & ~df[columns].isna().any(axis=1)
            na_values = df.loc[na_rows, inputs].values

            df_copy.loc[na_rows, target_column] = knn_model.predict(na_values)

        return df_copy

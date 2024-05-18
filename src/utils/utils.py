import os
import pymongo
import pymongo.collection
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_curve
from pymongo import MongoClient
import configparser
import pandas as pd
import torch


def evaluate_classification_metrics(y_true, y_pred, positive_label):
    """
    Calculate various evaluation metrics for a classification model.

    Args:
        y_true (array-like): True labels of the data.
        positive_label: The label considered as the positive class.
        y_pred (array-like): Predicted labels by the model.

    Returns:
        dict: A dictionary containing various evaluation metrics.

    Metrics Calculated:
        - Confusion Matrix: [TN, FP, FN, TP]
        - Accuracy: (TP + TN) / (TP + TN + FP + FN)
        - Precision: TP / (TP + FP)
        - Recall (Sensitivity): TP / (TP + FN)
        - Specificity: TN / (TN + FP)
        - F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
    """
    # Map string labels to 0 or 1
    y_true_mapped = np.array([1 if label == positive_label else 0 for label in y_true])
    y_pred_mapped = np.array([1 if label == positive_label else 0 for label in y_pred])

    # Confusion Matrix
    tp = sum((y_pred_mapped == 1) & (y_true_mapped == 1))
    tn = sum((y_pred_mapped == 0) & (y_true_mapped == 0))
    fp = sum((y_pred_mapped == 1) & (y_true_mapped == 0))
    fn = sum((y_pred_mapped == 0) & (y_true_mapped == 1))

    # Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # Precision
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0

    # Recall (Sensitivity)
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0

    # Specificity
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0

    # F1 Score
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) != 0
        else 0
    )

    return {
        "Confusion Matrix": [tn, fp, fn, tp],
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "F1 Score": f1,
    }


def plot_confusion_matrix(confusion_list, ax=None, cbar=True):
    TN, FP, FN, TP = confusion_list
    tn, fp, fn, tp = np.array(confusion_list) / sum(confusion_list)
    text_list = np.array(
        [
            [f"TP\n{tp:.02%}\n{TP:d}", f"FP\n{fp:.02%}\n{FP:d}"],
            [f"FN\n{fn:.02%}\n{FN:d}", f"TN\n{tn:.02%}\n{TN:d}"],
        ]
    )
    if ax is None:
        ax = plt.subplot(3, 2, 5)
    g = sns.heatmap(
        np.array([[tp, fp], [fn, tn]]),
        annot=text_list,
        fmt="",
        xticklabels=[1, 0],
        yticklabels=[1, 0],
        cmap=["red", "green"],
        annot_kws={"weight": "bold"},
        ax=ax,
        cbar=cbar,
    )
    plt.xlabel("Actual value")
    plt.ylabel("Predicted value")
    plt.title("Confusion Matrix")
    return g


def plot_calibration_curve(
    y_true, y_probs, positive_label, n_bins=10, return_vals=False
):
    """
    Plot a calibration curve to evaluate the accuracy of predicted probabilities.

    This function creates a plot that compares the mean predicted probabilities
    in each bin with the fraction of positives (true outcomes) in that bin.
    This helps assess how well the probabilities are calibrated.

    Args:
        y_true (array-like): True labels of the data.
        y_probs (array-like): Predicted probabilities for the positive class (1).
        positive_label: The label considered as the positive class.
        n_bins (int, optional): Number of equally spaced bins to use for calibration. Defaults to 10.

    Returns:
        None: This function plots the calibration curve and does not return any value.

    """
    # Map string labels to 0 or 1
    y_true_mapped = np.array([1 if label == positive_label else 0 for label in y_true])

    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    true_proportions = np.zeros(n_bins)

    for i in range(n_bins):
        indices = (y_probs >= bins[i]) & (y_probs < bins[i + 1])
        if np.sum(indices) > 0:
            true_proportions[i] = np.mean(y_true_mapped[indices])

    if not return_vals:
        plt.subplot(3, 2, 1)
        plt.plot(bin_centers, true_proportions, marker="o")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.title("Calibration Curve")
        # plt.show()
    else:
        return bin_centers, true_proportions


def plot_probability_histograms(
    y_true, y_probs, positive_label, n_bins=10, return_vals=False
):
    """
    Plot probability histograms for the positive and negative classes separately.

    This function creates two histograms showing the distribution of predicted
    probabilities for each class. This helps in understanding how the model
    differentiates between the classes.

    Args:
        y_true (array-like): True labels of the data.
        y_probs (array-like): Predicted probabilities for the positive class.
        positive_label: The label considered as the positive class.
        n_bins (int, optional): Number of bins for the histograms. Defaults to 10.

    Returns:
        None: This function plots the histograms and does not return any value.

    """
    # Map string labels to 0 or 1
    y_true_mapped = np.array([1 if label == positive_label else 0 for label in y_true])

    # plt.figure(figsize=(12, 6))
    if not return_vals:
        # Histogram for positive class
        plt.subplot(3, 2, 3)
        plt.hist(y_probs[y_true_mapped == 1], bins=n_bins, color="green", alpha=0.7)
        plt.xlabel("Predicted Probability")
        plt.ylabel("Frequency")
        plt.title("Probability Histogram (Positive Class)")

        # Histogram for negative class
        plt.subplot(3, 2, 4)
        plt.hist(y_probs[y_true_mapped == 0], bins=n_bins, color="red", alpha=0.7)
        plt.xlabel("Predicted Probability")
        plt.ylabel("Frequency")
        plt.title("Probability Histogram (Negative Class)")

        plt.tight_layout()
        # plt.show()
    else:
        return y_probs, n_bins


def plot_roc(fpr, tpr, model_name="model"):
    # figure = plt.figure()
    plt.subplot(3, 2, 2)
    plt.plot(
        fpr,
        tpr,
        # marker="-",
        label=f"{model_name} (AUC = {abs(np.trapz(x = fpr, y = tpr)):.02f})",
    )
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.title("ROC Curve")
    # plt.show()
    # return figure


def plot_roc_curve(y_true, y_probs, positive_label, return_vals=False):
    """
    Plot the Receiver Operating Characteristic (ROC) curve.

    The ROC curve is a graphical representation of the diagnostic ability of a binary
    classifier system as its discrimination threshold is varied. It plots the True Positive
    Rate (TPR) against the False Positive Rate (FPR) at various threshold settings.

    Args:
        y_true (array-like): True labels of the data.
        y_probs (array-like): Predicted probabilities for the positive class (1).
        positive_label: The label considered as the positive class.

    Returns:
        None: This function plots the ROC curve.

    """
    # Map string labels to 0 or 1
    y_true_mapped = np.array([1 if label == positive_label else 0 for label in y_true])

    thresholds = np.linspace(0, 1, 101)
    tpr = []  # True Positive Rate
    fpr = []  # False Positive Rate

    for thresh in thresholds:
        y_pred = (y_probs >= thresh).astype(int)
        tp = np.sum((y_true_mapped == 1) & (y_pred == 1))
        fp = np.sum((y_true_mapped == 0) & (y_pred == 1))
        fn = np.sum((y_true_mapped == 1) & (y_pred == 0))
        tn = np.sum((y_true_mapped == 0) & (y_pred == 0))

        tpr.append(tp / (tp + fn) if tp + fn != 0 else 0)
        fpr.append(fp / (fp + tn) if fp + tn != 0 else 0)

    tpr.append(0)
    fpr.append(0)
    if not return_vals:
        plot_roc(fpr, tpr)
    else:
        return fpr, tpr


def classification_report(
    y_true,
    y_probs,
    positive_label,
    threshold=0.5,
    n_bins=10,
    figsize=(10, 10),
    title="Classification Report",
):
    """
    Create a classification report using the auxiliary functions developed during Lab2_1

    Args:
        y_true (array-like): True labels of the data.
        y_probs (array-like): Predicted probabilities for the positive class.
        positive_label: The label considered as the positive class.
        threshold (float): Threshold to transform probabilities to predictions. Defaults to 0.5.
        n_bins (int, optional): Number of bins for the histograms and equally spaced
                                bins to use for calibration. Defaults to 10.

    Returns:
        dict: A dictionary containing various evaluation metrics.

    Metrics Calculated:
        - Confusion Matrix: [TN, FP, FN, TP]
        - Accuracy: (TP + TN) / (TP + TN + FP + FN)
        - Precision: TP / (TP + FP)
        - Recall (Sensitivity): TP / (TP + FN)
        - Specificity: TN / (TN + FP)
        - F1 Score: 2 * (Precision * Recall) / (Precision + Recall)

    """
    plt.figure(figsize=figsize)
    plt.suptitle(title)

    plot_calibration_curve(y_true, y_probs, positive_label, n_bins)
    plot_probability_histograms(y_true, y_probs, positive_label, n_bins)
    plot_roc_curve(y_true, y_probs, positive_label)
    metrics = evaluate_classification_metrics(
        y_true, (y_probs > threshold).astype(int), positive_label
    )
    plot_confusion_matrix(metrics["Confusion Matrix"])

    plt.subplot(3, 2, 6)
    metrics_titles = ["Accuracy", "Precision", "Recall", "Specificity", "F1 Score"]
    ax = sns.barplot(
        x=metrics_titles,
        y=[metrics[m] for m in metrics_titles],
        color="blue",
    )
    ax.bar_label(ax.containers[0], fontsize=10, fmt=lambda v: f"{v:.02%}")
    plt.ylim(0, 1)
    plt.show()
    return metrics


def cross_validation(
    model,
    X,
    y,
    pipeline,
    nFolds,
    params_initialization=None,
    params_of_fit=None,
    return_model=False,
    device="cpu",
    metric=accuracy_score,
    perform_extensive_analysis: bool = False,
    model_requires_df: bool = False,
):
    """
    Perform cross-validation on a given machine learning model to evaluate its performance.

    This function manually implements n-fold cross-validation if a specific number of folds is provided.
    If nFolds is set to -1, Leave One Out (LOO) cross-validation is performed instead, which uses each
    data point as a single test set while the rest of the data serves as the training set.

    Parameters:
    - model: scikit-learn-like estimator
        The machine learning model to be evaluated. This model must implement the .fit() and .score() methods
        similar to scikit-learn models.
    - X: array-like of shape (n_samples, n_features)
        The input features to be used for training and testing the model.
    - y: array-like of shape (n_samples,)
        The target values (class labels in classification, real numbers in regression) for the input samples.
    - nFolds: int
        The number of folds to use for cross-validation. If set to -1, LOO cross-validation is performed.

    Returns:
    - mean_score: float
        The mean score across all cross-validation folds.
    - std_score: float
        The standard deviation of the scores across all cross-validation folds, indicating the variability
        of the score across folds.

    Example:
    --------
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.datasets import make_classification

    # Generate a synthetic dataset
    X, y = make_classification(n_samples=100, n_features=20, n_classes=2, random_state=42)

    # Initialize a kNN model
    model = KNeighborsClassifier(n_neighbors=5)

    # Perform 5-fold cross-validation
    mean_score, std_score = cross_validation(model, X, y, nFolds=5)

    print(f'Mean CV Score: {mean_score}, Std Deviation: {std_score}')
    """
    if nFolds == -1:
        # Implement Leave One Out CV
        nFolds = X.shape[0]

    # TODO: Calculate fold_size based on the number of folds
    fold_size = X.shape[0] // nFolds

    # TODO: Initialize a list to store the accuracy values of the model for each fold
    accuracy_scores = []

    if params_of_fit is None:
        params_of_fit = {}

    if params_initialization is None:
        params_initialization = {}

    complete_metrics = {
        "Confusion Matrix": [],
        "Accuracy": [],
        "Precision": [],
        "Recall": [],
        "Specificity": [],
        "F1 Score": [],
        "roc": {
            "fpr": [],
            "tpr": [],
        },
        "calibration": {"bin_centers": [], "true_proportions": []},
        "histograms": {"y_probs": [], "n_bins": []},
    }

    for i in range(nFolds):

        if i == nFolds - 1:
            valid_last = X.shape[0]
            train_last = (i + 1) * fold_size
        else:
            valid_last = (i + 1) * fold_size
            train_last = X.shape[0]

        # TODO: Generate indices of samples for the validation set for the fold
        valid_indices = np.arange(i * fold_size, valid_last)

        # TODO: Generate indices of samples for the training set for the fold
        train_indices = np.concatenate(
            (np.arange(0, i * fold_size), np.arange((i + 1) * fold_size, train_last))
        )

        # TODO: Split the dataset into training and validation
        X_train_df, X_valid_df = X.iloc[train_indices, :], X.iloc[valid_indices, :]
        y_train_df, y_valid_df = y.iloc[train_indices], y.iloc[valid_indices]

        if model_requires_df:
            _, _ = pipeline.fit_transform(X_train_df, y_train_df)
            _, y_valid = pipeline.transform(X_valid_df, y_valid_df)
            X_train, y_train = X_train_df.copy(), y_train_df.copy()
            X_valid = X_valid_df.copy()

        else:
            X_train, y_train = pipeline.fit_transform(X_train_df, y_train_df)
            X_valid, y_valid = pipeline.transform(X_valid_df, y_valid_df)

        # TODO: Train the model with the training set
        my_model = model(**params_initialization)

        if device == "cuda":
            my_model.to(device)
            if isinstance(X_train, torch.Tensor):
                X_train = X_train.to(device)
            else:
                X_train = torch.Tensor(X_train).to(device)
            if isinstance(y_train, torch.Tensor):
                y_train = y_train.to(device)
            else:
                y_train = torch.Tensor(y_train).to(device)

            if isinstance(X_valid, torch.Tensor):
                X_valid = X_valid.to(device)
            else:
                X_valid = torch.Tensor(X_valid).to(device)

        my_model.fit(X_train, y_train, **params_of_fit)
        # TODO: Calculate the accuracy of the my_ with the validation set and store it in accuracy_scores

        if isinstance(y_valid, torch.Tensor):
            y_valid = y_valid.cpu().numpy()
        predictions = my_model.predict(X_valid)
        accuracy_scores.append(metric(y_valid, predictions))

        if perform_extensive_analysis:
            try:
                probabilities = my_model.predict_proba(X_valid)
                if np.ndim(probabilities) > 1:
                    predictions = probabilities[:, 1]
                else:
                    predictions = probabilities
            except:
                pass

            metrics = evaluate_classification_metrics(
                y_valid, (predictions > 0.5).astype(int), 1
            )
            for k, v in metrics.items():
                complete_metrics[k].append(v)

            bin_centers, true_proportions = plot_calibration_curve(
                y_valid, predictions, 1, return_vals=True
            )
            complete_metrics["calibration"]["bin_centers"].append(bin_centers)
            complete_metrics["calibration"]["true_proportions"].append(true_proportions)

            y_probs, n_bins = plot_probability_histograms(
                y_valid, predictions, 1, return_vals=True
            )
            complete_metrics["histograms"]["y_probs"].append(y_probs)
            complete_metrics["histograms"]["n_bins"].append(n_bins)
            fpr, tpr = plot_roc_curve(y_valid, predictions, 1, return_vals=True)
            complete_metrics["roc"]["fpr"].append(fpr)
            complete_metrics["roc"]["tpr"].append(tpr)

    # TODO: Return the mean and standard deviation of the accuracy_scores
    if return_model:
        to_return = (
            np.mean(accuracy_scores),
            np.std(accuracy_scores) / np.sqrt(nFolds),
            my_model,
        )
    elif perform_extensive_analysis:
        to_return = (
            np.mean(accuracy_scores),
            np.std(accuracy_scores) / np.sqrt(nFolds),
            complete_metrics,
        )
    else:
        to_return = (
            np.mean(accuracy_scores),
            np.std(accuracy_scores) / np.sqrt(nFolds),
        )
    return to_return


def cross_validation_og(
    model,
    X,
    y,
    nFolds,
    params_initialization=None,
    params_of_fit=None,
    return_model=False,
    device="cpu",
    metric=accuracy_score,
):
    """
    Perform cross-validation on a given machine learning model to evaluate its performance.

    This function manually implements n-fold cross-validation if a specific number of folds is provided.
    If nFolds is set to -1, Leave One Out (LOO) cross-validation is performed instead, which uses each
    data point as a single test set while the rest of the data serves as the training set.

    Parameters:
    - model: scikit-learn-like estimator
        The machine learning model to be evaluated. This model must implement the .fit() and .score() methods
        similar to scikit-learn models.
    - X: array-like of shape (n_samples, n_features)
        The input features to be used for training and testing the model.
    - y: array-like of shape (n_samples,)
        The target values (class labels in classification, real numbers in regression) for the input samples.
    - nFolds: int
        The number of folds to use for cross-validation. If set to -1, LOO cross-validation is performed.

    Returns:
    - mean_score: float
        The mean score across all cross-validation folds.
    - std_score: float
        The standard deviation of the scores across all cross-validation folds, indicating the variability
        of the score across folds.

    Example:
    --------
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.datasets import make_classification

    # Generate a synthetic dataset
    X, y = make_classification(n_samples=100, n_features=20, n_classes=2, random_state=42)

    # Initialize a kNN model
    model = KNeighborsClassifier(n_neighbors=5)

    # Perform 5-fold cross-validation
    mean_score, std_score = cross_validation(model, X, y, nFolds=5)

    print(f'Mean CV Score: {mean_score}, Std Deviation: {std_score}')
    """
    if nFolds == -1:
        # Implement Leave One Out CV
        nFolds = X.shape[0]

    # TODO: Calculate fold_size based on the number of folds
    fold_size = X.shape[0] // nFolds

    # TODO: Initialize a list to store the accuracy values of the model for each fold
    accuracy_scores = []

    if params_of_fit is None:
        params_of_fit = {}

    if params_initialization is None:
        params_initialization = {}

    for i in range(nFolds):

        if i == nFolds - 1:
            valid_last = X.shape[0]
            train_last = (i + 1) * fold_size
        else:
            valid_last = (i + 1) * fold_size
            train_last = X.shape[0]

        # TODO: Generate indices of samples for the validation set for the fold
        valid_indices = np.arange(i * fold_size, valid_last)

        # TODO: Generate indices of samples for the training set for the fold
        train_indices = np.concatenate(
            (np.arange(0, i * fold_size), np.arange((i + 1) * fold_size, train_last))
        )

        # TODO: Split the dataset into training and validation
        X_train, X_valid = X[train_indices, :], X[valid_indices, :]
        y_train, y_valid = y[train_indices], y[valid_indices]

        # TODO: Train the model with the training set
        my_model = model(**params_initialization)

        if device == "cuda":
            my_model.to(device)
            if isinstance(X_train, torch.Tensor):
                X_train = X_train.to(device)
            else:
                X_train = torch.Tensor(X_train).to(device)
            if isinstance(y_train, torch.Tensor):
                y_train = y_train.to(device)
            else:
                y_train = torch.Tensor(y_train).to(device)

            if isinstance(X_valid, torch.Tensor):
                X_valid = X_valid.to(device)
            else:
                X_valid = torch.Tensor(X_valid).to(device)

        my_model.fit(X_train, y_train, **params_of_fit)
        # TODO: Calculate the accuracy of the my_ with the validation set and store it in accuracy_scores

        if isinstance(y_valid, torch.Tensor):
            y_valid = y_valid.cpu().numpy()
        predictions = my_model.predict(X_valid)
        accuracy_scores.append(metric(y_valid, predictions))

    # TODO: Return the mean and standard deviation of the accuracy_scores
    if return_model:
        to_return = (
            np.mean(accuracy_scores),
            np.std(accuracy_scores) / np.sqrt(nFolds),
            my_model,
        )
    else:
        to_return = (
            np.mean(accuracy_scores),
            np.std(accuracy_scores) / np.sqrt(nFolds),
        )
    return to_return


def read_config() -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    config.read("configuracion.ini")
    return config


def get_collection(config) -> pymongo.collection.Collection:
    client = MongoClient(config["MONGODB"]["connection"])
    db = client[config["MONGODB"]["database"]]
    collection = db[config["MONGODB"]["collection"]]

    return collection


def load(
    data_dir,
    target_y,
    load_unsupervised: bool = False,
    sep_processed_data: str = "\t",
    return_df: bool = False,
    return_tensors: bool = False,
    device: str | torch.device = "cpu",
) -> tuple:

    return_df = return_tensors or return_df

    train_path = os.path.join(data_dir, "train.dat")
    test_path = os.path.join(data_dir, "test.dat")

    train_df = pd.read_csv(train_path, sep=sep_processed_data)
    test_df = pd.read_csv(test_path, sep=sep_processed_data)

    target_y = "RiskPerformance"

    y = train_df[target_y]
    X = train_df.drop(target_y, axis=1)

    y_test = test_df[target_y]
    X_test = test_df.drop(target_y, axis=1)

    if not return_df:
        y = y.values
        X = X.values
        X_test = X_test.values
        y_test = y_test.values

    if return_tensors:
        X = torch.tensor(X).to(device)
        y = torch.tensor(y).to(device)
        X_test = torch.tensor(X_test).to(device)
        y_test = torch.tensor(y_test).to(device)

    if load_unsupervised:
        unsupervised_path = os.path.join(data_dir, "unsupervised.dat")
        unsupervised_df = pd.read_csv(unsupervised_path, sep=sep_processed_data)
        y_unsupervised = unsupervised_df[target_y]
        X_unsupervised = unsupervised_df.drop(target_y, axis=1)

        if not return_df:
            X_unsupervised = X_unsupervised.values
            y_unsupervised = y_unsupervised.values

        if return_tensors:
            X_unsupervised = torch.tensor(X_unsupervised).to(device)
            y_unsupervised = torch.tensor(y_unsupervised).to(device)

        return X, y, X_test, y_test, X_unsupervised, y_unsupervised

    return X, y, X_test, y_test

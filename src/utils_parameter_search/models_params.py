from models.Models import LDA, QDA, LogisticRegressor, Naive_Bayes, knn
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)

from models.NN_Models import (
    MLP,
    DNeighborCompressedNN,
    NeighborNN,
    NeighborCompressedNN,
)
from xgboost import XGBClassifier
import numpy as np

models = {
    "logistic_regression": {
        "model": LogisticRegressor,
        "params_of_fit": {
            "penalty": {"values": [None, "lasso", "ridge", "elasticnet"]},
            "C": {
                "range": [3, 7],
                "log": True,
                "special_case": ["lasso", "ridge", "elasticnet"],
            },
            "l1_ratio": {"range": [0.1, 0.9], "special_case": ["elasticnet"]},
            "learning_rate": {"range": [-3, -1], "log": True},
            "num_iterations": {"values": [3000, 4000]},
        },
        "n_models": 300,
        "dataset": ["complete", "complete_one_hot"],
        "model_type": "regression",
    },
    "knn": {
        "model": knn,
        "params_of_fit": {
            "k": {"range": [90, 200], "int": True},
            "p": {"values": [1], "int": True},
        },
        "n_models": 200,
        "dataset": ["complete", "complete_one_hot"],
        "model_type": "knn",
    },
    "tree": {
        "model": DecisionTreeClassifier,
        "params_initialization": {
            "criterion": {"values": ["gini", "entropy", "log_loss"]},
            "splitter": {"values": ["best", "random"]},
            "max_depth": {"range": [1, 500], "int": True},
            "min_samples_split": {"range": [2, 200], "int": True},
        },
        "n_models": 200,
        "dataset": [
            "complete_pca",
            # "trees",
            # "trees_cleaned_special",
            # "trees_cleaned_special_one_hot",
            # "complete",
        ],
        "model_type": "trees",
    },
    "random_forest": {
        "model": RandomForestClassifier,
        "params_initialization": {
            "n_estimators": {"range": [900, 1200], "int": True},
            "criterion": {"values": ["gini", "entropy", "log_loss"]},
            "max_depth": {"range": [1, 1000], "int": True},
            "min_samples_split": {"range": [2, 150], "int": True},
            "max_features": {"values": [None, "sqrt", "log2"]},
        },
        "n_models": 400,
        "dataset": [
            # "complete_pca",
            "trees",
            # "trees_cleaned_special",
            # "trees_cleaned_special_one_hot",
            "complete",
        ],
        "model_type": "trees",
    },
    "gradient_boosting": {
        "model": GradientBoostingClassifier,
        "params_initialization": {
            "loss": {"values": ["log_loss"]},
            "n_estimators": {"range": [50, 500], "int": True},
            "min_samples_split": {"range": [2, 200], "int": True},
            "max_depth": {"range": [1, 500], "int": True},
            "max_features": {"values": [None, "sqrt", "log2"]},
            "criterion": {"values": ["friedman_mse", "squared_error"]},
            "learning_rate": {"range": [0, 5]},
        },
        "n_models": 400,
        "dataset": [
            "complete_pca",
            # "trees",
            # "trees_cleaned_special",
            # "trees_cleaned_special_one_hot",
            # "complete",
        ],
        "model_type": "trees",
    },
    "ada_boost": {
        "model": AdaBoostClassifier,
        "params_initialization": {
            "n_estimators": {"range": [50, 1000], "int": True},
            "learning_rate": {"range": [0, 1]},
        },
        "n_models": 400,
        "dataset": [
            "complete_pca",
            # "trees",
            # "trees_cleaned_special",
            # "trees_cleaned_special_one_hot",
            # "complete",
        ],
        "model_type": "trees",
    },
    "xgboost": {
        "model": XGBClassifier,
        "params_initialization": {
            "n_estimators": {"range": [50, 1000], "int": True},
            "max_depth": {"range": [1, 3], "int": True},
            "learning_rate": {"range": [0, 0.25]},
        },
        "n_models": 400,
        "dataset": [
            # "complete_pca",
            "trees",
            # "trees_cleaned_special",
            # "trees_cleaned_special_one_hot",
            "complete",
            "complete_one_hot",
        ],
        "model_type": "trees",
    },
    "rbf_svm": {
        "model": SVC,
        "params_initialization": {
            "kernel": {"values": ["rbf"]},
            "degree": {"values": [2, 3], "int": True, "special_case": ["poly"]},
            "C": {"range": [-2, 0], "log": True},
            "gamma": {
                "range": [-2, -0.8],
                "log": True,
                "special_case": ["poly", "rbf"],
            },
            "cache_size": {"values": [7000]},
        },
        "n_models": 100,
        "dataset": ["complete_pca", "complete", "complete_one_hot"],
        "model_type": "svm",
    },
    "linear_svm": {
        "model": SVC,
        "params_initialization": {
            "kernel": {"values": ["linear"]},
            "C": {"range": [-2.5, -1.5], "log": True},
            "cache_size": {"values": [7000]},
        },
        "n_models": 100,
        "dataset": ["complete_pca"],  # , "complete", "complete_one_hot"],
        "model_type": "svm",
    },
    "poly_svm": {
        "model": SVC,
        "params_initialization": {
            "kernel": {"values": ["poly"]},
            "C": {"range": [-2, -1.5], "log": True},
            "cache_size": {"values": [7000]},
            "p": {"range": [2, 4], "int": True},
        },
        "n_models": 100,
        "dataset": ["complete"],  # , "complete", "complete_one_hot"],
        "model_type": "svm",
    },
    "lda": {
        "model": LDA,
        "model_type": "bayes",
        "dataset": ["complete_pca"],  # , "complete"],
        "n_models": 1,
    },
    "qda": {
        "model": QDA,
        "model_type": "bayes",
        "dataset": ["complete_pca"],  # , "complete"],
        "n_models": 1,
    },
    "naive_bayes": {
        "model": Naive_Bayes,
        "model_type": "bayes",
        "params_of_fit": {
            "alpha": {"range": [1, 500], "int": True},
            "anderson_statistic_threshold": {"range": [5, 25]},
            "use_bins": {"values": [True, False]},
        },
        "dataset": ["complete_pca"],  # , "naive_bayes"],
        "n_models": 100,
    },
    "mlp": {
        "model": MLP,
        "model_type": "neural_networks",
        "params_initialization": {
            "hidden": {"range": [1, 10], "int": True},
            "activation_function": {"values": ["tanh", "relu"]},
            "n_layers": {"range": [1, 3], "int": True},
        },
        "params_of_fit": {
            "epochs": {"range": [30, 1000], "int": True},
            "batch_size": {"values": [-1, 64, 128, 256]},
            "device": {"values": ["cuda"]},
            "lr": {"range": [-4, -1], "log": True},
            "l2_lambda": {"range": [-6, np.log10(0.02)], "log": True},
        },
        "device": "cuda",
        "dataset": ["complete_pca"],  # , "complete", "complete_one_hot"],
        "n_models": 100,
    },
    "neighbor_nn": {
        "model": NeighborNN,
        "model_type": "neural_networks",
        "params_initialization": {
            "n_neighbors": {"range": [1, 126], "int": True},
            "hidden": {"range": [1, 10], "int": True},
            "activation_function": {"values": ["tanh", "relu"]},
            "n_layers": {"range": [1, 3], "int": True},
        },
        "params_of_fit": {
            "epochs": {"range": [30, 1000], "int": True},
            "batch_size": {"values": [-1, 64, 128, 256]},
            "device": {"values": ["cuda"]},
            "lr": {"range": [-4, -1], "log": True},
            "l2_lambda": {"range": [-6, 0.5], "log": True},
        },
        "device": "cuda",
        "dataset": ["complete_pca"],  # , "complete", "complete_one_hot"],
        "n_models": 100,
    },
    "neighbor_compressed_nn": {
        "model": NeighborCompressedNN,
        "model_type": "neural_networks",
        "params_initialization": {
            "n_neighbors": {"range": [1, 200], "int": True},
            "compression_size": {"range": [1, 126], "int": True},
            "hidden": {"range": [1, 10], "int": True},
            "activation_function": {"values": ["tanh", "relu"]},
            "n_layers": {"range": [1, 3], "int": True},
        },
        "params_of_fit": {
            "epochs": {"range": [30, 1000], "int": True},
            "batch_size": {"values": [-1, 64, 128, 256]},
            "device": {"values": ["cuda"]},
            "lr": {"range": [-4, -1], "log": True},
            "l2_lambda": {"range": [np.log10(0.04), 0], "log": True},
        },
        "device": "cuda",
        "dataset": ["complete", "complete_pca"],  # , "complete", "complete_one_hot"],
        "n_models": 100,
    },
    "deep_neighbor_compressed_nn": {
        "model": DNeighborCompressedNN,
        "model_type": "neural_networks",
        "params_initialization": {
            "n_neighbors": {"range": [1, 200], "int": True},
            "compression_size": {"range": [1, 126], "int": True},
            "neighbor_size": {"range": [2, 15], "int": True},
            "hidden": {"range": [1, 10], "int": True},
            "activation_function": {"values": ["tanh"]},
            "n_layers": {"range": [1, 3], "int": True},
        },
        "params_of_fit": {
            "epochs": {"range": [30, 1000], "int": True},
            "batch_size": {"values": [-1, 64, 128, 256]},
            "device": {"values": ["cuda"]},
            "lr": {"range": [-4, -1], "log": True},
            "l2_lambda": {"range": [np.log10(0.04), 0], "log": True},
        },
        "device": "cuda",
        "dataset": ["complete"],  # , "complete", "complete_one_hot"],
        "n_models": 100,
    },
}

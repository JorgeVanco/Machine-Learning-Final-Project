import os
from typing import Any
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from cleaning_data.EDA_utils import box_cox, standarize
from sklearn.impute import KNNImputer


class Pipeline:
    def __init__(
        self,
        data_path: str,
        store_dict: str,
        target_column: str,
        impute_special_columns: bool = False,
        create_one_hot_special: bool = False,
        box_cox_cols: list | None = None,
        standarize_cols: list | None = None,
        knn_impute: bool = False,
        leave_na_if_not_knn_imputed: bool = False,
        sep_data_path: str = ";",
        sep_processed_data: str = "\t",
        int_cols: list | None = None,
        random_state: int = 33,
    ) -> None:

        self.data_path = data_path
        self.store_dict = store_dict
        self.target_column = target_column
        self.impute_special_columns = impute_special_columns
        self.create_one_hot_special = create_one_hot_special
        self.box_cox_cols = box_cox_cols
        self.standarize_cols = standarize_cols
        self.knn_impute = knn_impute
        self.leave_na_if_not_knn_imputed = leave_na_if_not_knn_imputed
        self.random_state = random_state
        self.sep_data_path = sep_data_path
        self.sep_processed_data = sep_processed_data
        self.int_cols = int_cols

    def clean_and_divide(self) -> Any:

        print(f"Cleaning data to {self.store_dict}")

        data = pd.read_csv(self.data_path, sep=self.sep_data_path)
        inputs = list(data.columns)
        inputs.remove(self.target_column)

        data = self.drop_9(data, inputs)

        if self.impute_special_columns:
            data = self.impute_special_cols_78(data)

        supervised_data = data[~data[self.target_column].isna()]
        unsupervised_data = data

        X = supervised_data[inputs]
        y = supervised_data[self.target_column]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=0.8, random_state=self.random_state
        )

        # Shuffle unsupervised data
        unsupervised_data = unsupervised_data.sample(
            frac=1, random_state=self.random_state
        )
        unsupervised_X = unsupervised_data[inputs]
        unsupervised_y = unsupervised_data[self.target_column]

        if self.box_cox_cols is not None and len(self.box_cox_cols) > 0:
            X_train, lambdas = box_cox(X_train, self.box_cox_cols)
            X_test, _ = box_cox(X_test, self.box_cox_cols, lambdas)

            unsupervised_X, _ = box_cox(unsupervised_X, self.box_cox_cols)

        if self.standarize_cols is not None and len(self.standarize_cols) > 0:
            X_train, value_params = standarize(X_train, self.standarize_cols)
            X_test, _ = standarize(X_test, self.standarize_cols, value_params)
            unsupervised_X, _ = standarize(unsupervised_X, self.standarize_cols)

        if self.knn_impute:
            imp = KNNImputer(n_neighbors=20)

            X_train.loc[:, inputs] = imp.fit_transform(
                X_train[inputs].values, y_train.values
            )
            X_test.loc[:, inputs] = imp.transform(X_test[inputs].values)

            imp_unsupervised = KNNImputer(n_neighbors=20)
            unsupervised_X.loc[:, inputs] = imp_unsupervised.fit_transform(
                unsupervised_X[inputs].values, unsupervised_y.values
            )

            if self.int_cols is not None:
                for int_col in self.int_cols:
                    X_train[int_col] = X_train[int_col].apply(int)
                    X_test[int_col] = X_test[int_col].apply(int)
                    unsupervised_X[int_col] = unsupervised_X[int_col].apply(int)

        elif not self.leave_na_if_not_knn_imputed:
            X_train, y_train = self.drop_na(X_train, y_train, inputs)
            X_test, y_test = self.drop_na(X_test, y_test, inputs)
            unsupervised_X, unsupervised_y = self.drop_na(
                unsupervised_X, unsupervised_y, inputs
            )

        # Join X and y
        X_train[self.target_column] = y_train
        X_test[self.target_column] = y_test
        unsupervised_X[self.target_column] = unsupervised_y

        # Save
        os.makedirs(self.store_dict, exist_ok=True)

        train_path = os.path.join(self.store_dict, "train.dat")
        test_path = os.path.join(self.store_dict, "test.dat")
        unsupervised_path = os.path.join(self.store_dict, "unsupervised.dat")
        X_train.to_csv(train_path, sep=self.sep_processed_data, index=False)
        X_test.to_csv(test_path, sep=self.sep_processed_data, index=False)
        unsupervised_X.to_csv(
            unsupervised_path, sep=self.sep_processed_data, index=False
        )

    def drop_9(self, data_real, inputs) -> pd.DataFrame:
        data = data_real.copy()
        data[data == -9] = np.nan
        data = data.dropna(subset=inputs, how="all")
        return data

    def impute_special_cols_78(self, data_real) -> pd.DataFrame:
        data = data_real.copy()
        for column in data.columns:
            if any(data[column] == -8):

                if self.create_one_hot_special:
                    new_col = column + "-8"
                    data[new_col] = 0
                    data.loc[data[column] == -8, new_col] = 1

                data.loc[data[column] == -8, column] = np.nan

            if any(data[column] == -7):

                if self.create_one_hot_special:
                    new_col = column + "-7"
                    data[new_col] = 0
                    data.loc[data[column] == -7, new_col] = 1

                data.loc[data[column] == -7, column] = np.nan

        return data

    def drop_na(self, X_real, y_real=None, inputs=None):
        X = X_real.copy()
        if y_real is not None:
            y = y_real.copy()

        if y_real is not None:
            X["target"] = y
        X = X.dropna()
        if y_real is not None:
            y = X["target"]
            X = X[inputs]
        else:
            return X
        return X, y


class PipelineCV(Pipeline):
    def __init__(
        self,
        target_column: str,
        impute_special_columns: bool = False,
        create_one_hot_special: bool = False,
        box_cox_cols: list | None = None,
        standarize_cols: list | None = None,
        knn_impute: bool = False,
        sep_data_path: str = ";",
        sep_processed_data: str = "\t",
        int_cols: list | None = None,
        n_pca_components: int | None = None,
    ) -> None:

        self.target_column = target_column
        self.impute_special_columns = impute_special_columns
        self.create_one_hot_special = create_one_hot_special
        self.box_cox_cols = box_cox_cols
        self.standarize_cols = standarize_cols
        self.knn_impute = knn_impute
        self.sep_data_path = sep_data_path
        self.sep_processed_data = sep_processed_data
        self.int_cols = int_cols
        self.n_pca_components = n_pca_components

    def fit_transform(self, X, y):
        inputs = list(X.columns)
        self.inputs = inputs
        data = X.copy()
        data[self.target_column] = y

        data = self.drop_9(data, inputs)

        if self.impute_special_columns:
            data = self.impute_special_cols_78(data)

        supervised_data = data[~data[self.target_column].isna()]

        X_train = supervised_data[inputs].copy()
        y_train = supervised_data[self.target_column].copy()

        if self.box_cox_cols is not None and len(self.box_cox_cols) > 0:
            X_train, lambdas = box_cox(X_train, self.box_cox_cols)
            self.lambdas = lambdas

        if self.standarize_cols is not None and len(self.standarize_cols) > 0:
            X_train, value_params = standarize(X_train, self.standarize_cols)
            self.value_params = value_params

        if self.knn_impute:
            imp = KNNImputer(n_neighbors=20)

            X_train.loc[:, inputs] = imp.fit_transform(
                X_train[inputs].values, y_train.values
            )

            self.imp = imp

            if self.int_cols is not None:
                for int_col in self.int_cols:
                    X_train[int_col] = X_train[int_col].apply(int)

        else:
            X_train, y_train = self.drop_na(X_train, y_train, inputs)

        if self.n_pca_components is not None:
            self.pca = PCA(self.n_pca_components)
            X_train_pca = self.pca.fit_transform(X_train[inputs].values)
            return X_train_pca, y_train.values

        return X_train.values, y_train.values

    def transform(self, X, y=None):
        data = X.copy()
        if y is not None:
            data[self.target_column] = y.copy()
        data = self.drop_9(data, self.inputs)

        if self.impute_special_columns:
            data = self.impute_special_cols_78(data)

        if y is not None:
            supervised_data = data[~data[self.target_column].isna()]
        else:
            supervised_data = data.copy()

        X_test = supervised_data[self.inputs].copy()

        if y is not None:
            y_test = supervised_data[self.target_column].copy()
        else:
            y_test = None

        if self.box_cox_cols is not None and len(self.box_cox_cols) > 0:
            X_test, _ = box_cox(X_test, self.box_cox_cols, self.lambdas)

        if self.standarize_cols is not None and len(self.standarize_cols) > 0:
            X_test, _ = standarize(X_test, self.standarize_cols, self.value_params)

        if self.knn_impute:
            X_test.loc[:, self.inputs] = self.imp.transform(X_test[self.inputs].values)

            if self.int_cols is not None:
                for int_col in self.int_cols:
                    X_test[int_col] = X_test[int_col].apply(int)

        else:
            X_test, y_test = self.drop_na(X_test, y_test, self.inputs)

        if self.n_pca_components is not None:
            X_test_pca = self.pca.transform(X_test[self.inputs].values)
            if y_test is None:
                return X_test_pca
            return X_test_pca, y_test.values
        if y_test is None:
            return X_test.values
        return X_test.values, y_test.values

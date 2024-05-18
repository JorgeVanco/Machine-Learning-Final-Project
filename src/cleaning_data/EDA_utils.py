from sklearn.linear_model import LinearRegression
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns


def impute_regression(df, c1, c2):
    copy = df.copy()
    copy = copy.dropna(subset=[c1, c2], axis=0, how="all")

    data_interpolate = copy[~copy[[c1, c2]].isna().any(axis=1)][[c1, c2]]

    linear_regression = LinearRegression()
    linear_regression.fit(
        data_interpolate[c1].to_numpy().reshape((-1, 1)),
        data_interpolate[c2].to_numpy().reshape((-1, 1)),
    )

    w, b = linear_regression.coef_[0][0], linear_regression.intercept_[0]

    c2_nan = copy[c2].isna()
    copy.loc[c2_nan, c2] = w * copy.loc[c2_nan, c1] + b

    c1_nan = copy[c1].isna()
    copy.loc[c1_nan, c1] = (copy.loc[c1_nan, c2] - b) / w

    return copy


def standarize(df, columns, values=None):
    copy = df.copy()
    values = {}
    for column in columns:
        values[column] = {"mean": df[column].mean(), "std": df[column].std(ddof=1)}
        copy[column] = (df[column] - values[column]["mean"]) / values[column]["std"]
    return copy, values


def scale(df, columns, values=None):
    copy = df.copy()
    if values is None:
        values = {}
    for column in columns:
        if column not in values:
            values[column] = {"range": copy[column].max() - copy[column].min()}
        copy[column] = df[column] / values[column]["range"]
    return copy, values


def jeoyonsohn(df, columns, lambdas=None):
    copy = df.copy()

    def jeoyonsohn_transform(x, lmbda):
        if x >= 0 and lmbda != 0:
            y = ((x + 1) ** lmbda - 1) / lmbda
        elif x >= 0 and lmbda == 0:
            y = np.log(x + 1)
        elif x < 0 and lmbda != 2:
            y = -((-x + 1) ** (2 - lmbda) - 1) / (2 - lmbda)
        elif x < 0 and lmbda == 2:
            y = -np.log(-x + 1)
        return y

    if lambdas is None:
        lambdas = {}

    for col in columns:

        if col in lambdas:
            lamb = lambdas[col]
            transformed = copy[col].apply(lambda x: jeoyonsohn_transform(x, lamb))
        else:
            min_statistic = None
            for lamb in np.linspace(-20, 20, 2001):
                test_transformed = copy[col].apply(
                    lambda x: jeoyonsohn_transform(x, lamb)
                )

                res = stats.anderson(test_transformed.values)

                if min_statistic is None or res.statistic < min_statistic:
                    min_statistic = res.statistic
                    transformed = test_transformed
                    lambdas[col] = lamb

        copy[col] = transformed
    return copy, lambdas


def box_cox(df, columns, lambdas=None):

    copy = df.copy()

    def box_cox_transform(column, lamb) -> float:
        return np.log(column) if lamb == 0 else (column**lamb - 1) / lamb

    if lambdas is None:
        lambdas = {}

    for col in columns:

        if 0 == copy[col].min():
            copy[col] += 0.001

        if col in lambdas:
            lamb = lambdas[col]
            transformed = box_cox_transform(copy.loc[~copy[col].isna(), col], lamb)
        else:
            min_statistic = None
            for lamb in np.linspace(-5, 5, 501):
                test_transformed = box_cox_transform(
                    copy.loc[~copy[col].isna(), col], lamb
                )

                res = stats.anderson(test_transformed.values)

                if min_statistic is None or res.statistic < min_statistic:
                    min_statistic = res.statistic
                    transformed = test_transformed
                    lambdas[col] = lamb

        copy.loc[~copy[col].isna(), col] = transformed
    return copy, lambdas


import math


def plot_qq_plot(df, hue=None, color_palette=None):
    figure, axes = plt.subplots(
        math.ceil(len(df.columns) / 2), 4, figsize=(20, 20), constrained_layout=True
    )

    axes = axes.reshape((-1, 2))

    for ax, column in zip(axes, df.columns):
        sns.histplot(
            df,
            x=column,
            ax=ax[0],
            hue=hue,
            palette=color_palette if hue is not None else None,
        )
        stats.probplot(df[column], plot=ax[1])
    plt.show()

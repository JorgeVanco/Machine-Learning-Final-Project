import json
import math
import os
from matplotlib import pyplot as plt
import pandas as pd
import random
from cleaning_data.DataPipeline import PipelineCV
from utils.utils import cross_validation, evaluate_classification_metrics, load
import numpy as np
import time
import datetime
import seaborn as sns


def get_values(value_dict, n):
    values = value_dict.get("values", None)
    if values is not None:
        return random.choices(values, k=n)
    else:
        ranges = value_dict.get("range")
        is_int = value_dict.get("int", False)
        is_log = value_dict.get("log", False)

        if is_int:
            return random.choices(range(ranges[0], ranges[1] + 1), k=n)
        elif is_log:
            choices = np.random.uniform(*ranges, n)
            return [10**i for i in choices]
        else:
            return np.random.uniform(*ranges, n).tolist()


def get_params_of_fit(param_dict, key, n, number_tries=50):
    params_of_fit = param_dict.get(key, None)
    if params_of_fit is None:
        return [None] * n

    param_chosen_dict = {}

    special_case = False

    special_cases = set()
    special_case_params = {}
    params_to_fit = []
    tries = 0
    while (length_created := len(params_to_fit)) < n and tries < number_tries:
        for param in params_of_fit:
            param_values = get_values(params_of_fit[param], n - length_created)
            param_chosen_dict[param] = param_chosen_dict.get(param, []) + param_values
            new_special_cases = params_of_fit[param].get("special_case", None)
            if new_special_cases is not None:
                special_cases = special_cases | set(new_special_cases)
                special_case_params[param] = set(new_special_cases)
                special_case = True

        # params_to_fit = pd.DataFrame(param_chosen_dict).drop_duplicates().to_dict(orient="records")
        params_to_fit = [
            dict(zip(param_chosen_dict, t)) for t in zip(*param_chosen_dict.values())
        ]
        unique_values = []
        final_params_to_fit = []

        for record in params_to_fit:
            if special_case:
                record_values = record.values()
                for param, param_special_cases in special_case_params.items():
                    if (param_special_cases & set(record_values)) <= set():
                        record.pop(param)

            values = list(record.values())
            if values not in unique_values:
                unique_values.append(values)
                final_params_to_fit.append(record.copy())
        # params_to_fit = pd.DataFrame(params_to_fit).drop_duplicates().to_dict(orient="records")
        # params_to_fit = pd.DataFrame(params_to_fit).apply(lambda x : x.dropna().to_dict(),axis=1)

        tries += 1
        params_to_fit = final_params_to_fit.copy()
        # for record in params_to_fit:
        #     items = list(record.items())
        #     for k, v in items:
        #         if v is not None and pd.isna(v):
        #             record.pop(k)

    return params_to_fit


def create_model_dict(
    model_name, params_init, params_fit, accuracy, std, dataset, model_type
) -> dict:
    model_dict = {
        "model": model_name,
        "model_type": model_type,
        "accuracy": accuracy,
        "std": std,
    }
    if params_init is not None:
        model_dict["params_initialization"] = params_init
    if params_fit is not None:
        model_dict["params_of_fit"] = params_fit
    model_dict["dataset"] = dataset
    model_dict["time"] = datetime.datetime.fromtimestamp(time.time())

    return model_dict


def parameter_search(
    models,
    target_y,
    datasets_params,
    data_dir: str = "data/split_data",
    nFolds=10,
    collection=None,
    sep_processed_data: str = "\t",
    default_dataset="complete",
    models_to_be_fitted: list | None = None,
) -> None:

    MAXIMUM_ACCURACY_REACHED = 0
    MAXIMUM_ACCURACY_MODEL = []

    X, y, _, _ = load(
        data_dir, target_y, sep_processed_data=sep_processed_data, return_df=True
    )

    n_total_models = 0
    for model_name, param_dict in models.items():

        if models_to_be_fitted is not None and model_name not in models_to_be_fitted:
            continue

        print(f"Performing parameter search on {model_name}:")
        model = param_dict["model"]

        N = param_dict.get("n_models", 10)
        nFolds = param_dict.get("nFolds", nFolds)

        params_of_fit = get_params_of_fit(param_dict, "params_of_fit", N)
        params_initialization = get_params_of_fit(
            param_dict, "params_initialization", N
        )

        max_accuracy = 0
        N = len(params_of_fit)
        print(f"\tTraining {N} models")

        n_total_models += N

        percent = max(N // 20, 1)
        # datasets_loaded = {}

        pipelines_loaded = {}

        posible_datasets = param_dict.get("dataset", [default_dataset])
        device = param_dict.get("device", "cpu")

        for dset in posible_datasets:

            if dset not in datasets_params:
                print(f"\tNo se han encontrado los parametros para el dataset {dset}")
                dataset_pipeline = default_dataset
            else:
                dataset_pipeline = dset

            pipeline = PipelineCV(
                target_column=target_y, **datasets_params[dataset_pipeline]
            )
            pipelines_loaded[dataset_pipeline] = pipeline
            # X, y, _, _ = load(
            #     dataset_path,
            #     target_y,
            #     False,
            #     sep_processed_data,
            #     param_dict.get("return_df", False),
            #     param_dict.get("return_tensors", False),
            #     device,
            # )
            # datasets_loaded[dset] = (X, y)

        best_params = {}
        for model_i, (fit, init) in enumerate(
            zip(*[params_of_fit, params_initialization])
        ):

            # dataset = random.choices(param_dict.get("dataset", [default_dataset]), k=1)[
            #     0
            # ]
            dataset = random.choice(posible_datasets)

            # if dataset not in datasets:
            #     dataset = default_dataset

            pipeline = pipelines_loaded.get(
                dataset, pipelines_loaded.get(default_dataset, None)
            )

            accuracy, std = cross_validation(
                model,
                X,
                y,
                pipeline,
                nFolds,
                params_initialization=init,
                params_of_fit=fit,
                device=device,
            )
            model_dict = create_model_dict(
                model_name, init, fit, accuracy, std, dataset, param_dict["model_type"]
            )
            if accuracy > max_accuracy:
                max_accuracy = accuracy
                best_params = dict(params_initialization=init, params_of_fit=fit)

            if (model_i + 1) % percent == 0:
                print(f"\t{(model_i + 1) / N:.01%}: Max accuracy: {max_accuracy:.04%}")
            if collection is not None:
                collection.insert_one(model_dict)

        print(f"\tMax accuracy reached: {max_accuracy:.04%}")
        print(f"\tBest params for {model_name}:", end=" ")
        print(f"\t{json.dumps(best_params)}")
        if max_accuracy >= MAXIMUM_ACCURACY_REACHED:
            if max_accuracy > MAXIMUM_ACCURACY_REACHED:
                MAXIMUM_ACCURACY_MODEL = [model_name]
            else:
                MAXIMUM_ACCURACY_MODEL.append(model_name)
            MAXIMUM_ACCURACY_REACHED = max_accuracy
        print()
    print(
        f"\nThe maximum accuracy reached is {MAXIMUM_ACCURACY_REACHED:04%} by {', '.join(MAXIMUM_ACCURACY_MODEL)}"
    )
    print(f"Trained a total of {n_total_models} models")


def visualize_accuracy_vs_hyperparameter(
    df, hyperparameters, accuracy_field="accuracy", acc_threshold=0
):
    figure, axes = plt.subplots(
        math.ceil(len(hyperparameters) / 2),
        2,
        figsize=(10, 10),
        constrained_layout=True,
    )
    axes = np.reshape(axes, (-1))
    for ax, column in zip(axes, hyperparameters):
        sns.boxplot(df, x=column, y=accuracy_field, ax=ax)

    return figure

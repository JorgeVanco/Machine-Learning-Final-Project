from DataPipeline import Pipeline
from configparser import ConfigParser


box_cox_cols = [
    "NetFractionRevolvingBurden",
    "AverageMInFile",
    "MSinceOldestTradeOpen",
    "PercentInstallTrades",
    "NumSatisfactoryTrades",
    "NumTotalTrades",
]

standarize_cols = [
    "ExternalRiskEstimate",
    "NetFractionRevolvingBurden",
    "AverageMInFile",
    "MSinceOldestTradeOpen",
    "PercentTradesWBalance",
    "PercentInstallTrades",
    "NumSatisfactoryTrades",
    "NumTotalTrades",
    "PercentTradesNeverDelq",
    "MSinceMostRecentInqexcl7days",
]

standarize_naive_bayes_cols = [
    "ExternalRiskEstimate",
    "NetFractionRevolvingBurden",
    "AverageMInFile",
    "MSinceOldestTradeOpen",
    "PercentTradesWBalance",
    "PercentInstallTrades",
    "NumSatisfactoryTrades",
    "NumTotalTrades",
    # "PercentTradesNeverDelq",
]
standarize_naive_bayes_cols2 = [
    "ExternalRiskEstimate",
    "NetFractionRevolvingBurden",
    "AverageMInFile",
    "MSinceOldestTradeOpen",
    "PercentTradesWBalance",
    "PercentInstallTrades",
    "NumSatisfactoryTrades",
    "NumTotalTrades",
    "PercentTradesNeverDelq",
]

if __name__ == "__main__":

    config = ConfigParser()
    config.read("configuracion.ini")
    data_path = config["DATA"]["data_path"]

    target_y = config["DATA"]["target_y"]

    pipeline = Pipeline(
        data_path=data_path,
        store_dict="data/split_data",
        target_column=target_y,
        leave_na_if_not_knn_imputed=True,
    )

    pipeline.copy()

    # pipeline.clean_and_divide()

    # p = Pipeline(
    #     target_column=target_y,
    #     impute_special_columns=False,
    #     create_one_hot_special=False,
    #     box_cox_cols=None,
    #     standarize_cols=None,
    #     knn_impute=True,
    # )
    # p(data_path, "data/trees")

    # p = Pipeline(
    #     target_column=target_y,
    #     impute_special_columns=True,
    #     create_one_hot_special=False,
    #     box_cox_cols=None,
    #     standarize_cols=None,
    #     knn_impute=True,
    # )
    # p(data_path, "data/trees_cleaned_special")

    # p = Pipeline(
    #     target_column=target_y,
    #     impute_special_columns=True,
    #     create_one_hot_special=True,
    #     box_cox_cols=None,
    #     standarize_cols=None,
    #     knn_impute=True,
    # )
    # p(data_path, "data/trees_cleaned_special_one_hot")

    # p = Pipeline(
    #     target_column=target_y,
    #     impute_special_columns=True,
    #     create_one_hot_special=False,
    #     box_cox_cols=box_cox_cols,
    #     standarize_cols=None,
    #     knn_impute=None,
    # )
    # p(data_path, "data/box_cox")

    # p = Pipeline(
    #     target_column=target_y,
    #     impute_special_columns=True,
    #     create_one_hot_special=False,
    #     box_cox_cols=box_cox_cols,
    #     standarize_cols=standarize_naive_bayes_cols,
    #     knn_impute=True,
    # )
    # p(
    #     data_path,
    #     "data/naive_bayes",
    #     int_cols=["MSinceMostRecentInqexcl7days", "PercentTradesNeverDelq"],
    # )

    # p = Pipeline(
    #     target_column=target_y,
    #     impute_special_columns=True,
    #     create_one_hot_special=False,
    #     box_cox_cols=box_cox_cols,
    #     standarize_cols=standarize_naive_bayes_cols2,
    #     knn_impute=True,
    # )
    # p(
    #     data_path,
    #     "data/naive_bayes_only_one_discrete",
    #     int_cols=["MSinceMostRecentInqexcl7days", "PercentTradesNeverDelq"],
    # )

    # p = Pipeline(
    #     target_column=target_y,
    #     impute_special_columns=True,
    #     create_one_hot_special=False,
    #     box_cox_cols=box_cox_cols,
    #     standarize_cols=standarize_cols,
    #     knn_impute=True,
    # )
    # p(data_path, "data/complete")

    # p = Pipeline(
    #     target_column=target_y,
    #     impute_special_columns=True,
    #     create_one_hot_special=True,
    #     box_cox_cols=box_cox_cols,
    #     standarize_cols=standarize_cols,
    #     knn_impute=True,
    # )
    # p(data_path, "data/complete_one_hot")

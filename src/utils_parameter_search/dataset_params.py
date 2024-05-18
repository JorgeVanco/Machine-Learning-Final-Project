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


datasets = {
    "complete": {
        "impute_special_columns": True,
        "create_one_hot_special": False,
        "box_cox_cols": box_cox_cols,
        "standarize_cols": standarize_cols,
        "knn_impute": True,
    },
    "complete_one_hot": {
        "impute_special_columns": True,
        "create_one_hot_special": True,
        "box_cox_cols": box_cox_cols,
        "standarize_cols": standarize_cols,
        "knn_impute": True,
    },
    "complete_pca": {
        "impute_special_columns": True,
        "create_one_hot_special": False,
        "box_cox_cols": box_cox_cols,
        "standarize_cols": standarize_cols,
        "knn_impute": True,
        "n_pca_components": 6,
    },
    "trees": {
        "impute_special_columns": False,
        "create_one_hot_special": False,
        "box_cox_cols": None,
        "standarize_cols": None,
        "knn_impute": True,
    },
    "trees_cleaned_special": {
        "impute_special_columns": True,
        "create_one_hot_special": False,
        "box_cox_cols": None,
        "standarize_cols": None,
        "knn_impute": True,
    },
    "trees_cleaned_special_one_hot": {
        "impute_special_columns": True,
        "create_one_hot_special": True,
        "box_cox_cols": None,
        "standarize_cols": None,
        "knn_impute": True,
    },
    "box_cox": {
        "impute_special_columns": True,
        "create_one_hot_special": False,
        "box_cox_cols": box_cox_cols,
        "standarize_cols": None,
        "knn_impute": None,
    },
    "naive_bayes": {
        "impute_special_columns": True,
        "create_one_hot_special": False,
        "box_cox_cols": box_cox_cols,
        "standarize_cols": standarize_naive_bayes_cols,
        "knn_impute": True,
        "int_cols": ["MSinceMostRecentInqexcl7days", "PercentTradesNeverDelq"],
    },
    "naive_bayes_only_one_discrete": {
        "impute_special_columns": True,
        "create_one_hot_special": False,
        "box_cox_cols": box_cox_cols,
        "standarize_cols": standarize_naive_bayes_cols2,
        "knn_impute": True,
        "int_cols": ["MSinceMostRecentInqexcl7days", "PercentTradesNeverDelq"],
    },
}

from utils_parameter_search.utils_parameter_search import parameter_search
from utils_parameter_search.models_params import models
from utils_parameter_search.dataset_params import datasets
from sklearnex import patch_sklearn
from utils.utils import read_config, get_collection


if __name__ == "__main__":

    modelos_a_entrenar = ["lda"]

    patch_sklearn()
    # Lee configuración de config
    config = read_config()
    collection = get_collection(config)
    target_y = config["DATA"]["target_y"]
    # sep_processed_data = config["DATA"]["sep_processed_data"] # No funciona bien \t

    sep_processed_data = "\t"

    parameter_search(
        models,
        target_y,
        datasets,
        collection=collection,
        sep_processed_data=sep_processed_data,
        models_to_be_fitted=modelos_a_entrenar,
    )

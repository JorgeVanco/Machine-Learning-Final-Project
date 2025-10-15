The write-up of the project is [this pdf](https://github.com/JorgeVanco/Machine-Learning-Final-Project/blob/main/Memoria_proyecto_final_ml_Jorge_Vanco_Sampedro.pdf)

--- 

Todos los modelos entrenados se encuentran en /src/models

Para limpiar los datos, los códigos se encuentran en /src/cleaning_data. Ejecutando prepare_datasets.py se crea el train-test split en la carpeta /src/data/split_data

Para ejecutar realizar random grid search, hay que poner los diccionarios de los modelso en utils_parameter_search/models_params, los pipelines posibles en utils_parameter_search/datasets_params y ejecutar parameter_search.py. Se pueden elegir los modelos a entrenar pasando una lista a models_to_be_fitted.

En unsupervised.ipynb se encuentran los códigos de los métodos no supervisados.

En special_nn.ipynb hay pruebas con redes neuronales.

En logistic_regression hay unas gráficas de validación cruzada, es importante la de lasso.

En explainability.ipynb están los gráficos de explicabilidad.

Los ficheros .obj guardan información de parametros usados en los jupyter notebooks. El modelo final entrenado se encuentra en /src/final_model. Para hacer inferencia, se ejecuta inference.ipynb

En configuracion.ini se encuentran los parámetros de la conexión con MongoDB y algun parámetro que se tiene en cuenta en parameter_search.py

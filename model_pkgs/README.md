# Steps to create `MLflow Model` package
1. Create the following directory structure at the root of the repo
```
model_pkgs/
└── mlflow_pkg
    ├── model_code
    └── model_data
```
2. Download model weights and auxiliary data to `model-data` directory
by running the following command at the root of the repo
```
$ transcriptformer download all --checkpoint-dir model_pkgs/mlflow_pkg/model_data/
```
3. Create `tf_mlflow_model.py` that creates a custom `MLflow PythonModel` that provides an uniform interface to wrap `transcriptformer` inference interface

4. Create `package.py` to save the 3 model variants as a `pyfunc` model

5. Create `predict.py` to invoke prediction on the `pyfunc` wrapped model

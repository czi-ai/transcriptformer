# Steps to create and test `MLflow Model` package
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

6. Run `package.py` to create `MLflow Model` artifact

```
$ python model_pkgs/mlflow_pkg/package.py --model-variant tf_sapiens   --checkpoint-path model_pkgs/mlflow_pkg/model_data/tf_sapiens --output-dir model_pkgs/mlflow_pkg/mlflow_models --requirements model_pkgs/mlflow_pkg/requirements-mlflow-pkg.txt
```

```
$ tree model_pkgs/mlflow_pkg/mlflow_models/

model_pkgs/mlflow_pkg/mlflow_models/
└── transcriptformer_tf_sapiens
    ├── MLmodel
    ├── artifacts
    │   └── tf_sapiens
    │       ├── config.json
    │       ├── model_weights.pt
    │       └── vocabs
    │           ├── assay_vocab.json
    │           └── homo_sapiens_gene.h5
    ├── code
    │   └── model_code
    │       ├── __init__.py
    │       └── tf_mlflow_model.py
    ├── conda.yaml
    ├── python_env.yaml
    ├── python_model.pkl
    └── requirements.txt
```

7. Run `predict.py` using the `MLflow Model` artifact

```
$ python model_pkgs/mlflow_pkg/predict.py --model-path model_pkgs/mlflow_pkg/mlflow_models/transcriptformer_tf_sapiens --input-file ~/.cz-benchmarks/datasets/tsv2_bladder.h5ad --output-file model_pkgs/mlflow_pkg/tf_results/tsv2_bladder_embeddings.h5ad --gene-col-name ensembl_id --precision 16-mixed
```

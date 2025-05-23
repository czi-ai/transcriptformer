from pathlib import Path

import mlflow

model_uri = "mlflow_models/transcriptformer_tf_sapiens"
input_file = Path("/path/to/input_data.h5ad")
output_file = Path("/path/to/output_embeddings.h5ad")

result = mlflow.models.predict(
    model_uri=model_uri,
    input_data=str(input_file),
    params={"output_file": str(output_file), "batch_size": 32, "gene_col_name": "ensembl_id", "precision": "16-mixed"},
    env_manager="uv",
)

print(result)

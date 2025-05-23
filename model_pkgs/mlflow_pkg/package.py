import mlflow.pyfunc
from model_code.tf_mlflow_model import TranscriptformerMLflowModel

variants = [
    {
        "name": "tf_sapiens",
        "checkpoint": "/checkpoints/tf_sapiens",
        "embedding": "/checkpoints/all_embeddings/homo_sapiens_gene.h5",
    },
    {
        "name": "tf_exemplar",
        "checkpoint": "/checkpoints/tf_exemplar",
        "embedding": "/checkpoints/all_embeddings/exemplar.h5",
    },
    {
        "name": "tf_metazoa",
        "checkpoint": "/checkpoints/tf_metazoa",
        "embedding": "/checkpoints/all_embeddings/metazoa.h5",
    },
]

for variant in variants:
    model_path = f"mlflow_models/transcriptformer_{variant['name']}"
    mlflow.pyfunc.save_model(
        path=model_path,
        python_model=TranscriptformerMLflowModel(),
        artifacts={"checkpoint_path": variant["checkpoint"], "pretrained_embedding": variant["embedding"]},
        pip_requirements="requirements-mlflow-pkg.txt",
        code_path=["model_code"],
        env_manager="uv",
        metadata={"tags": {"model_variant": variant["name"]}},
    )
    print(f"Saved model: {model_path} with variant: {variant['name']}")

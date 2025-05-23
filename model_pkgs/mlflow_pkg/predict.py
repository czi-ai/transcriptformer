"""
Run inference using a packaged MLflow Transcriptformer model.

Example usage:

    python predict.py \
        --model-path mlflow_models/transcriptformer_tf_sapiens \
        --input-file /data/sample_input.h5ad \
        --output-file /data/output_embeddings.h5ad \
        --gene-col-name ensembl_id \
        --precision 16-mixed \
        --pretrained-embedding /data/optional_embedding.h5
"""

import argparse

import mlflow


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference using an MLflow-wrapped Transcriptformer model")
    parser.add_argument("--model-path", required=True, help="Path to saved MLflow model directory")
    parser.add_argument("--input-file", required=True, help="Path to input .h5ad file")
    parser.add_argument("--output-file", required=True, help="Path to output embeddings .h5ad file")

    # Optional CLI arguments
    parser.add_argument("--batch-size", type=int, help="Batch size to use")
    parser.add_argument("--gene-col-name", default="ensembl_id", help="Gene column name")
    parser.add_argument("--precision", default="16-mixed", help="Precision setting")
    parser.add_argument("--pretrained-embedding", help="Optional path to pretrained embedding file")

    return parser.parse_args()


def main():
    args = parse_args()

    params = {"output_file": args.output_file, "gene_col_name": args.gene_col_name, "precision": args.precision}

    if args.batch_size is not None:
        params["batch_size"] = args.batch_size

    if args.pretrained_embedding:
        params["pretrained_embedding"] = args.pretrained_embedding

    # Load the model without specifying env_manager
    model = mlflow.pyfunc.load_model(model_uri=args.model_path)

    # Perform prediction with custom parameters
    result = model.predict(args.input_file, params=params)

    print("Inference completed.")
    print("Output written to:", result["output_file"])


if __name__ == "__main__":
    main()

import subprocess
from pathlib import Path

import mlflow.pyfunc


class TranscriptformerMLflowModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.checkpoint_path = context.artifacts["checkpoint_path"]
        self.pretrained_embedding = context.artifacts["pretrained_embedding"]
        self.model_variant = context.tags.get("model_variant", "unknown")

    def _get_default_batch_size(self) -> int:
        # Empirical defaults for Tesla T4
        if self.model_variant == "tf_sapiens":
            return 32
        elif self.model_variant == "tf_exemplar":
            return 8
        elif self.model_variant == "tf_metazoa":
            return 2
        return 16  # Safe fallback

    def predict(self, context, model_input: str | Path, params: dict | None = None) -> dict[str, str]:
        input_filepath = Path(model_input)

        if not input_filepath.is_file():
            raise ValueError(f"model_input must be a valid file path: {input_filepath}")

        if not params or "output_file" not in params:
            raise ValueError("params must include 'output_file'.")

        output_file = Path(params["output_file"])
        output_path = output_file.parent
        output_filename = output_file.name

        # Handle batch size
        batch_size = params.get("batch_size")
        if batch_size is None:
            batch_size = self._get_default_batch_size()
        else:
            batch_size = str(batch_size)

        gene_col_name = params.get("gene_col_name", "ensembl_id")
        precision = params.get("precision", "16-mixed")

        cmd = [
            "transcriptformer",
            "inference",
            "--checkpoint-path",
            str(self.checkpoint_path),
            "--data-file",
            str(input_filepath),
            "--output-path",
            str(output_path),
            "--output-filename",
            output_filename,
            "--batch-size",
            batch_size,
            "--gene-col-name",
            gene_col_name,
            "--precision",
            precision,
            "--pretrained-embedding",
            str(self.pretrained_embedding),
        ]

        subprocess.run(cmd, check=True)

        return {"output_file": str(output_file)}

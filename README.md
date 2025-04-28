# TranscriptFormer

## Description

Transcriptformer is a deep learning model for cross-species single-cell RNA sequencing analysis. It uses transformer-based architectures to learn representations of gene expression data across multiple species, leveraging protein sequence information through ESM-2 embeddings.

## Installation

Transcriptformer requires Python >=3.11.

#### Install from source with uv

```bash
# Clone the repository
git clone https://github.com/czi-ai/transcriptformer.git
cd transcriptformer

# Create and activate a virtual environment with Python 3.11
uv venv --python=3.11
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
uv pip install -e .
```

#### Install from PyPI with uv

```bash
# Create and activate a virtual environment
uv venv --python=3.11
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install from PyPI
uv pip install transcriptformer
```

### Requirements

Transcriptformer has the following core dependencies:
- PyTorch (<=2.5.1, as 2.6.0+ may cause pickle errors)
- PyTorch Lightning
- anndata
- scanpy
- numpy
- pandas
- h5py
- hydra-core

See the `pyproject.toml` file for the complete list of dependencies.

### Hardware Requirements
You'll need a Python environment with GPU access to run this model. While we've tested it on NVIDIA A100 GPUs, you can use smaller GPUs like the T4 for the smaller version of the model.

## Downloading Model Weights

Model weights and artifacts are available via AWS S3. You can download them using the provided `download_artifacts.py` script:

```bash
# Download a specific model
python download_artifacts.py tf-sapiens
python download_artifacts.py tf-exemplar
python download_artifacts.py tf-metazoa

# Download all models and embeddings
python download_artifacts.py all

# Download only the embedding files
python download_artifacts.py all-embeddings

# Specify a custom checkpoint directory
python download_artifacts.py tf-sapiens --checkpoint-dir /path/to/custom/dir
```

The script will download and extract the following files to the `./checkpoints` directory (or your specified directory):
- `./checkpoints/tf_sapiens/`: Sapiens model weights
- `./checkpoints/tf_exemplar/`: Exemplar model weights
- `./checkpoints/tf_metazoa/`: Metazoa model weights
- `./checkpoints/all_embeddings/`: Embedding files for out-of-distribution species

The script includes progress bars for both download and extraction processes.

## Running Inference

The `inference.py` script provides a convenient interface for running inference with TranscriptFormer. The script uses Hydra for configuration management, allowing flexible parameter specification.

Basic usage:

```bash
python inference.py --config-name=inference_config.yaml model.checkpoint_path=./checkpoints/tf_sapiens
```

#### Key Parameters:

- `model.checkpoint_path`: Path to the checkpoint directory containing model weights and vocabulary files
- `model.inference_config.data_files`: Path(s) to input data files (H5AD format)
- `model.inference_config.pretrained_embedding`: Path(s) to pretrained embeddings (out-of-distribution species)
- `model.inference_config.output_path`: Directory to save inference results
- `model.inference_config.batch_size`: Batch size for inference (default: 32)
- `model.inference_config.precision`: Numerical precision (default: "16-mixed")

#### Example:
For in-distribution species (e.g. human with TF-Sapiens):
```bash
# Inference on in-distribution species
python inference.py --config-name=inference_config.yaml \
  model.checkpoint_path=./checkpoints/tf_sapiens \
  model.inference_config.data_files.0=test/data/human_val.h5ad \
  model.inference_config.batch_size=8
```

For out-of-distribution species (e.g. mouse with TF-Sapiens) supply the embedding file:

```bash
# Inference on out-of-distribution species
python inference.py --config-name=inference_config.yaml \
  model.checkpoint_path=./checkpoints/tf_sapiens \
  model.inference_config.data_files.0=test/data/mouse_val.h5ad \
  model.inference_config.pretrained_embedding=./checkpoints/all_embeddings/mus_musculus_gene.h5
  model.inference_config.batch_size=8
```

To specify multiple input files, use indexed notation:

```bash
python inference.py --config-name=inference_config.yaml \
  model.checkpoint_path=./checkpoints/tf_sapiens \
  model.inference_config.data_files.0=test/data/human_val.h5ad \
  model.inference_config.data_files.1=test/data/mouse_val.h5ad
```

Or use the list notation:

```bash
python inference.py --config-name=inference_config.yaml \
  model.checkpoint_path=../checkpoints/tf_sapiens \
  "model.inference_config.data_files=[test/data/human_val.h5ad,test/data/mouse_val.h5ad]"
```

#### Input Data Format:

Input data files should be in H5AD format (AnnData objects) with the following requirements:

- **Gene IDs**: The `var` dataframe must contain an `ensembl_id` column with Ensembl gene identifiers
  - Out-of-vocabulary gene IDs will be automatically filtered out during processing
  - Only genes present in the model's vocabulary will be used for inference

- **Expression Data**: Raw count data should be stored in the `adata.X` matrix
  - The model expects raw (non-normalized) counts
  - Log-transformed or normalized data may lead to unexpected results

- **Cell Metadata**: Any cell metadata in the `obs` dataframe will be preserved in the output

#### Output Format:

The inference results will be saved to the specified output directory (default: `./inference_results`) in a file named `embeddings.h5ad`. This is an AnnData object where:

- Cell embeddings are stored in `obsm['embeddings']`
- Original cell metadata is preserved in the `obs` dataframe
- Log-likelihood scores (if available) are stored in `uns['llh']`

#### Output:

The inference results will be saved to the specified output directory. The script will generate:

1. Gene embeddings for each cell
2. Log-likelihood scores
3. Metadata from the original dataset

Results are saved in HDF5 format with the same structure as the input data, with additional embedding matrices and likelihood scores.

For detailed configuration options, see the `conf/inference_config.yaml` file.

## Contributing
This project adheres to the Contributor Covenant code of conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to opensource@chanzuckerberg.com.

## Reporting Security Issues
Please note: If you believe you have found a security issue, please responsibly disclose by contacting us at security@chanzuckerberg.com.

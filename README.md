# TranscriptFormer

<p align="center">
  <img src="assets/model_overview.png" width="600" alt="TranscriptFormer Overview">
  <br>
  <em>Overview of TranscriptFormer pretraining data (A), model (B), outputs (C) and downstream tasks (D).
</em>
</p>

**Authors:** James D Pearce, Sara E Simmonds*, Gita Mahmoudabadi*, Lakshmi Krishnan*, Giovanni
Palla, Ana-Maria Istrate, Alexander Tarashansky, Benjamin Nelson, Omar Valenzuela,
Donghui Li, Stephen R Quake, Theofanis Karaletsos (Chan Zuckerberg Initiative)

*Equal contribution

## Description

TranscriptFormer is a family of generative foundation models representing a cross-species generative cell atlas trained on up to 112 million cells spanning 1.53 billion years of evolution across 12 species. The models include three distinct versions:

- **TF-Metazoa**: Trained on 112 million cells spanning all twelve species. The set covers six vertebrates (human, mouse, rabbit, chicken, African clawed frog, zebrafish), four invertebrates (sea urchin, C. elegans, fruit fly, freshwater sponge), plus a fungus (yeast) and a protist (malaria parasite).
The model includes 444 million trainable parameters and 633 million non-trainable
parameters (from frozen pretrained embeddings). Vocabulary size: 247,388.

- **TF-Exemplar**: Trained on 110 million cells from human and four model organisms: mouse (M. musculus), zebrafish (D. rerio), fruit fly (D. melanogaster ), and C. ele-
gans. Total trainable parameters: 542 million; non-trainable: 282 million. Vocabulary size:
110,290.

- **TF-Sapiens**: Trained on 57 million human-only cells. This model has 368 million trainable parameters and 61 million non-trainable parameters. Vocabulary size: 23,829.


TranscriptFormer is designed to learn rich, context-aware representations of single-cell transcriptomes while jointly modeling genes and transcripts using a novel generative architecture. It employs a generative autoregressive joint model over genes and their expression levels per cell across species, with a transformer-based architecture, including a novel coupling between gene and transcript heads, expression-aware multi-head self-attention, causal masking, and a count likelihood to capture transcript-level variability. TranscriptFormer demonstrates robust zero-shot performance for cell type classification across species, disease state identification in human cells, and prediction of cell type specific transcription factors and gene-gene regulatory relationships. This work establishes a powerful framework for integrating and interrogating cellular diversity across species as well as offering a foundation for in-silico experimentation with a generative single-cell atlas model.

For more details, please refer to our manuscript: [A Cross-Species Generative Cell Atlas Across 1.5 Billion Years of Evolution: The TranscriptFormer Single-cell Model](https://www.biorxiv.org/content/10.1101/2025.04.25.650731v1)


## Installation

Transcriptformer requires Python >=3.11.

### Install from PyPI

```bash
# Create and activate a virtual environment
uv venv --python=3.11
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install from PyPI
uv pip install transcriptformer
```

### Install from source

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
- GPU (A100 40GB recommended) for efficient inference and embedding extraction.
- Can also use a GPU with a lower amount of VRAM (16GB) by setting the inference batch size to 1-4.


## Using the TranscriptFormer CLI

After installing the package, you'll have access to the `transcriptformer` command-line interface (CLI), which provides easy access to download model artifacts, download training datasets, and run inference.

### Downloading Model Weights

Use the CLI to download model weights and artifacts from AWS S3:

```bash
# Download a specific model
transcriptformer download tf-sapiens
transcriptformer download tf-exemplar
transcriptformer download tf-metazoa

# Download all models and embeddings
transcriptformer download all

# Download only the embedding files
transcriptformer download all-embeddings

# Specify a custom checkpoint directory
transcriptformer download tf-sapiens --checkpoint-dir /path/to/custom/dir
```

The command will download and extract the following files to the `./checkpoints` directory (or your specified directory):
- `./checkpoints/tf_sapiens/`: Sapiens model weights
- `./checkpoints/tf_exemplar/`: Exemplar model weights
- `./checkpoints/tf_metazoa/`: Metazoa model weights
- `./checkpoints/all_embeddings/`: Embedding files for out-of-distribution species

#### Available Protein Embeddings

The following protein embeddings are available for download with `transcriptformer download all-embeddings`:

| Scientific Name | Common Name | TF-Metazoa | TF-Exemplar | TF-Sapiens | Notes |
|-----------------|-------------|------------|-------------|------------|-------|
| *Homo sapiens* | Human | ✓ | ✓ | ✓ | Primary training species |
| *Mus musculus* | Mouse | ✓ | ✓ | - | Model organism |
| *Danio rerio* | Zebrafish | ✓ | ✓ | - | Model organism |
| *Drosophila melanogaster* | Fruit fly | ✓ | ✓ | - | Model organism |
| *Caenorhabditis elegans* | C. elegans | ✓ | ✓ | - | Model organism |
| *Oryctolagus cuniculus* | Rabbit | ✓ | - | - | Vertebrate |
| *Gallus gallus* | Chicken | ✓ | - | - | Vertebrate |
| *Xenopus laevis* | African clawed frog | ✓ | - | - | Vertebrate |
| *Lytechinus variegatus* | Sea urchin | ✓ | - | - | Invertebrate |
| *Spongilla lacustris* | Freshwater sponge | ✓ | - | - | Invertebrate |
| *Saccharomyces cerevisiae* | Yeast | ✓ | - | - | Fungus |
| *Plasmodium falciparum* | Malaria parasite | ✓ | - | - | Protist |
| *Rattus norvegicus* | Rat | - | - | - | Out-of-distribution |
| *Sus scrofa* | Pig | - | - | - | Out-of-distribution |
| *Pan troglodytes* | Chimpanzee | - | - | - | Out-of-distribution |
| *Gorilla gorilla* | Gorilla | - | - | - | Out-of-distribution |
| *Macaca mulatta* | Rhesus macaque | - | - | - | Out-of-distribution |
| *Callithrix jacchus* | Marmoset | - | - | - | Out-of-distribution |
| *Xenopus tropicalis* | Western clawed frog | - | - | - | Out-of-distribution |
| *Ornithorhynchus anatinus* | Platypus | - | - | - | Out-of-distribution |
| *Monodelphis domestica* | Opossum | - | - | - | Out-of-distribution |
| *Heterocephalus glaber* | Naked mole-rat | - | - | - | Out-of-distribution |
| *Petromyzon marinus* | Sea lamprey | - | - | - | Out-of-distribution |
| *Stylophora pistillata* | Coral | - | - | - | Out-of-distribution |

**Legend:**
- ✓ = Species included in model training data
- \- = Species not included in model training (out-of-distribution)

### Generating Protein Embeddings for New Species

The pre-generated embeddings cover the most commonly used species. If you need to work with a species not included in the downloaded embeddings, you can generate protein embeddings using the ESM-2 models.

**Note**: This is only necessary for new species that don't have pre-generated embeddings available for download.

For detailed instructions on generating protein embeddings for additional species, see the [protein_embeddings/README.md](protein_embeddings/README.md) documentation.

### Downloading Training Datasets

Use the CLI to download single-cell RNA sequencing datasets from the CellxGene Discover portal:

```bash
# Download human datasets
transcriptformer download-data --species "homo sapiens" --output-dir ./data/human

# Download multiple species datasets
transcriptformer download-data --species "homo sapiens,mus musculus" --output-dir ./data/multi_species

# Download with custom settings
transcriptformer download-data \
  --species "homo sapiens" \
  --output-dir ./data/human \
  --processes 8 \
  --max-retries 3 \
  --no-metadata
```

The `download-data` command provides the following options:

- `--species`: Comma-separated list of species to download (required). Common species names include:
  - "homo sapiens" (human)
  - "mus musculus" (mouse)
  - "danio rerio" (zebrafish)
  - "drosophila melanogaster" (fruit fly)
  - "caenorhabditis elegans" (C. elegans)
- `--output-dir`: Directory where datasets will be saved (default: `./data/cellxgene`)
- `--processes`: Number of parallel download processes (default: 4)
- `--max-retries`: Maximum retry attempts per dataset (default: 5)
- `--no-metadata`: Skip saving dataset metadata to JSON file

**Note:** You can also use the module directly for programmatic access:
```python
# Direct module usage
python -m transcriptformer.data.bulk_download --species "homo sapiens" --output-dir ./data/human
```

**Downloaded Data Structure:**
```
output_dir/
├── dataset_metadata.json          # Metadata for all downloaded datasets
├── dataset_id_1/
│   ├── full.h5ad                  # Raw dataset in AnnData format
│   └── __success__                # Download completion marker
├── dataset_id_2/
│   ├── full.h5ad
│   └── __success__
└── ...
```

Each dataset is downloaded as an AnnData object in H5AD format, containing raw count data suitable for use with TranscriptFormer models. The metadata JSON file contains detailed information about each dataset including cell counts, tissue types, and experimental conditions.

### Running Inference

Use the CLI to run inference with TranscriptFormer models:

```bash
# Basic inference on in-distribution species (e.g., human with TF-Sapiens)
transcriptformer inference \
  --checkpoint-path ./checkpoints/tf_sapiens \
  --data-file test/data/human_val.h5ad \
  --output-path ./inference_results \
  --batch-size 8

# Inference on out-of-distribution species (e.g., mouse with TF-Sapiens)
transcriptformer inference \
  --checkpoint-path ./checkpoints/tf_sapiens \
  --data-file test/data/mouse_val.h5ad \
  --pretrained-embedding ./checkpoints/all_embeddings/mus_musculus_gene.h5 \
  --batch-size 8

# Extract contextual gene embeddings instead of cell embeddings
transcriptformer inference \
  --checkpoint-path ./checkpoints/tf_sapiens \
  --data-file test/data/human_val.h5ad \
  --emb-type cge \
  --batch-size 8
```

You can also use the CLI it run inference on the ESM2-CE baseline model discussed in the paper:

transcriptformer inference \
  --checkpoint-path ./checkpoints/tf_sapiens \
  --data-file test/data/human_val.h5ad \
  --model-type esm2ce \
  --batch-size 8
```

### Advanced Configuration

For advanced configuration options not exposed as CLI arguments, use the `--config-override` parameter:

```bash
transcriptformer inference \
  --checkpoint-path ./checkpoints/tf_sapiens \
  --data-file test/data/human_val.h5ad \
  --config-override model.data_config.normalize_to_scale=10000 \
  --config-override model.inference_config.obs_keys.0=cell_type
```

To see all available CLI options:

```bash
transcriptformer inference --help
transcriptformer download --help
transcriptformer download-data --help
```

### CLI Options for `inference`:

- `--checkpoint-path PATH`: Path to the model checkpoint directory (required).
- `--data-file PATH`: Path to input AnnData file (required).
- `--output-path DIR`: Directory for saving results (default: `./inference_results`).
- `--output-filename NAME`: Filename for the output embeddings (default: `embeddings.h5ad`).
- `--batch-size INT`: Number of samples to process in each batch (default: 8).
- `--gene-col-name NAME`: Column name in AnnData.var containing gene identifiers (default: `ensembl_id`).
- `--precision {16-mixed,32}`: Numerical precision for inference (default: `16-mixed`).
- `--pretrained-embedding PATH`: Path to pretrained embeddings for out-of-distribution species.
- `--clip-counts INT`: Maximum count value (higher values will be clipped) (default: 30).
- `--filter-to-vocabs`: Whether to filter genes to only those in the vocabulary (default: True).
- `--use-raw {True,False,auto}`: Whether to use raw counts from `AnnData.raw.X` (True), `adata.X` (False), or auto-detect (auto/None) (default: None).
- `--embedding-layer-index INT`: Index of the transformer layer to extract embeddings from (-1 for last layer, default: -1). Use with `transcriptformer` model type.
- `--model-type {transcriptformer,esm2ce}`: Type of model to use (default: `transcriptformer`). Use `esm2ce` to extract raw ESM2-CE gene embeddings.
- `--emb-type {cell,cge}`: Type of embeddings to extract (default: `cell`). Use `cell` for mean-pooled cell embeddings or `cge` for contextual gene embeddings.
- `--config-override key.path=value`: Override any configuration value directly.

### Input Data Format and Preprocessing:

Input data files should be in H5AD format (AnnData objects) with the following requirements:

- **Gene IDs**: The `var` dataframe must contain an `ensembl_id` column with Ensembl gene identifiers
  - Out-of-vocabulary gene IDs will be automatically filtered out during processing
  - Only genes present in the model's vocabulary will be used for inference
  - The column name can be changed using `model.data_config.gene_col_name`

- **Expression Data**: The model expects unnormalized count data and will look for it in the following order:
  1. `adata.raw.X` (if available)
  2. `adata.X`

  This behavior can be controlled using `model.data_config.use_raw`:
  - `None` (default): Try `adata.raw.X` first, then fall back to `adata.X`
  - `True`: Use only `adata.raw.X`
  - `False`: Use only `adata.X`

- **Count Processing**:
  - Count values are clipped at 30 by default (as was done in training)
  - If this seems too low, you can either:
    1. Use `model.data_config.normalize_to_scale` to scale total counts to a specific value (e.g., 1e3-1e4)
    2. Increase `model.data_config.clip_counts` to a value > 30

- **Cell Metadata**: Any cell metadata in the `obs` dataframe will be preserved in the output

No other data preprocessing is necessary - the model handles all required transformations internally. You do not need to perform any additional normalization, scaling, or transformation of the count data before input.

### Output Format:

The inference results will be saved to the specified output directory (default: `./inference_results`) in a file named `embeddings.h5ad`. This is an AnnData object where:

**For cell embeddings (`--emb-type cell`, default):**
- Cell embeddings are stored in `obsm['embeddings']`
- Original cell metadata is preserved in the `obs` dataframe
- Log-likelihood scores (if available) are stored in `uns['llh']`

**For contextual gene embeddings (`--emb-type cge`):**
- Contextual gene embeddings are stored in `uns['cge_embeddings']` as a 2D array (n_gene_instances, embedding_dim)
- Cell indices for each gene embedding are stored in `uns['cge_cell_indices']`
- Gene names for each embedding are stored in `uns['cge_gene_names']`
- Original cell metadata is preserved in the `obs` dataframe
- Log-likelihood scores (if available) are stored in `uns['llh']`

#### Contextual Gene Embeddings (CGE)

Contextual gene embeddings provide gene-specific representations that capture how each gene is contextualized within the cell sentence. Unlike cell embeddings which are mean-pooled across all genes, CGEs represent the individual embedding for each gene as computed by the transformer.

Example usage:
```bash
# Extract contextual gene embeddings
transcriptformer inference \
  --checkpoint-path ./checkpoints/tf_sapiens \
  --data-file test/data/human_val.h5ad \
  --emb-type cge \
  --output-filename cge_embeddings.h5ad
```

To access CGE data in Python:
```python
import anndata as ad
import numpy as np

# Load the results
adata = ad.read_h5ad("./inference_results/cge_embeddings.h5ad")

# Access all contextual gene embeddings
cge_embeddings = adata.uns['cge_embeddings']  # Shape: (n_gene_instances, embedding_dim)
cell_indices = adata.uns['cge_cell_indices']   # Which cell each embedding belongs to
gene_names = adata.uns['cge_gene_names']       # Gene name for each embedding

# Get all gene embeddings for the first cell (cell index 0)
cell_0_mask = cell_indices == 0
cell_0_embeddings = cge_embeddings[cell_0_mask]
cell_0_genes = gene_names[cell_0_mask]

# Get embedding for a specific gene in the first cell
gene_mask = (cell_indices == 0) & (gene_names == 'ENSG00000000003')
if np.any(gene_mask):
    gene_embedding = cge_embeddings[gene_mask][0]  # Returns numpy array
else:
    gene_embedding = None  # Gene not found in this cell
```

For detailed configuration options, see the `src/transcriptformer/cli/conf/inference_config.yaml` file.

## Contributing
This project adheres to the Contributor Covenant code of conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to opensource@chanzuckerberg.com.

## Reporting Security Issues
Please note: If you believe you have found a security issue, please responsibly disclose by contacting us at security@chanzuckerberg.com.

## Citation

If you use TranscriptFormer in your research, please cite:
Pearce, J. D., et. al. (2025). A Cross-Species Generative Cell Atlas Across 1.5 Billion Years of Evolution: The TranscriptFormer Single-cell Model. bioRxiv. Retrieved April 29, 2025, from https://www.biorxiv.org/content/10.1101/2025.04.25.650731v1

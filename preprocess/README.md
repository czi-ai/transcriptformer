# Protein Embeddings Generation

This directory contains scripts for generating protein embeddings using Facebook's ESM-2 (Evolutionary Scale Modeling) models. The pipeline downloads protein sequences from Ensembl, processes them with pre-trained ESM-2 models, and outputs gene-level embeddings suitable for inputs to TranscriptFormer.

## Overview

The protein embedding pipeline consists of three main components:

1. **`protein_embedding.py`** - Main script for generating protein embeddings using ESM-2 models
2. **`get_stable_id_mapping.py`** - Utility functions for mapping between gene, transcript, and protein stable IDs
3. **`fasta_manifest_pep.json`** - Configuration file containing download URLs for protein FASTA files from Ensembl

## Installation

### Using pip (traditional)

Install the required dependencies:

```bash
pip install -r requirements.txt
pip install fair-esm
```

For GPU acceleration (recommended):
```bash
# For CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Using uv (recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver:

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate virtual environment
uv venv protein-embeddings
source protein-embeddings/bin/activate  # On Windows: protein-embeddings\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
uv pip install fair-esm

# For GPU acceleration
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### System Requirements

- **Memory**: At least 16GB RAM (32GB+ recommended for large models)
- **GPU**: NVIDIA GPU with 8GB+ VRAM (optional but highly recommended)
- **Storage**: Several GB for downloaded FASTA files and generated embeddings
- **Network**: Internet connection for downloading protein sequences from Ensembl

## Usage

### Basic Usage

Generate protein embeddings for a single species:

```bash
python protein_embedding.py --organism_key homo_sapiens
```

### Advanced Usage

```bash
python protein_embedding.py \
    --organism_key mus_musculus \
    --output_dir /path/to/output \
    --batch_size 32 \
    --use_large_model true
```

### Command Line Arguments

- `--organism_key`: Species to process (see [Supported Species](#supported-species))
- `--output_dir`: Directory to save embeddings (default: current directory `./`)
- `--batch_size`: Batch size for processing (default: 16)
- `--use_large_model`: Use ESM-2 15B parameter model instead of 3B (default: false)


## Supported Species

The pipeline supports the following species (from Ensembl release 110/113):

| Species | Organism Key | Common Name |
|---------|-------------|-------------|
| Homo sapiens | `homo_sapiens` | Human |
| Mus musculus | `mus_musculus` | Mouse |
| Rattus norvegicus | `rattus_norvegicus` | Rat |
| Sus scrofa | `sus_scrofa` | Pig |
| Oryctolagus cuniculus | `oryctolagus_cuniculus` | Rabbit |
| Macaca mulatta | `macaca_mulatta` | Rhesus macaque |
| Pan troglodytes | `pan_troglodytes` | Chimpanzee |
| Gorilla gorilla | `gorilla_gorilla` | Gorilla |
| Callithrix jacchus | `callithrix_jacchus` | Marmoset |
| Microcebus murinus | `microcebus_murinus` | Mouse lemur |
| Gallus gallus | `gallus_gallus` | Chicken |
| Danio rerio | `danio_rerio` | Zebrafish |
| Xenopus tropicalis | `xenopus_tropicalis` | Frog |
| Drosophila melanogaster | `drosophila_melanogaster` | Fruit fly |
| Petromyzon marinus | `petromyzon_marinus` | Sea lamprey |
| Ornithorhynchus anatinus | `ornithorhynchus_anatinus` | Platypus |
| Monodelphis domestica | `monodelphis_domestica` | Opossum |
| Heterocephalus glaber | `heterocephalus_glaber` | Naked mole-rat |
| Stylophora pistillata | `stylophora_pistillata` | Coral |

## Adding New Species

To add support for a new species, you need to update the `fasta_manifest_pep.json` file with the appropriate Ensembl download URLs.

### Step 1: Find Ensembl URLs

1. Visit the [Ensembl FTP site](https://ftp.ensembl.org/pub/) or [Ensembl Genomes](https://ftp.ebi.ac.uk/ensemblgenomes/pub/) for non-vertebrates
2. Navigate to the latest release (e.g., `release-113/`)
3. Find your species under `fasta/{species_name}/pep/`
4. Copy the URL for the `.pep.all.fa.gz` file

### Step 2: Update the Manifest

Add an entry to `fasta_manifest_pep.json`:

```json
{
  "new_species_name": {
    "fa": "https://ftp.ensembl.org/pub/release-113/fasta/new_species/pep/New_species.Assembly.pep.all.fa.gz"
  }
}
```

### Step 3: Generate Embeddings

```bash
python protein_embedding.py --organism_key new_species_name
```

### Example: Adding Sheep (Ovis aries)

```json
{
  "ovis_aries": {
    "fa": "https://ftp.ensembl.org/pub/release-113/fasta/ovis_aries/pep/Ovis_aries_rambouillet.ARS-UI_Ramb_v2.0.pep.all.fa.gz"
  }
}
```

### Notes

- Use lowercase with underscores for organism keys (e.g., `ovis_aries`)
- Ensure the FASTA file contains protein sequences (`.pep.` not `.cdna.` or `.dna.`)
- Some species may be in Ensembl Genomes rather than main Ensembl
- Check that the assembly and release versions are current

## Output Format

The script generates embeddings in HDF5 format with the following structure:

```python
import h5py

# Load embeddings
with h5py.File('homo_sapiens_gene.h5', 'r') as f:
    keys = f['keys'][:]  # Gene IDs
    embeddings = f['arrays']  # Group containing embedding arrays

    # Access specific gene embedding
    gene_id = 'ENSG00000139618'  # Example: BRCA2
    embedding = embeddings[gene_id][:]  # Shape: (2560,) for ESM-2 3B model
```

### Output Files

- **Standard model**: `{organism}_gene.h5` (d=2560, TranscriptFormer default)
- **Large model**: `{organism}_gene_large.h5` (d=5120, UCE default)

## Pipeline Details

### Processing Steps

1. **Download**: Automatically downloads protein FASTA files from Ensembl FTP
2. **Parse**: Extracts gene IDs from protein sequence headers
3. **Deduplicate**: Removes duplicate sequences for the same gene
4. **Clean**: Replaces invalid amino acids (*) with `<unk>` tokens
5. **Embed**: Generates embeddings using ESM-2 model (layer 33 for 3B, layer 48 for 15B)
6. **Average**: Averages embeddings across all protein isoforms per gene
7. **Save**: Stores final gene-level embeddings in HDF5 format

### Models Used

- **ESM-2 3B** (`esm2_t36_3B_UR50D`): 36-layer, 3 billion parameter model
- **ESM-2 15B** (`esm2_t48_15B_UR50D`): 48-layer, 15 billion parameter model

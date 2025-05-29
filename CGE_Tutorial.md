# Contextual Gene Embeddings (CGE) Tutorial

This tutorial demonstrates how to generate and work with Contextual Gene Embeddings (CGE) using TranscriptFormer. CGE provides gene-specific representations that capture how each gene is contextualized within the full transcriptome of a cell, offering a more granular view compared to traditional mean-pooled cell embeddings.

## Prerequisites

Before starting this tutorial, you need to:

### 1. Install TranscriptFormer
Create a virtual environment with Python 3.11+ and install TranscriptFormer from PyPI using `uv pip install transcriptformer`.

### 2. Download Model Weights and Artifacts
TranscriptFormer requires pre-trained model weights and vocabulary files. Use the CLI command `transcriptformer download tf-sapiens` to download the human-only model, or `transcriptformer download all` for all models and embeddings. Files will be downloaded to `./checkpoints/` by default.

### 3. Prepare Your Data
Your input data should be in H5AD format (AnnData objects) with:
- **Gene IDs**: `var` dataframe must contain an `ensembl_id` column
- **Expression Data**: Unnormalized count data in `adata.X` or `adata.raw.X`
- **Cell Metadata**: Any metadata in `obs` will be preserved

## What are Contextual Gene Embeddings?

Traditional cell embeddings are created by mean-pooling gene representations across all genes in a cell. While useful for cell-level analysis, this approach loses gene-specific information and the contextual relationships between genes.

**Contextual Gene Embeddings (CGE)** preserve the individual embedding for each gene as computed by the transformer, capturing:

- **Gene-specific context**: How each gene's expression is influenced by other genes in the cell
- **Positional relationships**: The transformer's understanding of gene-gene interactions
- **Cell-type specific patterns**: How the same gene behaves differently across cell types
- **Regulatory networks**: Implicit gene regulatory relationships learned by the model

### Key Differences:

| Feature | Cell Embeddings | Contextual Gene Embeddings |
|---------|----------------|---------------------------|
| **Granularity** | One embedding per cell | One embedding per gene per cell |
| **Information** | Mean-pooled across genes | Gene-specific contextual information |
| **Use Cases** | Cell clustering, classification | Gene analysis, regulatory networks |
| **Output Size** | (n_cells, embedding_dim) | (n_gene_instances, embedding_dim) |

## Output Format

The CGE output is stored in a flattened format optimized for HDF5 compatibility with the following components:

1. **`adata.obs`**: Original cell metadata (preserved from input)
2. **`adata.uns['cge_embeddings']`**: 2D array of shape `(n_gene_instances, embedding_dim)`
3. **`adata.uns['cge_cell_indices']`**: Array indicating which cell each embedding belongs to
4. **`adata.uns['cge_gene_names']`**: Array of gene names corresponding to each embedding

## Applications

CGE enables several novel analyses:

- **Gene Regulatory Network Analysis**: Identify genes with similar contextual patterns
- **Cell Type-Specific Gene Analysis**: Compare how genes behave differently across cell types
- **Pathway Analysis**: Understand pathway-level gene interactions in embedding space
- **Co-expression Analysis**: Find genes that cluster together in specific cellular contexts

This tutorial will walk through generating CGE embeddings and demonstrate these analysis approaches using real single-cell data. 
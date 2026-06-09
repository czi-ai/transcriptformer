import logging
import os
import random
from collections import Counter

import anndata
import numpy as np
import scanpy as sc
import torch
from scipy.sparse import csc_matrix, csr_matrix
from torch import tensor
from torch.utils.data import BatchSampler, Dataset

from transcriptformer.data.dataclasses import BatchData
from transcriptformer.tokenizer.tokenizer import (
    BatchGeneTokenizer,
    BatchObsTokenizer,
)


def load_data(file_path, *, backed: bool = False):
    """Load H5AD file.

    Args:
        file_path: Path to .h5ad file
        backed: If True, use memory-mapped backed='r' mode (for streaming); otherwise fully load into memory
    """
    try:
        if backed:
            adata = anndata.read_h5ad(file_path, backed="r")
        else:
            adata = sc.read_h5ad(file_path)
        return adata, True
    except Exception as e:
        logging.error(f"Failed to read file {file_path}: {e}")
        return None, False


def apply_filters(
    X,
    obs,
    gene_names,
    file_path,
    filter_to_vocab,
    vocab,  # gene  vocab
    filter_outliers,
    min_expressed_genes,
):
    """Apply filters to the data."""
    n_cells = X.shape[0]

    if filter_to_vocab:
        filter_idx = [i for i, name in enumerate(gene_names) if name in vocab]
        X = X[:, filter_idx]
        logging.info(f"Filtered {len(gene_names)} genes to {len(filter_idx)} genes in vocab")
        gene_names = gene_names[filter_idx]
        if X.shape[1] == 0:
            logging.warning(f"Warning: Filtered all genes from {file_path}")
            logging.warning(f"Available genes: {len(gene_names)}")
            logging.warning(f"Number of non-zero genes: {np.sum(X > 0, axis=1).mean()}")
            return None, None, None

    if filter_outliers > 0:
        expr_counts = X.sum(axis=1)
        count_std = np.std(expr_counts)
        count_mean = np.mean(expr_counts)
        filter_idx = (expr_counts > count_mean - count_std * filter_outliers) & (
            expr_counts < count_mean + count_std * filter_outliers
        )
        X = X[filter_idx]
        obs = obs.iloc[filter_idx]

    if min_expressed_genes > 0:
        filter_idx = (X > 0).sum(axis=1) >= min_expressed_genes
        X = X[filter_idx]
        obs = obs.iloc[filter_idx]

    logging.info(f"Filtered {n_cells} cells to {X.shape[0]} cells")

    return X, obs, gene_names


def process_batch(
    x_batch,
    obs_batch,
    gene_names,
    gene_tokenizer,
    aux_tokenizer,
    sort_genes,
    randomize_order,
    max_len,
    pad_zeros,
    pad_token,
    gene_vocab,
    normalize_to_scale,
    clip_counts,
    aux_vocab,
):
    """Process a batch of data, including sorting, tokenization, and normalization."""
    x_batch = tensor(x_batch, dtype=torch.float32)

    # Sort genes or randomize order
    if sort_genes:
        ids_batch = torch.argsort(x_batch, dim=1, descending=True)
    else:
        ids_batch = torch.zeros_like(x_batch, dtype=torch.long)
        for i, sample in enumerate(x_batch):
            non_zero_indices = torch.nonzero(sample, as_tuple=True)[0]
            zero_indices = torch.nonzero(sample == 0, as_tuple=True)[0]
            if randomize_order:
                non_zero_indices = non_zero_indices[torch.randperm(len(non_zero_indices))]
                zero_indices = zero_indices[torch.randperm(len(zero_indices))]
            sample_ids = torch.cat([non_zero_indices, zero_indices])
            ids_batch[i] = sample_ids

    # Limit to max_len and gather counts
    if ids_batch.shape[1] > max_len:
        ids_batch = ids_batch[:, :max_len]

    counts_batch = torch.gather(x_batch, 1, ids_batch)

    # Tokenize gene names
    gene_names_batch = gene_names[ids_batch.numpy()]
    gene_tokens_batch = gene_tokenizer(gene_names_batch)

    # Apply padding and normalization
    if pad_zeros:
        gene_tokens_batch = gene_tokens_batch.masked_fill(counts_batch == 0, gene_vocab[pad_token])

    # Pad ids_batch to max_len
    tok_bz, tok_sq = gene_tokens_batch.shape
    if tok_sq < max_len:
        padding = torch.full(
            (tok_bz, max_len - tok_sq),
            gene_vocab[pad_token],
            dtype=gene_tokens_batch.dtype,
        )
        gene_tokens_batch = torch.cat([gene_tokens_batch, padding], dim=1)
        gene_names_batch = np.hstack(
            [
                gene_names_batch,
                np.full((tok_bz, max_len - tok_sq), pad_token),
            ]
        )

        counts_batch = torch.cat([counts_batch, torch.zeros_like(padding, dtype=counts_batch.dtype)], dim=1)

    # Normalize to scale if specified
    if normalize_to_scale is not None and normalize_to_scale > 0:
        row_sums = counts_batch.sum(dim=1, keepdim=True)
        counts_batch = counts_batch / row_sums * normalize_to_scale

    # Clip counts if specified
    if clip_counts is not None:
        counts_batch = counts_batch.clamp(min=0, max=clip_counts)

    # Prepare result dictionary
    result = {
        "gene_counts": counts_batch,
        "gene_token_indices": gene_tokens_batch,
    }

    # Add auxiliary and tokens if specified
    if aux_vocab is not None:
        aux_tokens_batch = torch.stack([aux_tokenizer(obs) for _, obs in obs_batch.iterrows()])
        result["aux_token_indices"] = aux_tokens_batch

    return result


def get_counts_layer(adata: anndata.AnnData, use_raw: bool | None):
    if use_raw is True:
        if adata.raw is not None:
            logging.info("Using 'raw.X' layer from AnnData object")
            return adata.raw.X
        else:
            raise ValueError("raw.X not found in AnnData object")
    elif use_raw is False:
        if adata.X is not None:
            logging.info("Using 'X' layer from AnnData object")
            return adata.X
        else:
            raise ValueError("X not found in AnnData object")
    else:  # None - try raw first, then fallback to X
        if adata.raw is not None:
            logging.info("Using 'raw.X' layer from AnnData object")
            return adata.raw.X
        elif adata.X is not None:
            logging.info("Using 'X' layer from AnnData object")
            return adata.X
        else:
            raise ValueError("No valid data layer found in AnnData object")


def to_dense(X: np.ndarray | csr_matrix | csc_matrix) -> np.ndarray:
    if isinstance(X, csr_matrix | csc_matrix):
        return X.toarray()
    elif isinstance(X, np.ndarray):
        return X
    else:
        raise TypeError(f"Expected numpy array or sparse matrix, got {type(X)}")


def is_raw_counts(X: np.ndarray | csr_matrix | csc_matrix) -> bool:
    """Check if a matrix looks like raw counts (integer-valued where non-zero).

    Handles both dense numpy arrays and sparse CSR/CSC matrices without densifying the full matrix.
    """
    # Sparse path: operate on non-zero data directly
    if isinstance(X, csr_matrix | csc_matrix):
        data = X.data
        if data.size == 0:
            return False
        # Sample if very large
        if data.size > 1000:
            idx = np.random.choice(data.size, 1000, replace=False)
            data = data[idx]
        return np.all(np.abs(data - np.round(data)) < 1e-6)

    # Dense path
    non_zero_mask = X > 0
    if not np.any(non_zero_mask):
        return False
    non_zero_values = X[non_zero_mask]
    if non_zero_values.size > 1000:
        idx = np.random.choice(non_zero_values.size, 1000, replace=False)
        non_zero_values = non_zero_values.flatten()[idx]
    return np.all(np.abs(non_zero_values - np.round(non_zero_values)) < 1e-6)


def load_gene_features(
    adata: anndata.AnnData, gene_col_name: str, remove_duplicate_genes: bool, use_raw: bool | None = None
):
    try:
        # Select the appropriate var depending on which matrix will be used
        using_raw = bool(use_raw is True or (use_raw is None and getattr(adata, "raw", None) is not None))
        has_raw = getattr(adata, "raw", None) is not None
        using_raw = bool(use_raw is True or (use_raw is None and has_raw))
        var_df = adata.raw.var if using_raw and has_raw else adata.var

        # Prefer requested column; otherwise use index which aligns with matrix columns for that layer
        if gene_col_name in var_df.columns:
            gene_names = np.array(list(var_df[gene_col_name].values))
        else:
            raise ValueError(
                f"Gene column '{gene_col_name}' not found in var DataFrame columns: {list(var_df.columns)}"
            )

        # Remove version numbers from gene names
        gene_names = np.array([id.split(".")[0] for id in gene_names])

        gene_counts = Counter(gene_names)
        duplicates = {gene for gene, count in gene_counts.items() if count > 1}
        dedup_col_indices = None
        if len(duplicates) > 0:
            if remove_duplicate_genes:
                seen = set()
                unique_indices = []
                for i, gene in enumerate(gene_names):
                    if gene not in seen:
                        seen.add(gene)
                        unique_indices.append(i)
                gene_names = gene_names[unique_indices]
                logging.warning(
                    f"Removed {len(duplicates)} duplicate genes after removing version numbers. Kept first occurrence."
                )
                if adata.isbacked:
                    # Cannot copy a backed AnnData object; return indices so the caller
                    # can incorporate them into its column-index filter instead.
                    dedup_col_indices = unique_indices
                else:
                    adata = adata[:, unique_indices].copy()
            else:
                raise ValueError(
                    "Found duplicate genes after removing version numbers. "
                    "Remove duplicates or pass --remove-duplicate-genes."
                )

        return gene_names, True, adata, dedup_col_indices
    except KeyError:
        return None, False, adata, None


def validate_gene_dimension(X: np.ndarray, gene_names: np.ndarray, gene_col_name: str):
    if X.shape[1] != len(gene_names):
        raise ValueError(
            f"Mismatch between expression matrix columns ({X.shape[1]}) and gene names length ({len(gene_names)}). "
            f"Ensure 'adata.var[{gene_col_name}]' exists and aligns with the matrix columns."
        )


def compute_row_stats_chunked(
    X_layer,
    filter_idx: list[int] | None = None,
    chunk_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-row expression sums and nonzero counts in chunks.

    This avoids loading the full matrix into memory for backed AnnData layers.
    """
    n_rows = int(X_layer.shape[0])
    expr_counts = np.zeros(n_rows, dtype=np.float64)
    nnz_counts = np.zeros(n_rows, dtype=np.int32)

    for start in range(0, n_rows, chunk_size):
        end = min(start + chunk_size, n_rows)
        block = X_layer[start:end]

        if isinstance(block, csr_matrix | csc_matrix):
            if filter_idx is not None:
                block = block[:, filter_idx]
            expr = np.asarray(block.sum(axis=1)).ravel()
            nnz = np.asarray(block.getnnz(axis=1)).ravel()
        else:
            arr = np.asarray(block)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if filter_idx is not None:
                arr = arr[:, filter_idx]
            expr = arr.sum(axis=1)
            nnz = (arr > 0).sum(axis=1)

        expr_counts[start:end] = expr
        nnz_counts[start:end] = nnz

    return expr_counts, nnz_counts


class AnnDataset(Dataset):
    def __init__(
        self,
        files_list: list[str] | list[anndata.AnnData],
        gene_vocab: dict[str, str],
        data_dir: str = None,
        aux_vocab: dict[str, dict[str, str]] = None,
        max_len: int = 2048,
        normalize_to_scale: bool = None,
        sort_genes: bool = False,
        randomize_order: bool = False,
        pad_zeros: bool = True,
        gene_col_name: str = "ensembl_id",
        filter_to_vocab: bool = True,
        filter_outliers: float = 0.0,
        min_expressed_genes: int = 0,
        seed: int = 0,
        pad_token: str = "[PAD]",
        clip_counts: float = 1e10,
        inference: bool = False,
        obs_keys: list[str] = None,
        use_raw: bool = None,
        remove_duplicate_genes: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.files_list = files_list
        self.gene_vocab = gene_vocab
        self.aux_vocab = aux_vocab
        self.max_len = max_len
        self.normalize_to_scale = normalize_to_scale
        self.sort_genes = sort_genes
        self.randomize_order = randomize_order
        self.pad_zeros = pad_zeros
        self.gene_col_name = gene_col_name
        self.filter_to_vocab = filter_to_vocab
        self.filter_outliers = filter_outliers
        self.min_expressed_genes = min_expressed_genes
        self.seed = seed
        self.pad_token = pad_token
        self.clip_counts = clip_counts
        self.inference = inference
        self.obs_keys = obs_keys
        self.use_raw = use_raw
        self.remove_duplicate_genes = remove_duplicate_genes
        self.filter_metadata: list[dict] = []

        self.gene_tokenizer = BatchGeneTokenizer(gene_vocab)
        if aux_vocab is not None:
            self.aux_tokenizer = BatchObsTokenizer(aux_vocab)

        random.seed(self.seed)

        logging.info("Loading and processing all data")
        self.data = self.load_and_process_all_data()

    def _get_batch_from_file(self, file: str | anndata.AnnData) -> BatchData | None:
        if isinstance(file, str):
            file_path = file
            if self.data_dir is not None:
                file_path = os.path.join(self.data_dir, file_path)

            adata, success = load_data(file_path)
        elif isinstance(file, anndata.AnnData):
            adata = file
            success = True
            file_path = None
        else:
            raise ValueError(f"Invalid file type: {type(file)}")

        if not success:
            logging.error(f"Failed to load data from {file_path}")
            return None

        gene_names, success, adata, _dedup_col_indices = load_gene_features(
            adata, self.gene_col_name, self.remove_duplicate_genes, use_raw=self.use_raw
        )
        if not success:
            logging.error(f"Failed to load gene features from {file_path}")
            return None

        X = get_counts_layer(adata, self.use_raw)
        # AnnDataset loads and processes all data in-memory; convert to dense for batching
        X = to_dense(X)
        obs = adata.obs

        # Validate that gene dimension matches number of gene names
        validate_gene_dimension(X, gene_names, self.gene_col_name)

        # Check if the data appears to be raw counts
        logging.info("Checking if data is raw counts")
        if not is_raw_counts(X):
            logging.warning(
                "Data does not appear to be raw counts. TranscriptFormer expects unnormalized count data. "
                "If your data is normalized, consider using the original count matrix instead."
            )

        original_gene_count = int(len(gene_names))
        original_cell_count = int(X.shape[0])

        logging.info("Applying filters")
        vocab = self.gene_vocab
        X, obs, gene_names = apply_filters(
            X,
            obs,
            gene_names,
            file_path,
            self.filter_to_vocab,
            vocab,
            self.filter_outliers,
            self.min_expressed_genes,
        )

        if X is None:
            self.filter_metadata.append(
                {
                    "file": file_path,
                    "original_genes": original_gene_count,
                    "kept_genes": 0,
                    "removed_genes": original_gene_count,
                    "original_cells": original_cell_count,
                    "kept_cells": 0,
                    "removed_cells": original_cell_count,
                    "filter_to_vocab": bool(self.filter_to_vocab),
                    "filter_outliers": float(self.filter_outliers),
                    "min_expressed_genes": int(self.min_expressed_genes),
                }
            )
            logging.warning(f"Data was filtered out completely for {file_path}")
            return None

        self.filter_metadata.append(
            {
                "file": file_path,
                "original_genes": original_gene_count,
                "kept_genes": int(len(gene_names)),
                "removed_genes": int(original_gene_count - len(gene_names)),
                "original_cells": original_cell_count,
                "kept_cells": int(X.shape[0]),
                "removed_cells": int(original_cell_count - X.shape[0]),
                "filter_to_vocab": bool(self.filter_to_vocab),
                "filter_outliers": float(self.filter_outliers),
                "min_expressed_genes": int(self.min_expressed_genes),
            }
        )

        logging.info("Processing data")
        batch = process_batch(
            X,
            obs,
            gene_names,
            self.gene_tokenizer,
            getattr(self, "aux_tokenizer", None),
            self.sort_genes,
            self.randomize_order,
            self.max_len,
            self.pad_zeros,
            self.pad_token,
            self.gene_vocab,
            self.normalize_to_scale,
            self.clip_counts,
            self.aux_vocab,
        )
        batch["file_path"] = np.array([file_path] * X.shape[0])

        if self.obs_keys is not None:
            obs_data = {}
            if "all" in self.obs_keys:
                # Keep all columns from obs
                self.obs_keys = obs.columns
                for col in obs.columns:
                    obs_data[col] = np.array(obs[col].tolist())[:, None]
            else:
                # Keep only specified columns
                for col in self.obs_keys:
                    obs_data[col] = np.array(obs[col].tolist())[:, None]
            batch["obs"] = obs_data

        return BatchData(**batch)

    def load_and_process_all_data(self):
        all_data = []
        for i, file in enumerate(self.files_list):
            logging.info(f"Processing data file {i+1} of {len(self.files_list)}")
            file_batch = self._get_batch_from_file(file)
            if file_batch is None:
                continue

            all_data.append(file_batch)

        # Add check for empty all_data list
        if not all_data:
            raise ValueError(
                "No valid data was loaded from any files. "
                "Check if files exist and contain valid data after filtering."
            )

        concatenated_batch = BatchData(
            gene_counts=torch.concat([batch.gene_counts for batch in all_data]),
            gene_token_indices=torch.concat([batch.gene_token_indices for batch in all_data]),
            file_path=None,
            aux_token_indices=(
                torch.concat([batch.aux_token_indices for batch in all_data])
                if all_data[0].aux_token_indices is not None
                else None
            ),
            obs=(
                {col: np.vstack([batch.obs[col] for batch in all_data]) for col in self.obs_keys}
                if self.obs_keys is not None
                else None
            ),
        )

        return concatenated_batch

    def __len__(self):
        return len(self.data.gene_counts)

    def __getitem__(self, idx):
        data_dict = {}
        for key, value in self.data.__dict__.items():
            if value is None:
                data_dict[key] = None
            elif isinstance(value, dict):
                data_dict[key] = {k: v[idx] for k, v in value.items()}
            else:
                data_dict[key] = value[idx]
        return BatchData(**data_dict)

    @staticmethod
    def collate_fn(batch: BatchData | list[BatchData]) -> BatchData:
        if isinstance(batch, BatchData):
            return batch

        collated_batch = BatchData(
            gene_counts=torch.stack([item.gene_counts for item in batch]),
            gene_token_indices=torch.stack([item.gene_token_indices for item in batch]),
            file_path=None,
            aux_token_indices=(
                torch.stack([item.aux_token_indices for item in batch])
                if batch[0].aux_token_indices is not None
                else None
            ),
            obs=(
                {col: np.vstack([item.obs[col] for item in batch]) for col in batch[0].obs.keys()}
                if batch[0].obs is not None
                else None
            ),
        )
        return collated_batch


class AnnDatasetOOM(Dataset):
    """Map-style OOM-safe dataset using backed reads and per-item processing.

    Designed to provide OOM-safe iteration while leveraging PyTorch's
    DistributedSampler for automatic sharding across DDP ranks.
    """

    collate_fn = staticmethod(AnnDataset.collate_fn)

    def __init__(
        self,
        files_list: list[str],
        gene_vocab: dict[str, str],
        data_dir: str | None = None,
        aux_vocab: dict[str, dict[str, str]] | None = None,
        max_len: int = 2048,
        normalize_to_scale: float | None = None,
        sort_genes: bool = False,
        randomize_order: bool = False,
        pad_zeros: bool = True,
        pad_token: str = "[PAD]",
        gene_col_name: str = "ensembl_id",
        filter_to_vocab: bool = True,
        filter_outliers: float = 0.0,
        min_expressed_genes: int = 0,
        clip_counts: float = 1e10,
        obs_keys: list[str] | None = None,
        use_raw: bool | None = None,
        remove_duplicate_genes: bool = False,
        stats_chunk_size: int = 1024,
    ):
        super().__init__()
        self.files_list = files_list
        self.data_dir = data_dir
        self.gene_vocab = gene_vocab
        self.aux_vocab = aux_vocab
        self.max_len = max_len
        self.normalize_to_scale = normalize_to_scale
        self.sort_genes = sort_genes
        self.randomize_order = randomize_order
        self.pad_zeros = pad_zeros
        self.pad_token = pad_token
        self.gene_col_name = gene_col_name
        self.filter_to_vocab = filter_to_vocab
        self.filter_outliers = filter_outliers
        self.min_expressed_genes = min_expressed_genes
        self.clip_counts = clip_counts
        self.obs_keys = obs_keys
        self.use_raw = use_raw
        self.remove_duplicate_genes = remove_duplicate_genes
        self.stats_chunk_size = max(1, int(stats_chunk_size))
        self.filter_metadata: list[dict] = []

        self.gene_tokenizer = BatchGeneTokenizer(gene_vocab)
        if aux_vocab is not None:
            self.aux_tokenizer = BatchObsTokenizer(aux_vocab)

        # Open backed handles and build cumulative row offsets
        self._handles: list[anndata.AnnData] = []
        self._gene_names_per_file: list[np.ndarray] = []
        self._filter_idx_per_file: list[list[int] | None] = []
        self._keep_rows_per_file: list[np.ndarray] = []
        self._X_per_file: list = []
        self._n_rows: list[int] = []
        for file in self.files_list:
            file_path = file if self.data_dir is None else os.path.join(self.data_dir, file)
            adata = anndata.read_h5ad(file_path, backed="r")
            gene_names, success, adata, dedup_col_indices = load_gene_features(
                adata, self.gene_col_name, self.remove_duplicate_genes, use_raw=self.use_raw
            )
            if not success:
                raise ValueError(f"Failed to load gene features from {file_path}")

            original_gene_count = int(len(gene_names))
            original_cell_count = int(adata.n_obs)

            # Optional vocab filtering at token level.
            # When dedup_col_indices is set (backed mode deduplication), those indices are
            # into the *original* adata column space; vocab filtering indices are into the
            # deduplicated gene_names list, so the two must be composed.
            filter_idx = None
            if self.filter_to_vocab:
                vocab_positions = [i for i, name in enumerate(gene_names) if name in self.gene_vocab]
                if dedup_col_indices is not None:
                    filter_idx = [dedup_col_indices[i] for i in vocab_positions]
                else:
                    filter_idx = vocab_positions
                gene_names = gene_names[np.array(vocab_positions)]
                logging.info(
                    f"Filtered {original_gene_count} genes to {len(gene_names)} genes in vocab for file {file_path}"
                )
                if len(gene_names) == 0:
                    raise ValueError(f"No genes remaining after filtering for file {file_path}")
            elif dedup_col_indices is not None:
                # No vocab filter but deduplication was deferred; use the dedup indices as filter_idx
                filter_idx = dedup_col_indices

            X_layer = get_counts_layer(adata, self.use_raw)
            if self.filter_outliers > 0 or self.min_expressed_genes > 0:
                expr_counts, nnz_counts = compute_row_stats_chunked(
                    X_layer,
                    filter_idx=filter_idx,
                    chunk_size=self.stats_chunk_size,
                )

                keep_mask = np.ones(original_cell_count, dtype=bool)
                if self.filter_outliers > 0:
                    count_std = np.std(expr_counts)
                    count_mean = np.mean(expr_counts)
                    keep_mask &= (expr_counts > count_mean - count_std * self.filter_outliers) & (
                        expr_counts < count_mean + count_std * self.filter_outliers
                    )
                if self.min_expressed_genes > 0:
                    keep_mask &= nnz_counts >= self.min_expressed_genes

                keep_rows = np.nonzero(keep_mask)[0].astype(np.int64)

                logging.info(
                    f"Filtered {original_cell_count} cells to {len(keep_rows)} cells for file {file_path}"
                )
            else:
                keep_rows = np.arange(original_cell_count, dtype=np.int64)
            if keep_rows.size == 0:
                raise ValueError(f"No cells remaining after filtering for file {file_path}")

            self._handles.append(adata)
            self._gene_names_per_file.append(gene_names)
            self._filter_idx_per_file.append(filter_idx)
            self._keep_rows_per_file.append(keep_rows)
            self._X_per_file.append(X_layer)
            self._n_rows.append(int(keep_rows.size))

            self.filter_metadata.append(
                {
                    "file": file_path,
                    "original_genes": original_gene_count,
                    "kept_genes": int(len(gene_names)),
                    "removed_genes": int(original_gene_count - len(gene_names)),
                    "original_cells": original_cell_count,
                    "kept_cells": int(keep_rows.size),
                    "removed_cells": int(original_cell_count - keep_rows.size),
                    "filter_to_vocab": bool(self.filter_to_vocab),
                    "filter_outliers": float(self.filter_outliers),
                    "min_expressed_genes": int(self.min_expressed_genes),
                }
            )

        self._offsets = np.cumsum([0] + self._n_rows)

    def __len__(self) -> int:
        return int(self._offsets[-1])

    @property
    def num_files(self) -> int:
        return len(self._n_rows)

    def file_num_rows(self, file_id: int) -> int:
        return int(self._n_rows[file_id])

    def file_offset(self, file_id: int) -> int:
        return int(self._offsets[file_id])

    def _loc(self, idx: int) -> tuple[int, int]:
        file_id = int(np.searchsorted(self._offsets, idx, side="right") - 1)
        row = int(idx - self._offsets[file_id])
        return file_id, row

    @staticmethod
    def _ensure_2d_array(x_rows) -> np.ndarray:
        if isinstance(x_rows, csr_matrix | csc_matrix):
            arr = x_rows.toarray()
        else:
            arr = np.asarray(x_rows)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr

    def _build_obs_dict(self, obs_rows) -> dict[str, np.ndarray] | None:
        if self.obs_keys is None:
            return None

        obs_dict = {}
        cols = list(obs_rows.columns) if "all" in self.obs_keys else list(self.obs_keys or [])
        for col in cols:
            obs_dict[col] = np.array(obs_rows[col].tolist())[:, None]
        return obs_dict

    def _build_batch_from_rows(self, file_id: int, rows: list[int]) -> BatchData:
        actual_rows = self._keep_rows_per_file[file_id][np.asarray(rows, dtype=np.int64)]
        actual_rows = np.asarray(actual_rows, dtype=np.int64)

        X = self._X_per_file[file_id]
        # Backed/h5ad row reads are typically better behaved (and often faster) with monotonic indices
        sort_order = np.argsort(actual_rows, kind="stable")
        sorted_rows = actual_rows[sort_order]
        x_rows = self._ensure_2d_array(X[sorted_rows])
        if x_rows.shape[0] > 1:
            x_rows = x_rows[np.argsort(sort_order)]

        filter_idx = self._filter_idx_per_file[file_id]
        if filter_idx is not None:
            x_rows = x_rows[:, filter_idx]

        adata = self._handles[file_id]
        obs_rows = adata.obs.iloc[actual_rows]
        gene_names = self._gene_names_per_file[file_id]

        batch = process_batch(
            x_rows,
            obs_rows,
            gene_names,
            self.gene_tokenizer,
            getattr(self, "aux_tokenizer", None),
            self.sort_genes,
            self.randomize_order,
            self.max_len,
            self.pad_zeros,
            self.pad_token,
            self.gene_vocab,
            self.normalize_to_scale,
            self.clip_counts,
            self.aux_vocab,
        )

        return BatchData(
            gene_counts=batch["gene_counts"],
            gene_token_indices=batch["gene_token_indices"],
            file_path=None,
            aux_token_indices=batch.get("aux_token_indices"),
            obs=self._build_obs_dict(obs_rows),
        )

    def __getitem__(self, idx: int) -> BatchData:
        # Build a 1-row batch
        file_id, row = self._loc(idx)
        batch = self._build_batch_from_rows(file_id, [row])
        return BatchData(
            gene_counts=batch.gene_counts[0],
            gene_token_indices=batch.gene_token_indices[0],
            file_path=None,
            aux_token_indices=(batch.aux_token_indices[0] if batch.aux_token_indices is not None else None),
            obs=({col: value[0:1] for col, value in batch.obs.items()} if batch.obs is not None else None),
        )

    def __getitems__(self, indices: list[int]) -> BatchData | list[BatchData]:
        # Build a multi-row batch if all indices are from the same file; otherwise fallback to individual retrieval
        if not indices:
            return []

        locations = [self._loc(int(idx)) for idx in indices]
        file_ids = {file_id for file_id, _ in locations}
        if len(file_ids) != 1:
            return [self[int(idx)] for idx in indices]

        file_id = locations[0][0]
        rows = [row for _, row in locations]
        return self._build_batch_from_rows(file_id, rows)


class FileAwareBatchSampler(BatchSampler):
    """Yield file-local batches while interleaving files across an epoch."""

    def __init__(
        self,
        dataset: AnnDatasetOOM,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        batches_per_file: int = 1,
        seed: int = 0,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batches_per_file <= 0:
            raise ValueError("batches_per_file must be positive")

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.batches_per_file = int(batches_per_file)
        self.seed = int(seed)
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def _build_all_batches(self) -> list[list[int]]:
        rng = random.Random(self.seed + self._epoch)
        file_ids = list(range(self.dataset.num_files))
        if self.shuffle:
            rng.shuffle(file_ids)

        # Validation-friendly path: deterministic, file-sequential traversal.
        # This minimizes backed-file switching and is typically faster than interleaving.
        if not self.shuffle:
            all_batches: list[list[int]] = []
            for file_id in file_ids:
                local_rows = list(range(self.dataset.file_num_rows(file_id)))
                offset = self.dataset.file_offset(file_id)
                for start in range(0, len(local_rows), self.batch_size):
                    batch_rows = local_rows[start : start + self.batch_size]
                    if len(batch_rows) < self.batch_size and self.drop_last:
                        continue
                    all_batches.append([offset + row for row in batch_rows])
            return all_batches

        active_files: list[dict] = []
        for file_id in file_ids:
            local_rows = list(range(self.dataset.file_num_rows(file_id)))
            if self.shuffle:
                rng.shuffle(local_rows)

            batches = []
            offset = self.dataset.file_offset(file_id)
            for start in range(0, len(local_rows), self.batch_size):
                batch_rows = local_rows[start : start + self.batch_size]
                if len(batch_rows) < self.batch_size and self.drop_last:
                    continue
                batches.append([offset + row for row in batch_rows])

            if batches:
                active_files.append({"batches": batches, "cursor": 0})

        all_batches: list[list[int]] = []
        while active_files:
            round_order = list(range(len(active_files)))
            if self.shuffle:
                rng.shuffle(round_order)

            next_active = []
            for pos in round_order:
                state = active_files[pos]
                start = state["cursor"]
                stop = min(start + self.batches_per_file, len(state["batches"]))
                all_batches.extend(state["batches"][start:stop])
                if stop < len(state["batches"]):
                    state["cursor"] = stop
                    next_active.append(state)
            active_files = next_active

        return all_batches

    def _rank_batches(self, all_batches: list[list[int]]) -> list[list[int]]:
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return all_batches

        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        if world_size <= 1 or not all_batches:
            return all_batches

        if self.drop_last:
            total = len(all_batches) - (len(all_batches) % world_size)
            all_batches = all_batches[:total]
        else:
            remainder = len(all_batches) % world_size
            if remainder:
                padding = world_size - remainder
                all_batches = all_batches + all_batches[:padding]

        return all_batches[rank::world_size]

    def __iter__(self):
        rank_batches = self._rank_batches(self._build_all_batches())
        self._epoch += 1
        yield from rank_batches

    def __len__(self) -> int:
        all_batches = self._build_all_batches()
        rank_batches = self._rank_batches(all_batches)
        return len(rank_batches)

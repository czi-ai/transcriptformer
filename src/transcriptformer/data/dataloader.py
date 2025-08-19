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
from torch.utils.data import Dataset

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
        if len(duplicates) > 0:
            if remove_duplicate_genes:
                seen = set()
                unique_indices = []
                for i, gene in enumerate(gene_names):
                    if gene not in seen:
                        seen.add(gene)
                        unique_indices.append(i)
                adata = adata[:, unique_indices].copy()
                gene_names = gene_names[unique_indices]
                logging.warning(
                    f"Removed {len(duplicates)} duplicate genes after removing version numbers. Kept first occurrence."
                )
            else:
                raise ValueError(
                    "Found duplicate genes after removing version numbers. "
                    "Remove duplicates or pass --remove-duplicate-genes."
                )

        return gene_names, True, adata
    except KeyError:
        return None, False, adata


def validate_gene_dimension(X: np.ndarray, gene_names: np.ndarray, gene_col_name: str):
    if X.shape[1] != len(gene_names):
        raise ValueError(
            f"Mismatch between expression matrix columns ({X.shape[1]}) and gene names length ({len(gene_names)}). "
            f"Ensure 'adata.var[{gene_col_name}]' exists and aligns with the matrix columns."
        )


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

        gene_names, success, adata = load_gene_features(
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
            logging.warning(f"Data was filtered out completely for {file_path}")
            return None

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
        clip_counts: float = 1e10,
        obs_keys: list[str] | None = None,
        use_raw: bool | None = None,
        remove_duplicate_genes: bool = False,
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
        self.clip_counts = clip_counts
        self.obs_keys = obs_keys
        self.use_raw = use_raw
        self.remove_duplicate_genes = remove_duplicate_genes

        self.gene_tokenizer = BatchGeneTokenizer(gene_vocab)
        if aux_vocab is not None:
            self.aux_tokenizer = BatchObsTokenizer(aux_vocab)

        # Open backed handles and build cumulative row offsets
        self._handles: list[anndata.AnnData] = []
        self._gene_names_per_file: list[np.ndarray] = []
        self._filter_idx_per_file: list[list[int] | None] = []
        self._X_per_file: list = []
        self._n_rows: list[int] = []
        for file in self.files_list:
            file_path = file if self.data_dir is None else os.path.join(self.data_dir, file)
            adata = anndata.read_h5ad(file_path, backed="r")
            gene_names, success, adata = load_gene_features(
                adata, self.gene_col_name, self.remove_duplicate_genes, use_raw=self.use_raw
            )
            if not success:
                raise ValueError(f"Failed to load gene features from {file_path}")
            # Optional vocab filtering at token level
            filter_idx = None
            if self.filter_to_vocab:
                original_gene_count = len(gene_names)
                filter_idx = [i for i, name in enumerate(gene_names) if name in self.gene_vocab]
                gene_names = gene_names[filter_idx]
                logging.info(
                    f"Filtered {original_gene_count} genes to {len(gene_names)} genes in vocab for file {file_path}"
                )
                if len(gene_names) == 0:
                    raise ValueError(f"No genes remaining after filtering for file {file_path}")

            self._handles.append(adata)
            self._gene_names_per_file.append(gene_names)
            self._filter_idx_per_file.append(filter_idx)
            X_layer = get_counts_layer(adata, self.use_raw)
            self._X_per_file.append(X_layer)
            self._n_rows.append(int(adata.n_obs))

        self._offsets = np.cumsum([0] + self._n_rows)

    def __len__(self) -> int:
        return int(self._offsets[-1])

    def _loc(self, idx: int) -> tuple[int, int]:
        file_id = int(np.searchsorted(self._offsets, idx, side="right") - 1)
        row = int(idx - self._offsets[file_id])
        return file_id, row

    def __getitem__(self, idx: int) -> BatchData:
        file_id, row = self._loc(idx)
        adata = self._handles[file_id]
        gene_names = self._gene_names_per_file[file_id]
        filter_idx = self._filter_idx_per_file[file_id]

        X = self._X_per_file[file_id]
        # Some backed sparse implementations use __getitem__ returning 2D; ensure 1D
        x_row = X[row]
        # Only convert to dense if the row is actually sparse
        if isinstance(x_row, csr_matrix | csc_matrix):
            x_row = x_row.toarray().ravel()
        else:
            x_row = np.asarray(x_row).ravel()
        if filter_idx is not None:
            x_row = x_row[filter_idx]

        obs_row = adata.obs.iloc[row : row + 1]

        # Build a 1-row batch and reuse existing processing pipeline
        x_batch = np.expand_dims(x_row, axis=0)
        batch = process_batch(
            x_batch,
            obs_row,
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

        # Convert to BatchData for collate_fn compatibility
        obs_dict = None
        if self.obs_keys is not None:
            obs_dict = {}
            cols = list(obs_row.columns) if "all" in self.obs_keys else list(self.obs_keys or [])
            for col in cols:
                obs_dict[col] = np.array(obs_row[col].tolist())[:, None]

        return BatchData(
            gene_counts=batch["gene_counts"][0],
            gene_token_indices=batch["gene_token_indices"][0],
            file_path=None,
            aux_token_indices=(
                batch.get("aux_token_indices")[0] if batch.get("aux_token_indices") is not None else None
            ),
            obs=obs_dict,
        )

"""Utilities for adapting Transcriptformer to expanded assay vocabularies."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class AssayInitConfig:
    """Configuration for initializing new assay embedding rows."""

    default_source: str = "unknown"
    mean_pool_fallback: bool = True


def _validate_vocab_indices(vocab: dict[str, int]) -> None:
    ids = set(vocab.values())
    expected = set(range(len(vocab)))
    if ids != expected:
        raise ValueError("Assay vocab indices must be contiguous [0, ..., N-1]")


def build_expanded_assay_embedding_weight(
    old_weight: torch.Tensor,
    old_vocab: dict[str, int],
    new_vocab: dict[str, int],
    init_map: dict[str, str] | None = None,
    cfg: AssayInitConfig | None = None,
) -> torch.Tensor:
    """Create an expanded embedding matrix for assay tokens.

    Args:
        old_weight: Existing assay embedding matrix with shape [old_vocab_size, dim].
        old_vocab: Original assay vocab mapping token -> index.
        new_vocab: Expanded assay vocab mapping token -> index.
        init_map: Optional explicit mapping new_token -> existing_source_token.
        cfg: Initialization policy.

    Returns
    -------
        Expanded embedding matrix with shape [new_vocab_size, dim].
    """
    _validate_vocab_indices(old_vocab)
    _validate_vocab_indices(new_vocab)

    if cfg is None:
        cfg = AssayInitConfig()
    if init_map is None:
        init_map = {}

    dim = old_weight.shape[1]
    device = old_weight.device
    dtype = old_weight.dtype

    new_weight = torch.empty((len(new_vocab), dim), device=device, dtype=dtype)

    # Random init baseline similar to nn.Embedding default behavior.
    nn.init.normal_(new_weight, mean=0.0, std=1.0)

    # Precompute fallback vector.
    if cfg.mean_pool_fallback:
        fallback_vec = old_weight.mean(dim=0)
    else:
        source_token = cfg.default_source
        if source_token not in old_vocab:
            raise ValueError(f"default_source token '{source_token}' is missing in old vocab")
        fallback_vec = old_weight[old_vocab[source_token]]

    for token, new_idx in new_vocab.items():
        if token in old_vocab:
            new_weight[new_idx] = old_weight[old_vocab[token]]
            continue

        source_token = init_map.get(token, cfg.default_source)
        if source_token in old_vocab:
            new_weight[new_idx] = old_weight[old_vocab[source_token]]
        else:
            new_weight[new_idx] = fallback_vec

    return new_weight


def apply_freeze_policy(
    model: nn.Module,
    freeze_transformer: bool = False,
    freeze_gene_embeddings: bool = True,
    freeze_count_head: bool = False,
    freeze_gene_head: bool = False,
    train_aux_only: bool = False,
) -> None:
    """Apply parameter freezing policy in-place."""

    if train_aux_only:
        for param in model.parameters():
            param.requires_grad = False
        if hasattr(model, "aux_embeddings"):
            for param in model.aux_embeddings.parameters():
                param.requires_grad = True
        return

    if freeze_transformer and hasattr(model, "transformer_encoder"):
        for param in model.transformer_encoder.parameters():
            param.requires_grad = False

    if freeze_gene_embeddings and hasattr(model, "gene_embeddings"):
        for param in model.gene_embeddings.parameters():
            param.requires_grad = False

    if freeze_count_head and hasattr(model, "mu"):
        for param in model.mu.parameters():
            param.requires_grad = False

    if freeze_gene_head and hasattr(model, "gene_id_head"):
        for param in model.gene_id_head.parameters():
            param.requires_grad = False


def count_trainable_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (trainable_params, total_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total

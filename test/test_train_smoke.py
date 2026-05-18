"""Smoke tests for training module."""

import pytest
import torch

from transcriptformer.data.dataclasses import BatchData, DataConfig, LossConfig, ModelConfig
from transcriptformer.model.model import Transcriptformer
from transcriptformer.train.engine import TranscriptformerTrainModule


def _tiny_model() -> Transcriptformer:
    gene_vocab = {
        "unknown": 0,
        "[PAD]": 1,
        "[START]": 2,
        "[END]": 3,
        "[RD]": 4,
        "[CELL]": 5,
        "[MASK]": 6,
        "g1": 7,
        "g2": 8,
        "g3": 9,
    }
    aux_vocab = {"assay": {"unknown": 0, "new_assay": 1}}

    data_config = DataConfig(
        aux_vocab_path=".",
        pin_memory=False,
        aux_cols=["assay"],
        gene_col_name="ensembl_id",
        clip_counts=30,
        filter_to_vocabs=True,
        filter_outliers=0.0,
        pad_zeros=True,
        normalize_to_scale=0,
        n_data_workers=0,
        sort_genes=False,
        randomize_genes=False,
        min_expressed_genes=0,
        gene_pad_token="[PAD]",
        aux_pad_token="unknown",
    )

    model_config = ModelConfig(
        log_counts_eps=1e-6,
        num_heads=2,
        num_layers=1,
        model_dim=16,
        embed_dim=8,
        dropout=0.1,
        activation="gelu",
        attn_bias=False,
        fw_bias=False,
        mu_link_fn="softplus",
        softcap=0,
        seq_len=3,
        aux_len=1,
        block_len=2,
        compile_block_mask=False,
    )

    loss_config = LossConfig(gene_id_loss_weight=1.0, softplus_approx=True)

    emb_matrix = torch.randn(len(gene_vocab), model_config.embed_dim)

    return Transcriptformer(
        data_config=data_config,
        model_config=model_config,
        loss_config=loss_config,
        gene_vocab_dict=gene_vocab,
        aux_vocab_dict=aux_vocab,
        emb_matrix=emb_matrix,
    )


def test_train_smoke_forward_backward():
    """Smoke test for training forward/backward (skipped on CPU due to FlexAttention limitation)."""
    # FlexAttention does not support backward on CPU, so skip this test if no GPU available
    if not torch.cuda.is_available():
        pytest.skip("FlexAttention backward not supported on CPU; GPU required for this test")
    
    device = torch.device("cuda")
    model = _tiny_model()
    model = model.to(device)
    
    module = TranscriptformerTrainModule(
        model=model,
        lr=1e-4,
        weight_decay=0.0,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        warmup_ratio=0.1,
        min_lr_ratio=0.1,
        gene_loss_weight=1.0,
        count_loss_weight=1.0,
        shuffle_expressed_each_batch=False,
    )

    batch = BatchData(
        gene_counts=torch.tensor([[5.0, 3.0, 2.0], [2.0, 1.0, 4.0]], dtype=torch.float32, device=device),
        gene_token_indices=torch.tensor([[7, 8, 9], [8, 7, 9]], dtype=torch.int64, device=device),
        aux_token_indices=torch.tensor([[0], [1]], dtype=torch.int64, device=device),
    )

    loss = module.training_step(batch, 0)
    assert torch.isfinite(loss).item()
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)

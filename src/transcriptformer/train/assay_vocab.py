"""Assay vocabulary helpers for adaptation workflows."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad


def load_vocab(path: Path) -> dict[str, int]:
    with path.open() as f:
        vocab = json.load(f)
    if "unknown" not in vocab:
        raise ValueError("Input vocab must include an 'unknown' token")
    return vocab


def write_vocab(vocab: dict[str, int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(vocab, f, indent=2, sort_keys=True)
        f.write("\n")


def discover_tokens(data_files: list[str], obs_col: str, min_frequency: int = 1) -> list[str]:
    counts: dict[str, int] = {}
    for file in data_files:
        adata = ad.read_h5ad(file, backed="r")
        if obs_col not in adata.obs.columns:
            raise ValueError(f"Column '{obs_col}' not found in {file}")
        values = adata.obs[obs_col].astype(str).values
        for token in values:
            counts[token] = counts.get(token, 0) + 1
    return sorted([token for token, n in counts.items() if n >= min_frequency])


def expand_vocab(existing_vocab: dict[str, int], new_tokens: list[str]) -> dict[str, int]:
    expanded = dict(existing_vocab)
    next_id = max(expanded.values()) + 1 if expanded else 0

    for token in new_tokens:
        if token in expanded:
            continue
        expanded[token] = next_id
        next_id += 1

    return expanded

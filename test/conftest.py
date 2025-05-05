"""Test fixtures for TranscriptFormer CLI tests."""

import json
import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def mock_checkpoint_dir(tmp_path):
    """Create a mock checkpoint directory structure for testing."""
    # Create main directories
    checkpoint_dir = tmp_path / "checkpoints"
    model_dir = checkpoint_dir / "tf_sapiens"
    vocabs_dir = model_dir / "vocabs"

    os.makedirs(vocabs_dir, exist_ok=True)

    # Create a mock config.json
    config = {
        "model": {
            "data_config": {
                "gene_col_name": "ensembl_id",
                "aux_vocab_path": str(vocabs_dir),
                "esm2_mappings_path": str(vocabs_dir),
            },
            "inference_config": {"batch_size": 8, "precision": "16-mixed"},
        }
    }

    with open(model_dir / "config.json", "w") as f:
        json.dump(config, f)

    # Create dummy model weights file
    with open(model_dir / "model_weights.pt", "w") as f:
        f.write("dummy model weights")

    # Create dummy vocabulary files
    with open(vocabs_dir / "gene_vocab.json", "w") as f:
        json.dump({"gene1": 1, "gene2": 2}, f)

    return checkpoint_dir


@pytest.fixture
def mock_h5ad_file(tmp_path):
    """Create a mock h5ad file for testing."""
    # Create a mock h5ad file
    data_dir = tmp_path / "data"
    os.makedirs(data_dir, exist_ok=True)

    # Create a dummy h5ad file
    data_file = data_dir / "test_data.h5ad"
    with open(data_file, "w") as f:
        f.write("dummy h5ad data")

    return data_file


@pytest.fixture
def mock_output_dir(tmp_path):
    """Create a mock output directory for testing."""
    output_dir = tmp_path / "output"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


@pytest.fixture
def cli_env():
    """Set up a clean environment for CLI tests and restore original environment after."""
    # Save original environment
    original_env = os.environ.copy()
    original_argv = os.sys.argv.copy()

    # Use a temporary directory for the tests
    with tempfile.TemporaryDirectory() as tempdir:
        # Return the temp directory path
        yield Path(tempdir)

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)
    os.sys.argv = original_argv

import json
import logging
import os
import pickle

import h5py
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_embeddings(embeddings_path):
    with open(embeddings_path, "rb") as f:
        if embeddings_path.endswith(".pkl"):
            embeddings = pickle.load(f)
        elif embeddings_path.endswith(".h5"):
            embeddings = load_from_hdf5(embeddings_path)
    return embeddings


def load_from_hdf5(file_path):
    """Load dictionary from HDF5 file.

    Args:
        file_path (str): Path to HDF5 file containing embeddings data. The file should have
            a 'keys' dataset containing gene names and an 'arrays' group containing the
            corresponding embedding arrays.

    Returns
    -------
        dict: Dictionary mapping gene names (str) to their embedding arrays (numpy.ndarray).
            The keys are decoded from bytes to UTF-8 strings.
    """
    data_dict = {}
    with h5py.File(file_path, "r") as f:
        # Get the keys
        keys = [k.decode("utf-8") for k in f["keys"][:]]

        # Load the arrays
        arrays_group = f["arrays"]
        for key in keys:
            data_dict[key] = arrays_group[str(key)][:]

    return data_dict


def stack_dict(output):
    concatenated_data = {}
    for key in output[0].keys():
        if isinstance(output[0][key], torch.Tensor):
            if output[0][key].dim() == 0:  # Scalar tensor
                concatenated_data[key] = [batch[key].item() for batch in output]
            else:
                concatenated_data[key] = torch.cat([batch[key] for batch in output], dim=0)
        elif isinstance(output[0][key], np.ndarray):
            concatenated_data[key] = np.concatenate([batch[key] for batch in output], axis=0)
        elif isinstance(output[0][key], dict):
            concatenated_data[key] = {
                k: np.vstack([batch[key][k] for batch in output]).flatten() for k in output[0][key].keys()
            }
        elif isinstance(output[0][key], int | float):  # Python scalar
            concatenated_data[key] = [batch[key] for batch in output]
        elif isinstance(output[0][key], list):
            concatenated_data[key] = sum([batch[key] for batch in output], [])
        else:  # Handle other types
            concatenated_data[key] = [batch[key] for batch in output]
    return concatenated_data


def save_as_hdf5(data_dict, output_path):
    """Save dictionary as HDF5 file."""
    with h5py.File(output_path, "w") as f:
        # Store the keys as a dataset
        keys = list(data_dict.keys())
        f.create_dataset("keys", data=np.array(keys, dtype="S"))

        # Create a group for the arrays
        arrays_group = f.create_group("arrays")
        for key, value in data_dict.items():
            arrays_group.create_dataset(str(key), data=value)


def filter_minimum_class(
    X: np.ndarray, y: np.ndarray | pd.Series, min_class_size: int = 10
) -> tuple[np.ndarray, np.ndarray | pd.Series]:
    logging.info(f"Label composition ({y.name}):")
    value_counts = y.value_counts()
    logging.info(f"Total classes before filtering: {len(value_counts)}")

    filtered_counts = value_counts[value_counts >= min_class_size]
    logging.info(f"Total classes after filtering (min_class_size={min_class_size}): {len(filtered_counts)}")

    y = pd.Series(y) if isinstance(y, np.ndarray) else y
    class_counts = y.value_counts()

    valid_classes = class_counts[class_counts >= min_class_size].index
    valid_indices = y.isin(valid_classes)

    X_filtered = X[valid_indices]
    y_filtered = y[valid_indices]

    return X_filtered, pd.Categorical(y_filtered)


def merge_checkpoint_with_cfg(checkpoint_path: str, base_cfg):
    """Merge checkpoint config.json with a provided base cfg (YAML + overrides).

    The merge order matches the CLI behavior:
    - Start from checkpoint config (mlflow_cfg)
    - Merge base_cfg on top (YAML + any overrides)
    - Set derived paths consistently
    """
    # Load model config from checkpoint
    config_path = os.path.join(checkpoint_path, "config.json")
    with open(config_path) as f:
        model_config = json.load(f)
    mlflow_cfg = OmegaConf.create(model_config)

    # Merge: checkpoint first, then base cfg overrides
    cfg = OmegaConf.merge(mlflow_cfg, base_cfg)

    # Set derived paths
    cfg.model.inference_config.load_checkpoint = os.path.join(checkpoint_path, "model_weights.pt")
    cfg.model.data_config.aux_vocab_path = os.path.join(checkpoint_path, "vocabs")
    cfg.model.data_config.esm2_mappings_path = os.path.join(checkpoint_path, "vocabs")
    cfg.model.inference_config.checkpoint_path = checkpoint_path

    return cfg


def print_progress(current: int, total: int, prefix: str = "", suffix: str = "", length: int = 50) -> None:
    """Print a simple ASCII progress bar.

    Args:
        current: Current progress value
        total: Total value representing 100%
        prefix: Text prefix to display before the bar
        suffix: Text suffix to display after the percentage
        length: Character length of the bar
    """
    filled = int(length * current / total) if total > 0 else 0
    bar = "#" * filled + "-" * (length - filled)
    percent = int(100 * current / total) if total > 0 else 0
    print(f"\r{prefix} |{bar}| {percent}% {suffix}", end="", flush=True)
    if total > 0 and current >= total:
        print()


class ProgressTracker:
    """Rate-limited progress tracker to avoid excessive stdout updates."""

    def __init__(self, prefix: str = "", min_update_interval: float = 0.1):
        self.prefix = prefix
        self.min_update_interval = min_update_interval
        self.last_update_time = 0.0
        self.last_percent = -1

    def update(self, current: int, total: int) -> None:
        import time

        now = time.time()
        current_percent = int(100 * current / total) if total > 0 else 0

        should_update = (
            now - self.last_update_time >= self.min_update_interval
            or current_percent - self.last_percent >= 2
            or (total > 0 and current >= total)
        )

        if should_update:
            print_progress(current, total, prefix=self.prefix)
            self.last_update_time = now
            self.last_percent = current_percent

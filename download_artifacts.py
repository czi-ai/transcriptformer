#!/usr/bin/env python3

"""
Download and extract TranscriptFormer model artifacts from a public S3 bucket.

This script provides a convenient way to download and extract TranscriptFormer model weights
from a public S3 bucket. It supports downloading individual models or all models at once,
with progress indicators for both download and extraction processes.

Usage:
    python download_artifacts.py [model] [--checkpoint-dir DIR]

    model: The model to download. Options are:
        - tf-sapiens: Download the sapiens model
        - tf-exemplar: Download the exemplar model
        - tf-metazoa: Download the metazoa model
        - all: Download all models and embeddings
        - all-embeddings: Download only the embedding files

    --checkpoint-dir: Optional directory to store the downloaded checkpoints.
                      Defaults to './checkpoints'

Examples
--------
    # Download the sapiens model
    python download_artifacts.py tf-sapiens

    # Download all models and embeddings
    python download_artifacts.py all

    # Download only the embeddings file
    python download_artifacts.py all-embeddings

    # Download the exemplar model to a custom directory
    python download_artifacts.py tf-exemplar --checkpoint-dir /path/to/models

The downloaded models will be extracted to:
    ./checkpoints/tf_sapiens/
    ./checkpoints/tf_exemplar/
    ./checkpoints/tf_metazoa/
    ./checkpoints/all_embeddings/
"""

import argparse
import math
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path


def print_progress(current, total, prefix="", suffix="", length=50):
    """Print a simple progress bar."""
    filled = int(length * current / total)
    bar = "█" * filled + "░" * (length - filled)
    percent = math.floor(100 * current / total)
    print(f"\r{prefix} |{bar}| {percent}% {suffix}", end="", flush=True)
    if current == total:
        print()


def download_and_extract(model_name: str, checkpoint_dir: str = "./checkpoints"):
    """Download and extract a model artifact from S3."""
    s3_path = f"https://czi-transcriptformer.s3.amazonaws.com/weights/{model_name}.tar.gz"
    output_dir = Path(checkpoint_dir) / model_name

    # Create checkpoint directory if it doesn't exist
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {model_name} from {s3_path}...")

    try:
        # Create a temporary file to store the tar.gz
        with tempfile.NamedTemporaryFile(suffix=".tar.gz") as tmp_file:
            # Download the file using urllib with progress bar
            try:

                def report_hook(count, block_size, total_size):
                    """Callback function to report download progress."""
                    if total_size > 0:
                        print_progress(
                            count * block_size,
                            total_size,
                            prefix=f"Downloading {model_name}",
                        )

                urllib.request.urlretrieve(s3_path, filename=tmp_file.name, reporthook=report_hook)
                print()  # New line after download completes
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    print(f"Error: The model {model_name} was not found at {s3_path}")
                else:
                    print(f"Error downloading file: HTTP {e.code}")
                sys.exit(1)
            except urllib.error.URLError as e:
                print(f"Error downloading file: {str(e)}")
                sys.exit(1)

            # Reset the file pointer to the beginning
            tmp_file.seek(0)

            # Extract the tar.gz file
            try:
                print(f"Extracting {model_name}...")
                with tarfile.open(fileobj=tmp_file, mode="r:gz") as tar:
                    members = tar.getmembers()
                    total_files = len(members)
                    for i, member in enumerate(members, 1):
                        tar.extract(member, path=str(output_dir.parent))
                        print_progress(
                            i,
                            total_files,
                            prefix=f"Extracting {model_name}",
                        )
                print()  # New line after extraction completes
            except tarfile.ReadError:
                print(f"Error: The downloaded file for {model_name} is not a valid tar.gz archive")
                sys.exit(1)

        print(f"Successfully downloaded and extracted {model_name} to {output_dir}")

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Download and extract TranscriptFormer model artifacts")
    parser.add_argument(
        "model",
        choices=["tf-sapiens", "tf-exemplar", "tf-metazoa", "all", "all-embeddings"],
        help="Model to download (or 'all' for all models and embeddings, 'all-embeddings' for just embeddings)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="./checkpoints",
        help="Directory to store the downloaded checkpoints (default: ./checkpoints)",
    )

    args = parser.parse_args()

    models = {
        "tf-sapiens": "tf_sapiens",
        "tf-exemplar": "tf_exemplar",
        "tf-metazoa": "tf_metazoa",
        "all-embeddings": "all_embeddings",
    }

    if args.model == "all":
        # Download all models and embeddings
        for model in ["tf_sapiens", "tf_exemplar", "tf_metazoa", "all_embeddings"]:
            download_and_extract(model, args.checkpoint_dir)
    elif args.model == "all-embeddings":
        # Download only embeddings
        download_and_extract("all_embeddings", args.checkpoint_dir)
    else:
        download_and_extract(models[args.model], args.checkpoint_dir)


if __name__ == "__main__":
    main()

"""
Module for bulk downloading and processing single-cell RNA sequencing datasets from the CellxGene Discover portal.

This module provides functionality to:
- Download datasets from the CellxGene Discover portal API
- Process and validate the downloaded h5ad files
- Split large datasets into smaller chunks for efficient processing
- Handle multiprocessing for parallel downloads
- Filter datasets by organism/species

The module expects datasets to be in h5ad format and validates that they contain raw count data
before processing.

Example usage:
    # Download human datasets
    python -m transcriptformer.data.bulk_download --species "homo sapiens" --output-dir ./data/human

    # Download multiple species
    python -m transcriptformer.data.bulk_download --species "homo sapiens,mus musculus" --output-dir ./data/multi_species

    # Use CLI
    transcriptformer download-data --species "homo sapiens" --output-dir ./data/human
"""

import json
import logging
import warnings
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import requests

# Suppress annoying warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="anndata")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*read_.*from.*anndata.*deprecated.*")

# Set up logging
logger = logging.getLogger(__name__)

# CellxGene Discover API configuration
CELLXGENE_DOMAIN = "cellxgene.cziscience.com"
API_BASE_URL = f"https://api.{CELLXGENE_DOMAIN}"
DATASETS_ENDPOINT = "/curation/v1/datasets"


def download_single_dataset(dataset_info: dict[str, Any], save_path: Path, max_retries: int = 5) -> bool:
    """
    Download a single dataset from the CellxGene Discover portal.

    Args:
        dataset_info: Dictionary containing dataset metadata including assets and dataset_id
        save_path: Base directory where the dataset should be saved
        max_retries: Maximum number of download retry attempts

    Returns
    -------
        bool: True if download succeeded, False otherwise
    """
    asset = dataset_info["assets"][0]
    dataset_id = dataset_info["dataset_id"]
    download_filename = f"{dataset_id}.{asset['filetype']}"
    dataset_dir = save_path / dataset_id

    # Check if dataset already exists
    success_file = dataset_dir / "__success__"
    if success_file.exists():
        logger.info(f"Dataset {dataset_id} already exists, skipping...")
        return True

    # Create dataset directory
    dataset_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading {download_filename}...")

    for retry in range(max_retries):
        try:
            with requests.get(asset["url"], stream=True, timeout=120) as response:  # Increased from 30 to 120 seconds
                response.raise_for_status()

                # Get file size for progress tracking
                total_size = int(response.headers.get("Content-Length", 0))

                output_file = dataset_dir / "full.h5ad"
                with open(output_file, "wb") as file:
                    downloaded = 0
                    chunk_size = 8 * 1024 * 1024  # 8MB chunks

                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            file.write(chunk)
                            downloaded += len(chunk)

                            if total_size > 0:
                                progress = (downloaded / total_size) * 100
                                print(f"\r{dataset_id}: {progress:.1f}% downloaded", end="", flush=True)

                print()  # New line after progress

                # Mark download as successful
                success_file.write_text("success")
                logger.info(f"Successfully downloaded {dataset_id}")
                return True

        except Exception as e:
            logger.warning(f"Attempt {retry + 1}/{max_retries} failed for dataset {dataset_id}: {e}")
            if retry == max_retries - 1:
                logger.error(f"Failed to download dataset {dataset_id} after {max_retries} attempts")
                # Clean up partial download
                if (dataset_dir / "full.h5ad").exists():
                    (dataset_dir / "full.h5ad").unlink()
                return False

    return False


def download_dataset_wrapper(save_path: Path, max_retries: int, dataset_info: dict[str, Any]) -> bool:
    """
    Wrapper function for multiprocessing that handles a single dataset download.

    Args:
        save_path: Base directory where datasets should be saved
        max_retries: Maximum number of retry attempts
        dataset_info: Dictionary containing dataset metadata

    Returns
    -------
        bool: True if download succeeded, False otherwise
    """
    # Skip non-primary datasets
    if not dataset_info.get("is_primary_data", [False])[0]:
        logger.debug(f"Skipping {dataset_info['dataset_id']} because it is not the primary dataset")
        return False

    return download_single_dataset(dataset_info, save_path, max_retries)


def download_datasets_parallel(
    datasets: list[dict[str, Any]], save_path: Path, n_processes: int = 4, max_retries: int = 5
) -> int:
    """
    Download multiple datasets in parallel using multiprocessing.

    Args:
        datasets: List of dataset dictionaries to download
        save_path: Base directory where datasets should be saved
        n_processes: Number of parallel processes to use
        max_retries: Maximum number of retry attempts per dataset

    Returns
    -------
        int: Number of successfully downloaded datasets
    """
    logger.info(f"Starting parallel download of {len(datasets)} datasets using {n_processes} processes")

    with Pool(processes=n_processes) as pool:
        # Create a partial function with fixed arguments
        download_fn = partial(download_dataset_wrapper, save_path, max_retries)

        # Map the download function across all datasets
        results = pool.map(download_fn, datasets)

    successful_downloads = sum(1 for result in results if result)
    logger.info(f"Successfully downloaded {successful_downloads}/{len(datasets)} datasets")

    return successful_downloads


def fetch_all_datasets() -> list[dict[str, Any]]:
    """
    Fetch all available datasets from the CellxGene Discover API.

    Returns
    -------
        List[Dict]: List of dataset dictionaries containing metadata

    Raises
    ------
        requests.RequestException: If API request fails after all retries
    """
    datasets_url = f"{API_BASE_URL}{DATASETS_ENDPOINT}"
    headers = {"Content-Type": "application/json"}

    logger.info("Fetching dataset list from CellxGene Discover API...")

    # Retry configuration
    max_retries = 3
    base_timeout = 60  # Increased from 30 seconds

    for attempt in range(max_retries):
        try:
            timeout = base_timeout * (2**attempt)  # Exponential backoff
            logger.info(f"Attempt {attempt + 1}/{max_retries} with timeout {timeout}s...")

            response = requests.get(url=datasets_url, headers=headers, timeout=timeout)
            response.raise_for_status()
            datasets = response.json()

            logger.info(f"Found {len(datasets)} total datasets")
            return datasets

        except requests.exceptions.Timeout as e:
            logger.warning(f"Timeout on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt == max_retries - 1:
                logger.error("All retry attempts failed due to timeout")
                raise
        except requests.RequestException as e:
            logger.warning(f"Request failed on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt == max_retries - 1:
                logger.error("All retry attempts failed")
                raise


def filter_datasets_by_species(datasets: list[dict[str, Any]], species_list: list[str]) -> list[dict[str, Any]]:
    """
    Filter datasets by organism/species.

    Args:
        datasets: List of dataset dictionaries
        species_list: List of species names to filter by (case-insensitive)

    Returns
    -------
        List[Dict]: Filtered list of datasets matching the specified species
    """
    n_before = len(datasets)

    # Normalize species names for comparison
    normalized_species = [species.lower().strip() for species in species_list]
    logger.info(f"Filtering datasets for species: {normalized_species}")

    filtered_datasets = []
    for dataset in datasets:
        # Skip non-primary datasets
        if not dataset.get("is_primary_data", [False])[0]:
            continue

        # Get organism information
        organism_info = dataset.get("organism", None)
        if not organism_info or len(organism_info) != 1:
            continue

        organism_name = organism_info[0]["label"].lower()
        if organism_name in normalized_species:
            filtered_datasets.append(dataset)

    n_after = len(filtered_datasets)
    logger.info(f"Filtered out {n_before - n_after} datasets, kept {n_after} datasets")

    return filtered_datasets


def save_dataset_metadata(datasets: list[dict[str, Any]], output_path: Path) -> None:
    """
    Save dataset metadata to a JSON file.

    Args:
        datasets: List of dataset dictionaries
        output_path: Path where to save the metadata JSON file
    """
    metadata_file = output_path / "dataset_metadata.json"

    with open(metadata_file, "w") as f:
        json.dump(datasets, f, indent=2)

    logger.info(f"Saved metadata for {len(datasets)} datasets to {metadata_file}")


def download_cellxgene_data(
    species: list[str],
    output_dir: str = "./data/cellxgene",
    n_processes: int = 4,
    max_retries: int = 5,
    save_metadata: bool = True,
) -> int:
    """
    Main function to download CellxGene Discover datasets.

    Args:
        species: List of species names to download (e.g., ["homo sapiens", "mus musculus"])
        output_dir: Directory where datasets will be saved
        n_processes: Number of parallel processes for downloading
        max_retries: Maximum number of retry attempts per dataset
        save_metadata: Whether to save dataset metadata to JSON file

    Returns
    -------
        int: Number of successfully downloaded datasets
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # Fetch all available datasets
        all_datasets = fetch_all_datasets()

        # Save complete metadata if requested
        if save_metadata:
            save_dataset_metadata(all_datasets, output_path)

        # Filter datasets by species
        filtered_datasets = filter_datasets_by_species(all_datasets, species)

        if not filtered_datasets:
            logger.warning("No datasets found matching the specified species")
            return 0

        # Download filtered datasets
        successful_downloads = download_datasets_parallel(
            filtered_datasets, output_path, n_processes=n_processes, max_retries=max_retries
        )

        logger.info(f"Download completed. {successful_downloads} datasets downloaded successfully.")
        return successful_downloads

    except Exception as e:
        logger.error(f"Error during download process: {e}")
        raise


if __name__ == "__main__":
    # Simple command-line interface for testing
    import argparse

    parser = argparse.ArgumentParser(description="Download CellxGene Discover datasets")
    parser.add_argument(
        "--species",
        required=True,
        help="Comma-separated list of species to download (e.g., 'homo sapiens,mus musculus')",
    )
    parser.add_argument("--output-dir", default="./data/cellxgene", help="Output directory for downloaded datasets")
    parser.add_argument("--processes", type=int, default=4, help="Number of parallel processes")
    parser.add_argument("--max-retries", type=int, default=5, help="Maximum retry attempts per dataset")

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Parse species list
    species_list = [s.strip() for s in args.species.split(",")]

    # Download data
    download_cellxgene_data(
        species=species_list, output_dir=args.output_dir, n_processes=args.processes, max_retries=args.max_retries
    )

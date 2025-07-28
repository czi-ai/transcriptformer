"""
CLI command for downloading CellxGene Discover datasets.

This module provides a command-line interface for downloading single-cell RNA sequencing
datasets from the CellxGene Discover portal with support for species filtering,
parallel downloads, and robust error handling.
"""

import logging
import warnings

from transcriptformer.data.bulk_download import download_cellxgene_data, fetch_all_datasets

# Suppress annoying warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="anndata")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*read_.*from.*anndata.*deprecated.*")

# Set up logging
logger = logging.getLogger(__name__)


def test_api_connectivity() -> bool:
    """
    Test connectivity to the CellxGene Discover API.

    Returns
    -------
        bool: True if API is accessible, False otherwise
    """
    try:
        logger.info("Testing connectivity to CellxGene Discover API...")
        datasets = fetch_all_datasets()
        logger.info(f"✅ Successfully connected! Found {len(datasets)} datasets.")
        return True
    except Exception as e:
        logger.error(f"❌ API connectivity test failed: {e}")
        return False


def main(
    species: list[str],
    output_dir: str = "./data/cellxgene",
    n_processes: int = 4,
    max_retries: int = 5,
    save_metadata: bool = True,
    test_only: bool = False,
    **kwargs,
) -> int:
    """
    Main function for the download-data CLI command.

    Args:
        species: List of species names to download
        output_dir: Directory where datasets will be saved
        n_processes: Number of parallel processes for downloading
        max_retries: Maximum number of retry attempts per dataset
        save_metadata: Whether to save dataset metadata to JSON file
        test_only: Only test API connectivity, don't download
        **kwargs: Additional keyword arguments (ignored)

    Returns
    -------
        int: Number of successfully downloaded datasets
    """
    if test_only:
        return 1 if test_api_connectivity() else 0

    logger.info(f"Starting download for species: {', '.join(species)}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Using {n_processes} parallel processes")

    try:
        successful_downloads = download_cellxgene_data(
            species=species,
            output_dir=output_dir,
            n_processes=n_processes,
            max_retries=max_retries,
            save_metadata=save_metadata,
        )

        if successful_downloads > 0:
            logger.info(f"Successfully completed download of {successful_downloads} datasets")
            return successful_downloads
        else:
            logger.warning("No datasets were downloaded")
            return 0

    except Exception as e:
        logger.error(f"Download failed with error: {e}")
        logger.info("💡 Try running with --test-only to check API connectivity")
        raise


if __name__ == "__main__":
    # Simple command-line interface for testing
    import argparse

    parser = argparse.ArgumentParser(description="Download CellxGene Discover datasets")
    parser.add_argument("--species", required=True, help="Comma-separated list of species to download")
    parser.add_argument("--output-dir", default="./data/cellxgene", help="Output directory for downloaded datasets")
    parser.add_argument("--processes", type=int, default=4, help="Number of parallel processes")
    parser.add_argument("--max-retries", type=int, default=5, help="Maximum retry attempts per dataset")
    parser.add_argument("--no-metadata", action="store_true", help="Skip saving dataset metadata")
    parser.add_argument("--test-only", action="store_true", help="Only test API connectivity, don't download")

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Parse species list
    species_list = [s.strip() for s in args.species.split(",")]

    # Run download
    main(
        species=species_list,
        output_dir=args.output_dir,
        n_processes=args.processes,
        max_retries=args.max_retries,
        save_metadata=not args.no_metadata,
        test_only=args.test_only,
    )

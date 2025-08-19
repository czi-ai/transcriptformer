import argparse
import os

import anndata as ad
import numpy as np
from scipy.stats import pearsonr, ttest_rel


def compare_embeddings(file1, file2, tolerance=1e-5):
    """
    Compare embeddings from two AnnData files.

    Parameters
    ----------
    file1 : str
        Path to first AnnData file
    file2 : str
        Path to second AnnData file
    tolerance : float, optional
        Numerical tolerance for considering embeddings identical, by default 1e-5

    Returns
    -------
    bool
        True if embeddings are identical within tolerance, False otherwise
    """
    print(f"Comparing embeddings in {file1} and {file2}")

    # Load AnnData files
    adata1 = ad.read_h5ad(file1)
    adata2 = ad.read_h5ad(file2)

    # Check if embeddings exist
    if "embeddings" not in adata1.obsm or "embeddings" not in adata2.obsm:
        missing = []
        if "embeddings" not in adata1.obsm:
            missing.append(f"'embeddings' not found in {file1}")
        if "embeddings" not in adata2.obsm:
            missing.append(f"'embeddings' not found in {file2}")
        print(f"Error: {', '.join(missing)}")
        return False

    # Get embeddings
    emb1 = adata1.obsm["embeddings"]
    emb2 = adata2.obsm["embeddings"]

    # Check shapes
    if emb1.shape != emb2.shape:
        print(f"Error: Embedding shapes differ: {emb1.shape} vs {emb2.shape}")
        return False

    # Check if embeddings are identical within tolerance
    if np.allclose(emb1, emb2, atol=tolerance):
        print("Embeddings are identical within specified tolerance.")
        return True
    else:
        # Calculate differences
        abs_diff = np.abs(emb1 - emb2)
        max_diff = np.max(abs_diff)
        mean_diff = np.mean(abs_diff)

        # Calculate correlation
        emb1_flat = emb1.flatten()
        emb2_flat = emb2.flatten()
        corr, _ = pearsonr(emb1_flat, emb2_flat)

        # Calculate z-score and p-value for difference test
        # Using paired t-test to test null hypothesis that embeddings are the same
        diff = emb1_flat - emb2_flat
        t_stat, p_value = ttest_rel(emb1_flat, emb2_flat)

        # Calculate z-score manually from the differences
        z_score = np.mean(diff) / (np.std(diff) / np.sqrt(len(diff)))

        print("Embeddings differ:")
        print(f"  Max absolute difference: {max_diff:.6e}")
        print(f"  Mean absolute difference: {mean_diff:.6e}")
        print(f"  Pearson correlation: {corr:.6f}")
        print(f"  Z-score (difference): {z_score:.6f}")
        print(f"  P-value (paired t-test): {p_value:.6e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare embeddings in two AnnData files")
    parser.add_argument("file1", type=str, help="Path to first AnnData file")
    parser.add_argument("file2", type=str, help="Path to second AnnData file")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-5,
        help="Numerical tolerance for considering embeddings identical (default: 1e-5)",
    )

    args = parser.parse_args()

    # Check if files exist
    for file_path in [args.file1, args.file2]:
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            exit(1)

    # Compare embeddings
    identical = compare_embeddings(args.file1, args.file2, args.tolerance)

    # Exit with appropriate status code
    exit(0 if identical else 1)

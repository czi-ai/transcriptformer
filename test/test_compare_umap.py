import argparse
import os
import warnings

import anndata as ad
import numpy as np
from scipy.spatial import procrustes
from scipy.stats import pearsonr


def _require_umap():
    try:
        import umap.umap_ as umap  # type: ignore
    except Exception as exc:  # pragma: no cover - informative error path
        raise RuntimeError(
            "This script requires the 'umap-learn' package. Install with 'pip install umap-learn'."
        ) from exc
    return umap


def compute_umap(
    embeddings: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    random_state: int | None = 42,
) -> np.ndarray:
    """
    Compute a 2D UMAP from input embeddings.

    Parameters
    ----------
    embeddings : np.ndarray
        2D array of shape (n_samples, n_features)
    n_neighbors : int
        UMAP n_neighbors parameter
    min_dist : float
        UMAP min_dist parameter
    metric : str
        Distance metric for UMAP
    random_state : int | None
        Random seed for reproducibility

    Returns
    -------
    np.ndarray
        2D array of shape (n_samples, 2) with UMAP coordinates
    """
    umap = _require_umap()

    # Suppress noisy warnings sometimes emitted by numba/umap during fit
    warnings.filterwarnings("ignore", message=".*cannot cache compiled function.*")
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        n_components=2,
        random_state=random_state,
    )
    return reducer.fit_transform(embeddings)


def compare_umaps(
    file1: str,
    file2: str,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    random_state: int | None = 42,
    procrustes_tolerance: float = 1e-2,
    save_plot: str | None = None,
    obs_key: str | None = "cell_type",
) -> bool:
    """
    Load embeddings from two AnnData files, compute UMAPs independently, and compare via Procrustes.

    Returns True if the Procrustes disparity <= procrustes_tolerance.
    """
    # Load AnnData files
    adata1 = ad.read_h5ad(file1)
    adata2 = ad.read_h5ad(file2)

    if "embeddings" not in adata1.obsm or "embeddings" not in adata2.obsm:
        missing = []
        if "embeddings" not in adata1.obsm:
            missing.append(f"'embeddings' not found in {file1}")
        if "embeddings" not in adata2.obsm:
            missing.append(f"'embeddings' not found in {file2}")
        raise ValueError("; ".join(missing))

    emb1 = np.asarray(adata1.obsm["embeddings"])  # (n, d)
    emb2 = np.asarray(adata2.obsm["embeddings"])  # (n, d)

    if emb1.shape[0] != emb2.shape[0]:
        raise ValueError(f"Number of rows differ between embeddings: {emb1.shape[0]} vs {emb2.shape[0]}")

    # Compute UMAPs independently
    umap1 = compute_umap(emb1, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=random_state)
    umap2 = compute_umap(emb2, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=random_state)

    # Procrustes analysis aligns scale/rotation/translation; returns disparity (lower is better)
    mtx1, mtx2, disparity = procrustes(umap1, umap2)

    # Simple additional similarity: correlation of flattened coordinates after alignment
    corr, _ = pearsonr(mtx1.ravel(), mtx2.ravel())

    print("UMAP comparison:")
    print(f"  Procrustes disparity: {disparity:.6e}")
    print(f"  Pearson correlation (aligned): {corr:.6f}")

    if save_plot is not None:
        try:
            import matplotlib.pyplot as plt  # Lazy import for optional plotting

            # Prepare colors using a consistent palette across both datasets
            labels1 = None
            labels2 = None
            color_map = None
            if obs_key is not None:
                try:
                    series1 = adata1.obs[obs_key]
                    series2 = adata2.obs[obs_key]
                    labels1 = series1.astype(str).to_numpy()
                    labels2 = series2.astype(str).to_numpy()
                    categories = sorted(set(labels1).union(set(labels2)))

                    import matplotlib as mpl

                    if len(categories) <= 20:
                        cmap = mpl.cm.get_cmap("tab20", len(categories))
                        palette = [mpl.colors.to_hex(cmap(i)) for i in range(len(categories))]
                    else:
                        # fallback palette for many categories
                        cmap = mpl.cm.get_cmap("hsv", len(categories))
                        palette = [mpl.colors.to_hex(cmap(i)) for i in range(len(categories))]
                    color_map = {cat: palette[i] for i, cat in enumerate(categories)}
                except Exception as e:
                    print(f"Warning: could not color by '{obs_key}': {e}")

            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            if labels1 is not None and color_map is not None:
                colors1 = [color_map.get(lbl, "#000000") for lbl in labels1]
                axes[0].scatter(mtx1[:, 0], mtx1[:, 1], s=3, alpha=0.6, c=colors1)
            else:
                axes[0].scatter(mtx1[:, 0], mtx1[:, 1], s=3, alpha=0.6)
            axes[0].set_title("UMAP 1 (aligned)")
            axes[0].set_xticks([])
            axes[0].set_yticks([])

            if labels2 is not None and color_map is not None:
                colors2 = [color_map.get(lbl, "#000000") for lbl in labels2]
                axes[1].scatter(mtx2[:, 0], mtx2[:, 1], s=3, alpha=0.6, c=colors2)
            else:
                axes[1].scatter(mtx2[:, 0], mtx2[:, 1], s=3, alpha=0.6, color="tab:orange")
            axes[1].set_title("UMAP 2 (aligned)")
            axes[1].set_xticks([])
            axes[1].set_yticks([])

            # Optional legend if number of categories is reasonable
            if color_map is not None and len(color_map) > 0 and len(color_map) <= 20:
                import matplotlib.patches as mpatches

                handles = [mpatches.Patch(color=clr, label=cat) for cat, clr in color_map.items()]
                fig.legend(handles=handles, loc="lower center", ncol=min(5, len(handles)), frameon=False)

            plt.tight_layout()
            out_dir = os.path.dirname(save_plot)
            if out_dir and not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)
            plt.savefig(save_plot, dpi=150)
            plt.close(fig)
            print(f"Saved comparison plot to {save_plot}")
        except Exception as e:  # pragma: no cover - plotting not essential in tests
            print(f"Warning: failed to save plot: {e}")

    # Pass criterion
    return disparity <= procrustes_tolerance


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute UMAPs from two AnnData files' obsm['embeddings'] independently and compare them via Procrustes."
        )
    )
    parser.add_argument("file1", type=str, help="Path to first AnnData .h5ad file")
    parser.add_argument("file2", type=str, help="Path to second AnnData .h5ad file")
    parser.add_argument("--n-neighbors", type=int, default=15, help="UMAP n_neighbors (default: 15)")
    parser.add_argument("--min-dist", type=float, default=0.1, help="UMAP min_dist (default: 0.1)")
    parser.add_argument("--metric", type=str, default="euclidean", help="UMAP metric (default: euclidean)")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for UMAP (default: 42)")
    parser.add_argument(
        "--procrustes-tolerance",
        type=float,
        default=1e-2,
        help="Maximum acceptable Procrustes disparity to consider UMAPs similar (default: 1e-2)",
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default="./umap_comparison.png",
        help="Optional path to save a side-by-side aligned UMAP comparison plot (PNG)",
    )
    parser.add_argument(
        "--obs-key",
        type=str,
        default="cell_type",
        help="Column in .obs to color points by (default: cell_type). Use 'none' to disable.",
    )

    args = parser.parse_args()

    # Existence check
    for fp in [args.file1, args.file2]:
        if not os.path.exists(fp):
            print(f"Error: File not found: {fp}")
            raise SystemExit(1)

    ok = compare_umaps(
        args.file1,
        args.file2,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        random_state=args.random_state,
        procrustes_tolerance=args.procrustes_tolerance,
        save_plot=args.save_plot,
        obs_key=(None if (args.obs_key is None or str(args.obs_key).lower() == "none") else args.obs_key),
    )

    if ok:
        print("UMAPs are similar within tolerance.")
        raise SystemExit(0)
    else:
        print("UMAPs differ beyond tolerance.")
        raise SystemExit(2)


if __name__ == "__main__":
    main()

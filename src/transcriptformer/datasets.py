import os
from typing import Any, Literal

import anndata as ad
from scanpy.readwrite import _check_datafile_present_and_download

PathLike = os.PathLike | str


def lymphnode_tabula_sapiens(
    path: PathLike = "~/.cache/transcriptformer/lymphnode_tsv2.h5ad",
    force_download: bool = False,
    version: Literal["v1", "v2"] = "v2",
    **kwargs: Any,
) -> ad.AnnData:
    """
    Lymph node dataset from Tabula Sapiens.

    Args:
        path: Path to save the dataset.
        force_download: Whether to force download the dataset.
        version: Version of the dataset to load. `v1` is in-distribution for Transcriptformer, `v2` is out-of-distribution.

    Returns
    -------
        AnnData object.
    """
    adata = _load_dataset_from_url(
        path,
        file_type="h5ad",
        backup_url="https://datasets.cellxgene.cziscience.com/71021707-8e1f-4871-abe4-8f645f2cdb9e.h5ad",
        force_download=force_download,
        **kwargs,
    )

    adata = filter_anndata_by_tissue_and_version(adata, version=version)

    return adata


def heart_tabula_sapiens(
    path: PathLike = "~/.cache/transcriptformer/heart_tsv2.h5ad",
    force_download: bool = False,
    version: Literal["v1", "v2"] = "v2",
    **kwargs: Any,
) -> ad.AnnData:
    """
    Heart dataset from Tabula Sapiens.

    Args:
        path: Path to save the dataset.
        force_download: Whether to force download the dataset.
        version: Version of the dataset to load. `v1` is in-distribution for Transcriptformer, `v2` is out-of-distribution.

    Returns
    -------
        AnnData object.
    """
    adata = _load_dataset_from_url(
        path,
        file_type="h5ad",
        backup_url="https://datasets.cellxgene.cziscience.com/97516b79-8d08-46a6-b329-5d0a25b0be98.h5ad",
        force_download=force_download,
        **kwargs,
    )

    adata = filter_anndata_by_tissue_and_version(adata, version=version)

    return adata


def ear_tabula_sapiens(
    path: PathLike = "~/.cache/transcriptformer/ear_tsv2.h5ad",
    force_download: bool = False,
    version: Literal["v1", "v2"] = "v2",
    **kwargs: Any,
) -> ad.AnnData:
    """
    Ear dataset from Tabula Sapiens.

    Args:
        path: Path to save the dataset.
        force_download: Whether to force download the dataset.
        version: Version of the dataset to load. `v1` is in-distribution for Transcriptformer, `v2` is out-of-distribution.

    Returns
    -------
        AnnData object.
    """
    adata = _load_dataset_from_url(
        path,
        file_type="h5ad",
        backup_url="https://datasets.cellxgene.cziscience.com/ffa57bc0-78ca-4aa4-be6b-adc40b2f5214.h5ad",
        force_download=force_download,
        **kwargs,
    )

    adata = filter_anndata_by_tissue_and_version(adata, version=version)

    return adata


def _load_dataset_from_url(
    fpath: PathLike,
    file_type: Literal["h5ad"],
    *,
    backup_url: str,
    force_download: bool = False,
    **kwargs: Any,
) -> ad.AnnData:
    fpath = os.path.expanduser(fpath)
    assert file_type in ["h5ad"], f"Invalid type `{file_type}`. Must be one of `['h5ad']`."
    if not fpath.endswith(file_type):
        fpath += f".{file_type}"
    if force_download and os.path.exists(fpath):
        os.remove(fpath)
    if not _check_datafile_present_and_download(backup_url=backup_url, path=fpath):
        raise FileNotFoundError(f"File `{fpath}` not found or download failed.")
    data = ad.read_h5ad(filename=fpath, **kwargs)

    return data


def filter_anndata_by_tissue_and_version(
    adata: ad.AnnData,
    version: Literal["v1", "v2"],
    min_gene_counts: int = 10,
) -> ad.AnnData:
    if version == "v1":
        mask = adata.obs["donor_id"].str.split("TSP").str[-1].astype(int) < 16
    elif version == "v2":
        mask = adata.obs["donor_id"].str.split("TSP").str[-1].astype(int) > 16
    else:
        raise ValueError(f"Invalid version: {version}. Must be one of ['v1', 'v2']")

    adata_filtered = adata[mask].copy()

    gene_sums = adata_filtered.X.sum(axis=0)
    genes_to_keep = gene_sums >= min_gene_counts
    adata_filtered = adata_filtered[:, genes_to_keep].copy()

    return adata_filtered


if __name__ == "__main__":
    lymphnode_tabula_sapiens()

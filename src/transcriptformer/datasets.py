import os
from typing import Any, Literal

import anndata as ad
from scanpy.readwrite import _check_datafile_present_and_download

PathLike = os.PathLike | str


def lymphnode_tsv2(
    path: PathLike = "~/.cache/transcriptformer/lymphnode_tsv2.h5ad",
    force_download: bool = False,
    **kwargs: Any,
) -> ad.AnnData:
    """Lymph node dataset from Tabula Sapiens v2."""
    adata = _load_dataset_from_url(
        path,
        file_type="h5ad",
        backup_url="https://datasets.cellxgene.cziscience.com/71021707-8e1f-4871-abe4-8f645f2cdb9e.h5ad",
        force_download=force_download,
        **kwargs,
    )

    print(adata)

    # if adata.shape != expected_shape:
    #     raise ValueError(f"Expected AnnData object to have shape `{expected_shape}`, found `{adata.shape}`.")

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


if __name__ == "__main__":
    lymphnode_tsv2()

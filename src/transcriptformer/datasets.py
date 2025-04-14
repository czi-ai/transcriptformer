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
    return _load_dataset_from_url(
        path,
        file_type="h5ad",
        backup_url="https://figshare.com/ndownloader/files/40569779",
        expected_shape=(54134, 2000),
        force_download=force_download,
        **kwargs,
    )


def _load_dataset_from_url(
    fpath: PathLike,
    file_type: Literal["h5ad"],
    *,
    backup_url: str,
    expected_shape: tuple[int, int],
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

    if data.shape != expected_shape:
        raise ValueError(f"Expected AnnData object to have shape `{expected_shape}`, found `{data.shape}`.")

    return data

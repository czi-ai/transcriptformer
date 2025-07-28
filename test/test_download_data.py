"""Tests for the download-data functionality."""

from unittest.mock import MagicMock, patch

from transcriptformer.data.bulk_download import (
    fetch_all_datasets,
    filter_datasets_by_species,
    save_dataset_metadata,
)


def test_filter_datasets_by_species():
    """Test filtering datasets by species."""
    mock_datasets = [
        {
            "dataset_id": "1",
            "is_primary_data": [True],
            "organism": [{"label": "Homo sapiens"}],
        },
        {
            "dataset_id": "2",
            "is_primary_data": [True],
            "organism": [{"label": "Mus musculus"}],
        },
        {
            "dataset_id": "3",
            "is_primary_data": [False],  # Not primary
            "organism": [{"label": "Homo sapiens"}],
        },
        {
            "dataset_id": "4",
            "is_primary_data": [True],
            "organism": [{"label": "Danio rerio"}],
        },
    ]

    # Test filtering for human only
    human_datasets = filter_datasets_by_species(mock_datasets, ["homo sapiens"])
    assert len(human_datasets) == 1
    assert human_datasets[0]["dataset_id"] == "1"

    # Test filtering for human and mouse
    multi_species = filter_datasets_by_species(mock_datasets, ["homo sapiens", "mus musculus"])
    assert len(multi_species) == 2
    dataset_ids = {d["dataset_id"] for d in multi_species}
    assert dataset_ids == {"1", "2"}


def test_save_dataset_metadata(tmp_path):
    """Test saving dataset metadata to JSON."""
    mock_datasets = [
        {"dataset_id": "1", "title": "Test Dataset 1"},
        {"dataset_id": "2", "title": "Test Dataset 2"},
    ]

    save_dataset_metadata(mock_datasets, tmp_path)

    metadata_file = tmp_path / "dataset_metadata.json"
    assert metadata_file.exists()

    import json

    with open(metadata_file) as f:
        loaded_data = json.load(f)

    assert len(loaded_data) == 2
    assert loaded_data[0]["dataset_id"] == "1"


@patch("transcriptformer.data.bulk_download.requests.get")
def test_fetch_all_datasets(mock_get):
    """Test fetching datasets from API."""
    mock_response = MagicMock()
    mock_response.json.return_value = [{"dataset_id": "1", "title": "Test Dataset"}]
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    datasets = fetch_all_datasets()

    assert len(datasets) == 1
    assert datasets[0]["dataset_id"] == "1"
    mock_get.assert_called_once()


def test_cli_import():
    """Test that CLI modules can be imported without errors."""
    from transcriptformer.cli.download_data import main as download_data_main
    from transcriptformer.data.bulk_download import download_cellxgene_data

    # Basic import test - functions should be callable
    assert callable(download_data_main)
    assert callable(download_cellxgene_data)

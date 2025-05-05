"""Tests for the download_artifacts module."""

import sys
from unittest import mock

import pytest

from transcriptformer.cli.download_artifacts import (
    download_and_extract,
    main,
    print_progress,
)


class TestDownloadArtifacts:
    """Tests for the download_artifacts module."""

    def test_print_progress(self, capsys):
        """Test the print_progress function."""
        # Test with various progress values
        print_progress(50, 100, prefix="Test", suffix="Done", length=20)
        out, _ = capsys.readouterr()
        assert "Test |██████████░░░░░░░░░░| 50% Done" in out

        print_progress(100, 100, prefix="Test", suffix="Done", length=20)
        out, _ = capsys.readouterr()
        assert "Test |████████████████████| 100% Done" in out

    @mock.patch("urllib.request.urlretrieve")
    @mock.patch("tarfile.open")
    def test_download_and_extract(self, mock_tarfile_open, mock_urlretrieve, tmp_path):
        """Test the download_and_extract function."""
        # Setup mocks
        mock_tar = mock.MagicMock()
        mock_tarfile_open.return_value.__enter__.return_value = mock_tar
        mock_tar.getmembers.return_value = [mock.MagicMock() for _ in range(3)]

        # Test with single model
        model_name = "tf_sapiens"
        output_dir = tmp_path / "checkpoints"

        download_and_extract(model_name, str(output_dir))

        # Check URL was correct
        expected_url = f"https://czi-transcriptformer.s3.amazonaws.com/weights/{model_name}.tar.gz"
        mock_urlretrieve.assert_called_once()
        call_args = mock_urlretrieve.call_args[0]
        assert call_args[0] == expected_url

        # Check tarfile was extracted
        mock_tarfile_open.assert_called_once()
        mock_tar.extract.assert_called()
        assert mock_tar.extract.call_count == 3  # 3 mock members

    @mock.patch("urllib.request.urlretrieve")
    def test_download_and_extract_http_error(self, mock_urlretrieve, capsys):
        """Test download_and_extract handling of HTTP errors."""
        from urllib.error import HTTPError

        # Mock HTTP 404 error
        mock_urlretrieve.side_effect = HTTPError("url", 404, "Not Found", {}, None)

        with pytest.raises(SystemExit) as exc_info:
            download_and_extract("not_a_real_model")

        assert exc_info.value.code == 1
        out, _ = capsys.readouterr()
        assert "Error: The model not_a_real_model was not found" in out

    @mock.patch("transcriptformer.cli.download_artifacts.download_and_extract")
    def test_main(self, mock_download_extract, monkeypatch):
        """Test the main function."""
        # Test with a specific model
        monkeypatch.setattr(sys, "argv", ["download_artifacts.py", "tf-sapiens"])
        main()
        mock_download_extract.assert_called_once_with("tf_sapiens", "./checkpoints")

        # Test with "all" option
        mock_download_extract.reset_mock()
        monkeypatch.setattr(sys, "argv", ["download_artifacts.py", "all"])
        main()
        assert mock_download_extract.call_count == 4

        # Test with custom checkpoint directory
        mock_download_extract.reset_mock()
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "download_artifacts.py",
                "tf-exemplar",
                "--checkpoint-dir",
                "/custom/path",
            ],
        )
        main()
        mock_download_extract.assert_called_once_with("tf_exemplar", "/custom/path")


class TestDownloadArtifactsIntegration:
    """Integration tests for the download_artifacts module."""

    @pytest.mark.skip(reason="This test would download real files, only run manually")
    def test_download_extract_integration(self, tmp_path):
        """Real integration test that downloads and extracts files.

        This test is skipped by default as it would download real files.
        """
        model_name = "tf_sapiens"
        checkpoint_dir = tmp_path / "checkpoints"

        # Run the actual download and extract
        download_and_extract(model_name, str(checkpoint_dir))

        # Check that files were downloaded and extracted
        model_dir = checkpoint_dir / model_name
        assert model_dir.exists()
        assert (model_dir / "model_weights.pt").exists()
        assert (model_dir / "config.json").exists()
        assert (model_dir / "vocabs").exists()

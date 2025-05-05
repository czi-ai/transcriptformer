"""Integration tests for the TranscriptFormer CLI."""

import subprocess
import sys
from unittest import mock

import pytest


@pytest.mark.skip(reason="Integration test requiring actual model execution")
class TestCLIIntegration:
    """Integration tests for the CLI that run the actual commands."""

    def test_help_command(self):
        """Test that the help command works."""
        result = subprocess.run(
            ["transcriptformer", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "TranscriptFormer command-line interface" in result.stdout
        assert "inference" in result.stdout
        assert "download" in result.stdout

    def test_download_help(self):
        """Test that the download help command works."""
        result = subprocess.run(
            ["transcriptformer", "download", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Download and extract TranscriptFormer model artifacts" in result.stdout

    def test_inference_help(self):
        """Test that the inference help command works."""
        result = subprocess.run(
            ["transcriptformer", "inference", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Run inference with a TranscriptFormer model" in result.stdout


class TestCLIMockedIntegration:
    """Integration tests for the CLI that mock the actual execution."""

    @mock.patch("transcriptformer.cli.run_inference_cli")
    def test_inference_command(self, mock_run_inference):
        """Test that the inference command runs with appropriate arguments."""
        # Create a subprocess with a mocked entry point that just logs the arguments
        with mock.patch.object(
            sys,
            "argv",
            [
                "transcriptformer",
                "inference",
                "--checkpoint-path",
                "/path/to/checkpoint",
                "--data-file",
                "/path/to/data.h5ad",
                "--output-path",
                "/path/to/output",
                "--batch-size",
                "16",
                "--gene-col-name",
                "gene_symbol",
            ],
        ):
            # Import and run the main function
            from transcriptformer.cli import main

            main()

        # Check that run_inference_cli was called
        mock_run_inference.assert_called_once()

        # Check that args were passed correctly
        args = mock_run_inference.call_args[0][0]
        assert args.checkpoint_path == "/path/to/checkpoint"
        assert args.data_file == "/path/to/data.h5ad"
        assert args.output_path == "/path/to/output"
        assert args.batch_size == 16
        assert args.gene_col_name == "gene_symbol"

    @mock.patch("transcriptformer.cli.run_download_cli")
    def test_download_command(self, mock_run_download):
        """Test that the download command runs with appropriate arguments."""
        # Create a subprocess with a mocked entry point that just logs the arguments
        with mock.patch.object(
            sys,
            "argv",
            [
                "transcriptformer",
                "download",
                "tf-sapiens",
                "--checkpoint-dir",
                "/path/to/checkpoints",
            ],
        ):
            # Import and run the main function
            from transcriptformer.cli import main

            main()

        # Check that run_download_cli was called
        mock_run_download.assert_called_once()

        # Check that args were passed correctly
        args = mock_run_download.call_args[0][0]
        assert args.model == "tf-sapiens"
        assert args.checkpoint_dir == "/path/to/checkpoints"

    @mock.patch("transcriptformer.cli.run_download_cli")
    def test_download_all_command(self, mock_run_download):
        """Test that the download all command works."""
        with mock.patch.object(sys, "argv", ["transcriptformer", "download", "all"]):
            # Import and run the main function
            from transcriptformer.cli import main

            main()

        # Check that run_download_cli was called
        mock_run_download.assert_called_once()

        # Check that args were passed correctly
        args = mock_run_download.call_args[0][0]
        assert args.model == "all"
        assert args.checkpoint_dir == "./checkpoints"  # default

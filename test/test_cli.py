"""Tests for the TranscriptFormer CLI module."""

import sys
from unittest import mock

import pytest

from transcriptformer.cli import (
    main,
    run_download_cli,
    run_inference_cli,
    setup_download_parser,
    setup_inference_parser,
)


class TestCLIMain:
    """Tests for the main CLI entry point."""

    def test_main_no_args(self, monkeypatch, capsys):
        """Test CLI with no arguments prints help and exits."""
        # Mock sys.argv and sys.exit
        monkeypatch.setattr(sys, "argv", ["transcriptformer"])
        with mock.patch("sys.exit") as mock_exit:
            main()
            mock_exit.assert_called_once_with(1)

        # Check help was printed
        captured = capsys.readouterr()
        assert "usage: " in captured.out
        assert "TranscriptFormer command-line interface" in captured.out

    def test_main_help(self, monkeypatch, capsys):
        """Test CLI with --help argument prints help."""
        monkeypatch.setattr(sys, "argv", ["transcriptformer", "--help"])
        with pytest.raises(SystemExit):
            main()

        captured = capsys.readouterr()
        assert "usage: " in captured.out
        assert "TranscriptFormer command-line interface" in captured.out


class TestInferenceCommand:
    """Tests for the inference command."""

    @mock.patch("transcriptformer.cli.run_inference_cli")
    def test_inference_command(self, mock_run_inference, monkeypatch):
        """Test that inference command runs with required arguments."""
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "transcriptformer",
                "inference",
                "--checkpoint-path",
                "/path/to/checkpoint",
                "--data-file",
                "/path/to/data.h5ad",
            ],
        )

        main()
        mock_run_inference.assert_called_once()

    @mock.patch("transcriptformer.cli.inference.main")
    def test_run_inference_cli(self, mock_inference_main, monkeypatch):
        """Test run_inference_cli function properly calls inference.main."""
        args = mock.MagicMock()
        args.checkpoint_path = "/path/to/checkpoint"
        args.data_file = "/path/to/data.h5ad"
        args.output_path = "./inference_results"
        args.output_filename = "embeddings.h5ad"
        args.batch_size = 8
        args.gene_col_name = "ensembl_id"
        args.precision = "16-mixed"
        args.pretrained_embedding = None
        args.config_override = []

        # Test that the function properly sets up Hydra config
        original_argv = sys.argv.copy()
        run_inference_cli(args)
        mock_inference_main.assert_called_once()
        # Check sys.argv was restored
        assert sys.argv == original_argv


class TestDownloadCommand:
    """Tests for the download command."""

    @mock.patch("transcriptformer.cli.run_download_cli")
    def test_download_command(self, mock_run_download, monkeypatch):
        """Test that download command runs with required arguments."""
        monkeypatch.setattr(sys, "argv", ["transcriptformer", "download", "tf-sapiens"])

        main()
        mock_run_download.assert_called_once()

    @mock.patch("transcriptformer.cli.download_artifacts.download_and_extract")
    def test_run_download_cli_single_model(self, mock_download, monkeypatch):
        """Test run_download_cli with a single model."""
        args = mock.MagicMock()
        args.model = "tf-sapiens"
        args.checkpoint_dir = "./checkpoints"

        run_download_cli(args)
        mock_download.assert_called_once_with("tf_sapiens", "./checkpoints")

    @mock.patch("transcriptformer.cli.download_artifacts.download_and_extract")
    def test_run_download_cli_all_models(self, mock_download, monkeypatch):
        """Test run_download_cli with 'all' option."""
        args = mock.MagicMock()
        args.model = "all"
        args.checkpoint_dir = "./checkpoints"

        run_download_cli(args)
        assert mock_download.call_count == 4

        # Check all models were downloaded
        mock_download.assert_any_call("tf_sapiens", "./checkpoints")
        mock_download.assert_any_call("tf_exemplar", "./checkpoints")
        mock_download.assert_any_call("tf_metazoa", "./checkpoints")
        mock_download.assert_any_call("all_embeddings", "./checkpoints")


class TestCLIParsers:
    """Tests for CLI parsers setup."""

    def test_inference_parser_setup(self):
        """Test that inference parser is set up correctly."""
        parser = mock.MagicMock()
        subparsers = mock.MagicMock()
        subparsers.add_parser.return_value = parser

        setup_inference_parser(subparsers)

        # Check required arguments are added
        subparsers.add_parser.assert_called_once_with(
            "inference",
            help="Run inference with a TranscriptFormer model",
            description="Run inference with a TranscriptFormer model on scRNA-seq data.",
        )

        # Check that required arguments are added with required=True
        parser.add_argument.assert_any_call(
            "--checkpoint-path",
            required=True,
            help="Path to the model checkpoint directory",
        )
        parser.add_argument.assert_any_call(
            "--data-file",
            required=True,
            help="Path to input AnnData file to run inference on",
        )

    def test_download_parser_setup(self):
        """Test that download parser is set up correctly."""
        parser = mock.MagicMock()
        subparsers = mock.MagicMock()
        subparsers.add_parser.return_value = parser

        setup_download_parser(subparsers)

        # Check parser is created
        subparsers.add_parser.assert_called_once_with(
            "download",
            help="Download and extract TranscriptFormer model artifacts",
            description="Download and extract TranscriptFormer model artifacts from a public S3 bucket.",
        )

        # Check that model argument is added with choices
        parser.add_argument.assert_any_call(
            "model",
            choices=[
                "tf-sapiens",
                "tf-exemplar",
                "tf-metazoa",
                "all",
                "all-embeddings",
            ],
            help="Model to download ('all' for all models and embeddings, 'all-embeddings' for just embeddings)",
        )

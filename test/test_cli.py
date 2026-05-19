"""Tests for the TranscriptFormer CLI module."""

import sys
from unittest import mock

import pytest

from transcriptformer.cli import (
    main,
    run_train_cli,
    setup_inference_parser,
    setup_train_parser,
)


class TestCLIMain:
    """Tests for the main CLI entry point."""

    def test_main_no_args(self, monkeypatch, capsys):
        """Test CLI with no arguments prints help and exits."""
        monkeypatch.setattr(sys, "argv", ["transcriptformer"])
        with mock.patch("sys.exit") as mock_exit:
            main()
            mock_exit.assert_called_once_with(1)

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


class TestTrainCommand:
    """Tests for the train command."""

    @mock.patch("transcriptformer.cli.run_train_cli")
    def test_train_command(self, mock_run_train, monkeypatch):
        """Test that train command runs with required arguments."""
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "transcriptformer",
                "train",
                "--checkpoint-dir",
                "/path/to/checkpoint",
                "--output-dir",
                "/path/to/output",
                "--train-file",
                "/path/to/train.h5ad",
            ],
        )

        main()
        mock_run_train.assert_called_once()

    @mock.patch("transcriptformer.cli.run_train_from_dict")
    @mock.patch("transcriptformer.cli.setup_runtime_for_training")
    def test_run_train_cli(self, mock_setup_runtime, mock_run_train_from_dict):
        """Test run_train_cli parameter mapping."""
        args = mock.MagicMock()
        args.checkpoint_dir = "/path/to/checkpoint"
        args.resume_artifact_dir = None
        args.resume_mode = "weights"
        args.output_dir = "/path/to/output"
        args.train_file = ["/path/to/train.h5ad"]
        args.val_file = []
        args.expanded_assay_vocab = None
        args.obs_assay_col = "assay"
        args.gene_col_name = "ensembl_id"
        args.filter_to_vocabs = True
        args.filter_outliers = 0.0
        args.sort_genes = False
        args.randomize_genes = False
        args.min_expressed_genes = 0
        args.n_data_workers = 4
        args.batch_size = 2
        args.num_workers = 0
        args.max_epochs = 1
        args.precision = "32"
        args.devices = "1"
        args.num_nodes = 1
        args.accelerator = "cpu"
        args.lr = 1e-4
        args.weight_decay = 0.0
        args.adam_beta1 = 0.9
        args.adam_beta2 = 0.95
        args.adam_eps = 1e-8
        args.warmup_ratio = 0.1
        args.min_lr_ratio = 0.1
        args.gene_id_loss_weight = 1.0
        args.softplus_approx = True
        args.init_default_source = "unknown"
        args.assay_init_map = []
        args.freeze_transformer = False
        args.freeze_gene_embeddings = False
        args.freeze_count_head = False
        args.freeze_gene_head = False
        args.train_aux_only = False
        args.shuffle_expressed_each_batch = False
        args.clip_counts = 30.0
        args.normalize_to_scale = 0.0
        args.use_raw = False
        args.remove_duplicate_genes = False
        args.use_oom_dataloader = False
        args.seed = 42

        # Mock return value
        mock_run_train_from_dict.return_value = {"output_dir": "/path/to/output"}

        # Call the function
        run_train_cli(args)

        # Verify setup was called
        mock_setup_runtime.assert_called_once()
        
        # Verify run_train_from_dict was called with correct config
        mock_run_train_from_dict.assert_called_once()
        call_args = mock_run_train_from_dict.call_args[0][0]
        assert call_args["checkpoint_dir"] == "/path/to/checkpoint"
        assert call_args["output_dir"] == "/path/to/output"
        assert call_args["train_files"] == ["/path/to/train.h5ad"]
        assert call_args["data_config"]["gene_col_name"] == "ensembl_id"
        assert call_args["loss_config"]["gene_id_loss_weight"] == 1.0


class TestCLIParsers:
    """Tests for CLI parsers setup."""

    def test_inference_parser_setup(self):
        """Test that inference parser is set up correctly."""
        parser = mock.MagicMock()
        subparsers = mock.MagicMock()
        subparsers.add_parser.return_value = parser

        setup_inference_parser(subparsers)

        subparsers.add_parser.assert_called_once_with(
            "inference",
            help="Run inference with a TranscriptFormer model",
            description="Run inference with a TranscriptFormer model on scRNA-seq data.",
        )

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

        parser.add_argument.assert_any_call(
            "--emb-type",
            default="cell",
            choices=["cell", "cge"],
            help="Type of embeddings to extract: 'cell' for mean-pooled cell embeddings or 'cge' for contextual gene embeddings (default: cell)",
        )

    def test_train_parser_setup(self):
        """Test that train parser is set up correctly."""
        parser = mock.MagicMock()
        subparsers = mock.MagicMock()
        subparsers.add_parser.return_value = parser

        setup_train_parser(subparsers)

        subparsers.add_parser.assert_called_once_with(
            "train",
            help="Train or continue training with expanded assay vocab",
            description="Fine-tune TranscriptFormer with expanded assay tokens and optional freezing.",
        )

        parser.add_argument.assert_any_call(
            "--checkpoint-dir",
            required=True,
            help="Base artifact directory with config.json/model_weights.pt",
        )
        parser.add_argument.assert_any_call(
            "--output-dir",
            required=True,
            help="Output artifact directory",
        )
        parser.add_argument.assert_any_call(
            "--train-file",
            action="append",
            required=True,
            help="Training .h5ad file (repeatable)",
        )

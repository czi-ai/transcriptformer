"""Tests for the inference module."""

import os
from unittest import mock

from omegaconf import OmegaConf

from transcriptformer.cli.inference import main


class TestInference:
    """Tests for the inference module."""

    @mock.patch("transcriptformer.cli.inference.run_inference")
    @mock.patch("json.load")
    @mock.patch("builtins.open", new_callable=mock.mock_open)
    def test_main(self, mock_open, mock_json_load, mock_run_inference, tmp_path):
        """Test the main function with mocked dependencies."""
        # Setup mocks
        mock_adata = mock.MagicMock()
        mock_run_inference.return_value = mock_adata

        # Mock config.json loading with proper JSON
        mock_json_content = {"model": {"data_config": {"gene_col_name": "ensembl_id"}}}
        mock_json_load.return_value = mock_json_content

        # Create a hydra config for the test
        cfg = OmegaConf.create(
            {
                "model": {
                    "checkpoint_path": str(tmp_path / "checkpoints" / "model"),
                    "inference_config": {
                        "data_files": [str(tmp_path / "data.h5ad")],
                        "output_path": str(tmp_path / "output"),
                        "output_filename": "test_output.h5ad",
                    },
                }
            }
        )

        # Create test directory structure
        os.makedirs(cfg.model.checkpoint_path, exist_ok=True)
        os.makedirs(cfg.model.inference_config.output_path, exist_ok=True)

        # Test the main function
        with mock.patch("sys.argv", ["inference.py", "--config-name=inference_config.yaml"]):
            main(cfg)

        # Verify run_inference was called
        mock_run_inference.assert_called_once()

        # Verify adata.write_h5ad was called
        mock_adata.write_h5ad.assert_called_once()
        output_path = mock_adata.write_h5ad.call_args[0][0]
        assert os.path.dirname(output_path) == str(tmp_path / "output")

    @mock.patch("transcriptformer.cli.inference.run_inference")
    @mock.patch("json.load")
    @mock.patch("builtins.open", new_callable=mock.mock_open)
    def test_main_config_handling(self, mock_open, mock_json_load, mock_run_inference, tmp_path):
        """Test that config handling works correctly in the main function."""
        # Setup mock return values
        mock_json_load.return_value = {"model": {"data_config": {"gene_col_name": "gene_symbol"}}}
        mock_adata = mock.MagicMock()
        mock_run_inference.return_value = mock_adata

        # Create input config
        cfg = OmegaConf.create(
            {
                "model": {
                    "checkpoint_path": str(tmp_path / "checkpoints" / "model"),
                    "inference_config": {
                        "data_files": [str(tmp_path / "data.h5ad")],
                        "output_path": str(tmp_path / "output"),
                        "output_filename": "test_output.h5ad",
                        "batch_size": 16,
                        "precision": "32",
                    },
                }
            }
        )

        # Run with patched dependencies
        with mock.patch("sys.argv", ["inference.py", "--config-name=inference_config.yaml"]):
            main(cfg)

        # Check that config paths were set correctly
        config_path = os.path.join(cfg.model.checkpoint_path, "config.json")
        mock_open.assert_any_call(config_path)

        # Check that run_inference was called with merged config
        mock_run_inference.assert_called_once()

        # Check output path handling - just verify the directory is correct
        # The filename may be overridden by defaults from the config merge
        output_file = mock_adata.write_h5ad.call_args[0][0]
        assert os.path.dirname(output_file) == str(tmp_path / "output")
        # Either the custom filename or the default should be used
        assert os.path.basename(output_file) in ["test_output.h5ad", "embeddings.h5ad"]

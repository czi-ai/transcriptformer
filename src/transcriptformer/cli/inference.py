"""
Script to perform inference with Transcriptformer models.

Example usage:
    python inference.py --config-name=inference_config.yaml \
  model.checkpoint_path=./checkpoints/tf_sapiens \
  model.inference_config.data_files.0=test/data/human_val.h5ad \
  model.inference_config.output_path=./custom_results_dir \
  model.inference_config.output_filename=custom_embeddings.h5ad \
  model.inference_config.batch_size=8
"""

import logging
import os

import hydra
from omegaconf import DictConfig, OmegaConf

from transcriptformer.model.inference import run_inference
from transcriptformer.utils.utils import merge_checkpoint_with_cfg

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@hydra.main(
    config_path=os.path.join(os.path.dirname(__file__), "conf"),
    config_name="inference_config.yaml",
    version_base=None,
)
def main(cfg: DictConfig):
    logging.debug(OmegaConf.to_yaml(cfg))

    # Get checkpoint path from either location for backward compatibility
    checkpoint_path = getattr(cfg.model, "checkpoint_path", None) or cfg.model.inference_config.checkpoint_path

    cfg = merge_checkpoint_with_cfg(checkpoint_path, cfg)

    adata_output = run_inference(cfg, data_files=cfg.model.inference_config.data_files)

    # Save the output adata
    output_path = cfg.model.inference_config.output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Get output filename from config or use default
    output_filename = getattr(cfg.model.inference_config, "output_filename", "embeddings.h5ad")
    save_file = os.path.join(output_path, output_filename)

    adata_output.write_h5ad(save_file)
    logging.info(f"Saved embeddings to {save_file}")


if __name__ == "__main__":
    main()

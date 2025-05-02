"""
Script to perform inference with Transcriptformer models.

Example usage:
    python inference.py --config-name=inference_config.yaml \
  model.checkpoint_path=./checkpoints/tf_sapiens \
  model.inference_config.data_files.0=test/data/human_val.h5ad \
  model.inference_config.output_path=./inference_results \
  model.inference_config.batch_size=8
"""

import json
import logging
import os

import hydra
from omegaconf import DictConfig, OmegaConf

from transcriptformer.model.inference import run_inference

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

print(
    """
\033[38;2;138;43;226m ___________  ___   _   _  _____           _       _  ______ ______________  ___ ___________
\033[38;2;138;43;226m|_   _| ___ \\/ _ \\ | \\ | |/  ___|         (_)     | | |  ___|  _  | ___ \\  \\/  ||  ___| ___ \\
\033[38;2;132;57;207m  | | | |_/ / /_\\ \\|  \\| |\\ `--.  ___ _ __ _ _ __ | |_| |_  | | | | |_/ / .  . || |__ | |_/ /
\033[38;2;126;71;188m  | | |    /|  _  || . ` | `--. \\/ __| '__| | '_ \\| __|  _| | | | |    /| |\\/| ||  __||    /
\033[38;2;120;85;169m  | | | |\\ \\| | | || |\\  |/\\__/ / (__| |  | | |_) | |_| |   \\ \\_/ / |\\ \\| |  | || |___| |\\ \\
\033[38;2;114;99;150m  \\_/ \\_| \\_\\_| |_/\\_| \\_/\\____/ \\___|_|  |_| .__/ \\__\\_|    \\___/\\_| \\_\\_|  |_/\\____/\\_| \\_|
\033[38;2;108;113;131m                                            | |
\033[38;2;108;113;131m                                            |_|
\033[0m"""
)


@hydra.main(
    config_path="../../../conf", config_name="inference_config.yaml", version_base=None
)
def main(cfg: DictConfig):
    logging.debug(OmegaConf.to_yaml(cfg))

    config_path = os.path.join(cfg.model.checkpoint_path, "config.json")
    with open(config_path) as f:
        config_dict = json.load(f)
    mlflow_cfg = OmegaConf.create(config_dict)

    # Merge the MLflow config with the main config
    cfg = OmegaConf.merge(mlflow_cfg, cfg)

    # Set the checkpoint paths based on the unified checkpoint_path
    cfg.model.inference_config.load_checkpoint = os.path.join(
        cfg.model.checkpoint_path, "model_weights.pt"
    )
    cfg.model.data_config.aux_vocab_path = os.path.join(
        cfg.model.checkpoint_path, "vocabs"
    )
    cfg.model.data_config.esm2_mappings_path = os.path.join(
        cfg.model.checkpoint_path, "vocabs"
    )

    adata_output = run_inference(cfg, data_files=cfg.model.inference_config.data_files)

    # Save the output adata
    output_path = cfg.model.inference_config.output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    save_file = os.path.join(output_path, "embeddings.h5ad")
    adata_output.write_h5ad(save_file)
    logging.info(f"Saved embeddings to {save_file}")


if __name__ == "__main__":
    main()

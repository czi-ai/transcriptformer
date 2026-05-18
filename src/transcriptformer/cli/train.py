"""Hydra entrypoint for training Transcriptformer assay adaptations."""

from __future__ import annotations

import logging
import os

import hydra
from omegaconf import DictConfig, OmegaConf

from transcriptformer.train.engine import run_train_from_dict, setup_runtime_for_training

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@hydra.main(
    config_path=os.path.join(os.path.dirname(__file__), "conf"),
    config_name="train_config.yaml",
    version_base=None,
)
def main(cfg: DictConfig):
    setup_runtime_for_training()
    logging.info("Training config:\n%s", OmegaConf.to_yaml(cfg))
    result = run_train_from_dict(OmegaConf.to_container(cfg.train, resolve=True))
    logging.info("Training complete: %s", result)


if __name__ == "__main__":
    main()

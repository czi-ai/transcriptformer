import logging
import warnings

import anndata
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

from transcriptformer.data.dataloader import AnnDataset, AnnDatasetOOM
from transcriptformer.model.embedding_surgery import change_embedding_layer
from transcriptformer.tokenizer.vocab import load_vocabs_and_embeddings
from transcriptformer.utils.device import resolve_accelerator_and_devices, resolve_checkpoint_map_location
from transcriptformer.utils.utils import stack_dict

# Set float32 matmul precision for better performance with Tensor Cores
torch.set_float32_matmul_precision("high")

torch._dynamo.config.optimize_ddp = False
torch._dynamo.config.cache_size_limit = 1000


# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def run_inference(cfg, data_files: list[str] | list[anndata.AnnData]):
    """Run inference using the provided config and AnnData object.

    Args:
        cfg: OmegaConf configuration object
        data_files: list of data files to load
    Returns:
        AnnData: Processed data with embeddings and likelihood scores
    """
    warnings.filterwarnings(
        "ignore", message="The 'predict_dataloader' does not have many workers which may be a bottleneck"
    )
    warnings.filterwarnings(
        "ignore",
        message="Your `IterableDataset` has `__len__` defined. In combination with multi-process data loading",
        category=UserWarning,
    )
    warnings.filterwarnings("ignore", message="Transforming to str index.", category=UserWarning)

    # Load vocabs and embeddings
    (gene_vocab, aux_vocab), emb_matrix = load_vocabs_and_embeddings(cfg)

    # Determine model class based on config
    model_type = cfg.model.get("model_type", "transcriptformer")  # Default to transcriptformer if not set
    if model_type == "esm2ce":
        from transcriptformer.model.model import ESM2CE as ModelClass

        logging.info("Instantiating ESM2CE model")
    elif model_type == "transcriptformer":
        from transcriptformer.model.model import Transcriptformer as ModelClass

        logging.info("Instantiating Transcriptformer model")
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Instantiate the selected model
    logging.info(f"Instantiating the {model_type} model")
    model = ModelClass(
        data_config=cfg.model.data_config,
        model_config=cfg.model.model_config,
        loss_config=cfg.model.loss_config,
        inference_config=cfg.model.inference_config,
        gene_vocab_dict=gene_vocab,
        aux_vocab_dict=aux_vocab,
        emb_matrix=emb_matrix,
    )
    model.eval()

    logging.info("Model instantiated successfully")

    # Check if checkpoint is supplied
    if not hasattr(cfg.model.inference_config, "load_checkpoint") or not cfg.model.inference_config.load_checkpoint:
        raise ValueError(
            "No checkpoint provided for inference. Please specify a checkpoint path in "
            "model.inference_config.load_checkpoint"
        )

    logging.info("Loading model checkpoint")
    device_preference = getattr(cfg.model.inference_config, "device", "auto")
    map_location = resolve_checkpoint_map_location(device_preference)

    # Instead of loading full checkpoint, just load weights with proper device mapping
    state_dict = torch.load(cfg.model.inference_config.load_checkpoint, weights_only=True, map_location=map_location)

    # Validate and load weights
    # converter.validate_loaded_weights(model, state_dict)
    model.load_state_dict(state_dict)
    logging.info("Model weights loaded successfully")

    # Perform embedding surgery if specified in config
    if cfg.model.inference_config.pretrained_embedding is not None:
        logging.info("Performing embedding surgery")
        # Check if pretrained_embedding_paths is a list, if not convert it to a list
        if not isinstance(cfg.model.inference_config.pretrained_embedding, list):
            pretrained_embedding_paths = [cfg.model.inference_config.pretrained_embedding]
        else:
            pretrained_embedding_paths = cfg.model.inference_config.pretrained_embedding
        model, gene_vocab = change_embedding_layer(model, pretrained_embedding_paths)

    # Load dataset
    data_kwargs = {
        "gene_vocab": gene_vocab,
        "aux_vocab": aux_vocab,
        "max_len": cfg.model.model_config.seq_len,
        "pad_zeros": cfg.model.data_config.pad_zeros,
        "pad_token": cfg.model.data_config.gene_pad_token,
        "sort_genes": cfg.model.data_config.sort_genes,
        "filter_to_vocab": cfg.model.data_config.filter_to_vocabs,
        "filter_outliers": cfg.model.data_config.filter_outliers,
        "gene_col_name": cfg.model.data_config.gene_col_name,
        "normalize_to_scale": cfg.model.data_config.normalize_to_scale,
        "randomize_order": cfg.model.data_config.randomize_genes,
        "min_expressed_genes": cfg.model.data_config.min_expressed_genes,
        "clip_counts": cfg.model.data_config.clip_counts,
        "obs_keys": cfg.model.inference_config.obs_keys,
        "remove_duplicate_genes": cfg.model.data_config.remove_duplicate_genes,
        "use_raw": cfg.model.data_config.use_raw,
    }
    use_oom = bool(getattr(cfg.model.inference_config, "use_oom_dataloader", False))
    can_use_oom = all(isinstance(file, str) for file in data_files)
    if use_oom and can_use_oom:
        dataset = AnnDatasetOOM(
            data_files,
            gene_vocab,
            aux_vocab=aux_vocab,
            max_len=cfg.model.model_config.seq_len,
            normalize_to_scale=cfg.model.data_config.normalize_to_scale,
            sort_genes=cfg.model.data_config.sort_genes,
            randomize_order=cfg.model.data_config.randomize_genes,
            pad_zeros=cfg.model.data_config.pad_zeros,
            gene_col_name=cfg.model.data_config.gene_col_name,
            filter_to_vocab=cfg.model.data_config.filter_to_vocabs,
            clip_counts=cfg.model.data_config.clip_counts,
            obs_keys=cfg.model.inference_config.obs_keys,
            use_raw=cfg.model.data_config.use_raw,
            remove_duplicate_genes=cfg.model.data_config.remove_duplicate_genes,
        )
    else:
        if use_oom and not can_use_oom:
            logging.warning(
                "use_oom_dataloader=true with in-memory AnnData inputs; falling back to AnnDataset for compatibility"
            )
        dataset = AnnDataset(data_files, **data_kwargs)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.model.inference_config.batch_size,
        num_workers=cfg.model.data_config.n_data_workers,
        drop_last=False,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    num_gpus = getattr(cfg.model.inference_config, "num_gpus", 1)
    accelerator, devices = resolve_accelerator_and_devices(device_preference, num_gpus)

    logging.info(f"Using {devices} device(s) with accelerator: {accelerator}")

    # Create Trainer
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        num_nodes=1,
        precision=cfg.model.inference_config.precision,
        limit_predict_batches=None,
        logger=CSVLogger("logs", name="inference"),
        strategy="ddp" if devices > 1 else "auto",  # Use DDP for multi-GPU
    )

    # Run prediction
    output = trainer.predict(model, dataloaders=dataloader)

    # Combine predictions
    logging.info("Combining predictions")
    concat_output = stack_dict(output)

    # Create pandas DataFrames from the obs and uns data in concat_output
    obs_df = pd.DataFrame(concat_output["obs"])
    uns = {"llh": pd.DataFrame({"llh": concat_output["llh"]})} if "llh" in concat_output else None
    obsm = {}

    # Add all other output keys to the obsm
    for k in cfg.model.inference_config.output_keys:
        if k in concat_output:
            if k == "embeddings" and cfg.model.inference_config.emb_type == "cge":
                # Handle contextual gene embeddings - store in a flattened format for HDF5 compatibility
                cge_list = concat_output[k]

                # Flatten all embeddings and track their metadata
                all_embeddings = []
                cell_indices = []
                gene_names = []

                for cell_idx, cell_dict in enumerate(cge_list):
                    for gene_name, embedding in cell_dict.items():
                        all_embeddings.append(embedding)
                        cell_indices.append(cell_idx)
                        gene_names.append(gene_name)

                # Convert to numpy arrays
                if all_embeddings:
                    embeddings_array = np.stack(all_embeddings)
                    cell_indices_array = np.array(cell_indices)
                    gene_names_array = np.array(gene_names)

                    # Store in uns (unstructured data)
                    if uns is None:
                        uns = {}
                    uns["cge_embeddings"] = embeddings_array
                    uns["cge_cell_indices"] = cell_indices_array
                    uns["cge_gene_names"] = gene_names_array
                else:
                    # Handle empty case
                    if uns is None:
                        uns = {}
                    uns["cge_embeddings"] = np.array([])
                    uns["cge_cell_indices"] = np.array([])
                    uns["cge_gene_names"] = np.array([])
            else:
                # Regular embeddings or other outputs
                obsm[k] = concat_output[k].numpy() if hasattr(concat_output[k], "numpy") else concat_output[k]

    # Create a new AnnData object with the embeddings
    output_adata = anndata.AnnData(
        obs=obs_df,
        obsm=obsm,
        uns=uns,
    )

    return output_adata

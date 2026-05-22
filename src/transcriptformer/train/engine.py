"""Training engine for assay-vocab expansion fine-tuning."""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from transcriptformer.data.dataclasses import BatchData, DataConfig, LossConfig, ModelConfig
from transcriptformer.data.dataloader import AnnDataset, AnnDatasetOOM
from transcriptformer.model.assay_adaptation import (
    AssayInitConfig,
    apply_freeze_policy,
    build_expanded_assay_embedding_weight,
    count_trainable_parameters,
)
from transcriptformer.model.model import Transcriptformer
from transcriptformer.tokenizer.vocab import construct_gene_embeddings, open_vocabs


class LiveMetricsSummaryCallback(pl.Callback):
    """Export per-epoch metric summaries and live plots during training."""

    def __init__(self, output_dir: Path, enabled: bool = True, every_n_epochs: int = 1):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.enabled = enabled
        self.every_n_epochs = max(1, int(every_n_epochs))
        self.live_dir = self.output_dir / "live_metrics"
        self.epoch_csv_path = self.live_dir / "epoch_metrics.csv"

    @staticmethod
    def _to_float(value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().item()
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_float(text: str | None):
        if text is None or text == "":
            return None
        try:
            return float(text)
        except ValueError:
            return None

    @staticmethod
    def _extract_csv_logger_path(trainer: pl.Trainer) -> Path | None:
        logger = trainer.logger
        if logger is None:
            return None
        if hasattr(logger, "log_dir"):
            return Path(logger.log_dir) / "metrics.csv"
        if hasattr(logger, "_logger_iterable"):
            for lg in logger._logger_iterable:  # Lightning internal logger collection
                if hasattr(lg, "log_dir"):
                    return Path(lg.log_dir) / "metrics.csv"
        return None

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        if not self.enabled:
            return
        self.live_dir.mkdir(parents=True, exist_ok=True)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # For training without validation, emit at train epoch end.
        if trainer.val_dataloaders:
            return
        self._export(trainer)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # For training with validation, emit once per epoch after validation.
        self._export(trainer)

    def _export(self, trainer: pl.Trainer) -> None:
        if not self.enabled:
            return
        epoch_idx = int(trainer.current_epoch)
        if (epoch_idx + 1) % self.every_n_epochs != 0:
            return

        callback_metrics = trainer.callback_metrics
        row = {
            "epoch": epoch_idx,
            "global_step": int(trainer.global_step),
            "train/total_loss": self._to_float(callback_metrics.get("train/total_loss")),
            "val/total_loss": self._to_float(callback_metrics.get("val/total_loss")),
            "train/count_loss": self._to_float(callback_metrics.get("train/count_loss")),
            "val/count_loss": self._to_float(callback_metrics.get("val/count_loss")),
            "train/gene_loss": self._to_float(callback_metrics.get("train/gene_loss")),
            "val/gene_loss": self._to_float(callback_metrics.get("val/gene_loss")),
        }

        file_exists = self.epoch_csv_path.exists()
        with self.epoch_csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

        self._render_epoch_bar_plot()
        self._render_step_curve_plot(trainer)

    def _render_epoch_bar_plot(self) -> None:
        try:
            import matplotlib.pyplot as plt
        except Exception:
            return

        if not self.epoch_csv_path.exists():
            return

        with self.epoch_csv_path.open("r", newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return

        last = rows[-1]
        labels = []
        values = []
        for key in ["train/total_loss", "val/total_loss", "train/count_loss", "val/count_loss", "train/gene_loss", "val/gene_loss"]:
            value = self._safe_float(last.get(key))
            if value is None:
                continue
            labels.append(key)
            values.append(value)

        if not labels:
            return

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(labels, values)
        ax.set_ylabel("Loss")
        ax.set_title(f"Epoch {last.get('epoch', '?')} Metrics")
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        fig.savefig(self.live_dir / "epoch_metrics_bar_latest.png", dpi=140)
        plt.close(fig)

    def _render_step_curve_plot(self, trainer: pl.Trainer) -> None:
        csv_metrics = self._extract_csv_logger_path(trainer)
        if csv_metrics is None or not csv_metrics.exists():
            return

        try:
            import matplotlib.pyplot as plt
        except Exception:
            return

        with csv_metrics.open("r", newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return

        step_key = "step"
        candidate_metric_keys = [
            "train/total_loss_step",
            "train/count_loss_step",
            "train/gene_loss_step",
        ]

        # Keep only metrics that actually exist in this run's CSV.
        metric_keys = [k for k in candidate_metric_keys if any((r.get(k) not in (None, "")) for r in rows)]
        if not metric_keys:
            return

        curves = {k: {"x": [], "y": []} for k in metric_keys}
        for r in rows:
            step_val = self._safe_float(r.get(step_key))
            if step_val is None:
                continue
            for k in metric_keys:
                y = self._safe_float(r.get(k))
                if y is None:
                    continue
                curves[k]["x"].append(step_val)
                curves[k]["y"].append(y)

        fig, ax = plt.subplots(figsize=(10, 4))
        have_curve = False
        for k in metric_keys:
            xs = curves[k]["x"]
            ys = curves[k]["y"]
            if not xs:
                continue
            ax.plot(xs, ys, label=k)
            have_curve = True

        if not have_curve:
            plt.close(fig)
            return

        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Training Step Metrics")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(self.live_dir / "step_metrics_curve.png", dpi=140)
        plt.close(fig)


def _parse_init_map(items: list[str] | None) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Invalid assay init map '{item}', expected new=source")
        new_tok, src_tok = item.split("=", 1)
        new_tok = new_tok.strip()
        src_tok = src_tok.strip()
        if not new_tok or not src_tok:
            raise ValueError(f"Invalid assay init map '{item}'")
        mapping[new_tok] = src_tok
    return mapping


def _sanitize_data_config(base_data_cfg: dict, cfg: dict) -> DataConfig:
    if "data_config" not in cfg or not isinstance(cfg["data_config"], dict):
        raise ValueError("Missing required nested 'data_config' in training config")

    out = dict(base_data_cfg)
    data_cfg = dict(cfg["data_config"])
    data_cfg.pop("_target_", None)
    # Only override base checkpoint data config when values are explicitly provided.
    for key, value in data_cfg.items():
        if value is not None:
            out[key] = value

    out["aux_vocab_path"] = cfg["source_vocab_dir"]
    out["pin_memory"] = True
    out["aux_cols"] = cfg.get("obs_assay_col", "assay")
    out["pad_zeros"] = True
    out["n_data_workers"] = int(out.get("n_data_workers", cfg.get("num_workers", 4)))
    out["gene_pad_token"] = out.get("gene_pad_token") or "[PAD]"
    out["aux_pad_token"] = out.get("aux_pad_token") or "unknown"

    out.pop("_target_", None)
    return DataConfig(**out)


def _sanitize_model_config(base_model_cfg: dict) -> ModelConfig:
    cfg = dict(base_model_cfg)
    cfg.pop("_target_", None)
    return ModelConfig(**cfg)


def _sanitize_loss_config(base_loss_cfg: dict, cfg: dict) -> LossConfig:
    out = dict(base_loss_cfg)
    train_loss_cfg = cfg.get("loss_config", {})
    if not isinstance(train_loss_cfg, dict):
        raise ValueError("'loss_config' must be a nested mapping in training config")
    train_loss_cfg = dict(train_loss_cfg)
    train_loss_cfg.pop("_target_", None)
    for key, value in train_loss_cfg.items():
        if value is not None:
            out[key] = value
    out.pop("_target_", None)
    return LossConfig(**out)


def _parse_devices(devices: str | int):
    if isinstance(devices, int):
        return devices
    if devices in {"auto", "-1"}:
        return devices
    return int(devices)


def _shuffle_expressed_genes(batch: BatchData, pad_idx: int) -> BatchData:
    tokens = batch.gene_token_indices
    counts = batch.gene_counts

    shuffled_tokens = tokens.clone()
    shuffled_counts = counts.clone()

    valid = tokens != pad_idx
    for i in range(tokens.shape[0]):
        idx = torch.nonzero(valid[i], as_tuple=True)[0]
        if idx.numel() <= 1:
            continue
        perm = idx[torch.randperm(idx.numel(), device=idx.device)]
        shuffled_tokens[i, idx] = tokens[i, perm]
        shuffled_counts[i, idx] = counts[i, perm]

    return BatchData(
        gene_counts=shuffled_counts,
        gene_token_indices=shuffled_tokens,
        aux_token_indices=batch.aux_token_indices,
        file_path=batch.file_path,
        obs=batch.obs,
    )


class TranscriptformerTrainModule(pl.LightningModule):
    def __init__(
        self,
        model: Transcriptformer,
        lr: float,
        weight_decay: float,
        beta1: float,
        beta2: float,
        eps: float,
        warmup_ratio: float,
        min_lr_ratio: float,
        shuffle_expressed_each_batch: bool,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.warmup_ratio = warmup_ratio
        self.min_lr_ratio = min_lr_ratio
        self.shuffle_expressed_each_batch = shuffle_expressed_each_batch

    def _shared_step(self, batch: BatchData, stage: str) -> torch.Tensor:
        if stage == "train" and self.shuffle_expressed_each_batch:
            batch = _shuffle_expressed_genes(batch, self.model.gene_vocab.pad_idx)

        out = self.model(batch)

        count_loss = self.model.criterion(mu=out["mu"], input_counts=out["input_counts"], mask=out["mask"])
        gene_loss = torch.tensor(0.0, device=count_loss.device)
        if self.model.loss_config.gene_id_loss_weight > 0 and "gene_logit" in out:
            gene_loss = self.model.gene_id_criterion(
                logits=out["gene_logit"],
                input_ids=out["input_gene_token_indices"],
                mask=out["mask"],
            )

        total = count_loss + self.model.loss_config.gene_id_loss_weight * gene_loss
        # explict batch size for logging to avoid warning from nested batch dictionary 
        bs = int(batch.gene_counts.shape[0])
        self.log(f"{stage}/total_loss", total, prog_bar=True, on_step=(stage == "train"), on_epoch=True, batch_size=bs)
        self.log(f"{stage}/count_loss", count_loss, on_step=False, on_epoch=True, batch_size=bs)
        self.log(f"{stage}/gene_loss", gene_loss, on_step=False, on_epoch=True, batch_size=bs)
        return total

    def training_step(self, batch: BatchData, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch: BatchData, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="val")

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("No trainable parameters found")

        optimizer = AdamW(
            params,
            lr=self.lr,
            betas=(self.beta1, self.beta2),
            eps=self.eps,
            weight_decay=self.weight_decay,
        )

        total_steps = max(1, int(self.trainer.estimated_stepping_batches))
        warmup_steps = int(total_steps * self.warmup_ratio)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(max(1, warmup_steps))
            progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }


def _build_dataset(
    files: list[str],
    cfg: dict,
    data_config: DataConfig,
    gene_vocab: dict,
    aux_vocab: dict,
    seq_len: int,
    is_train: bool,
):
    for file in files:
        if not Path(file).exists():
            raise FileNotFoundError(f"Dataset file not found: {file}")
    
    kwargs = {
        "gene_vocab": gene_vocab,
        "aux_vocab": aux_vocab,
        "max_len": seq_len,
        "normalize_to_scale": data_config.normalize_to_scale,
        "sort_genes": bool(data_config.sort_genes),
        "randomize_order": bool(data_config.randomize_genes),
        "pad_zeros": True,
        "gene_col_name": data_config.gene_col_name,
        "filter_to_vocab": bool(data_config.filter_to_vocabs),
        "clip_counts": data_config.clip_counts,
        "use_raw": data_config.use_raw,
        "remove_duplicate_genes": bool(data_config.remove_duplicate_genes),
    }

    if cfg.get("use_oom_dataloader", False):
        return AnnDatasetOOM(
            files_list=files,
            filter_outliers=float(data_config.filter_outliers),
            min_expressed_genes=int(data_config.min_expressed_genes),
            **kwargs,
        )

    return AnnDataset(
        files_list=files,
        filter_outliers=float(data_config.filter_outliers),
        min_expressed_genes=int(data_config.min_expressed_genes),
        seed=int(cfg.get("seed", 42)) + (0 if is_train else 17),
        inference=False,
        **kwargs,
    )


def _copy_vocab_assets(source_vocab_dir: Path, output_vocab_dir: Path, assay_vocab_to_copy: Path) -> None:
    output_vocab_dir.mkdir(parents=True, exist_ok=True)
    for file in source_vocab_dir.glob("*"):
        if file.is_file() and file.name != "assay_vocab.json":
            shutil.copy2(file, output_vocab_dir / file.name)
    shutil.copy2(assay_vocab_to_copy, output_vocab_dir / "assay_vocab.json")


def _save_plain_weights(lightning_ckpt_path: Path, output_model_path: Path) -> None:
    ckpt = torch.load(lightning_ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"]
    cleaned = {k[len("model.") :]: v for k, v in state_dict.items() if k.startswith("model.")}
    torch.save(cleaned, output_model_path)


def _save_dataset_filter_metadata(output_dir: Path, train_dataset, val_dataset=None) -> None:
    summary = {
        "train": getattr(train_dataset, "filter_metadata", None),
        "val": getattr(val_dataset, "filter_metadata", None) if val_dataset is not None else None,
    }
    if summary["train"] is None and summary["val"] is None:
        return

    with (output_dir / "data_filtering_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")


def _prepare_source_dirs(cfg: dict) -> tuple[Path, Path, Path]:
    resume_dir = cfg.get("resume_artifact_dir")
    if resume_dir:
        source_dir = Path(resume_dir)
    else:
        source_dir = Path(cfg["checkpoint_dir"])

    config_path = source_dir / "config.json"
    weights_path = source_dir / "model_weights.pt"
    vocab_dir = source_dir / "vocabs"

    if not config_path.exists() or not weights_path.exists() or not vocab_dir.exists():
        raise FileNotFoundError(f"Invalid artifact directory: {source_dir}")

    return config_path, weights_path, vocab_dir


def run_train_from_dict(cfg: dict) -> dict:
    pl.seed_everything(int(cfg.get("seed", 42)), workers=True)

    config_path, weights_path, vocab_dir = _prepare_source_dirs(cfg)

    with config_path.open() as f:
        config_json = json.load(f)
    model_json = config_json["model"]

    cfg = dict(cfg)
    cfg["source_vocab_dir"] = str(vocab_dir)

    data_config = _sanitize_data_config(model_json["data_config"], cfg)
    model_config = _sanitize_model_config(model_json["model_config"])
    loss_config = _sanitize_loss_config(model_json["loss_config"], cfg)

    obs_assay_col = cfg.get("obs_assay_col", "assay")
    aux_vocab = open_vocabs(str(vocab_dir), cols_to_load=None)
    if obs_assay_col not in aux_vocab:
        raise ValueError(f"Aux key '{obs_assay_col}' not found in {vocab_dir}")

    source_assay_vocab = aux_vocab[obs_assay_col]
    expanded_path = cfg.get("expanded_assay_vocab")
    if expanded_path:
        with Path(expanded_path).open() as f:
            target_assay_vocab = json.load(f)
        assay_vocab_path_for_output = Path(expanded_path)
    else:
        target_assay_vocab = source_assay_vocab
        assay_vocab_path_for_output = vocab_dir / "assay_vocab.json"

    aux_vocab[obs_assay_col] = target_assay_vocab

    emb_files = [str(vocab_dir / file_name) for file_name in data_config.esm2_mappings]
    gene_vocab, emb_matrix = construct_gene_embeddings(emb_files, data_config.special_tokens)
    emb_matrix = torch.tensor(emb_matrix)

    base_model = Transcriptformer(
        data_config=data_config,
        model_config=model_config,
        loss_config=loss_config,
        inference_config=None,
        gene_vocab_dict=gene_vocab,
        aux_vocab_dict=aux_vocab,
        emb_matrix=emb_matrix,
    )

    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    aux_key = f"aux_embeddings.{obs_assay_col}.weight"
    if aux_key not in state_dict:
        raise KeyError(f"Missing checkpoint key: {aux_key}")

    init_map = _parse_init_map(cfg.get("assay_init_map", []))
    state_dict[aux_key] = build_expanded_assay_embedding_weight(
        old_weight=state_dict[aux_key],
        old_vocab=source_assay_vocab,
        new_vocab=target_assay_vocab,
        init_map=init_map,
        cfg=AssayInitConfig(default_source=cfg.get("init_default_source", "unknown"), mean_pool_fallback=True),
    )

    base_model.load_state_dict(state_dict, strict=False)

    apply_freeze_policy(
        base_model,
        freeze_transformer=bool(cfg.get("freeze_transformer", False)),
        freeze_gene_embeddings=bool(cfg.get("freeze_gene_embeddings", False)),
        freeze_count_head=bool(cfg.get("freeze_count_head", False)),
        freeze_gene_head=bool(cfg.get("freeze_gene_head", False)),
        train_aux_only=bool(cfg.get("train_aux_only", False)),
    )

    trainable, total = count_trainable_parameters(base_model)

    train_dataset = _build_dataset(
        files=list(cfg["train_files"]),
        cfg=cfg,
        data_config=data_config,
        gene_vocab=gene_vocab,
        aux_vocab=aux_vocab,
        seq_len=model_config.seq_len,
        is_train=True,
    )
    val_files = list(cfg.get("val_files", []))
    val_dataset = (
        _build_dataset(
            files=val_files,
            cfg=cfg,
            data_config=data_config,
            gene_vocab=gene_vocab,
            aux_vocab=aux_vocab,
            seq_len=model_config.seq_len,
            is_train=False,
        )
        if val_files
        else None
    )

    batch_size = int(cfg.get("batch_size", 8))
    num_workers = int(cfg.get("num_workers", 4))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=not bool(cfg.get("use_oom_dataloader", False)),
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=val_dataset.collate_fn,
        )

    lit_model = TranscriptformerTrainModule(
        model=base_model,
        lr=float(cfg.get("lr", 5.5e-5)),
        weight_decay=float(cfg.get("weight_decay", 0.05)),
        beta1=float(cfg.get("adam_beta1", 0.9)),
        beta2=float(cfg.get("adam_beta2", 0.95)),
        eps=float(cfg.get("adam_eps", 1e-8)),
        warmup_ratio=float(cfg.get("warmup_ratio", 0.1)),
        min_lr_ratio=float(cfg.get("min_lr_ratio", 0.1)),
        shuffle_expressed_each_batch=bool(cfg.get("shuffle_expressed_each_batch", False)),
    )


    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    _save_dataset_filter_metadata(output_dir, train_dataset, val_dataset)

    # Save config and train_args BEFORE training loop
    out_config = dict(config_json)
    out_config["model"] = dict(model_json)
    out_config["model"]["data_config"] = dict(out_config["model"]["data_config"])
    out_config["model"]["data_config"]["aux_cols"] = obs_assay_col
    out_config["train_runtime"] = cfg
    with (output_dir / "config.json").open("w") as f:
        json.dump(out_config, f, indent=2)
        f.write("\n")

    with (output_dir / "train_args.json").open("w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")


    # Checkpoint config
    ckpt_cfg = cfg.get("checkpoint", {})
    save_top_k = int(ckpt_cfg.get("save_top_k", 1))
    save_last = bool(ckpt_cfg.get("save_last", True))

    monitor_metric = "val/total_loss" if val_loader is not None else "train/total_loss"
    filename_template = "epoch-{epoch:03d}-step-{step:08d}-val-{val/total_loss:.4f}"
    if val_loader is None:
        filename_template = "epoch-{epoch:03d}-step-{step:08d}-train-{train/total_loss:.4f}"

    callbacks: list[pl.Callback] = []
    ckpt_topk = None
    ckpt_last = None

    # Top-K metric-based checkpoints
    if save_top_k > 0:
        ckpt_topk = ModelCheckpoint(
            dirpath=str(output_dir / "lightning_ckpts"),
            filename=filename_template,
            monitor=monitor_metric,
            mode="min",
            save_top_k=save_top_k,
            save_last=False,
            auto_insert_metric_name=False, # avoid duplicate metric name and "=" sign
        )
        callbacks.append(ckpt_topk)

    # Last checkpoint at each epoch end
    if save_last:
        ckpt_last = ModelCheckpoint(
            dirpath=str(output_dir / "lightning_ckpts"),
            filename="last",
            save_top_k=0,
            save_last=True,
            every_n_epochs=1,
        )
        callbacks.append(ckpt_last)

    devices = _parse_devices(cfg.get("devices", "1"))
    strategy = "auto"
    if isinstance(devices, int) and devices > 1:
        strategy = "ddp"

    # Gradient accumulation and clipping
    grad_clip = float(cfg.get("gradient_clip_val", 1.0))
    grad_accum = int(cfg.get("accumulate_grad_batches", 1))

    # CSV Logger for live metrics
    from pytorch_lightning.loggers import CSVLogger
    csv_logger = CSVLogger(save_dir=str(output_dir), name="csv_logs")

    live_cfg = cfg.get("live_metrics", {})
    live_metrics_cb = LiveMetricsSummaryCallback(
        output_dir=output_dir,
        enabled=bool(live_cfg.get("enabled", True)),
        every_n_epochs=int(live_cfg.get("every_n_epochs", 1)),
    )
    callbacks.append(live_metrics_cb)

    trainer = pl.Trainer(
        accelerator=cfg.get("accelerator", "auto"),
        devices=devices,
        num_nodes=int(cfg.get("num_nodes", 1)),
        precision=cfg.get("precision", "16-mixed"),
        max_epochs=int(cfg.get("max_epochs", 5)),
        log_every_n_steps=int(cfg.get("log_every_n_steps", 10)),
        callbacks=callbacks,
        strategy=strategy,
        gradient_clip_val=grad_clip,
        accumulate_grad_batches=grad_accum,
        logger=csv_logger,
    )

    resume_mode = cfg.get("resume_mode", "weights")
    resume_ckpt = None
    if cfg.get("resume_artifact_dir") and resume_mode == "lightning":
        resume_ckpt = Path(cfg["resume_artifact_dir"]) / "lightning_ckpts" / "last.ckpt"
        if not resume_ckpt.exists():
            raise FileNotFoundError(f"Missing resume checkpoint: {resume_ckpt}")

    trainer.fit(
        lit_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=str(resume_ckpt) if resume_ckpt else None,
    )

    best_ckpt_path = ""
    if ckpt_topk is not None and ckpt_topk.best_model_path:
        best_ckpt_path = ckpt_topk.best_model_path
    elif ckpt_last is not None:
        best_ckpt_path = str(Path(ckpt_last.dirpath) / "last.ckpt")

    best_ckpt = Path(best_ckpt_path)
    if not best_ckpt.exists():
        raise RuntimeError("No checkpoint generated")

    output_weights = output_dir / "model_weights.pt"
    _save_plain_weights(best_ckpt, output_weights)

    output_vocab_dir = output_dir / "vocabs"
    _copy_vocab_assets(vocab_dir, output_vocab_dir, assay_vocab_path_for_output)

    return {
        "output_dir": str(output_dir),
        "model_weights": str(output_weights),
        "trainable_params": trainable,
        "total_params": total,
    }


def setup_runtime_for_training() -> None:
    torch._dynamo.config.optimize_ddp = False
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import shutil
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from renewables_forecasting.models.config import ModelConfig, TrainingConfig
from renewables_forecasting.models.cnn import GenerationCNN
from renewables_forecasting.models.dataset import GenerationDataset


def train(
        features_dir: Path,
        config_path: Path,
        out_dir: Path,
        time_col: str = "time",
        value_col: str = "generation_mwh",
        early_stopping_patience: int = 10,
        num_workers: int = 0,
) -> None:
    """
    Trains a GenerationCNN model on the feature dataset produced by
    build_feature_dataset() and saves the best model checkpoint, training
    logs, and config to out_dir.

    Supports two output modes set via ModelConfig.output_capacity_factor:

    Raw MWh mode (output_capacity_factor=False):
        Model output is compared directly to target MWh. No reconstruction
        needed. total_capacity is ignored even if present in the batch.

    Capacity factor mode (output_capacity_factor=True):
        Model outputs a capacity factor in [0, 1]. The training loop
        reconstructs MWh before computing the loss:
            predicted_mwh = model_output × total_capacity_t
        Requires total_cap_per_t.csv to be present in features_dir
        (built with capacity_as_spatial_distribution=True or
        capacity_as_spatiotemporal_distribution=True).

    Loss is logged to loss_log.csv after each epoch so progress can be
    monitored mid-run. The model state dict from the epoch with the lowest
    validation loss is saved as best_model.pt.

    The config is copied to out_dir immediately on startup — before any
    training begins — so it is always available even if training is interrupted.

    Parameters
    ----------
    features_dir:
        Directory containing the feature dataset.
    config_path:
        Path to the YAML model config file.
    out_dir:
        Directory to write best_model.pt, loss_log.csv, config.yaml.
    time_col:
        Name of the datetime column in CSV files. Default: 'time'.
    value_col:
        Name of the generation column in target.csv. Default: 'generation_mwh'.
    early_stopping_patience:
        Epochs without val improvement before stopping. Default: 10.
    num_workers:
        DataLoader worker processes. Default: 0 (main process).
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    # Save config immediately — available even if training is interrupted
    shutil.copy(config_path, out_dir / "config.yaml")

    # ── Load config ────────────────────────────────────────────────────────────

    model_config = ModelConfig.from_yaml(config_path)
    training_config = TrainingConfig.from_yaml(config_path)

    print(f"Model config:    {config_path.name}")
    print(f"Features dir:    {features_dir}")
    print(f"Output dir:      {out_dir}")
    print(f"Max epochs:      {training_config.max_epochs}")
    print(f"Early stopping:  {early_stopping_patience} epochs patience")
    print(f"Batch size:      {training_config.batch_size}")
    print(f"Learning rate:   {training_config.learning_rate}")
    print(f"Weight decay:    {training_config.weight_decay}")
    print(f"Output mode:     {'capacity factor [0,1]' if model_config.output_capacity_factor else 'raw MWh'}")

    # ── Device ─────────────────────────────────────────────────────────────────

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # ── Datasets and dataloaders ───────────────────────────────────────────────

    print("\nLoading datasets ...")
    train_dataset = GenerationDataset(features_dir, split="train",
                                      time_col=time_col, value_col=value_col)
    val_dataset = GenerationDataset(features_dir, split="val",
                                    time_col=time_col, value_col=value_col)

    # Validate capacity factor mode requirements
    if model_config.output_capacity_factor:
        if train_dataset.total_capacity is None:
            raise ValueError(
                "output_capacity_factor=True requires total_cap_per_t.csv in "
                "features_dir. Rebuild features with capacity_as_spatial_distribution=True "
                "or capacity_as_spatiotemporal_distribution=True."
            )

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    # ── Model ──────────────────────────────────────────────────────────────────

    model = GenerationCNN(model_config).to(device)
    print(f"\nModel parameters: {model.n_parameters():,}")

    # ── Optimizer and loss ─────────────────────────────────────────────────────

    if training_config.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
        )
    elif training_config.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=training_config.learning_rate,
            momentum=0.9,
            weight_decay=training_config.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer '{training_config.optimizer}'")

    loss_fn = nn.MSELoss()

    # ── Training loop ──────────────────────────────────────────────────────────

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    loss_records = []
    loss_log_path = out_dir / "loss_log.csv"

    print(f"\n{'Epoch':>6}  {'Train loss':>12}  {'Val loss':>12}  {'Time':>8}  {'Status'}")
    print("─" * 60)

    for epoch in range(1, training_config.max_epochs + 1):
        epoch_start = time.time()

        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        train_losses = []

        for batch in train_loader:
            grid = batch["grid"].to(device)
            time_features = batch["time_features"].to(device)
            target = batch["target"].to(device)

            optimizer.zero_grad()
            pred = model(grid, time_features)

            # Capacity factor mode: reconstruct MWh before loss
            if model_config.output_capacity_factor:
                total_cap = batch["total_capacity"].to(device)
                pred = pred * total_cap

            loss = loss_fn(pred, target)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))

        # ── Validate ───────────────────────────────────────────────────────────
        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in val_loader:
                grid = batch["grid"].to(device)
                time_features = batch["time_features"].to(device)
                target = batch["target"].to(device)

                pred = model(grid, time_features)

                if model_config.output_capacity_factor:
                    total_cap = batch["total_capacity"].to(device)
                    pred = pred * total_cap

                loss = loss_fn(pred, target)
                val_losses.append(loss.item())

        val_loss = float(np.mean(val_losses))
        epoch_time = time.time() - epoch_start

        # ── Early stopping and checkpointing ───────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            status = "✓ best"
        else:
            epochs_without_improvement += 1
            status = f"  ({epochs_without_improvement}/{early_stopping_patience})"

        # ── Log ────────────────────────────────────────────────────────────────
        print(
            f"{epoch:>6}  "
            f"{train_loss:>12.2f}  "
            f"{val_loss:>12.2f}  "
            f"{epoch_time:>7.1f}s  "
            f"{status}"
        )

        loss_records.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epoch_time_s": round(epoch_time, 1),
            "best": val_loss == best_val_loss,
        })

        # Write after every epoch so progress is visible mid-run
        pd.DataFrame(loss_records).to_csv(loss_log_path, index=False)

        if epochs_without_improvement >= early_stopping_patience:
            print(f"\nEarly stopping — no improvement for {early_stopping_patience} epochs.")
            break

    # ── Final summary ──────────────────────────────────────────────────────────

    print(f"\nBest val loss:  {best_val_loss:.2f}")
    print(f"Best model:     {out_dir / 'best_model.pt'}")
    print(f"Loss log:       {loss_log_path}")
    print(f"Config saved:   {out_dir / 'config.yaml'}")

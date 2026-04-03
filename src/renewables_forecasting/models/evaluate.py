import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader

from renewables_forecasting.models.config import ModelConfig, TrainingConfig
from renewables_forecasting.models.cnn import GenerationCNN
from renewables_forecasting.models.dataset import GenerationDataset


def evaluate(
        features_dir: Path,
        config_path: Path,
        model_path: Path,
        out_dir: Path,
        split: str = "test",
        time_col: str = "time",
        value_col: str = "generation_mwh",
        num_workers: int = 0,
) -> pd.DataFrame:
    """
    Evaluates a trained GenerationCNN on a given split (default: test) and
    computes a comprehensive set of regression metrics.

    Saves predictions and metrics to out_dir and prints a summary to console.

    Metrics computed:
        MAE     — Mean Absolute Error (MWh) — average magnitude of errors
        RMSE    — Root Mean Squared Error (MWh) — penalises large errors more
        MAPE    — Mean Absolute Percentage Error (%) — relative error, excludes
                  near-zero hours to avoid division by zero
        nRMSE   — RMSE normalised by mean actual generation (%) — scale-free
        R²      — Coefficient of determination — fraction of variance explained
        Bias    — Mean signed error (MWh) — systematic over/under prediction
        Peak MAE — MAE restricted to hours where actual generation > 90th percentile

    Parameters
    ----------
    features_dir:
        Directory containing the feature dataset.
    config_path:
        Path to the YAML model config file.
    model_path:
        Path to the saved model state dict (best_model.pt).
    out_dir:
        Directory to write predictions.csv and metrics.csv.
    split:
        Which split to evaluate on. Default: 'test'.
    time_col:
        Name of the datetime column in CSV files. Default: 'time'.
    value_col:
        Name of the generation column. Default: 'generation_mwh'.
    num_workers:
        DataLoader worker processes. Default: 0.

    Returns
    -------
    DataFrame with one row per metric.
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load config and model ──────────────────────────────────────────────────

    model_config = ModelConfig.from_yaml(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = GenerationCNN(model_config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model from: {model_path}")
    print(f"Model parameters:  {model.n_parameters():,}")

    # ── Dataset ────────────────────────────────────────────────────────────────

    print(f"\nLoading {split} dataset ...")
    dataset = GenerationDataset(
        features_dir, split=split, time_col=time_col, value_col=value_col
    )
    loader = DataLoader(
        dataset,
        batch_size=256,  # Larger batch fine for inference
        shuffle=False,
        num_workers=num_workers,
    )

    # ── Inference ──────────────────────────────────────────────────────────────

    print("Running inference ...")
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            grid = batch["grid"].to(device)
            time_features = batch["time_features"].to(device)
            target = batch["target"]

            pred = model(grid, time_features)

            if model_config.output_capacity_factor:
                total_cap = batch["total_capacity"].to(device)
                pred = pred * total_cap

            all_preds.append(pred.cpu().numpy())
            all_targets.append(target.numpy())

    preds = np.concatenate(all_preds)      # (n_samples,)
    targets = np.concatenate(all_targets)  # (n_samples,)

    print(f"  Evaluated {len(preds):,} samples")

    # ── Compute metrics ────────────────────────────────────────────────────────

    errors = preds - targets
    abs_errors = np.abs(errors)

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    bias = float(np.mean(errors))

    # nRMSE: RMSE normalised by mean actual generation
    mean_actual = float(np.mean(targets))
    nrmse = rmse / mean_actual * 100 if mean_actual > 0 else np.nan

    # R²
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((targets - mean_actual) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    # MAPE — exclude near-zero hours (< 1% of max actual) to avoid instability
    threshold = np.max(targets) * 0.01
    nonzero_mask = targets > threshold
    if nonzero_mask.sum() > 0:
        mape = float(
            np.mean(np.abs(errors[nonzero_mask] / targets[nonzero_mask])) * 100
        )
    else:
        mape = np.nan

    # Peak MAE — hours above 90th percentile of actual generation
    p90 = np.percentile(targets, 90)
    peak_mask = targets > p90
    peak_mae = float(np.mean(abs_errors[peak_mask])) if peak_mask.sum() > 0 else np.nan

    metrics = {
        "MAE (MWh)": mae,
        "RMSE (MWh)": rmse,
        "Bias (MWh)": bias,
        "nRMSE (%)": nrmse,
        "MAPE (%)": mape,
        "R²": r2,
        "Peak MAE (MWh, >p90)": peak_mae,
        "Mean actual (MWh)": mean_actual,
        "Peak actual (MWh)": float(np.max(targets)),
        "n_samples": len(preds),
        "split": split,
    }

    # ── Print summary ──────────────────────────────────────────────────────────

    print(f"\n{'─' * 40}")
    print(f"  Evaluation results ({split} set)")
    print(f"{'─' * 40}")
    print(f"  MAE:              {mae:>10.1f} MWh")
    print(f"  RMSE:             {rmse:>10.1f} MWh")
    print(f"  Bias:             {bias:>10.1f} MWh  {'(over)' if bias > 0 else '(under)'}")
    print(f"  nRMSE:            {nrmse:>10.2f} %")
    print(f"  MAPE:             {mape:>10.2f} %")
    print(f"  R²:               {r2:>10.4f}")
    print(f"  Peak MAE (>p90):  {peak_mae:>10.1f} MWh")
    print(f"{'─' * 40}")
    print(f"  Mean actual:      {mean_actual:>10.1f} MWh")
    print(f"  Peak actual:      {float(np.max(targets)):>10.1f} MWh")
    print(f"  Samples:          {len(preds):>10,}")

    # ── Save ───────────────────────────────────────────────────────────────────

    # Predictions with timestamps
    splits_df = pd.read_csv(features_dir / "splits.csv")
    splits_df[time_col] = pd.to_datetime(splits_df[time_col])
    test_times = splits_df.loc[splits_df["split"] == split, time_col].values

    pred_df = pd.DataFrame({
        time_col: test_times,
        "actual_mwh": targets,
        "predicted_mwh": preds,
        "error_mwh": errors,
        "abs_error_mwh": abs_errors,
    })
    pred_path = out_dir / f"predictions_{split}.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"\n  Predictions saved to: {pred_path}")

    # Metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_path = out_dir / f"metrics_{split}.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"  Metrics saved to:     {metrics_path}")

    return metrics_df

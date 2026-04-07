import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from torch.utils.data import DataLoader

from renewables_forecasting.config.paths import SOLAR_FEATURES_DIR, PROJECT_ROOT
from renewables_forecasting.models.config import ModelConfig
from renewables_forecasting.models.cnn import GenerationCNN
from renewables_forecasting.models.dataset import GenerationDataset


# ── Config ─────────────────────────────────────────────────────────────────────


FEATURES_DIR = SOLAR_FEATURES_DIR
TIME_COL = "time"
VALUE_COL = "generation_mwh"

# Path to global results file shared across all model runs
GLOBAL_RESULTS_PATH = PROJECT_ROOT / "data" / "models" / "solar" / "all_results.csv"

# Set to True to append this run's metrics to the global results file
APPEND_TO_GLOBAL_RESULTS = True


# ── Model loading ──────────────────────────────────────────────────────────────

def _load_model(model_dir: Path, device: torch.device) -> GenerationCNN:
    model_config = ModelConfig.from_yaml(model_dir / "config.yaml")
    model = GenerationCNN(model_config).to(device)
    model.load_state_dict(torch.load(model_dir / "best_model.pt", map_location=device))
    model.eval()
    return model


# ── Inference ──────────────────────────────────────────────────────────────────

def _predict(
        model: GenerationCNN,
        features_dir: Path,
        split: str,
        device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (times, actuals, predictions) for a given split."""
    dataset = GenerationDataset(features_dir, split=split,
                                time_col=TIME_COL, value_col=VALUE_COL)
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            grid = batch["grid"].to(device)
            time_features = batch["time_features"].to(device)
            pred = model(grid, time_features)

            if model.config.output_capacity_factor:
                total_cap = batch["total_capacity"].to(device)
                pred = pred * total_cap

            all_preds.append(pred.cpu().numpy())
            all_targets.append(batch["target"].numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    splits_df = pd.read_csv(features_dir / "splits.csv")
    splits_df[TIME_COL] = pd.to_datetime(splits_df[TIME_COL])
    times = splits_df.loc[splits_df["split"] == split, TIME_COL].values

    return times, targets, preds


# ── Metrics ────────────────────────────────────────────────────────────────────

def _compute_metrics(
        times: np.ndarray,
        actuals: np.ndarray,
        preds: np.ndarray,
        split: str,
        model_name: str,
) -> dict:
    errors = preds - actuals
    abs_errors = np.abs(errors)

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mse = float(np.mean(errors ** 2))
    bias = float(np.mean(errors))
    mean_actual = float(np.mean(actuals))
    std_actual = float(np.std(actuals))
    peak_actual = float(np.max(actuals))

    nrmse = rmse / mean_actual * 100 if mean_actual > 0 else np.nan
    nrmse_peak = rmse / peak_actual * 100 if peak_actual > 0 else np.nan

    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((actuals - mean_actual) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    correlation = float(np.corrcoef(actuals, preds)[0, 1])

    threshold = peak_actual * 0.01
    nonzero_mask = actuals > threshold
    mape = float(
        np.mean(np.abs(errors[nonzero_mask] / actuals[nonzero_mask])) * 100
    ) if nonzero_mask.sum() > 0 else np.nan

    p90 = np.percentile(actuals, 90)
    peak_mask = actuals > p90
    peak_mae = float(np.mean(abs_errors[peak_mask])) if peak_mask.sum() > 0 else np.nan
    peak_rmse = float(
        np.sqrt(np.mean(errors[peak_mask] ** 2))
    ) if peak_mask.sum() > 0 else np.nan

    # Skill score vs mean predictor (equivalent to R²)
    mse_mean = float(np.mean((actuals - mean_actual) ** 2))
    skill_score = float(1 - mse / mse_mean) if mse_mean > 0 else np.nan

    # Skill score vs persistence (previous day same hour)
    # For each timestamp find the value 24 hours earlier in the series.
    # Samples within the first 24 hours have no previous day and are excluded.
    times_dt_full = pd.DatetimeIndex(times)
    time_to_idx = pd.Series(np.arange(len(actuals)), index=times_dt_full)
    persistence_errors_sq = []
    model_errors_sq_persist = []
    for i, t in enumerate(times_dt_full):
        prev = t - pd.Timedelta(hours=24)
        if prev in time_to_idx.index:
            j = time_to_idx[prev]
            persist_pred = actuals[j]
            persistence_errors_sq.append((persist_pred - actuals[i]) ** 2)
            model_errors_sq_persist.append(errors[i] ** 2)
    if len(persistence_errors_sq) > 0:
        mse_persistence = float(np.mean(persistence_errors_sq))
        mse_model_vs_persist = float(np.mean(model_errors_sq_persist))
        skill_score_persistence = float(
            1 - mse_model_vs_persist / mse_persistence
        ) if mse_persistence > 0 else np.nan
    else:
        mse_persistence = np.nan
        skill_score_persistence = np.nan

    times_dt = pd.DatetimeIndex(times)
    seasonal = {}
    for season, months in [("winter", [12, 1, 2]), ("spring", [3, 4, 5]),
                            ("summer", [6, 7, 8]), ("autumn", [9, 10, 11])]:
        mask = times_dt.month.isin(months)
        if mask.sum() > 0:
            seasonal[f"MAE_{season} (MWh)"] = float(np.mean(abs_errors[mask]))
            seasonal[f"RMSE_{season} (MWh)"] = float(np.sqrt(np.mean(errors[mask] ** 2)))
            seasonal[f"Bias_{season} (MWh)"] = float(np.mean(errors[mask]))

    return {
        "model": model_name,
        "split": split,
        "n_samples": len(preds),
        "MAE (MWh)": mae,
        "RMSE (MWh)": rmse,
        "MSE (MWh²)": mse,
        "Bias (MWh)": bias,
        "nRMSE_mean (%)": nrmse,
        "nRMSE_peak (%)": nrmse_peak,
        "MAPE (%)": mape,
        "R²": r2,
        "Pearson r": correlation,
        "Skill score": skill_score,
        "Skill score (vs persistence)": skill_score_persistence,
        "Peak MAE (MWh, >p90)": peak_mae,
        "Peak RMSE (MWh, >p90)": peak_rmse,
        "Mean actual (MWh)": mean_actual,
        "Std actual (MWh)": std_actual,
        "Peak actual (MWh)": peak_actual,
        **seasonal,
    }


def _print_metrics(metrics: dict) -> None:
    split = metrics["split"]
    print(f"\n{'─' * 50}")
    print(f"  {split.upper()} SET  ({metrics['n_samples']:,} samples)")
    print(f"{'─' * 50}")
    print(f"  MAE:              {metrics['MAE (MWh)']:>10.1f} MWh")
    print(f"  RMSE:             {metrics['RMSE (MWh)']:>10.1f} MWh")
    print(f"  Bias:             {metrics['Bias (MWh)']:>+10.1f} MWh  "
          f"({'over' if metrics['Bias (MWh)'] > 0 else 'under'})")
    print(f"  nRMSE (mean):     {metrics['nRMSE_mean (%)']:>10.2f} %")
    print(f"  nRMSE (peak):     {metrics['nRMSE_peak (%)']:>10.2f} %")
    print(f"  MAPE:             {metrics['MAPE (%)']:>10.2f} %")
    print(f"  R²:               {metrics['R²']:>10.4f}")
    print(f"  Pearson r:        {metrics['Pearson r']:>10.4f}")
    print(f"  Skill score:      {metrics['Skill score']:>10.4f}")
    print(f"  Skill (persist.): {metrics['Skill score (vs persistence)']:>10.4f}")
    print(f"  Peak MAE (>p90):  {metrics['Peak MAE (MWh, >p90)']:>10.1f} MWh")
    print(f"  Peak RMSE (>p90): {metrics['Peak RMSE (MWh, >p90)']:>10.1f} MWh")
    print(f"\n  Seasonal MAE:")
    for season in ("winter", "spring", "summer", "autumn"):
        key = f"MAE_{season} (MWh)"
        if key in metrics:
            print(f"    {season.capitalize():<8} {metrics[key]:>8.1f} MWh")


# ── Global results ─────────────────────────────────────────────────────────────

def _append_to_global_results(
        metrics_list: list[dict],
        results_path: Path,
        model_name: str,
) -> None:
    """
    Appends metrics for all splits of a model run to the global results CSV.
    If the file does not exist it is created. If the model already has an
    entry in the file it is overwritten — so re-running evaluation for the
    same model updates rather than duplicates its row.
    """
    new_df = pd.DataFrame(metrics_list)

    if results_path.exists():
        existing_df = pd.read_csv(results_path)
        # Remove any existing rows for this model to avoid duplicates
        existing_df = existing_df[existing_df["model"] != model_name]
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined = new_df

    # Sort by model name and split for readability
    split_order = {"train": 0, "val": 1, "test": 2}
    combined["_split_order"] = combined["split"].map(split_order)
    combined = combined.sort_values(["model", "_split_order"]).drop(
        columns=["_split_order"]
    )

    results_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(results_path, index=False)
    print(f"\nGlobal results updated: {results_path}")
    print(f"  Models in file: {sorted(combined['model'].unique().tolist())}")


# ── Results table plot ─────────────────────────────────────────────────────────

def plot_results_table(
        results_path: Path,
        splits: list[str] = ("train", "val", "test"),
        metrics: list[str] = ("MAE (MWh)", "RMSE (MWh)", "Bias (MWh)",
                              "nRMSE_mean (%)", "R²", "Skill score",
                              "Skill score (vs persistence)"),
        out_path: Path | None = None,
) -> None:
    """
    Reads the global results CSV and plots a formatted table comparing all
    models across the selected splits and metrics.

    Parameters
    ----------
    results_path:
        Path to the global results CSV produced by the evaluation script.
    splits:
        Which splits to include. Default: val and test.
    metrics:
        Which metrics to show in the table. Default: key summary metrics.
    out_path:
        Optional path to save the table as a PNG. If None, shows interactively.
    """
    df = pd.read_csv(results_path)
    df = df[df["split"].isin(splits)]

    # Build a multi-index table: rows = models, columns = (split, metric)
    rows = []
    model_names = sorted(df["model"].unique())

    for model in model_names:
        row = {"model": model}
        for split in splits:
            split_data = df[(df["model"] == model) & (df["split"] == split)]
            for metric in metrics:
                col_name = f"{split}/{metric}"
                if not split_data.empty and metric in split_data.columns:
                    row[col_name] = split_data[metric].values[0]
                else:
                    row[col_name] = np.nan
        rows.append(row)

    table_df = pd.DataFrame(rows).set_index("model")

    # Format values for display
    def _fmt(val, metric):
        if pd.isna(val):
            return "—"
        if "MWh" in metric:
            return f"{val:,.0f}"
        if "%" in metric:
            return f"{val:.2f}%"
        return f"{val:.4f}"

    display_data = []
    for model in model_names:
        row_vals = []
        for col in table_df.columns:
            metric = col.split("/", 1)[1]
            row_vals.append(_fmt(table_df.loc[model, col], metric))
        display_data.append(row_vals)

    # Shorten metric names for readability in column headers
    metric_short = {
        "MAE (MWh)": "MAE",
        "RMSE (MWh)": "RMSE",
        "Bias (MWh)": "Bias",
        "nRMSE_mean (%)": "nRMSE",
        "nRMSE_peak (%)": "nRMSE pk",
        "MAPE (%)": "MAPE",
        "R²": "R²",
        "Pearson r": "r",
        "Skill score": "Skill",
        "Skill score (vs persistence)": "Skill/pers",
        "Peak MAE (MWh, >p90)": "PkMAE",
        "Peak RMSE (MWh, >p90)": "PkRMSE",
    }
    col_labels = [
        f"{s}\n{metric_short.get(m, m)}"
        for s, m in [c.split("/", 1) for c in table_df.columns]
    ]

    fig_width = max(16, len(col_labels) * 2.4)
    fig_height = max(4, len(model_names) * 0.8 + 2.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=display_data,
        rowLabels=model_names,
        colLabels=col_labels,
        cellLoc="center",
        rowLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(col_labels) + 1)))

    # Increase row height for all cells
    for (row, col), cell in table.get_celld().items():
        cell.set_height(0.12)
    # Increase row height so header text is not clipped
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_height(0.12)
        else:
            cell.set_height(0.08)

    # Metrics where lower is better — all others assumed higher is better
    lower_is_better = {
        "MAE (MWh)", "RMSE (MWh)", "MSE (MWh²)", "nRMSE_mean (%)",
        "nRMSE_peak (%)", "MAPE (%)", "Peak MAE (MWh, >p90)",
        "Peak RMSE (MWh, >p90)",
    }

    # Find best and worst row index per column
    best_cells = set()
    worst_cells = set()

    for col_idx, col in enumerate(table_df.columns):
        metric = col.split("/", 1)[1]
        col_vals = table_df[col].values
        float_vals = col_vals.astype(float)
        if np.isnan(float_vals).all() or (~np.isnan(float_vals)).sum() < 2:
            continue
        if metric in lower_is_better:
            best_row = int(np.nanargmin(float_vals)) + 1
            worst_row = int(np.nanargmax(float_vals)) + 1
        else:
            best_row = int(np.nanargmax(float_vals)) + 1
            worst_row = int(np.nanargmin(float_vals)) + 1
        best_cells.add((best_row, col_idx))
        worst_cells.add((worst_row, col_idx))

    # Style all cells
    for (row, col), cell in table.get_celld().items():
        if row == 0 or col == -1:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif (row, col) in best_cells:
            cell.set_facecolor("#27ae60")
            cell.set_text_props(color="white", fontweight="bold")
        elif (row, col) in worst_cells:
            cell.set_facecolor("#e74c3c")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f2f2f2")

    ax.set_title("Model comparison", fontsize=12, fontweight="bold", pad=20)
    plt.tight_layout()

    if out_path is not None:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Table saved to: {out_path}")
    else:
        plt.show()
    plt.close()


# ── Plots ──────────────────────────────────────────────────────────────────────

def _plot_full_period(times, actuals, preds, metrics, out_dir, zoom_weeks=4):
    split = metrics["split"]
    times_dt = pd.DatetimeIndex(times)
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=False)

    ax = axes[0]
    ax.plot(times_dt, actuals, label="Actual", linewidth=0.8, alpha=0.9)
    ax.plot(times_dt, preds, label="Predicted", linewidth=0.8, alpha=0.7)
    ax.set_ylabel("Generation (MWh)")
    ax.set_title(
        f"{split.capitalize()} set — full period  |  "
        f"RMSE: {metrics['RMSE (MWh)']:,.0f} MWh  |  "
        f"MAE: {metrics['MAE (MWh)']:,.0f} MWh  |  "
        f"Bias: {metrics['Bias (MWh)']:+,.0f} MWh"
    )
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    ax2 = axes[1]
    zoom_end = times_dt[0] + pd.Timedelta(weeks=zoom_weeks)
    zoom_mask = times_dt <= zoom_end
    ax2.plot(times_dt[zoom_mask], actuals[zoom_mask], label="Actual",
             linewidth=1.2, alpha=0.9)
    ax2.plot(times_dt[zoom_mask], preds[zoom_mask], label="Predicted",
             linewidth=1.2, alpha=0.8)
    ax2.set_ylabel("Generation (MWh)")
    ax2.set_title(f"First {zoom_weeks} weeks (detail)")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator())
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    path = out_dir / f"predictions_{split}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def _plot_scatter(actuals, preds, metrics, out_dir):
    split = metrics["split"]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(actuals, preds, alpha=0.15, s=3, color="steelblue")
    lim = max(actuals.max(), preds.max()) * 1.05
    ax.plot([0, lim], [0, lim], "r--", linewidth=1, label="Perfect prediction")
    ax.set_xlabel("Actual (MWh)")
    ax.set_ylabel("Predicted (MWh)")
    ax.set_title(
        f"{split.capitalize()} set — scatter  |  "
        f"RMSE: {metrics['RMSE (MWh)']:,.0f} MWh  |  R²: {metrics['R²']:.4f}"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    plt.tight_layout()
    path = out_dir / f"scatter_{split}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def _plot_week(times, actuals, preds, split, out_dir, season):
    times_dt = pd.DatetimeIndex(times)
    month = 7 if season == "summer" else 1
    label = "Summer (July)" if season == "summer" else "Winter (January)"

    month_mask = times_dt.month == month
    if not month_mask.any():
        print(f"  No {season} data found in {split} set — skipping")
        return

    week_start = times_dt[month_mask][0]
    week_end = week_start + pd.Timedelta(weeks=1)
    week_mask = (times_dt >= week_start) & (times_dt < week_end)
    if week_mask.sum() == 0:
        return

    errors = preds[week_mask] - actuals[week_mask]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(times_dt[week_mask], actuals[week_mask], label="Actual",
            linewidth=1.5, alpha=0.9)
    ax.plot(times_dt[week_mask], preds[week_mask], label="Predicted",
            linewidth=1.5, alpha=0.8)
    ax.set_ylabel("Generation (MWh)")
    ax.set_title(
        f"{split.capitalize()} set — {label}  |  "
        f"RMSE: {np.sqrt(np.mean(errors**2)):,.0f} MWh  |  "
        f"MAE: {np.mean(np.abs(errors)):,.0f} MWh"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    path = out_dir / f"week_{season}_{split}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for num in range(6, 10):
        MODEL_DIR = PROJECT_ROOT / "data" / "models" / "solar" / f"solar_v{num}"
        if not MODEL_DIR.exists():
            print(f"Skipping model solar_v{num}. No directory found.")
            continue

        OUT_DIR = MODEL_DIR / "eval"
        model_name = MODEL_DIR.name  # e.g. "solar_v2"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device:     {device}")
        print(f"Model:      {MODEL_DIR}")
        print(f"Model name: {model_name}")

        model = _load_model(MODEL_DIR, device)
        print(f"Model parameters: {model.n_parameters():,}")

        OUT_DIR.mkdir(parents=True, exist_ok=True)

        all_metrics = []

        for split in ("train", "val", "test"):
            print(f"\nPredicting {split} set ...")
            times, actuals, preds = _predict(model, FEATURES_DIR, split, device)

            metrics = _compute_metrics(times, actuals, preds, split, model_name)
            _print_metrics(metrics)
            all_metrics.append(metrics)

            # Save per-split predictions CSV
            errors = preds - actuals
            pred_df = pd.DataFrame({
                TIME_COL: times,
                "actual_mwh": actuals,
                "predicted_mwh": preds,
                "error_mwh": errors,
                "abs_error_mwh": np.abs(errors),
            })
            pred_df.to_csv(OUT_DIR / f"predictions_{split}.csv", index=False)
            print(f"  Predictions saved to: {OUT_DIR / f'predictions_{split}.csv'}")

            # Plots for val and test only
            if split != "train":
                _plot_full_period(times, actuals, preds, metrics, OUT_DIR)
                _plot_scatter(actuals, preds, metrics, OUT_DIR)
                _plot_week(times, actuals, preds, split, OUT_DIR, "summer")
                _plot_week(times, actuals, preds, split, OUT_DIR, "winter")

        # Save per-model metrics CSV
        metrics_df = pd.DataFrame(all_metrics)
        metrics_path = OUT_DIR / "metrics_all_splits.csv"
        metrics_df.to_csv(metrics_path, index=False)
        print(f"\nModel metrics saved to: {metrics_path}")

        # Optionally append to global results file
        if APPEND_TO_GLOBAL_RESULTS:
            _append_to_global_results(all_metrics, GLOBAL_RESULTS_PATH, model_name)

        # Plot comparison table if global results file exists
        if GLOBAL_RESULTS_PATH.exists():
            plot_results_table(
                results_path=GLOBAL_RESULTS_PATH,
                splits=["train", "val", "test"],
                out_path=GLOBAL_RESULTS_PATH.parent / "results_table.png",
            )

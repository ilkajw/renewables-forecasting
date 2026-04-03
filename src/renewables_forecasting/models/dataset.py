import numpy as np
import pandas as pd
import xarray as xr
import torch
from torch.utils.data import Dataset
from pathlib import Path


class GenerationDataset(Dataset):
    """
    PyTorch Dataset for hourly renewable generation forecasting.

    Loads the feature dataset produced by build_feature_dataset() and serves
    one sample per timestamp consisting of:
        - grid: spatial feature grid (channels, lat, lon) as float32 tensor
        - time_features: cyclical time encodings (4,) as float32 tensor
        - target: generation in MWh as float32 scalar tensor

    Optionally loads total capacity per timestamp when the feature dataset
    was built with capacity_as_spatial_distribution=True or
    capacity_as_spatiotemporal_distribution=True, in which case total_capacity
    is also returned per sample for use in the training loop.

    Parameters
    ----------
    features_dir:
        Directory containing the feature dataset produced by
        build_feature_dataset(). Must contain features.zarr, target.csv,
        cyclical_features.csv, and splits.csv.
    split:
        Which split to load. One of 'train', 'val', 'test'.
    time_col:
        Name of the datetime column in CSV files. Default: 'time'.
    value_col:
        Name of the generation column in target.csv. Default: 'generation_mwh'.
    """

    def __init__(
            self,
            features_dir: Path,
            split: str,
            time_col: str = "time",
            value_col: str = "generation_mwh",
    ):
        assert split in ("train", "val", "test"), \
            f"split must be one of 'train', 'val', 'test', got '{split}'"

        self.features_dir = features_dir
        self.split = split

        # ── Load splits ───────────────────────────────────────────────────────
        splits_df = pd.read_csv(features_dir / "splits.csv")
        splits_df[time_col] = pd.to_datetime(splits_df[time_col])
        split_mask = splits_df["split"] == split
        self.indices = splits_df[split_mask].index.values
        self.times = splits_df.loc[split_mask, time_col].values

        print(f"  {split}: {len(self.indices):,} samples")

        # ── Load features ─────────────────────────────────────────────────────
        ds = xr.open_dataset(features_dir / "features.zarr", engine="zarr")
        self.features = torch.from_numpy(
            ds["features"].values.astype(np.float32)
        )  # (total_time, channels, lat, lon)

        # ── Load target ───────────────────────────────────────────────────────
        target_df = pd.read_csv(features_dir / "target.csv")
        self.target = torch.from_numpy(
            target_df[value_col].values.astype(np.float32)
        )  # (total_time,)

        # ── Load cyclical time features ───────────────────────────────────────
        cyclical_df = pd.read_csv(features_dir / "cyclical_features.csv")
        cyclical_cols = ["doy_sin", "doy_cos", "hod_sin", "hod_cos"]
        self.time_features = torch.from_numpy(
            cyclical_df[cyclical_cols].values.astype(np.float32)
        )  # (total_time, 4)

        # ── Load total capacity if present ────────────────────────────────────
        total_cap_path = features_dir / "total_cap_per_t.csv"
        if total_cap_path.exists():
            total_cap_df = pd.read_csv(total_cap_path)
            # Take the second column regardless of unit suffix in the name
            self.total_capacity = torch.from_numpy(
                total_cap_df.iloc[:, 1].values.astype(np.float32)
            )  # (total_time,)
        else:
            self.total_capacity = None

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        i = self.indices[idx]
        sample = {
            "grid": self.features[i],           # (channels, lat, lon)
            "time_features": self.time_features[i],  # (4,)
            "target": self.target[i],            # scalar
        }
        if self.total_capacity is not None:
            sample["total_capacity"] = self.total_capacity[i]  # scalar
        return sample
    
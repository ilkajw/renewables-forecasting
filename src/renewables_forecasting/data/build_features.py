import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List


_VALID_NORM_OPTIONS = ("none", "zscore", "minmax")


def _load_era5_variable(
        era5_dir: Path,
        variable: str,
) -> xr.DataArray:
    """Load and concatenate all monthly ERA5 NetCDF files for a single variable."""
    var_dir = era5_dir / variable
    files = sorted(var_dir.glob(f"{variable}_*.nc"))
    if not files:
        raise FileNotFoundError(f"No ERA5 files found for variable '{variable}' in {var_dir}")
    print(f"    Found {len(files)} monthly files for '{variable}'")
    ds = xr.open_mfdataset(files, combine="by_coords")
    return ds[variable]


def _load_capacity_grids(
        capacity_store_dir: Path,
) -> xr.DataArray:
    """Load and concatenate all monthly capacity grid Zarr stores."""
    stores = sorted(capacity_store_dir.glob("capacity_????_??.zarr"))
    if not stores:
        raise FileNotFoundError(
            f"No capacity grid stores found in {capacity_store_dir}. "
            f"Expected files matching 'capacity_YYYY_MM.zarr'."
        )
    print(f"    Found {len(stores)} monthly capacity stores")
    datasets = [xr.open_dataset(s, engine="zarr")["capacity_kw"] for s in stores]
    return xr.concat(datasets, dim="time")


def _broadcast_daily_capacity_to_hourly(
        capacity_da: xr.DataArray,
        hourly_times: pd.DatetimeIndex,
) -> np.ndarray:
    """
    Broadcasts a daily capacity grid (days × lat × lon) to hourly resolution
    by repeating each day's grid for all hours of that day.

    Parameters
    ----------
    capacity_da:
        Daily capacity grid DataArray with dims (time, latitude, longitude).
    hourly_times:
        Hourly timestamps from the ERA5 dataset to broadcast into.

    Returns
    -------
    np.ndarray of shape (len(hourly_times), lat, lon) as float32.
    """
    cap_time_index = pd.DatetimeIndex(capacity_da.time.values)
    hourly_dates_norm = hourly_times.normalize()

    positions = cap_time_index.get_indexer(hourly_dates_norm, method="nearest")

    n_missing = (positions == -1).sum()
    if n_missing > 0:
        raise ValueError(
            f"{n_missing} hourly timestamps could not be matched to a daily "
            f"capacity grid. Ensure capacity grids cover the full ERA5 time range. "
            f"Capacity range: {cap_time_index[0]} — {cap_time_index[-1]}. "
            f"ERA5 range: {hourly_times[0]} — {hourly_times[-1]}."
        )

    return capacity_da.values[positions].astype(np.float32)


def _compute_normalization_stats(
        features_np: np.ndarray,
        channel_names: List[str],
        train_mask: np.ndarray,
) -> pd.DataFrame:
    """
    Computes per-channel normalization statistics from the training split only.
    Statistics are collapsed over (time, lat, lon), one value per channel.

    Computes mean, std, min, and max — covering both zscore and minmax
    normalization. All four are always saved regardless of which normalization
    is applied, so they are available for any future use.

    Parameters
    ----------
    features_np:
        Feature array of shape (time, channels, lat, lon).
    channel_names:
        Ordered list of channel names matching the channel axis.
    train_mask:
        Boolean mask of shape (time,) marking training samples.

    Returns
    -------
    DataFrame with columns: channel, mean, std, min, max.
    """
    train_features = features_np[train_mask]  # (train_hours, channels, lat, lon)

    channel_mean = np.nanmean(train_features, axis=(0, 2, 3))  # (C,)
    channel_std = np.nanstd(train_features, axis=(0, 2, 3))    # (C,)
    channel_std = np.where(channel_std == 0, 1.0, channel_std)
    channel_min = np.nanmin(train_features, axis=(0, 2, 3))    # (C,)
    channel_max = np.nanmax(train_features, axis=(0, 2, 3))    # (C,)

    for ch, mu, sigma, mn, mx in zip(
            channel_names, channel_mean, channel_std, channel_min, channel_max
    ):
        print(f"    {ch:<20}  mean={mu:>12.4f}  std={sigma:>12.4f}  "
              f"min={mn:>12.4f}  max={mx:>12.4f}")

    return pd.DataFrame({
        "channel": channel_names,
        "mean": channel_mean,
        "std": channel_std,
        "min": channel_min,
        "max": channel_max,
    })


def _apply_normalization(
        features_np: np.ndarray,
        channel_indices: List[int],
        norm_df: pd.DataFrame,
        channel_names: List[str],
        method: str,
) -> None:
    """
    Applies normalization in-place to specified channel indices.

    Parameters
    ----------
    features_np:
        Feature array of shape (time, channels, lat, lon). Modified in-place.
    channel_indices:
        Indices of channels to normalize.
    norm_df:
        DataFrame with normalization statistics, indexed by channel name.
    channel_names:
        Full list of channel names corresponding to the channel axis.
    method:
        One of 'zscore' or 'minmax'.
    """
    norm_lookup = norm_df.set_index("channel")
    for i in channel_indices:
        ch = channel_names[i]
        if method == "zscore":
            mu = norm_lookup.loc[ch, "mean"]
            sigma = norm_lookup.loc[ch, "std"]
            features_np[:, i, :, :] = (features_np[:, i, :, :] - mu) / sigma
            print(f"    zscore normalized '{ch}'  (mean={mu:.4f}, std={sigma:.4f})")
        elif method == "minmax":
            mn = norm_lookup.loc[ch, "min"]
            mx = norm_lookup.loc[ch, "max"]
            rng = mx - mn if mx != mn else 1.0
            features_np[:, i, :, :] = (features_np[:, i, :, :] - mn) / rng
            print(f"    minmax normalized '{ch}'  (min={mn:.4f}, max={mx:.4f})")


def build_feature_dataset(
        era5_dir: Path,
        era5_variables: List[str],
        capacity_store_dir: Path,
        smard_csv_path: Path,
        cyclical_features_csv_path: Path,
        out_dir: Path,
        train_end_year: int = 2022,
        val_end_year: int = 2023,
        capacity_unit_divisor: float = 1.0,
        capacity_as_spatial_distribution: bool = False,
        capacity_as_spatiotemporal_distribution: bool = False,
        normalize_weather: str = "none",
        normalize_capacity: str = "none",
        time_col: str = "time",
        value_col: str = "generation_mwh",
) -> None:
    """
    Builds the CNN input feature dataset for either the solar or wind model.

    For each hourly timestamp in the study period, constructs a multi-channel
    2D grid (channels × lat × lon) by stacking ERA5 weather variables with
    the capacity grid for that day. Output is channels-first to match
    PyTorch's expected (batch, channels, lat, lon) convention.

    All arrays are cast to float32. Individual channel arrays are deleted
    immediately after stacking to minimise peak memory usage.

    ── Capacity channel modes ────────────────────────────────────────────────────

    Three mutually exclusive modes control how the capacity channel is expressed.
    capacity_as_spatial_distribution and capacity_as_spatiotemporal_distribution
    cannot both be True — a ValueError is raised if attempted.

    Option 1 — raw (both False, default):
        Absolute capacity values after capacity_unit_divisor is applied.

    Option 2 — spatial distribution (capacity_as_spatial_distribution=True):
        Each cell's share of total installed capacity for that hour:
            capacity[t, i, j] = P[t, i, j] / sum_{i,j} P[t, i, j]
        Values sum to 1 across lat/lon per hour. Absolute growth is removed —
        recoverable only via total_cap_per_t.csv which is saved alongside.

    Option 3 — spatiotemporal distribution (capacity_as_spatiotemporal_distribution=True):
        Each cell's value divided by the global sum over ALL training time
        steps and ALL grid cells:
            capacity[t, i, j] = P[t, i, j] / sum_{t_train} sum_{i,j} P[t, i, j]
        Denominator computed on training data only for consistency.
        total_cap_per_t.csv saved for potential future use.

    ── Normalization ─────────────────────────────────────────────────────────────

    normalize_weather and normalize_capacity each accept:
        "none"   — no normalization (default)
        "zscore" — subtract mean, divide by std (fit on training data only)
        "minmax" — scale to [0, 1] using min/max (fit on training data only)

    All four statistics (mean, std, min, max) are always computed from the
    training split and saved to normalization_stats.csv regardless of which
    method is chosen, so they remain available for inference time.

    Note: normalize_capacity is applied after the capacity channel mode
    transformation. For spatial/spatiotemporal distributions the values are
    already on a natural bounded scale — normalize_capacity="none" is usually
    appropriate in those cases.

    ── Output layout ─────────────────────────────────────────────────────────────
        features.zarr           — stacked grids (time, channels, lat, lon)
        normalization_stats.csv — mean, std, min, max per channel (train only)
        target.csv              — SMARD generation in MWh per timestamp
        cyclical_features.csv   — sin/cos time encodings per timestamp
        splits.csv              — train/val/test split label per timestamp

        Options 2 and 3 only:
        total_cap_per_t.csv     — total installed capacity scalar per timestamp

    Parameters
    ----------
    era5_dir:
        Root directory of ERA5 data with per-variable subdirectories.
    era5_variables:
        Ordered list of ERA5 variable names to stack as channels.
        The capacity channel is always appended last as 'capacity'.
    capacity_store_dir:
        Directory containing monthly capacity Zarr stores.
    smard_csv_path:
        Path to SMARD generation CSV.
    cyclical_features_csv_path:
        Path to cyclical time features CSV.
    out_dir:
        Directory to write all output files.
    train_end_year:
        Last year (inclusive) of the training split. Default: 2022.
    val_end_year:
        Last year (inclusive) of the validation split. Default: 2023.
    capacity_unit_divisor:
        Divisor applied to raw capacity values (kW) before processing.
        Default 1.0 (no conversion). Pass 1000.0 for kW → MW.
        Has no effect on spatial/spatiotemporal distributions.
    capacity_as_spatial_distribution:
        If True, capacity channel is per-hour spatial distribution.
        Mutually exclusive with capacity_as_spatiotemporal_distribution.
    capacity_as_spatiotemporal_distribution:
        If True, capacity channel is normalised by global training sum.
        Mutually exclusive with capacity_as_spatial_distribution.
    normalize_weather:
        Normalization method for ERA5 weather channels.
        One of "none", "zscore", "minmax". Default "none".
    normalize_capacity:
        Normalization method for the capacity channel.
        One of "none", "zscore", "minmax". Default "none".
    time_col:
        Name of the datetime column in input CSVs. Default: 'time'.
    value_col:
        Name of the generation column in the SMARD CSV. Default: 'generation_mwh'.
    """

    # ── Validate parameters ────────────────────────────────────────────────────

    if capacity_as_spatial_distribution and capacity_as_spatiotemporal_distribution:
        raise ValueError(
            "capacity_as_spatial_distribution and capacity_as_spatiotemporal_distribution "
            "are mutually exclusive — at most one can be True."
        )
    if normalize_weather not in _VALID_NORM_OPTIONS:
        raise ValueError(f"normalize_weather must be one of {_VALID_NORM_OPTIONS}")
    if normalize_capacity not in _VALID_NORM_OPTIONS:
        raise ValueError(f"normalize_capacity must be one of {_VALID_NORM_OPTIONS}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load ERA5 variables ────────────────────────────────────────────────────

    print("Loading ERA5 variables ...")
    era5_arrays = []
    ref_da = None

    for var in era5_variables:
        print(f"  {var}")
        da = _load_era5_variable(era5_dir, var)
        era5_arrays.append(da.values.astype(np.float32))
        if ref_da is None:
            ref_da = da

    hourly_times = pd.DatetimeIndex(ref_da.valid_time.values)
    lats = ref_da.latitude.values
    lons = ref_da.longitude.values

    print(f"\n  ERA5 time range: {hourly_times[0]} — {hourly_times[-1]}")
    print(f"  Grid:            {len(lats)} lat × {len(lons)} lon")

    # ── Load and broadcast capacity grids ──────────────────────────────────────

    print("\nLoading capacity grids ...")
    capacity_da = _load_capacity_grids(capacity_store_dir)

    if capacity_unit_divisor != 1.0:
        print(f"Converting capacity units (÷ {capacity_unit_divisor}) ...")
        capacity_da = capacity_da / capacity_unit_divisor

    print("Broadcasting daily capacity to hourly ...")
    capacity_hourly = _broadcast_daily_capacity_to_hourly(capacity_da, hourly_times)
    del capacity_da

    cap_unit = "MW" if capacity_unit_divisor == 1000.0 else "kW"

    # ── Compute total capacity per hour ────────────────────────────────────────

    total_capacity_hourly = capacity_hourly.sum(axis=(1, 2)).astype(np.float32)
    print(f"\n  Total capacity range: "
          f"{total_capacity_hourly.min():.1f} — {total_capacity_hourly.max():.1f} {cap_unit}")

    # ── Align timestamps (needed before spatiotemporal sum) ───────────────────
    # Load SMARD and cyclical here so we can compute train_mask before
    # building the capacity channel — needed for option 3 training-only sum.

    print("\nLoading SMARD target ...")
    df_target = pd.read_csv(smard_csv_path)
    df_target[time_col] = pd.to_datetime(df_target[time_col])
    df_target = df_target.set_index(time_col)[value_col]

    print("Loading cyclical time features ...")
    df_cyclical = pd.read_csv(cyclical_features_csv_path)
    df_cyclical[time_col] = pd.to_datetime(df_cyclical[time_col])
    df_cyclical = df_cyclical.set_index(time_col)

    print("\nAligning timestamps across ERA5, SMARD, and cyclical features ...")
    common_times = (
        hourly_times
        .intersection(pd.DatetimeIndex(df_target.index))
        .intersection(pd.DatetimeIndex(df_cyclical.index))
    )

    print(f"  ERA5 hours:    {len(hourly_times):>10,}")
    print(f"  SMARD hours:   {len(df_target):>10,}")
    print(f"  Common hours:  {len(common_times):>10,}")

    era5_time_to_idx = pd.Series(np.arange(len(hourly_times)), index=hourly_times)
    feature_indices = era5_time_to_idx.loc[common_times].values

    # Compute train mask on common_times — needed for option 3 and norm stats
    years = common_times.year
    train_mask = years <= train_end_year
    val_mask = (years > train_end_year) & (years <= val_end_year)
    test_mask = years > val_end_year

    print(f"\n  Split summary:")
    print(f"    Train (≤ {train_end_year}):      {train_mask.sum():>8,} hours")
    print(f"    Val   ({train_end_year + 1} – {val_end_year}): {val_mask.sum():>8,} hours")
    print(f"    Test  (> {val_end_year}):      {test_mask.sum():>8,} hours")

    # ── Build capacity channel ─────────────────────────────────────────────────

    if capacity_as_spatial_distribution:
        safe_total = np.where(total_capacity_hourly > 0, total_capacity_hourly, 1.0)
        capacity_channel = (
            capacity_hourly / safe_total[:, None, None]
        ).astype(np.float32)
        print(f"\n  Capacity channel: per-hour spatial distribution (cell / total_t)")
        print(f"  Value range: {capacity_channel.min():.6f} — {capacity_channel.max():.6f}")
        spatiotemporal_sum = None

    elif capacity_as_spatiotemporal_distribution:
        # Sum over training timestamps only — consistent with train-only norm stats
        train_capacity = capacity_hourly[feature_indices[train_mask]]
        spatiotemporal_sum = float(train_capacity.sum())
        capacity_channel = (capacity_hourly / spatiotemporal_sum).astype(np.float32)
        print(f"\n  Capacity channel: spatiotemporal distribution (cell / train_sum)")
        print(f"  Training spatiotemporal sum: {spatiotemporal_sum:.4f} {cap_unit}·hours")
        print(f"  Value range: {capacity_channel.min():.8f} — {capacity_channel.max():.8f}")

    else:
        capacity_channel = capacity_hourly
        spatiotemporal_sum = None
        print(f"\n  Capacity channel: absolute values ({cap_unit})")

    del capacity_hourly

    # ── Stack all channels → (time, channels, lat, lon) ───────────────────────

    print("\nStacking channels ...")
    all_channels = era5_variables + ["capacity"]
    all_arrays = era5_arrays + [capacity_channel]
    features_np = np.stack(all_arrays, axis=1)  # (time, C, lat, lon)
    del all_arrays, era5_arrays, capacity_channel

    # Filter to common timestamps
    features_np = features_np[feature_indices]

    print(f"  Feature array shape: {features_np.shape}  (time, channels, lat, lon)")
    print(f"  Feature array dtype: {features_np.dtype}")
    print(f"  Channels: {all_channels}")

    target_aligned = df_target.loc[common_times].values
    total_capacity_aligned = total_capacity_hourly[feature_indices]
    cyclical_aligned = df_cyclical.loc[common_times]

    # ── Normalization stats (always computed from training split) ──────────────

    print("\nComputing normalization statistics from training data ...")
    norm_df = _compute_normalization_stats(features_np, all_channels, train_mask)

    # ── Apply normalization ────────────────────────────────────────────────────

    weather_indices = list(range(len(era5_variables)))
    capacity_index = len(era5_variables)  # capacity is always last channel

    if normalize_weather != "none":
        print(f"\nNormalizing weather channels ({normalize_weather}) ...")
        _apply_normalization(
            features_np, weather_indices, norm_df, all_channels, normalize_weather
        )
    if normalize_capacity != "none":
        print(f"\nNormalizing capacity channel ({normalize_capacity}) ...")
        _apply_normalization(
            features_np, [capacity_index], norm_df, all_channels, normalize_capacity
        )

    if normalize_weather == "none" and normalize_capacity == "none":
        print("  No normalization applied — all channels saved raw.")

    # ── Build transformation descriptions for attrs ────────────────────────────

    def _channel_transform_description(ch: str) -> str:
        parts = []
        if ch == "capacity":
            if capacity_as_spatial_distribution:
                parts.append("per-hour spatial distribution: cell / sum_{i,j} P[t]")
            elif capacity_as_spatiotemporal_distribution:
                parts.append(
                    f"spatiotemporal distribution: cell / train_sum "
                    f"(train_sum={spatiotemporal_sum:.4f} {cap_unit}·hours)"
                )
            else:
                parts.append(
                    f"unit conversion: kW ÷ {capacity_unit_divisor}"
                    if capacity_unit_divisor != 1.0
                    else "unit: kW (no conversion)"
                )
            parts.append(
                f"{normalize_capacity} normalized (fit on train)"
                if normalize_capacity != "none"
                else "not normalized"
            )
        else:
            parts.append("raw ERA5 values")
            parts.append(
                f"{normalize_weather} normalized (fit on train)"
                if normalize_weather != "none"
                else "not normalized"
            )
        return " | ".join(parts)

    channel_transforms = {ch: _channel_transform_description(ch) for ch in all_channels}

    # ── Save ───────────────────────────────────────────────────────────────────

    print("\nSaving feature dataset ...")

    if capacity_as_spatial_distribution:
        cap_channel_mode = "spatial_distribution"
    elif capacity_as_spatiotemporal_distribution:
        cap_channel_mode = "spatiotemporal_distribution"
    else:
        cap_channel_mode = "absolute"

    attrs = {
        "train_end_year": train_end_year,
        "val_end_year": val_end_year,
        "n_train": int(train_mask.sum()),
        "n_val": int(val_mask.sum()),
        "n_test": int(test_mask.sum()),
        "channels": str(all_channels),
        "capacity_channel_mode": cap_channel_mode,
        "capacity_unit": cap_unit,
        "target": "raw_generation_mwh",
        "normalize_weather": normalize_weather,
        "normalize_capacity": normalize_capacity,
        **{f"transform_{ch}": desc for ch, desc in channel_transforms.items()},
    }
    if capacity_as_spatiotemporal_distribution:
        attrs["capacity_spatiotemporal_sum"] = spatiotemporal_sum
        attrs["capacity_spatiotemporal_sum_unit"] = f"{cap_unit}·hours"

    features_da = xr.DataArray(
        features_np,
        dims=("time", "channel", "latitude", "longitude"),
        coords={
            "time": common_times,
            "channel": all_channels,
            "latitude": lats,
            "longitude": lons,
        },
        attrs=attrs,
    )
    feature_store = out_dir / "features.zarr"
    features_da.to_dataset(name="features").to_zarr(feature_store, mode="w")
    print(f"  features.zarr       → {feature_store}")

    norm_path = out_dir / "normalization_stats.csv"
    norm_df.to_csv(norm_path, index=False)
    print(f"  normalization_stats → {norm_path}")

    target_df = pd.DataFrame({time_col: common_times, value_col: target_aligned})
    target_df.to_csv(out_dir / "target.csv", index=False)
    print(f"  target.csv          → {out_dir / 'target.csv'}")

    if capacity_as_spatial_distribution or capacity_as_spatiotemporal_distribution:
        total_cap_col = f"total_capacity_{cap_unit.lower()}"
        total_cap_df = pd.DataFrame({
            time_col: common_times,
            total_cap_col: total_capacity_aligned,
        })
        total_cap_df.to_csv(out_dir / "total_cap_per_t.csv", index=False)
        print(f"  total_cap_per_t.csv → {out_dir / 'total_cap_per_t.csv'}")

    cyclical_aligned.to_csv(out_dir / "cyclical_features.csv")
    print(f"  cyclical_features   → {out_dir / 'cyclical_features.csv'}")

    split_labels = np.where(train_mask, "train", np.where(val_mask, "val", "test"))
    splits_df = pd.DataFrame({time_col: common_times, "split": split_labels})
    splits_df.to_csv(out_dir / "splits.csv", index=False)
    print(f"  splits.csv          → {out_dir / 'splits.csv'}")

    print(f"\nDone. Feature dataset saved to: {out_dir.resolve()}")
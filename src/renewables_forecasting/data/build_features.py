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
    train_features = features_np[train_mask]

    channel_mean = np.nanmean(train_features, axis=(0, 2, 3))
    channel_std = np.nanstd(train_features, axis=(0, 2, 3))
    channel_std = np.where(channel_std == 0, 1.0, channel_std)
    channel_min = np.nanmin(train_features, axis=(0, 2, 3))
    channel_max = np.nanmax(train_features, axis=(0, 2, 3))

    for ch, mu, sigma, mn, mx in zip(
            channel_names, channel_mean, channel_std, channel_min, channel_max
    ):
        print(f"    {ch:<30}  mean={mu:>12.4f}  std={sigma:>12.4f}  "
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
        capacity_weighted_weather: bool = False,
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

    Four mutually exclusive modes control how capacity information enters the
    feature dataset. At most one of the four boolean flags can be True.

    Option 1 — raw (all False, default):
        Capacity channel contains absolute values after capacity_unit_divisor
        is applied. Weather channels are unchanged.

    Option 2 — spatial distribution (capacity_as_spatial_distribution=True):
        Capacity channel is each cell's share of total installed capacity for
        that hour:
            capacity[t, i, j] = P[t, i, j] / sum_{i,j} P[t, i, j]
        Values sum to 1 across lat/lon per hour. Absolute growth is removed —
        recoverable only via total_cap_per_t.csv saved alongside.

    Option 3 — spatiotemporal distribution (capacity_as_spatiotemporal_distribution=True):
        Capacity channel is each cell's value divided by the global sum over
        ALL training time steps and ALL grid cells:
            capacity[t, i, j] = P[t, i, j] / sum_{t_train} sum_{i,j} P[t, i, j]
        Denominator computed on training data only for consistency.
        total_cap_per_t.csv saved for potential future use.

    Option 4 — capacity-weighted weather (capacity_weighted_weather=True):
        Inspired by Lindas et al. Each weather channel is multiplied
        elementwise by the spatiotemporal capacity weight:
            weather_weighted[t, i, j] = weather[t, i, j] * w[t, i, j]
        where w[t, i, j] = P[t, i, j] / sum_{t_train} sum_{i,j} P[t, i, j]
        (training-only spatiotemporal weights).
        No separate capacity channel is included — capacity information is
        fully embedded in the weighted weather channels. Channel names are
        suffixed with '_cap_weighted', e.g. 'ssrd_cap_weighted'.
        normalize_capacity is ignored in this mode.
        total_cap_per_t.csv is saved for potential use at inference time.

        Normalization compatibility:
            "zscore" — recommended. The weighted channels have very small
                       absolute values (weather magnitude × tiny weights),
                       making z-score normalization practically necessary
                       for stable training.
            "minmax" — compatible but less robust to extreme events.
            "none"   — not recommended. Raw weighted values are very small
                       and may cause numerical issues during training.

    ── Normalization ─────────────────────────────────────────────────────────────

    normalize_weather and normalize_capacity each accept:
        "none"   — no normalization (default)
        "zscore" — subtract mean, divide by std (fit on training data only)
        "minmax" — scale to [0, 1] using min/max (fit on training data only)

    All four statistics (mean, std, min, max) are always computed from the
    training split and saved to normalization_stats.csv regardless of which
    method is chosen.

    normalize_capacity is ignored when capacity_weighted_weather=True since
    there is no separate capacity channel.

    ── Output layout ─────────────────────────────────────────────────────────────
        features.zarr           — stacked grids (time, channels, lat, lon)
        normalization_stats.csv — mean, std, min, max per channel (train only)
        target.csv              — SMARD generation in MWh per timestamp
        cyclical_features.csv   — sin/cos time encodings per timestamp
        splits.csv              — train/val/test split label per timestamp

        Options 2, 3 and 4 only:
        total_cap_per_t.csv     — total installed capacity scalar per timestamp

    Parameters
    ----------
    era5_dir:
        Root directory of ERA5 data with per-variable subdirectories.
    era5_variables:
        Ordered list of ERA5 variable names to stack as channels.
        In options 1-3 the capacity channel is always appended last as
        'capacity'. In option 4 channels are named '{var}_cap_weighted'.
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
        Has no effect on distributions or weighted weather.
    capacity_as_spatial_distribution:
        Option 2. Mutually exclusive with all other capacity modes.
    capacity_as_spatiotemporal_distribution:
        Option 3. Mutually exclusive with all other capacity modes.
    capacity_weighted_weather:
        Option 4. Mutually exclusive with all other capacity modes.
        normalize_capacity is ignored when this is True.
    normalize_weather:
        Normalization method for weather channels (or weighted weather
        channels in option 4). One of "none", "zscore", "minmax".
        Default "none". zscore strongly recommended for option 4.
    normalize_capacity:
        Normalization method for the capacity channel (options 1-3 only).
        Ignored in option 4. One of "none", "zscore", "minmax".
        Default "none".
    time_col:
        Name of the datetime column in input CSVs. Default: 'time'.
    value_col:
        Name of the generation column in the SMARD CSV. Default: 'generation_mwh'.
    """

    # ── Validate parameters ────────────────────────────────────────────────────

    capacity_modes = [
        capacity_as_spatial_distribution,
        capacity_as_spatiotemporal_distribution,
        capacity_weighted_weather,
    ]
    if sum(capacity_modes) > 1:
        raise ValueError(
            "capacity_as_spatial_distribution, capacity_as_spatiotemporal_distribution "
            "and capacity_weighted_weather are mutually exclusive — at most one can be True."
        )
    if normalize_weather not in _VALID_NORM_OPTIONS:
        raise ValueError(f"normalize_weather must be one of {_VALID_NORM_OPTIONS}")
    if normalize_capacity not in _VALID_NORM_OPTIONS:
        raise ValueError(f"normalize_capacity must be one of {_VALID_NORM_OPTIONS}")
    if capacity_weighted_weather and normalize_capacity != "none":
        print("  WARNING: normalize_capacity is ignored when capacity_weighted_weather=True "
              "— there is no separate capacity channel.")

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

    # ── Align timestamps ───────────────────────────────────────────────────────

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

    years = common_times.year
    train_mask = years <= train_end_year
    val_mask = (years > train_end_year) & (years <= val_end_year)
    test_mask = years > val_end_year

    print(f"\n  Split summary:")
    print(f"    Train (≤ {train_end_year}):      {train_mask.sum():>8,} hours")
    print(f"    Val   ({train_end_year + 1} – {val_end_year}): {val_mask.sum():>8,} hours")
    print(f"    Test  (> {val_end_year}):      {test_mask.sum():>8,} hours")

    # ── Build capacity channel / weighted weather ──────────────────────────────

    spatiotemporal_sum = None

    if capacity_as_spatial_distribution:
        # Option 2: per-hour spatial distribution
        safe_total = np.where(total_capacity_hourly > 0, total_capacity_hourly, 1.0)
        capacity_channel = (
            capacity_hourly / safe_total[:, None, None]
        ).astype(np.float32)
        print(f"\n  Capacity channel: per-hour spatial distribution (cell / total_t)")
        print(f"  Value range: {capacity_channel.min():.6f} — {capacity_channel.max():.6f}")
        all_channels = era5_variables + ["capacity"]
        all_arrays = era5_arrays + [capacity_channel]
        del capacity_channel

    elif capacity_as_spatiotemporal_distribution:
        # Option 3: spatiotemporal distribution — training sum only
        train_capacity = capacity_hourly[feature_indices[train_mask]]
        spatiotemporal_sum = float(train_capacity.sum())
        capacity_channel = (capacity_hourly / spatiotemporal_sum).astype(np.float32)
        print(f"\n  Capacity channel: spatiotemporal distribution (cell / train_sum)")
        print(f"  Training spatiotemporal sum: {spatiotemporal_sum:.4f} {cap_unit}·hours")
        print(f"  Value range: {capacity_channel.min():.8f} — {capacity_channel.max():.8f}")
        all_channels = era5_variables + ["capacity"]
        all_arrays = era5_arrays + [capacity_channel]
        del capacity_channel

    elif capacity_weighted_weather:
        # Option 4: capacity-weighted weather (Lindas et al. approach)
        # Compute spatiotemporal weights from training data only
        train_capacity = capacity_hourly[feature_indices[train_mask]]
        spatiotemporal_sum = float(train_capacity.sum())
        weights = (capacity_hourly / spatiotemporal_sum).astype(np.float32)
        print(f"\n  Capacity-weighted weather mode (Lindas et al.)")
        print(f"  Training spatiotemporal sum: {spatiotemporal_sum:.4f} {cap_unit}·hours")
        print(f"  Weight range: {weights.min():.8f} — {weights.max():.8f}")

        # Multiply each weather channel by the capacity weights
        weighted_arrays = []
        for arr in era5_arrays:
            weighted_arrays.append((arr * weights).astype(np.float32))
        del weights

        all_channels = [f"{var}_cap_weighted" for var in era5_variables]
        all_arrays = weighted_arrays
        del weighted_arrays

    else:
        # Option 1: absolute capacity values
        all_channels = era5_variables + ["capacity"]
        all_arrays = era5_arrays + [capacity_hourly]
        print(f"\n  Capacity channel: absolute values ({cap_unit})")

    del capacity_hourly

    # ── Stack all channels → (time, channels, lat, lon) ───────────────────────

    print("\nStacking channels ...")
    features_np = np.stack(all_arrays, axis=1)  # (time, C, lat, lon)
    del all_arrays, era5_arrays

    # Filter to common timestamps
    features_np = features_np[feature_indices]

    print(f"  Feature array shape: {features_np.shape}  (time, channels, lat, lon)")
    print(f"  Feature array dtype: {features_np.dtype}")
    print(f"  Channels: {all_channels}")

    target_aligned = df_target.loc[common_times].values
    total_capacity_aligned = total_capacity_hourly[feature_indices]
    cyclical_aligned = df_cyclical.loc[common_times]

    # ── Normalization stats ────────────────────────────────────────────────────

    print("\nComputing normalization statistics from training data ...")
    norm_df = _compute_normalization_stats(features_np, all_channels, train_mask)

    # ── Apply normalization ────────────────────────────────────────────────────

    if capacity_weighted_weather:
        # In weighted weather mode all channels are weather-derived — normalize all
        all_indices = list(range(len(all_channels)))
        if normalize_weather != "none":
            print(f"\nNormalizing capacity-weighted weather channels ({normalize_weather}) ...")
            _apply_normalization(features_np, all_indices, norm_df, all_channels, normalize_weather)
        else:
            print("  WARNING: normalize_weather='none' with capacity_weighted_weather=True. "
                  "Raw weighted values are very small — zscore normalization is recommended.")
    else:
        weather_indices = list(range(len(era5_variables)))
        capacity_index = len(era5_variables)

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
        if capacity_weighted_weather:
            parts.append(
                f"capacity-weighted: {ch.replace('_cap_weighted', '')} × "
                f"spatiotemporal_weight (train_sum={spatiotemporal_sum:.4f} {cap_unit}·hours)"
            )
            parts.append(
                f"{normalize_weather} normalized (fit on train)"
                if normalize_weather != "none"
                else "not normalized"
            )
        elif ch == "capacity":
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
    elif capacity_weighted_weather:
        cap_channel_mode = "capacity_weighted_weather"
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
        "normalize_capacity": normalize_capacity if not capacity_weighted_weather else "n/a",
        **{f"transform_{ch}": desc for ch, desc in channel_transforms.items()},
    }
    if capacity_as_spatiotemporal_distribution or capacity_weighted_weather:
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

    # total_cap_per_t.csv — saved in options 2, 3 and 4
    if capacity_as_spatial_distribution or capacity_as_spatiotemporal_distribution \
            or capacity_weighted_weather:
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

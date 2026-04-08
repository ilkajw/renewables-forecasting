from renewables_forecasting.config.paths import (
    ERA5_MASKED_SOLAR_DATA_DIR,
    SOLAR_CAPACITY_GRIDS_ZARR_STORE,
    SMARD_SOLAR_GENERATION_SERIES_MASKED_CSV,
    CYCLICAL_TIME_FEATURES_CSV,
    SOLAR_FEATURES_OPT1_DIR,
    SOLAR_FEATURES_OPT2_DIR,
    SOLAR_FEATURES_OPT3_DIR,
    SOLAR_FEATURES_OPT4_DIR,
    SOLAR_FEATURES_OPT5_DIR
)
from renewables_forecasting.data.build_features import build_feature_dataset


# ── Common arguments shared across all options ─────────────────────────────────

_COMMON = dict(
    era5_dir=ERA5_MASKED_SOLAR_DATA_DIR,
    capacity_store_dir=SOLAR_CAPACITY_GRIDS_ZARR_STORE,
    smard_csv_path=SMARD_SOLAR_GENERATION_SERIES_MASKED_CSV,
    cyclical_features_csv_path=CYCLICAL_TIME_FEATURES_CSV,
    train_end_year=2022,
    val_end_year=2023,
    capacity_unit_divisor=1.0,
    time_col="time",
    value_col="generation_mwh",
)


# ── Option 1 — Raw absolute values ────────────────────────────────────────────
#
# Baseline. Capacity channel contains absolute installed capacity in kW.
# Weather channels are raw ERA5 values with no normalization.
# Model predicts MWh directly (output_capacity_factor=False).

def task_build_solar_features_opt1(
        depends_on={
            "era5": ERA5_MASKED_SOLAR_DATA_DIR / ".complete",
            "capacity": SOLAR_CAPACITY_GRIDS_ZARR_STORE / ".complete",
            "smard": SMARD_SOLAR_GENERATION_SERIES_MASKED_CSV,
            "cyclical": CYCLICAL_TIME_FEATURES_CSV,
        },
        produces=SOLAR_FEATURES_OPT1_DIR / ".complete",
):
    build_feature_dataset(
        **_COMMON,
        era5_variables=["ssrd", "t2m"],
        out_dir=SOLAR_FEATURES_OPT1_DIR,
        capacity_as_spatial_distribution=False,
        capacity_as_spatiotemporal_distribution=False,
        capacity_weighted_weather=False,
        normalize_weather="none",
        normalize_capacity="none",
    )
    (SOLAR_FEATURES_OPT1_DIR / ".complete").touch()


# ── Option 2 — Spatial distribution capacity, z-score weather ─────────────────
#
# Capacity channel is each cell's share of total installed capacity for that
# hour: cell / sum_{i,j} P[t]. Values sum to 1 across lat/lon per hour.
# Total capacity saved as total_cap_per_t.csv for use in training loop.
# Weather channels z-score normalized using training statistics.
# Model predicts a capacity factor in [0,1] (output_capacity_factor=True),
# which is multiplied by total_capacity in the training loop to recover MWh.

def task_build_solar_features_opt2(
        depends_on={
            "era5": ERA5_MASKED_SOLAR_DATA_DIR / ".complete",
            "capacity": SOLAR_CAPACITY_GRIDS_ZARR_STORE / ".complete",
            "smard": SMARD_SOLAR_GENERATION_SERIES_MASKED_CSV,
            "cyclical": CYCLICAL_TIME_FEATURES_CSV,
        },
        produces=SOLAR_FEATURES_OPT2_DIR / ".complete",
):
    build_feature_dataset(
        **_COMMON,
        era5_variables=["ssrd", "t2m"],
        out_dir=SOLAR_FEATURES_OPT2_DIR,
        capacity_as_spatial_distribution=True,
        capacity_as_spatiotemporal_distribution=False,
        capacity_weighted_weather=False,
        normalize_weather="zscore",
        normalize_capacity="none",  # spatial distribution already on [0,1] scale
    )
    (SOLAR_FEATURES_OPT2_DIR / ".complete").touch()


# ── Option 3 — Spatial distribution capacity, raw absolute weather ─────────────────
#
# Same as option 2 but without normalising the weather data.

def task_build_solar_features_opt3(
        depends_on={
            "era5": ERA5_MASKED_SOLAR_DATA_DIR / ".complete",
            "capacity": SOLAR_CAPACITY_GRIDS_ZARR_STORE / ".complete",
            "smard": SMARD_SOLAR_GENERATION_SERIES_MASKED_CSV,
            "cyclical": CYCLICAL_TIME_FEATURES_CSV,
        },
        produces=SOLAR_FEATURES_OPT3_DIR / ".complete",
):
    build_feature_dataset(
        **_COMMON,
        era5_variables=["ssrd", "t2m"],
        out_dir=SOLAR_FEATURES_OPT3_DIR,
        capacity_as_spatial_distribution=True,
        capacity_as_spatiotemporal_distribution=False,
        capacity_weighted_weather=False,
        normalize_weather="none",
        normalize_capacity="none",  # spatial distribution already on [0,1] scale
    )
    (SOLAR_FEATURES_OPT3_DIR / ".complete").touch()


# ── Option 4 — Spatiotemporal distribution capacity, z-score weather ───────────────
#
# Capacity channel is each cell's value divided by the sum over ALL training
# timestamps and ALL grid cells: cell / sum_{t_train} sum_{i,j} P.
# Denominator computed on training data only for consistency with other
# training-only statistics. Weather channels are ERA5 values z-score normalised over
# training set. Model predicts MWh directly (output_capacity_factor=False).

def task_build_solar_features_opt4(
        depends_on={
            "era5": ERA5_MASKED_SOLAR_DATA_DIR / ".complete",
            "capacity": SOLAR_CAPACITY_GRIDS_ZARR_STORE / ".complete",
            "smard": SMARD_SOLAR_GENERATION_SERIES_MASKED_CSV,
            "cyclical": CYCLICAL_TIME_FEATURES_CSV,
        },
        produces=SOLAR_FEATURES_OPT4_DIR / ".complete",
):
    build_feature_dataset(
        **_COMMON,
        era5_variables=["ssrd", "t2m"],
        out_dir=SOLAR_FEATURES_OPT4_DIR,
        capacity_as_spatial_distribution=False,
        capacity_as_spatiotemporal_distribution=True,
        capacity_weighted_weather=False,
        normalize_weather="zscore",
        normalize_capacity="none",
    )
    (SOLAR_FEATURES_OPT4_DIR / ".complete").touch()


# ── Option 5 — Capacity-weighted ssrd only (Lindas et al.) ────────────────────
#
# Inspired by Lindas et al. No separate capacity channel.
# ssrd is multiplied elementwise by the spatiotemporal capacity weights:
#     ssrd_cap_weighted[t, i, j] = ssrd[t, i, j] × w[t, i, j]
# where w[t, i, j] = P[t, i, j] / sum_{t_train} sum_{i,j} P[t, i, j]
# t2m is excluded — weighting temperature by capacity is not physically
# meaningful since temperature acts as an efficiency modifier, not a
# generation driver in the same multiplicative sense as irradiance.
# The weighted channel is z-score normalized since raw values are very small
# (product of large irradiance and tiny weights). Model predicts MWh directly.
# Model config should use in_channels=1 (ssrd_cap_weighted only).

def task_build_solar_features_opt5(
        depends_on={
            "era5": ERA5_MASKED_SOLAR_DATA_DIR / ".complete",
            "capacity": SOLAR_CAPACITY_GRIDS_ZARR_STORE / ".complete",
            "smard": SMARD_SOLAR_GENERATION_SERIES_MASKED_CSV,
            "cyclical": CYCLICAL_TIME_FEATURES_CSV,
        },
        produces=SOLAR_FEATURES_OPT5_DIR / ".complete",
):
    build_feature_dataset(
        **_COMMON,
        era5_variables=["ssrd"],   # t2m excluded — see docstring above
        out_dir=SOLAR_FEATURES_OPT5_DIR,
        capacity_as_spatial_distribution=False,
        capacity_as_spatiotemporal_distribution=False,
        capacity_weighted_weather=True,
        normalize_weather="zscore",  # optional. could be useful as values are very small
        normalize_capacity="none",   # ignored in capacity_weighted_weather mode
    )
    (SOLAR_FEATURES_OPT5_DIR / ".complete").touch()

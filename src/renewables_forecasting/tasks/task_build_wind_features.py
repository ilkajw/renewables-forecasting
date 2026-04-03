from renewables_forecasting.config.paths import (
    ERA5_PROCESSED_WIND_DATA_DIR,
    WIND_CAPACITY_GRIDS_ZARR_STORE,
    SMARD_WIND_TOTAL_GENERATION_SERIES_CSV,
    CYCLICAL_TIME_FEATURES_CSV,
    WIND_FEATURES_DIR,
)
from renewables_forecasting.data.build_features import build_feature_dataset


def task_build_wind_features(
        depends_on={
            "era5": ERA5_PROCESSED_WIND_DATA_DIR / ".complete",
            "capacity": WIND_CAPACITY_GRIDS_ZARR_STORE / ".complete",
            "smard": SMARD_WIND_TOTAL_GENERATION_SERIES_CSV,
            "cyclical": CYCLICAL_TIME_FEATURES_CSV,
        },
        produces=WIND_FEATURES_DIR / ".complete",
):
    build_feature_dataset(
        era5_dir=ERA5_PROCESSED_WIND_DATA_DIR,
        era5_variables=["u100", "v100", "wind_speed_100"],
        capacity_store_dir=WIND_CAPACITY_GRIDS_ZARR_STORE,
        smard_csv_path=SMARD_WIND_TOTAL_GENERATION_SERIES_CSV,
        cyclical_features_csv_path=CYCLICAL_TIME_FEATURES_CSV,
        out_dir=WIND_FEATURES_DIR,
        train_end_year=2022,
        val_end_year=2023,
        capacity_unit_divisor=1.0,
        capacity_as_spatial_distribution=False,
        capacity_as_spatiotemporal_distribution=False,
        normalize_weather=False,
        time_col="time",
        value_col="generation_mwh",
    )

    (WIND_FEATURES_DIR / ".complete").touch()

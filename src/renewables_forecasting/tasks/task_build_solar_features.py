from renewables_forecasting.config.paths import (
    ERA5_MASKED_SOLAR_DATA_DIR,
    SOLAR_CAPACITY_GRIDS_ZARR_STORE,
    SMARD_SOLAR_GENERATION_SERIES_MASKED_CSV,
    CYCLICAL_TIME_FEATURES_CSV,
    SOLAR_FEATURES_DIR,
)
from renewables_forecasting.data.build_features import build_feature_dataset


def task_build_solar_features(
        depends_on={
            "era5": ERA5_MASKED_SOLAR_DATA_DIR / ".complete",
            "capacity": SOLAR_CAPACITY_GRIDS_ZARR_STORE / ".complete",
            "smard": SMARD_SOLAR_GENERATION_SERIES_MASKED_CSV,
            "cyclical": CYCLICAL_TIME_FEATURES_CSV,
        },
        produces=SOLAR_FEATURES_DIR / ".complete",
):
    build_feature_dataset(
        era5_dir=ERA5_MASKED_SOLAR_DATA_DIR,
        era5_variables=["ssrd", "t2m"],
        capacity_store_dir=SOLAR_CAPACITY_GRIDS_ZARR_STORE,
        smard_csv_path=SMARD_SOLAR_GENERATION_SERIES_MASKED_CSV,
        cyclical_features_csv_path=CYCLICAL_TIME_FEATURES_CSV,
        out_dir=SOLAR_FEATURES_DIR,
        train_end_year=2022,
        val_end_year=2023,
        capacity_unit_divisor=1.0,
        capacity_as_spatial_distribution=False,
        capacity_as_spatiotemporal_distribution=False,
        normalize_weather=False,
        time_col="time",
        value_col="generation_mwh",
    )

    (SOLAR_FEATURES_DIR / ".complete").touch()

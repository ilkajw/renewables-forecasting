from datetime import date

from renewables_forecasting.data.dwd import (
    download_cosmo_rea6,
    preprocess_solar_data,
    preprocess_wind_data,
    build_solar_features,
)

from renewables_forecasting.config.paths import DWD_SOLAR_DATA_DIR_RAW, DWD_SOLAR_DATA_DIR_PROCESSED, \
    DWD_SOLAR_DATA_DIR_FEATURES, DWD_WIND_DATA_DIR_RAW, DWD_WIND_DATA_DIR_FEATURES
from renewables_forecasting.config.technologies import SOLAR, WIND


# ---- Download raw DWD data ----

def task_download_dwd_solar(produces=DWD_SOLAR_DATA_DIR_RAW / ".complete"):

    DWD_SOLAR_DATA_DIR_RAW.mkdir(parents=True, exist_ok=True)

    download_cosmo_rea6(
        variables=SOLAR.weather_variables,
        start=date(2017, 2, 1),
        end=date(2017, 2, 1),
        output_dir=DWD_SOLAR_DATA_DIR_RAW,
    )

    # Sentinel file as directories cannot be tracked by pytask
    (DWD_SOLAR_DATA_DIR_RAW / ".complete").touch()


def task_download_dwd_wind(produces=DWD_WIND_DATA_DIR_RAW / ".complete"):

    DWD_WIND_DATA_DIR_RAW.mkdir(parents=True, exist_ok=True)

    download_cosmo_rea6(
        variables=WIND.weather_variables,
        start=date(2017, 2, 1),
        end=date(2017, 2, 1),
        output_dir=DWD_WIND_DATA_DIR_RAW,
    )

    (DWD_WIND_DATA_DIR_RAW / ".complete").touch()


# ---- Build zarr with solar variables on target grid ----

def task_build_solar_variable_grids(
    depends_on=DWD_SOLAR_DATA_DIR_RAW / ".complete",
    produces=DWD_SOLAR_DATA_DIR_PROCESSED / ".complete",
):

    DWD_SOLAR_DATA_DIR_PROCESSED.mkdir(parents=True, exist_ok=True)

    preprocess_solar_data(
        tech=SOLAR,
        zarr_store=DWD_SOLAR_DATA_DIR_PROCESSED,
        grid_resolution_km=30,
    )

    (DWD_SOLAR_DATA_DIR_PROCESSED / ".complete").touch()


# ---- Build solar feature grids (GHI) on target grid ----

def task_build_solar_feature_grids(
    depends_on=DWD_SOLAR_DATA_DIR_PROCESSED / ".complete",
    produces=DWD_SOLAR_DATA_DIR_FEATURES / ".complete",
):
    DWD_SOLAR_DATA_DIR_FEATURES.mkdir(parents=True, exist_ok=True)

    build_solar_features(
        vars_path=DWD_SOLAR_DATA_DIR_PROCESSED,
        store_path=DWD_SOLAR_DATA_DIR_FEATURES,
    )

    (DWD_SOLAR_DATA_DIR_FEATURES / ".complete").touch()


# ---- Build wind feature grids ----

def task_build_wind_feature_grids(
    depends_on=DWD_WIND_DATA_DIR_RAW / ".complete",
    produces=DWD_WIND_DATA_DIR_FEATURES / ".complete",
):
    DWD_WIND_DATA_DIR_FEATURES.mkdir(parents=True, exist_ok=True)

    preprocess_wind_data(
        tech=WIND,
        zarr_store=DWD_WIND_DATA_DIR_FEATURES,
        grid_resolution_km=30,
    )

    (DWD_WIND_DATA_DIR_FEATURES / ".complete").touch()

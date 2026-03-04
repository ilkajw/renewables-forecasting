from datetime import date

from renewables_forecasting.data.dwd import (
    download_cosmo_rea6,
    _build_and_save_epsg_3035_germany_target_grid,
    regrid_cosmo_rea6_solar,
    regrid_cosmo_rea6_wind,
    build_solar_feature_grid,
)

from renewables_forecasting.config.paths import DWD_SOLAR_DATA_DIR_RAW, DWD_SOLAR_DATA_DIR_PROCESSED, \
    DWD_SOLAR_DATA_DIR_FEATURES, DWD_WIND_DATA_DIR_RAW, DWD_WIND_DATA_DIR_FEATURES, TARGET_GRID_ZARR_STORE, \
    TARGET_GRID_REF_DS_STORE
from renewables_forecasting.config.technologies import SOLAR, WIND
from renewables_forecasting.config.grid import GRID_RESOLUTION_M, LON_MAX, LON_MIN, LAT_MIN, LAT_MAX


# ---- Download raw DWD data ----

def task_download_dwd_solar(produces=DWD_SOLAR_DATA_DIR_RAW / ".complete"):

    download_cosmo_rea6(
        variables=SOLAR.weather_variables,
        start=date(2017, 2, 1),
        end=date(2017, 2, 1),
        output_dir=DWD_SOLAR_DATA_DIR_RAW,
    )

    # Sentinel file as directories cannot be tracked by pytask
    (DWD_SOLAR_DATA_DIR_RAW / ".complete").touch()


def task_download_dwd_wind(produces=DWD_WIND_DATA_DIR_RAW / ".complete"):

    download_cosmo_rea6(
        variables=WIND.weather_variables,
        start=date(2017, 2, 1),
        end=date(2017, 2, 1),
        output_dir=DWD_WIND_DATA_DIR_RAW,
    )

    (DWD_WIND_DATA_DIR_RAW / ".complete").touch()


def task_build_and_save_target_grid(
        depends_on=TARGET_GRID_REF_DS_STORE,
        produces=TARGET_GRID_ZARR_STORE / ".complete"
):

    _build_and_save_epsg_3035_germany_target_grid(
        ref_ds=TARGET_GRID_REF_DS_STORE,
        store_path=TARGET_GRID_ZARR_STORE,
        cell_size_m=GRID_RESOLUTION_M
    )

    (TARGET_GRID_ZARR_STORE / ".complete").touch()

# ---- Build zarr with solar variables on target grid ----

def task_build_solar_variable_grids(
    depends_on=DWD_SOLAR_DATA_DIR_RAW / ".complete",
    produces=DWD_SOLAR_DATA_DIR_PROCESSED / ".complete",
):

    regrid_cosmo_rea6_solar(
        tech=SOLAR,
        data_zarr_store=DWD_SOLAR_DATA_DIR_PROCESSED,
        target_grid_zarr_store=TARGET_GRID_ZARR_STORE,
        grid_resolution_m=GRID_RESOLUTION_M,
    )

    (DWD_SOLAR_DATA_DIR_PROCESSED / ".complete").touch()


# ---- Build solar feature grids (GHI) on target grid ----

def task_build_solar_feature_grids(
    depends_on=DWD_SOLAR_DATA_DIR_PROCESSED / ".complete",
    produces=DWD_SOLAR_DATA_DIR_FEATURES / ".complete",
):

    build_solar_feature_grid(
        vars_path=DWD_SOLAR_DATA_DIR_PROCESSED,
        store_path=DWD_SOLAR_DATA_DIR_FEATURES,
    )

    (DWD_SOLAR_DATA_DIR_FEATURES / ".complete").touch()


# ---- Build wind feature grids ----

def task_build_wind_feature_grids(
    depends_on=DWD_WIND_DATA_DIR_RAW / ".complete",
    produces=DWD_WIND_DATA_DIR_FEATURES / ".complete",
):

    regrid_cosmo_rea6_wind(
        tech=WIND,
        data_zarr_store=DWD_WIND_DATA_DIR_FEATURES,
        target_grid_zarr_store=TARGET_GRID_ZARR_STORE,
        grid_resolution_m=GRID_RESOLUTION_M,
    )

    (DWD_WIND_DATA_DIR_FEATURES / ".complete").touch()

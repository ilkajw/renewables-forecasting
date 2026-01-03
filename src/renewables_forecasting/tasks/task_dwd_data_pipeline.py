from datetime import date

from renewables_forecasting.data.dwd import (
    download_cosmo_rea6,
    preprocess_solar_data,
    preprocess_wind_data,
    build_solar_features,
)
from renewables_forecasting.config.technologies import SOLAR, WIND


# Download raw DWD data

def task_download_dwd(
    produces={
        "solar": SOLAR.raw_subdir / ".complete",
        "wind": WIND.raw_subdir / ".complete",
    },
):
    # Ensure directories exist
    SOLAR.raw_subdir.mkdir(parents=True, exist_ok=True)
    WIND.raw_subdir.mkdir(parents=True, exist_ok=True)

    # Download (example range)
    download_cosmo_rea6(
        variables=SOLAR.variables,
        start=date(2017, 2, 1),
        end=date(2017, 2, 1),
        output_dir=SOLAR.raw_subdir,
    )

    download_cosmo_rea6(
        variables=WIND.variables,
        start=date(2017, 2, 1),
        end=date(2017, 2, 1),
        output_dir=WIND.raw_subdir,
    )

    # Sentinel files
    (SOLAR.raw_subdir / ".complete").touch()
    (WIND.raw_subdir / ".complete").touch()


# Build regridded solar variable zarr

def task_build_solar_var_grids(
    depends_on=SOLAR.raw_subdir / ".complete",
    produces=SOLAR.var_zarr_store / ".complete",
):
    assert SOLAR.var_zarr_store is not None

    SOLAR.var_zarr_store.mkdir(parents=True, exist_ok=True)

    preprocess_solar_data(
        tech=SOLAR,
        zarr_store=SOLAR.var_zarr_store,
        grid_resolution_km=30,
    )

    (SOLAR.var_zarr_store / ".complete").touch()


# Build solar feature grids (GHI)

def task_build_solar_feature_grids(
    depends_on=SOLAR.var_zarr_store / ".complete",
    produces=SOLAR.feature_zarr_store / ".complete",
):
    SOLAR.feature_zarr_store.mkdir(parents=True, exist_ok=True)

    build_solar_features(
        vars_path=SOLAR.var_zarr_store,
        store_path=SOLAR.feature_zarr_store,
    )

    (SOLAR.feature_zarr_store / ".complete").touch()


# Build wind feature grids

def task_build_wind_feature_grids(
    depends_on=WIND.raw_subdir / ".complete",
    produces=WIND.feature_zarr_store / ".complete",
):
    WIND.feature_zarr_store.mkdir(parents=True, exist_ok=True)

    preprocess_wind_data(
        tech=WIND,
        zarr_store=WIND.feature_zarr_store,
        grid_resolution_km=30,
    )

    (WIND.feature_zarr_store / ".complete").touch()

from datetime import date

from renewables_forecasting.config.data_sources import ERA5_SOLAR_VARIABLES
from renewables_forecasting.config.paths import (
    ERA5_RAW_SOLAR_DATA_DIR,
    DAYLIGHT_MASK_NETCDF_PATH,
    ERA5_CET_SOLAR_DATA_DIR,
    ERA5_MASKED_SOLAR_DATA_DIR
)
from renewables_forecasting.data.era5 import (
    download_era5,
    convert_era5_utc_to_german_time,
    build_daylight_mask,
    apply_daylight_mask_to_era5_variables
)


# ── Download era5 solar variables ───────────────────────────────────────────────────────────────

def task_download_era5_solar(produces=ERA5_RAW_SOLAR_DATA_DIR / ".complete"):

    # Coordinates bounding box defaults to Germany
    download_era5(
        variables=ERA5_SOLAR_VARIABLES,
        start=date(2015, 1, 1),
        end=date(2025, 12, 1),
        store_dir=ERA5_RAW_SOLAR_DATA_DIR,
    )

    (ERA5_RAW_SOLAR_DATA_DIR / ".complete").touch()


# ── Convert time dimension from UTC to CET ────────────────────────────────────────────────────────

def task_convert_era5_solar_data_to_cet(
        depends_on=ERA5_RAW_SOLAR_DATA_DIR / ".complete",
        produces=ERA5_CET_SOLAR_DATA_DIR / ".complete"
):
    convert_era5_utc_to_german_time(
        in_dir=ERA5_RAW_SOLAR_DATA_DIR,
        out_dir=ERA5_CET_SOLAR_DATA_DIR,
        variables=list(ERA5_SOLAR_VARIABLES.keys()),
        file_pattern="{variable}_{year}-{month:02d}.nc",
    )

    (ERA5_CET_SOLAR_DATA_DIR / ".complete").touch()


# ── Construct a mask of daylight hours + buffer hours ───────────────────────────────────────

def task_build_daylight_mask(
        depends_on=ERA5_CET_SOLAR_DATA_DIR / ".complete",
        produces=DAYLIGHT_MASK_NETCDF_PATH
):
    build_daylight_mask(
        ssrd_dir=ERA5_CET_SOLAR_DATA_DIR / "ssrd",
        out_path=DAYLIGHT_MASK_NETCDF_PATH,
        buffer_hours=2,
        file_pattern="ssrd_{year}-{month:02d}.nc"
    )


# ── Mask out night hours from solar data ────────────────────────────────────────────────────────

def task_apply_daylight_mask_to_solar_data(
        depends_on={
            "mask": DAYLIGHT_MASK_NETCDF_PATH,
            "cet_era5": ERA5_CET_SOLAR_DATA_DIR / ".complete"
        },
        produces=ERA5_MASKED_SOLAR_DATA_DIR / ".complete"
):

    apply_daylight_mask_to_era5_variables(
        mask_path=DAYLIGHT_MASK_NETCDF_PATH,
        variables=list(ERA5_SOLAR_VARIABLES.keys()),
        era5_dir=ERA5_CET_SOLAR_DATA_DIR,
        out_dir=ERA5_MASKED_SOLAR_DATA_DIR,
    )

    (ERA5_MASKED_SOLAR_DATA_DIR / ".complete").touch()

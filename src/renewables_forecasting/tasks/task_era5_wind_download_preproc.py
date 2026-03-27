from datetime import date

from renewables_forecasting.config.data_sources import ERA5_WIND_VARIABLES
from renewables_forecasting.config.data_constants import WIND_LAT_MAX
from renewables_forecasting.config.paths import (
    ERA5_RAW_WIND_DATA_DIR,
    ERA5_PROCESSED_WIND_DATA_DIR
)
from renewables_forecasting.data.era5 import (
    download_era5,
    calculate_wind_speed_from_components,
)


# ── Download era5 u100, v100 wind components ───────────────────────────────────────────────────────────────

def task_download_era5_wind(
        produces=ERA5_RAW_WIND_DATA_DIR / ".complete"
):

    # Coordinates bounding box defaults to Germany
    download_era5(
        variables=ERA5_WIND_VARIABLES,
        start=date(2015, 1, 1),
        end=date(2025, 12, 1),  # Defaulting to months. Includes full december
        store_dir=ERA5_RAW_WIND_DATA_DIR,
        lat_max=WIND_LAT_MAX  # 0.25 degrees more derived from north-most offshore wind park
    )

    (ERA5_RAW_WIND_DATA_DIR / ".complete").touch()


# ── Calculate wind speeds from wind components per grid cell ─────────────────────────────────────────────────────────

def task_calculate_wind_speed(
        depends_on=ERA5_RAW_WIND_DATA_DIR / ".complete",
        produces=ERA5_PROCESSED_WIND_DATA_DIR / ".complete"
):
    calculate_wind_speed_from_components(
        in_dir=ERA5_RAW_WIND_DATA_DIR,
        out_dir=ERA5_PROCESSED_WIND_DATA_DIR,
        file_pattern="{variable}_{year}-{month:02d}.nc",
    )

    # todo: this might become inconsistent if more functions write to proc wind dir
    (ERA5_PROCESSED_WIND_DATA_DIR / ".complete").touch()


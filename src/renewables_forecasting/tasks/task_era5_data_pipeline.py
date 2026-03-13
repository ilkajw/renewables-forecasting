from datetime import date

from renewables_forecasting.config.paths import ERA5_RAW_SOLAR_DATA_DIR, ERA5_EUROPE_SOLAR_DATA_DIR, \
    ERA5_EUROPE_WIND_DATA_DIR
from renewables_forecasting.config.data_sources import ERA5_SOLAR_VARIABLES, ERA5_WIND_VARIABLES
from renewables_forecasting.data.era5 import download_era5


def task_download_era5_germany_solar(produces=ERA5_RAW_SOLAR_DATA_DIR / ".complete"):

    # Coordinates bounding box defaults to Germany
    download_era5(
        variables=ERA5_SOLAR_VARIABLES,
        start=date(2015, 1, 1),
        end=date(2025, 12, 1),
        store_dir=ERA5_RAW_SOLAR_DATA_DIR,
    )

    (ERA5_RAW_SOLAR_DATA_DIR / ".complete").touch()


def task_download_era5_europe_solar(produces=ERA5_EUROPE_SOLAR_DATA_DIR / ".complete"):

    (ERA5_EUROPE_SOLAR_DATA_DIR / ".complete").touch()
    pass


def task_download_era5_europe_wind(produces=ERA5_EUROPE_WIND_DATA_DIR / ".complete"):
    pass


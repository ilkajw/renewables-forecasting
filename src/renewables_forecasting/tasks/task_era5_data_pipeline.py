from datetime import date

from renewables_forecasting.config.paths import ERA5_SOLAR_DATA_DIR
from renewables_forecasting.config.data_sources import ERA5_SOLAR_VARIABLES, ERA5_WIND_VARIABLES
from renewables_forecasting.data.era5 import download_era5


def task_download_era5_solar_data(produces=ERA5_SOLAR_DATA_DIR / ".complete"):

    download_era5(
        variables=ERA5_SOLAR_VARIABLES,
        start=date(2015, 1, 1),
        end=date(2025, 12, 1),
        store_dir=ERA5_SOLAR_DATA_DIR,
    )

    (ERA5_SOLAR_DATA_DIR / ".complete").touch()

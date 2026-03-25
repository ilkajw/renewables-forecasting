from datetime import datetime

from renewables_forecasting.config.paths import (
    SMARD_SOLAR_GENERATION_SERIES_CSV,
    DAYLIGHT_MASK_NETCDF_PATH,
    SMARD_SOLAR_GENERATION_SERIES_MASKED_CSV
)
from renewables_forecasting.config.data_sources import (
    SMARD_SOLAR_GENERATION_URL_TEMPLATE,
    SMARD_SOLAR_GENERATION_TIMESTAMPS_URL
)
from renewables_forecasting.data.smard import (
    download_smard_generation,
    apply_daylight_mask_to_generation
)


def task_smard_download_solar(
        produces=SMARD_SOLAR_GENERATION_SERIES_CSV
):

    download_smard_generation(
        gen_url_template=SMARD_SOLAR_GENERATION_URL_TEMPLATE,
        timestamps_url=SMARD_SOLAR_GENERATION_TIMESTAMPS_URL,
        out_path=SMARD_SOLAR_GENERATION_SERIES_CSV,
        start=datetime(2015, 1, 1),  # Defaults to 00:00:00
        end=datetime(2026, 1, 1),
        connect_timeout=30,
        read_timeout=600,
        # CSV column naming
        time_col="time",
        timestamp_ms_col="timestamp_ms",
        value_col="generation_mwh",
    )


def task_mask_night_hours(
        depends_on={
                    "generation": SMARD_SOLAR_GENERATION_SERIES_CSV,
                    "mask": DAYLIGHT_MASK_NETCDF_PATH,
        },
        produces=SMARD_SOLAR_GENERATION_SERIES_MASKED_CSV
):

    apply_daylight_mask_to_generation(
        mask_path=DAYLIGHT_MASK_NETCDF_PATH,
        generation_csv_path=SMARD_SOLAR_GENERATION_SERIES_CSV,
        out_csv_path=SMARD_SOLAR_GENERATION_SERIES_MASKED_CSV,
        # Column naming
        time_col="time",
    )



from datetime import datetime

from renewables_forecasting.config.data_sources import (
    SMARD_SOLAR_GENERATION_URL_TEMPLATE,
    SMARD_SOLAR_GENERATION_TIMESTAMPS_URL
)
from renewables_forecasting.config.paths import SMARD_SOLAR_GENERATION_SERIES_JSON_TEMPLATE
from renewables_forecasting.data.smard import download_smard_generation, apply_daylight_mask_to_generation


def task_smard_download_solar():

    download_smard_generation(
        gen_url_template=SMARD_SOLAR_GENERATION_URL_TEMPLATE,
        out_path_template=SMARD_SOLAR_GENERATION_SERIES_JSON_TEMPLATE,
        timestamps_url=SMARD_SOLAR_GENERATION_TIMESTAMPS_URL,
        start=datetime(2015, 1, 1),  # Defaults to 00:00:00
        end=datetime(2026, 1, 1),
        connect_timeout=30,
        read_timeout=600,
    )


def task_mask_night_hours_solar():
    # todo
    # apply_daylight_mask_to_generation()

    pass

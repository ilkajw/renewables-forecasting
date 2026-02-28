from datetime import datetime

from renewables_forecasting.data.smard import download_smard_solar_gen
from renewables_forecasting.config.data_sources import SMARD_SOLAR_GENERATION_URL_TEMPLATE
from renewables_forecasting.config.paths import SMARD_SOLAR_GENERATION_SERIES_JSON_TEMPLATE


def task_smard_download():

    download_smard_solar_gen(
        url_template=SMARD_SOLAR_GENERATION_URL_TEMPLATE,
        out_path_template=SMARD_SOLAR_GENERATION_SERIES_JSON_TEMPLATE,
        start=datetime(2015, 1, 1),  # Defaults to 00:00:00
        end=datetime(2025, 12, 31, 23, 59),
        connect_timeout=30,
        read_timeout=600,
    )

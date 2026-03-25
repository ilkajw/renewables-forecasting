from datetime import datetime

from renewables_forecasting.config.paths import (
    SMARD_WIND_ONSHORE_GENERATION_SERIES_CSV,
    SMARD_WIND_OFFSHORE_GENERATION_SERIES_CSV,
    SMARD_WIND_TOTAL_GENERATION_SERIES_CSV
)
from renewables_forecasting.config.data_sources import (
    SMARD_ONSHORE_WIND_GENERATION_URL_TEMPLATE,
    SMARD_OFFSHORE_WIND_GENERATION_URL_TEMPLATE,
    SMARD_ONSHORE_WIND_GENERATION_TIMESTAMPS_URL,
    SMARD_OFFSHORE_WIND_GENERATION_TIMESTAMPS_URL
)
from renewables_forecasting.data.smard import (
    download_smard_generation, combine_generation_series
)


def task_smard_download_onshore_wind(
        produces=SMARD_WIND_ONSHORE_GENERATION_SERIES_CSV
):

    download_smard_generation(
        gen_url_template=SMARD_ONSHORE_WIND_GENERATION_URL_TEMPLATE,
        timestamps_url=SMARD_ONSHORE_WIND_GENERATION_TIMESTAMPS_URL,
        out_path=SMARD_WIND_ONSHORE_GENERATION_SERIES_CSV,
        start=datetime(2015, 1, 1),  # Defaults to 00:00:00
        end=datetime(2026, 1, 1),
        connect_timeout=30,
        read_timeout=600,
        # CSV column naming
        time_col="time",
        timestamp_ms_col="timestamp_ms",
        value_col="generation_mwh",
    )


def task_smard_download_offshore_wind(
        produces=SMARD_WIND_OFFSHORE_GENERATION_SERIES_CSV
):

    download_smard_generation(
        gen_url_template=SMARD_OFFSHORE_WIND_GENERATION_URL_TEMPLATE,
        timestamps_url=SMARD_OFFSHORE_WIND_GENERATION_TIMESTAMPS_URL,
        out_path=SMARD_WIND_OFFSHORE_GENERATION_SERIES_CSV,
        start=datetime(2015, 1, 1),  # Defaults to 00:00:00
        end=datetime(2026, 1, 1),
        connect_timeout=30,
        read_timeout=600,
        # CSV column naming
        time_col="time",
        timestamp_ms_col="timestamp_ms",
        value_col="generation_mwh",
    )


def task_sum_onshore_offshore_wind_generation(
        depends_on = {
            "onshore_gen": SMARD_WIND_ONSHORE_GENERATION_SERIES_CSV,
            "offshore_gen": SMARD_WIND_OFFSHORE_GENERATION_SERIES_CSV
        },
        produces=SMARD_WIND_TOTAL_GENERATION_SERIES_CSV
):
    combine_generation_series(
        csv_path_1=SMARD_WIND_ONSHORE_GENERATION_SERIES_CSV,
        csv_path_2=SMARD_WIND_OFFSHORE_GENERATION_SERIES_CSV,
        out_path=SMARD_WIND_TOTAL_GENERATION_SERIES_CSV,
        timestamp_ms_col="timestamp_ms",
        time_col="time",
        value_col="generation_mwh"
    )




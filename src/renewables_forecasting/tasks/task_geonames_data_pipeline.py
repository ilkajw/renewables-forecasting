from renewables_forecasting.config.data_sources import GEONAMES_POSTAL_CODES_URL
from renewables_forecasting.config.paths import GEONAMES_POSTAL_CODE_DATA
from renewables_forecasting.data.capacity_grids import download_geonames_postal_code_data


def task_download_geonames_postal_code_data():

    download_geonames_postal_code_data(
        url=GEONAMES_POSTAL_CODES_URL,
        out_path=GEONAMES_POSTAL_CODE_DATA,
    )

from datetime import date
from renewables_forecasting.data.dwd import download_cosmo_rea6
from renewables_forecasting.config.technologies import *


def task_download_solar_one_month():

    download_cosmo_rea6(
        variables=SOLAR.variables,
        start=date(2016, 1, 1),
        end=date(2016, 2, 1),
        output_dir=Path(SOLAR.raw_subdir)
    )

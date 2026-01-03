from datetime import date
from renewables_forecasting.data.dwd import download_cosmo_rea6
from renewables_forecasting.config.technologies import *


def task_download_dwd():

    # currently two months only

    download_cosmo_rea6(
        variables=SOLAR.variables,
        start=date(2016, 12, 1),
        end=date(2017, 2, 1),
        output_dir=Path(SOLAR.raw_subdir)
    )

    download_cosmo_rea6(
        variables=WIND.variables,
        start=date(2016, 12, 1),
        end=date(2017, 2, 1),
        output_dir=Path(WIND.raw_subdir)
    )

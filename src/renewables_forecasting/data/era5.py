from typing import Dict
from pathlib import Path
from datetime import date
from dateutil.relativedelta import relativedelta
import cdsapi

from renewables_forecasting.config.paths import ERA5_DATA_DIR
from renewables_forecasting.config.grid import LON_MAX, LON_MIN, LAT_MAX, LAT_MIN


def download_era5(
        variables: Dict[str, str],
        start: date,
        end: date,
        store_dir: Path = ERA5_DATA_DIR
):

    assert start.day == 1
    assert end.day == 1

    store_dir.mkdir(parents=True, exist_ok=True)

    c = cdsapi.Client()

    for var_short, var_long in variables.items():
        var_dir = store_dir / var_short
        var_dir.mkdir(parents=True, exist_ok=True)
        d = start
        while d <= end:
            out_file = var_dir / f"{var_short}_{d.year}_{d.month:02d}.nc"
            if out_file.exists():
                continue

            year, month = d.year, d.month
            c.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": "reanalysis",
                    "variable": [var_long],
                    "year": str(year),
                    "month": f"{month:02d}",
                    "day": [f"{day:02d}" for day in range(1, 32)],
                    "time": [f"{h:02d}:00" for h in range(24)],
                    "area": [LAT_MAX, LON_MIN, LAT_MIN, LON_MAX],
                    "format": "netcdf",
                },
                str(out_file)
            )

            d += relativedelta(months=1)

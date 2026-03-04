import requests
from pathlib import Path
import xarray as xr
import sqlite3
import pandas as pd
import numpy as np

from renewables_forecasting.config.paths import MASTR_SOLAR_PLANTS_SQLITE_PATH


def download_geonames_postal_code_data(
        url: str,
        out_path: Path,
        connect_timeout: int = 30,
        read_timeout: int = 600,
) -> None:

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # tmp file for partial downloads avoids corrupted zip at out_path
    tmp_path = out_path.with_suffix(".tmp")

    try:
        with requests.get(url, stream=True, timeout=(connect_timeout, read_timeout)) as response:
            response.raise_for_status()

            with open(tmp_path, "wb") as tmp:
                for chunk in response.iter_content(chunk_size=1024*1024):
                    if chunk:
                        tmp.write(chunk)

            tmp_path.replace(out_path)

    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def assign_coords_to_plants():
    # todo
    pass


def build_initial_solar_cap_grid(
        ref_weather_ds: Path,
        store: Path
):
    ds = xr.open_dataset(ref_weather_ds)

    lats = ds["latitude"].values
    lons = ds["longitude"].values

    cap_grid = np.zeros(shape=(len(lats), len(lons)))

    with sqlite3.connect(MASTR_SOLAR_PLANTS_SQLITE_PATH) as conn:
        plants = pd.read_sql(
            """
            SELECT Breitengrad, Laengengrad, Nettonennleistung FROM einheiten_solar
            WHERE Inbetriebnahmedatum <= '2015-01-01'
            AND (DatumEndgueltigeStilllegung IS NULL OR DatumEndgueltigeStilllegung >= '2015-01-01')
            AND (
                DatumBeginnVoruebergehendeStilllegung IS NULL 
                OR '2015-01-01' < DatumBeginnVorruebergehendeStilllegung
                OR DatumWiederaufnahmeBetrieb <= '2015-01-01'
            )
            """,
            conn)

        for _, plant in plants.iterrows():
            lon_idx = np.argmin(np.abs(lons - plant["Laengengrad"]))
            lat_idx = np.argmin(np.abs(lats - plant["Breitengrad"]))

            cap_grid[lat_idx, lon_idx] += plant["Nettonennleistung"]

        cap_grid_da = xr.DataArray(
            cap_grid,
            dims=("latitude", "longitude"),
            coords={"latitude": lats, "longitude": lons},
            name="capacity_kw",
            attrs={"units": "kW"}
        )

        cap_grid_da.to_dataset(name="capacity_kw").to_zarr(store, mode="w")




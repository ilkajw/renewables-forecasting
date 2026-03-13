import requests
import xarray as xr
import sqlite3
import pandas as pd
import numpy as np
import zipfile
from pathlib import Path
from typing import Dict, Tuple
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta

from renewables_forecasting.config.paths import MASTR_SOLAR_PLANTS_FILTERED_SQLITE
from renewables_forecasting.config.data_constants import MISSING_PLZS_TO_COORDS_DICT


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


def _get_plz_to_lat_lon_mapping(
        plz_zip_path: Path,
        file_name: str = "DE.txt"
) -> Dict[str, Tuple[float, float]]:

    # Open geonames zip
    with zipfile.ZipFile(plz_zip_path) as zf:
        with zf.open(file_name) as f:
            # get data frame from txt file
            plz_df = pd.read_csv(
                f,
                sep='\t',
                header=None,
                names=['country', 'plz', 'city', 'state', 'state_code',
                       'district', 'dist_code', 'county', 'county_code',
                       'lat', 'lon', 'accuracy'],
                dtype={'plz': str}
            )

    # Filter out company and institution entries
    plz_df = plz_df[plz_df['state_code'].str.isalpha().fillna(False)]

    # Get plz: (lat, lon)) dict
    plz_to_coords = dict(zip(plz_df['plz'].astype(str),
                             zip(plz_df['lat'], plz_df['lon'])))

    # Add missing PLZs that were looked up manually
    plz_to_coords = MISSING_PLZS_TO_COORDS_DICT | plz_to_coords

    return plz_to_coords


def assign_coords_to_plants(
        plz_data_path: Path,
        plants_csv_path: Path,
        out_path: Path,
        keep_existing_coords: bool = False,
):
    plz_to_coords_dict = _get_plz_to_lat_lon_mapping(plz_data_path)

    plants_df = pd.read_csv(plants_csv_path, dtype={'Postleitzahl': str})

    # Warn about inconsistent coords
    for idx, plant in plants_df.iterrows():
        if pd.isna(plant["Breitengrad"]) and not pd.isna(plant["Laengengrad"]):
            print(f"Got Laengengrad but not Breitengrad for {plant['EinheitMastrNummer']}, overwriting both.")
        elif pd.isna(plant["Laengengrad"]) and not pd.isna(plant["Breitengrad"]):
            print(f"Got Breitengrad but not Laengengrad for {plant['EinheitMastrNummer']}, overwriting both.")

    # Vectorized coord assignment for missing values
    if keep_existing_coords:
        missing = pd.isna(plants_df["Breitengrad"]) | pd.isna(plants_df["Laengengrad"])
        coords = plants_df.loc[missing, "Postleitzahl"].astype(str).map(plz_to_coords_dict)
        plants_df.loc[missing, "Breitengrad"] = coords.map(lambda x: x[0] if pd.notna(x) else None)
        plants_df.loc[missing, "Laengengrad"] = coords.map(lambda x: x[1] if pd.notna(x) else None)

    else:
        # Extract coords with plant indices
        coords = plants_df["Postleitzahl"].astype(str).map(plz_to_coords_dict)
        plants_df["Breitengrad"] = coords.map(lambda x: x[0] if pd.notna(x) else None)
        plants_df["Laengengrad"] = coords.map(lambda x: x[1] if pd.notna(x) else None)

    plants_df.to_csv(out_path, index=False)


def _build_initial_solar_cap_grid(
        ref_weather_ds: Path,
        plants_db: Path,
        start: date,
        store: Path
):
    ds = xr.open_dataset(ref_weather_ds)

    lats = ds["latitude"].values
    lons = ds["longitude"].values

    start = start.isoformat()
    cap_grid = np.zeros(shape=(len(lats), len(lons)))

    with sqlite3.connect(plants_db) as conn:

        plants = pd.read_sql(
            f"""
            SELECT Breitengrad, Laengengrad, Nettonennleistung FROM einheiten_solar
            WHERE Inbetriebnahmedatum <= '{start}'
            AND (DatumEndgueltigeStilllegung IS NULL OR DatumEndgueltigeStilllegung >= '{start}')
            AND (
                DatumBeginnVoruebergehendeStilllegung IS NULL 
                OR '{start}' < DatumBeginnVoruebergehendeStilllegung
                OR DatumWiederaufnahmeBetrieb <= '{start}'
            )
            """,
            conn
        )

    for _, plant in plants.iterrows():
        lon_idx = np.argmin(np.abs(lons - plant["Laengengrad"]))
        lat_idx = np.argmin(np.abs(lats - plant["Breitengrad"]))

        cap_grid[lat_idx, lon_idx] += plant["Nettonennleistung"]

    cap_grid_da = xr.DataArray(
        cap_grid,
        dims=("latitude", "longitude"),
        coords={"latitude": lats, "longitude": lons},
        name="capacity_kw",
        attrs={"units": "kW", "date": start}
    )

    # Save
    cap_grid_da.to_dataset(name="capacity_kw").to_zarr(store, mode="w")


def _get_daily_delta(
        plants: pd.DataFrame,
        day: date,
        lats: np.ndarray,
        lons: np.ndarray
) -> np.ndarray:

    day_str = day.isoformat()
    delta_grid = np.zeros(shape=(len(lats), len(lons)))

    # Plants that came online today
    started = plants[plants["Inbetriebnahmedatum"] == day_str]
    for _, plant in started.iterrows():
        lon_idx = np.argmin(np.abs(lons - plant["Laengengrad"]))
        lat_idx = np.argmin(np.abs(lats - plant["Breitengrad"]))
        delta_grid[lat_idx, lon_idx] += plant["Nettonennleistung"]

    # Plants that went temporarily offline today
    tmp_offline = plants[plants["DatumBeginnVoruebergehendeStilllegung"] == day_str]
    for _, plant in tmp_offline.iterrows():
        lon_idx = np.argmin(np.abs(lons - plant["Laengengrad"]))
        lat_idx = np.argmin(np.abs(lats - plant["Breitengrad"]))
        delta_grid[lat_idx, lon_idx] -= plant["Nettonennleistung"]

    # Plants that came back on from being temporarily offline
    back_online = plants[plants["DatumWiederaufnahmeBetrieb"] == day_str]
    for _, plant in back_online.iterrows():
        lon_idx = np.argmin(np.abs(lons - plant["Laengengrad"]))
        lat_idx = np.argmin(np.abs(lats - plant["Breitengrad"]))
        delta_grid[lat_idx, lon_idx] += plant["Nettonennleistung"]

    # Plants that went offline today
    stopped = plants[plants["DatumEndgueltigeStilllegung"] == day_str]
    for _, plant in stopped.iterrows():
        lon_idx = np.argmin(np.abs(lons - plant["Laengengrad"]))
        lat_idx = np.argmin(np.abs(lats - plant["Breitengrad"]))
        delta_grid[lat_idx, lon_idx] -= plant["Nettonennleistung"]

    return delta_grid


def _build_cap_grids_through_deltas(
        initial_grid_store: Path,
        end_month: date,
        plants_csv: Path,
        out_dir: Path,
):
    # Load initial grid and derive start
    ds = xr.open_dataset(initial_grid_store, engine="zarr")
    start = date.fromisoformat(ds["capacity_kw"].attrs["date"])
    assert start <= end_month, f"start derived from initial grid is {start}. end_month must be afterwards"
    lats = ds["latitude"].values
    lons = ds["longitude"].values
    cap_grid = ds["capacity_kw"].values

    # Load plants
    plants_df = pd.read_csv(plants_csv, dtype={'Postleitzahl': str})

    # Month as first of month
    current_month = date(start.year, start.month, 1)
    assert start == current_month, f"start must be the 1st of a month, got {start}"

    # Loop over all months (from all years of interest)
    while current_month <= end_month:

        # Stop for next loop over days
        month_end = current_month + relativedelta(months=1) - timedelta(days=1)
        daily_grids = []

        # If we are in the start month, add the initial grid to grids
        # and start calculating deltas from second of month
        if current_month == date(start.year, start.month, 1):
            daily_grids.append(cap_grid)
            current_day = start + timedelta(days=1)

        # In any other month, start from first of month aka. 'current_month'
        else:
            current_day = current_month

        # Loop over days
        while current_day <= month_end:

            # Get delta to day before
            delta = _get_daily_delta(plants_df, current_day, lats, lons)

            # Build today's caps with delta
            cap_grid = cap_grid + delta
            daily_grids.append(cap_grid)
            current_day += timedelta(days=1)

        # Build time axis
        days = pd.date_range(current_month.isoformat(), month_end.isoformat())

        # Construct xr DataArray from grids
        cap_da = xr.DataArray(
            np.stack(daily_grids),
            dims=("time", "latitude", "longitude"),
            coords={"time": days, "latitude": lats, "longitude": lons},
            attrs={"units": "kW"}
        )

        # Save
        store_path = out_dir / f"capacity_{current_month.strftime('%Y_%m')}.zarr"
        cap_da.to_dataset(name="capacity_kw").to_zarr(store_path, mode="w")

        current_month += relativedelta(months=1)


def build_capacity_grids(
        ref_weather_ds: Path,
        plants_db: Path,
        start_month: date,
        end_month: date,
        plants_csv: Path,
        out_dir: Path,
):
    assert start_month.day == 1, f"start_month must be the 1st of a month, got {start_month}"
    assert start_month <= end_month, f"start_month must be before or equal to end_mont. got {start_month} " \
                                     f"for start and {end_month} for end"
    initial_store = out_dir / f"capacity_{start_month.strftime('%Y_%m')}_initial.zarr"
    _build_initial_solar_cap_grid(ref_weather_ds, plants_db, start_month, initial_store)
    _build_cap_grids_through_deltas(initial_store, end_month, plants_csv, out_dir)







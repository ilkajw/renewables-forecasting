
import xarray as xr
import pandas as pd
import numpy as np

from pathlib import Path
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta


def _build_initial_cap_grid(
        ref_weather_ds: Path,
        plants_csv: Path,
        start: date,
        store: Path
):
    ds = xr.open_dataset(ref_weather_ds)
    lats = ds["latitude"].values
    lons = ds["longitude"].values

    cap_grid = np.zeros(shape=(len(lats), len(lons)))

    plants_df = pd.read_csv(plants_csv, dtype={"Postleitzahl": str})

    start_str = start.isoformat()

    # Filter for all plants active on start date
    active = plants_df[
        (plants_df["InbetriebnahmedatumEffektiv"] <= start_str)
        & (
                plants_df["DatumEndgueltigeStilllegung"].isna()
                | (plants_df["DatumEndgueltigeStilllegung"] >= start_str)
        )
        & (
                plants_df["DatumBeginnVoruebergehendeStilllegung"].isna()
                | (start_str < plants_df["DatumBeginnVoruebergehendeStilllegung"])
                | (plants_df["DatumWiederaufnahmeBetrieb"] <= start_str)
        )
        ]

    # Loop through all plants
    for _, plant in active.iterrows():

        # Find grid indices the plant snaps to
        lon_idx = np.argmin(np.abs(lons - plant["Laengengrad"]))
        lat_idx = np.argmin(np.abs(lats - plant["Breitengrad"]))

        # Add its capacity to the grid cell
        cap_grid[lat_idx, lon_idx] += plant["Nettonennleistung"]

    # Save capacity grid as dataset in zarr format
    cap_grid_da = xr.DataArray(
        cap_grid,
        dims=("latitude", "longitude"),
        coords={"latitude": lats, "longitude": lons},
        name="capacity_kw",
        attrs={"units": "kW", "date": start_str}
    )

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
    started = plants[plants["InbetriebnahmedatumEffektiv"] == day_str]
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
        plants_csv: Path,
        start_month: date,
        end_month: date,
        out_dir: Path,
):
    assert start_month.day == 1, f"start_month must be the 1st of a month, got {start_month}"
    assert start_month <= end_month, f"start_month must be before or equal to end_mont. got {start_month} " \
                                     f"for start and {end_month} for end"
    initial_store = out_dir / f"capacity_{start_month.strftime('%Y_%m')}_initial.zarr"
    _build_initial_cap_grid(ref_weather_ds, plants_csv, start_month, initial_store)
    _build_cap_grids_through_deltas(initial_store, end_month, plants_csv, out_dir)







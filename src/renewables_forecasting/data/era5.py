import cdsapi
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date
from typing import Dict, List
from dateutil.relativedelta import relativedelta

from renewables_forecasting.config.paths import ERA5_RAW_DATA_DIR
from renewables_forecasting.config.data_constants import (
    GERMAN_TZ,
    GERMANY_LON_MAX,
    GERMANY_LON_MIN,
    GERMANY_LAT_MAX,
    GERMANY_LAT_MIN
)


def download_era5(
        variables: Dict[str, str],
        start: date,
        end: date,
        store_dir: Path = ERA5_RAW_DATA_DIR,
        lon_min: float = GERMANY_LON_MIN,
        lon_max: float = GERMANY_LON_MAX,
        lat_min: float = GERMANY_LAT_MIN,
        lat_max: float = GERMANY_LAT_MAX,
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
            out_file = var_dir / f"{var_short}_{d.year}-{d.month:02d}.nc"
            if out_file.exists():
                d += relativedelta(months=1)
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
                    "area": [lat_max, lon_min, lat_min, lon_max],
                    "format": "netcdf",
                },
                str(out_file)
            )

            d += relativedelta(months=1)


def build_daylight_mask(
        ssrd_dir: Path,
        out_path: Path,
        buffer_hours: int = 2,
        file_pattern: str = "ssrd_{year}-{month:02d}.nc",
) -> None:
    """
    Computes a boolean daylight mask from ERA5 surface solar radiation downwards
    (ssrd) across the full study period and saves it as a NetCDF file for
    reuse across the pipeline.

    All monthly ssrd files in ssrd_dir are concatenated along the time
    dimension before computing the mask, ensuring the mask covers the full
    study period (e.g. 2015-2025) with year- and month-specific day length
    correctly captured for every individual date.

    The mask is True for all hours where any ERA5 grid cell over Germany has
    non-zero irradiance, extended by buffer_hours on each side to include a
    few zero-irradiance hours around sunrise and sunset. This allows the model
    to learn the ramp-up and ramp-down behaviour of solar generation.

    All deep night hours (far from any daylight window) are False and should
    be excluded from training.

    The ssrd data should be in UTC, consistent with the rest of the pipeline.
    The mask is timezone-agnostic — ssrd > 0 is a physical condition that
    holds at the same physical moment regardless of timezone representation.

    Parameters
    ----------
    ssrd_dir:
        Directory containing monthly ERA5 ssrd NetCDF files in UTC.
        Files are discovered via file_pattern glob.
    out_path:
        Path to write the daylight mask NetCDF file. Contains a single boolean
        variable 'daylight_mask' with dimension (time) covering the full period
        spanned by the ssrd files in ssrd_dir.
    buffer_hours:
        Number of zero-irradiance hours to include on each side of the daylight
        window. Default is 2, giving the model context on the sunrise/sunset
        transition. Set to 0 for a strict irradiance > 0 mask.
    file_pattern:
        Glob pattern to discover ssrd files in ssrd_dir. Default matches
        ssrd_{year}-{month:02d}.nc, e.g. ssrd_2015-01.nc.
    """

    out_path.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(ssrd_dir.glob(file_pattern.replace("{year}", "*").replace("{month:02d}", "*")))

    if not files:
        raise FileNotFoundError(
            f"No ssrd files found in {ssrd_dir} matching pattern '{file_pattern}'. "
            f"Ensure ERA5 ssrd data has been downloaded first."
        )

    print(f"Found {len(files)} monthly ssrd files in {ssrd_dir}")
    print(f"  First: {files[0].name}")
    print(f"  Last:  {files[-1].name}")

    print("\nConcatenating all monthly ssrd files ...")
    ds = xr.open_mfdataset(files, combine="by_coords")

    print("Computing spatial maximum of ssrd across all grid cells ...")
    ssrd_max = ds["ssrd"].max(dim=["latitude", "longitude"])
    daylight = ssrd_max > 0

    print(f"Extending daylight window by {buffer_hours} hours on each side ...")

    if buffer_hours > 0:
        window = 2 * buffer_hours + 1
        daylight_extended = (
            daylight
            .rolling(time=window, center=True, min_periods=1)
            .max()
            .astype(bool)
        )
    else:
        daylight_extended = daylight

    n_total = len(daylight_extended.time)
    n_daylight = int(daylight_extended.sum())
    n_night = n_total - n_daylight
    pct = n_daylight / n_total * 100

    print(f"\n  Total hours:     {n_total:,}")
    print(f"  Daylight hours:  {n_daylight:,}  ({pct:.1f}%)")
    print(f"  Night hours:     {n_night:,}  ({100 - pct:.1f}%)")

    daylight_extended.name = "daylight_mask"
    daylight_extended.attrs = {
        "description": (
            "Boolean daylight mask derived from ERA5 ssrd over the full study "
            "period. True for hours within the daily solar window plus buffer. "
            "Apply to both ERA5 and SMARD data to exclude night hours. "
            "Timestamps are in UTC, consistent with the pipeline."
        ),
        "buffer_hours": buffer_hours,
        "source_dir": str(ssrd_dir),
        "n_files": len(files),
        "n_total_hours": n_total,
        "n_daylight_hours": n_daylight,
        "n_night_hours": n_night,
        "pct_daylight": round(pct, 2),
    }

    daylight_extended.to_dataset(name="daylight_mask").to_netcdf(out_path)
    print(f"\n  Daylight mask saved to: {out_path}")


def apply_daylight_mask_to_era5_variables(
        mask_path: Path,
        variables: List[str],
        era5_dir: Path,
        out_dir: Path,
        file_pattern: str = "{variable}_{year}-{month:02d}.nc",
) -> None:
    """
    Applies a precomputed daylight mask to a list of ERA5 variables, loading
    each monthly file from per-variable subdirectories in era5_dir, filtering
    to daylight hours, and saving to matching subdirectories in out_dir.

    This is the main entry point for masking ERA5 inputs in the solar pipeline.
    Run once after build_daylight_mask() and use the output directory as the
    source for all downstream solar model training steps.

    ERA5 data must be in UTC time when applying the mask, since the mask
    is built from ssrd data in UTC time.

    Note: do NOT apply this mask to ERA5 variables used for wind generation,
    as wind turbines operate around the clock and the solar daylight mask is
    not appropriate for wind.

    Parameters
    ----------
    mask_path:
        Path to the daylight mask NetCDF file produced by build_daylight_mask().
    variables:
        List of ERA5 variable names to mask, e.g. ['ssrd', 't2m'].
    era5_dir:
        Root directory containing per-variable subdirectories of ERA5 NetCDF
        files in UTC time. Structure: era5_dir/{variable}/files.
    out_dir:
        Root directory to write masked files to. Same per-variable subdirectory
        structure as era5_dir is created automatically.
    file_pattern:
        Pattern for ERA5 file names within each variable's subdirectory.
        Must contain {variable}, {year}, {month}.
        Default: {variable}_{year}-{month:02d}.nc
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    # Load mask once and extract daylight timestamps
    print(f"Loading daylight mask from {mask_path} ...")
    mask = xr.open_dataset(mask_path)["daylight_mask"]
    day_times = mask.time.values[mask.values]
    n_daylight = len(day_times)
    print(f"  Daylight hours in mask: {n_daylight:,}")

    # Infer years and months from the first variable's subdirectory
    ref_var = variables[0]
    ref_var_dir = era5_dir / ref_var

    years = sorted(set(
        int(f.stem.split("_")[1].split("-")[0])
        for f in ref_var_dir.glob(f"{ref_var}_*.nc")
    ))
    months = sorted(set(
        int(f.stem.split("_")[1].split("-")[1])
        for f in ref_var_dir.glob(f"{ref_var}_*.nc")
    ))

    for var in variables:
        print(f"\n  Variable: {var}")

        var_out_dir = out_dir / var
        var_out_dir.mkdir(parents=True, exist_ok=True)

        for year in years:
            for month in months:

                in_path = era5_dir / var / file_pattern.format(variable=var, year=year, month=month)
                out_path = var_out_dir / file_pattern.format(variable=var, year=year, month=month)

                if not in_path.exists():
                    print(f"    Skipping {in_path.name} — file not found")
                    continue

                ds = xr.open_dataset(in_path)

                # Select only daylight hours that fall within this month's
                # time range — avoids KeyError when day_times spans years
                # outside this particular file's coverage
                month_times = pd.DatetimeIndex(ds.time.values)
                mask_for_month = pd.DatetimeIndex(day_times)
                overlap = mask_for_month[
                    (mask_for_month >= month_times.min()) &
                    (mask_for_month <= month_times.max())
                    ]

                # Use boolean indexing instead of .sel() to handle duplicate timestamps
                # that arise from the October DST transition (25-hour day)
                time_mask = pd.DatetimeIndex(ds.time.values).isin(overlap)
                ds_filtered = ds.isel(time=time_mask)

                ds_filtered.to_netcdf(out_path)
                print(f"    {in_path.name} -> {out_path.name}  "
                      f"({time_mask.sum()} daylight hours)")

    print(f"\nDone. All masked variables saved to: {out_dir.resolve()}")


def calculate_wind_speed_from_components(
        in_dir: Path,
        out_dir: Path,
        file_pattern: str = "{variable}_{year}-{month:02d}.nc",
) -> None:
    """
    Calculates wind speed from u100 and v100 ERA5 wind components using the
    L2 norm: wind_speed = sqrt(u100² + v100²) and saves the result as a
    new NetCDF variable 'wind_speed_100m' per monthly file.

    Both u100 and v100 must already be present in in_dir as per-variable
    subdirectories, i.e. in_dir/u100/ and in_dir/v100/.

    Parameters
    ----------
    in_dir:
        Root directory containing per-variable subdirectories of ERA5 NetCDF
        files. Structure: in_dir/{variable}/{variable}_{year}-{month}.nc
    out_dir:
        Root directory to write wind speed files to.
        Structure: out_dir/wind_speed_100m/{wind_speed_100m}_{year}-{month}.nc
    file_pattern:
        Pattern for ERA5 file names. Default: {variable}_{year}-{month:02d}.nc
    """

    out_var = "wind_speed_100m"
    var_out_dir = out_dir / out_var
    var_out_dir.mkdir(parents=True, exist_ok=True)

    # Infer years and months from u100 subdirectory
    u100_dir = in_dir / "u100"
    years = sorted(set(
        int(f.stem.split("_")[1].split("-")[0])
        for f in u100_dir.glob("u100_*.nc")
    ))
    months = sorted(set(
        int(f.stem.split("_")[1].split("-")[1])
        for f in u100_dir.glob("u100_*.nc")
    ))

    print(f"Computing wind speed from u100 and v100 ...")
    print(f"  Years:  {years}")
    print(f"  Months: {months}")

    for year in years:
        for month in months:
            u_path = in_dir / "u100" / file_pattern.format(variable="u100", year=year, month=month)
            v_path = in_dir / "v100" / file_pattern.format(variable="v100", year=year, month=month)
            out_path = var_out_dir / file_pattern.format(variable=out_var, year=year, month=month)

            if not u_path.exists():
                print(f"    Skipping {year}-{month:02d} — u100 not found")
                continue
            if not v_path.exists():
                print(f"    Skipping {year}-{month:02d} — v100 not found")
                continue

            ds_u = xr.open_dataset(u_path)
            ds_v = xr.open_dataset(v_path)

            if "valid_time" in ds_u.coords and "time" not in ds_u.coords:
                ds_u = ds_u.rename({"valid_time": "time"})
            if "valid_time" in ds_v.coords and "time" not in ds_v.coords:
                ds_v = ds_v.rename({"valid_time": "time"})

            u = ds_u["u100"]
            v = ds_v["v100"]

            wind_speed = np.sqrt(u ** 2 + v ** 2)
            wind_speed.name = out_var
            wind_speed.attrs = {
                "units": "m s-1",
                "long_name": "Wind speed at 100m derived from u100 and v100 components",
                "description": "sqrt(u100^2 + v100^2)"
            }

            wind_speed.to_dataset(name=out_var).to_netcdf(out_path)
            print(f"    {year}-{month:02d} -> {out_path.name}")

    print(f"\nDone. Wind speed files saved to: {var_out_dir.resolve()}")

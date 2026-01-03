import requests
import bz2
import shutil
import xarray as xr
import numpy as np
from pathlib import Path
from datetime import date
from typing import Dict
from pyproj import Transformer, CRS

from renewables_forecasting.config.technologies import TechnologyConfig, VariableConfig

# todo: combine solar and wind in one pipeline as seems not rotated


# ---- Generic helpers -----

def _download_file(url, target):
    response = requests.get(url, stream=True, timeout=1000)
    # Fail loudly
    response.raise_for_status()
    with open(target, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)


def _ensure_uncompressed_grib(path: Path) -> Path:
    if path.suffix != ".bz2":
        return path

    out_path = path.with_suffix("")  # removes .bz2
    if out_path.exists():
        return out_path

    with bz2.open(path, "rb") as f_in, open(out_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    return out_path


def _open_dataset(path: Path) -> xr.Dataset:
    # For solar files
    if path.name.endswith(".grb.bz2"):
        path = _ensure_uncompressed_grib(path)
        return xr.open_dataset(path, engine="cfgrib")
    # For wind files
    else:
        return xr.open_dataset(path)


def _goto_next_month(current: date):
    year = current.year
    month = current.month
    month += 1
    if month == 13:
        month = 1
        year += 1
    return date(year, month, 1)


# ---- Generic downloader ----

def download_cosmo_rea6(
    variables: Dict[str, VariableConfig],
    start: date,
    end: date,
    output_dir: Path,
):
    # Working with months only.
    # No unambiguity by enforcing use of first of month
    assert start.day == 1
    assert end.day == 1

    output_dir.mkdir(parents=True, exist_ok=True)

    # One directory per var
    for var in variables.values():
        d = start
        while d <= end:
            year = d.year
            month = d.month

            # Construct server file name
            filename = var.filename_pattern.format(var=var.name, year=year, month=month)
            url = f"{var.base_url}/{filename}"

            # Define location to store to
            target = output_dir / var.name / filename
            target.parent.mkdir(parents=True, exist_ok=True)

            # Check for existing file. If so, skip
            if target.exists():
                d = _goto_next_month(d)
                continue

            _download_file(url, target)

            # Increment by month
            d = _goto_next_month(d)


def _regrid(
        dataset: xr.Dataset,
        target_lats: xr.DataArray,
        target_lons: xr.DataArray
) -> xr.Dataset:

    return dataset.interp(
        latitude=target_lats,
        longitude=target_lons,
        method="linear",
    )


# ---- Solar data processing ----

def build_solar_features(vars_path: Path, store_path: Path):

    # Open previously constructed zarr datasets on ASWDIR_S and ASWDIFD_S, respectively
    ds_dir = vars_path

    ds_dif = xr.open_zarr(ds_dir, group="ASWDIFD_S")
    ds_dirr = xr.open_zarr(ds_dir, group="ASWDIR_S")

    ghi = ds_dif["ASWDIFD_S"] + ds_dirr["ASWDIR_S"]

    # Drop cfgrib artefact that else breaks chunking
    ghi = ghi.drop_vars("valid_time", errors="ignore")

    # Explicit, clean coordinate selection
    coords = {
        "time": ghi["time"],
        "latitude": ghi["latitude"],
        "longitude": ghi["longitude"],
    }

    # Construct new dataset with GHI variable only and write to disk as zarr
    xr.Dataset(
        {"GHI": ghi},
        coords=coords,
        attrs={"description": "Global Horizontal Irradiance"},
    ).chunk({"time": 24}).to_zarr(store_path, mode="w")


def _build_solar_target_grid(
    reference_ds: xr.Dataset,
    resolution_km: float = 30.0
) -> tuple[xr.DataArray, xr.DataArray]:

    # Extract lats/lons + convert to numpy arrays
    lats = reference_ds.latitude.values
    lons = reference_ds.longitude.values

    # Lat/lon to meters projection via transformer
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    x, y = transformer.transform(lons, lats)  # x, y in meters

    # Extract domain bounds
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    # Build grid in meters
    dx = resolution_km * 1000  # Cell width in meters
    x_target = np.arange(xmin, xmax + dx, dx)
    y_target = np.arange(ymin, ymax + dx, dx)

    # Transform back to lat/lon
    inv_transformer = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)
    xx, yy = np.meshgrid(x_target, y_target)
    lon_grid, lat_grid = inv_transformer.transform(xx, yy)

    # Ensure 1d coords
    target_lats = lat_grid[:, 0]
    target_lons = lon_grid[0, :]

    # Build data array from np arrays
    target_lats = xr.DataArray(
        target_lats,
        dims=("latitude",),
        name="latitude",
    )

    target_lons = xr.DataArray(
        target_lons,
        dims=("longitude",),
        name="longitude",
    )

    return target_lats, target_lons


def build_solar_target_grid_from_raw(
        raw_base_dir: Path,
        variables: list[str],
        resolution_km: float = 30.0
) -> tuple[xr.DataArray, xr.DataArray]:

    # Extract first grib file to obtain a reference dataset
    ref_var = variables[0]
    var_dir = raw_base_dir / ref_var
    ref_file = next(p for p in var_dir.iterdir() if p.is_file() if p.is_file() and p.name.endswith(".grb.bz2"))

    # Open as xr.Dataset
    ref_ds = _open_dataset(ref_file)

    return _build_solar_target_grid(
        reference_ds=ref_ds,
        resolution_km=resolution_km
    )


def preprocess_solar_data(
    tech: TechnologyConfig,
    zarr_store: Path,
    grid_resolution_km: float = 30.0,
):
    # ------------------------------------------------------------------
    # 1. Build target grid once from first raw file
    #    (uses geographic lat/lon already provided by cfgrib)
    # ------------------------------------------------------------------
    target_lats, target_lons = build_solar_target_grid_from_raw(
        raw_base_dir=tech.raw_subdir,
        variables=list(tech.variables.keys()),
        resolution_km=grid_resolution_km,
    )

    # ------------------------------------------------------------------
    # 2. Loop over variables and files
    # ------------------------------------------------------------------
    for var in tech.variables:
        first_write = True
        var_dir = tech.raw_subdir / var

        for file_path in sorted(var_dir.iterdir()):
            if not file_path.is_file():
                continue

            # Ignore previously decompressed .grb files and cached .idx
            if file_path.name.endswith(".grb") or file_path.name.endswith(".idx"):
                continue

            # Sanity check
            print(f"Variable: {var}")
            print(f"File path: {file_path}")
            ds = _open_dataset(file_path)

            # ----------------------------------------------------------
            # Solar data facts:
            # - dims: (time, y, x)
            # - coords: latitude(y,x), longitude(y,x)
            # ----------------------------------------------------------

            if "latitude" not in ds.coords or "longitude" not in ds.coords:
                raise ValueError(f"{file_path.name} has no latitude/longitude coords")

            # ----------------------------------------------------------
            # 3. Convert 2D lat/lon -> 1D axes
            # ----------------------------------------------------------
            lat_1d = ds.latitude[:, 0].data
            lon_1d = ds.longitude[0, :].data

            # ----------------------------------------------------------
            # 4. Attach 1D coords and promote to dimensions (dims needed for interpolation)
            # ----------------------------------------------------------
            ds = ds.assign_coords(
                latitude=("y", lat_1d),
                longitude=("x", lon_1d),
            )

            ds = ds.swap_dims({"y": "latitude", "x": "longitude"})
            ds = ds.drop_vars(["y", "x", "latitude", "longitude"], errors="ignore")

            ds = ds.transpose("time", "latitude", "longitude")

            # ----------------------------------------------------------
            # 5. Regrid to target grid
            # ----------------------------------------------------------
            ds = ds.interp(
                latitude=target_lats,
                longitude=target_lons,
                method="linear",
            )

            # ----------------------------------------------------------
            # 6. Write / append to Zarr
            # ----------------------------------------------------------
            mode = "w" if first_write else "a"
            ds.to_zarr(
                zarr_store,
                group=var,
                mode=mode,
                append_dim="time" if not first_write else None,
            )

            first_write = False


# ---- Wind data ----

def _build_wind_target_grid(
    reference_ds: xr.Dataset,
    resolution_km: float = 30.0,
) -> tuple[xr.DataArray, xr.DataArray]:

    # Expect 2D geographic lat/lon already present
    if "latitude" not in reference_ds.coords or "longitude" not in reference_ds.coords:
        raise ValueError("Wind dataset has no geographic latitude/longitude coordinates")

    lat2d = reference_ds.latitude.values
    lon2d = reference_ds.longitude.values

    # Project to meters
    proj = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    x, y = proj.transform(lon2d, lat2d)

    dx = resolution_km * 1000
    x_t = np.arange(x.min(), x.max() + dx, dx)
    y_t = np.arange(y.min(), y.max() + dx, dx)

    xx, yy = np.meshgrid(x_t, y_t)

    inv_proj = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)
    lon_t, lat_t = inv_proj.transform(xx, yy)

    target_lats = xr.DataArray(lat_t[:, 0], dims=("latitude",), name="latitude")
    target_lons = xr.DataArray(lon_t[0, :], dims=("longitude",), name="longitude")

    return target_lats, target_lons


def build_wind_target_grid_from_raw(
    raw_base_dir: Path,
    variables: list[str],
    resolution_km: float = 30.0,
) -> tuple[xr.DataArray, xr.DataArray]:

    # Obtain reference grid
    ref_var = variables[0]
    ref_dir = raw_base_dir / ref_var
    ref_file = next(p for p in ref_dir.iterdir() if p.is_file())
    ref_ds = _open_dataset(ref_file)

    # Regrid
    return _build_wind_target_grid(
        reference_ds=ref_ds,
        resolution_km=resolution_km,
    )


def preprocess_wind_data(
    tech: TechnologyConfig,
    zarr_store: Path,
    grid_resolution_km: float = 30,
):

    for var in tech.variables:
        var_dir = tech.raw_subdir / var

        first_write = True

        for file_path in sorted(var_dir.iterdir()):
            if not file_path.is_file():
                continue

            ds = _open_dataset(file_path)

            # Attach geographic lat/lon
            ds = ds.assign_coords(
                latitude=(("rlat", "rlon"), ds["RLAT"].values),
                longitude=(("rlat", "rlon"), ds["RLON"].values),
            )

            # Reduce lat/lot to 1D coords for xarray.interp
            lat_1d = ds.latitude[:, 0]
            lon_1d = ds.longitude[0, :]

            ds = ds.assign_coords(
                latitude=lat_1d,
                longitude=lon_1d,
            )

            # Promote to dimensions for xr.interp
            ds = ds.swap_dims({"rlat": "latitude", "rlon": "longitude"})
            ds = ds.drop_vars(["rlat", "rlon"])

            # Build target grid once
            if first_write:

                proj = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)

                lon2d, lat2d = np.meshgrid(lon_1d.values, lat_1d.values)
                X, Y = proj.transform(lon2d, lat2d)

                dx = grid_resolution_km * 1000
                x_t = np.arange(X.min(), X.max() + dx, dx)
                y_t = np.arange(Y.min(), Y.max() + dx, dx)

                inv = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)
                XX, YY = np.meshgrid(x_t, y_t)
                lon_t, lat_t = inv.transform(XX, YY)

                target_lats = xr.DataArray(lat_t[:, 0], dims="latitude")
                target_lons = xr.DataArray(lon_t[0, :], dims="longitude")

            # 4. Regrid (dims remain rlat/rlon internally)
            ds = ds.interp(
                latitude=target_lats,
                longitude=target_lons,
                method="linear",
            )

            # 5. Write to disk
            mode = "w" if first_write else "a"
            ds.to_zarr(
                zarr_store,
                group=var,
                mode=mode,
                append_dim="time" if not first_write else None,
            )

            first_write = False



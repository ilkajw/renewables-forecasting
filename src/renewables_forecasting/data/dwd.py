import requests
import bz2
import shutil
import xarray as xr
from pyresample import geometry, kd_tree
import numpy as np
from pathlib import Path
from datetime import date
from typing import Dict, Tuple
from pyproj import Transformer, CRS

from renewables_forecasting.config.technologies import TechnologyConfig, VariableConfig


# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------

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
    # For solar files being compressed grib
    if path.name.endswith(".grb.bz2"):
        path = _ensure_uncompressed_grib(path)
        return xr.open_dataset(path, engine="cfgrib")
    # For wind files being NetCDF
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


# -----------------------------------------------------------------------------
# Shared downloader
# -----------------------------------------------------------------------------

def download_cosmo_rea6(
    variables: Dict[str, VariableConfig],
    start: date,
    end: date,
    output_dir: Path,
):
    # Raw data is summarized to monthly files.
    # No ambiguity by enforcing use of first of month
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


# --------------------------------------------------------------------------------
# Projected target grid helpers (regridding curvilinear source -> regular target)
# --------------------------------------------------------------------------------

def _normalize_lon_180(lon):
    return ((lon + 180) % 360) - 180


def _build_3035_target_grid_from_4326_source_grid(
    lat2d: xr.DataArray,
    lon2d: xr.DataArray,
    resolution_km: float = 30.0,
) -> Tuple[geometry.AreaDefinition, xr.DataArray, xr.DataArray]:

    dx = float(resolution_km) * 1000.0

    # Project EPSG:4326 lat/lon source grid to EPSG:3035 meters grid

    # EPSG:3035 is area-preserving (target grid cells are equal in size),
    # and not generally equidistant.
    # Equidistance between target grid cell centers is achieved through
    # target grid construction below.

    # x_ij, y_ij being EPSG:3035 grid coords, lat_ij, lon_ij being source grid coords:
    # Build Transformer which calculates projection P such that:
    # (x_ij, y_ij) = P(lon_ij, lat_ij)

    # Initialise transformer
    to_3035 = Transformer.from_crs(crs_from="EPSG:4326", crs_to="EPSG:3035", always_xy=True)

    # Apply projection, obtaining:
    # todo: are they really centers? affects bounding box calculation
    # x_m: 2D array of cell center x-coords: each COSMO REA6 source grid cell x-coord projected to EPSG:3035
    # y_m: 2D array of cell center y-coords: each COSMO REA6 source grid cell y-coord projected to EPSG:3035
    # in meters from EPSG:3035 origin
    # todo: what is the origin?
    x_m, y_m = to_3035.transform(_normalize_lon_180(lon2d).values, lat2d.values)

    # Get bounds of EPSG:3035 grid
    xmin = np.nanmin(x_m)
    xmax = np.nanmax(x_m)
    ymin = np.nanmin(y_m)
    ymax = np.nanmax(y_m)

    # Construct target grid from EPSG:3035 projection bounds
    # with dx=dy=30 km spacing
    # -> achieves equidistance between cell centers
    x = np.arange(xmin, xmax + dx, dx)
    y = np.arange(ymin, ymax + dx, dx)

    # Calculate outer grid edges
    # as AreaDefinition expects this info for area_extent
    area_extent = (
        float(x[0] - dx / 2),
        float(y[0] - dx / 2),
        float(x[-1] + dx / 2),
        float(y[-1] + dx / 2),
    )

    # Create a pyproj coordinate reference system object representing the EPSG:3035 system
    crs_3035 = CRS.from_epsg(3035)

    # Define EPSG:3035 target grid with target spacing as an AreaDefinition object
    area = geometry.AreaDefinition(
        area_id="target_3035",
        description=f"EPSG:3035 regular grid {resolution_km:.1f}km",
        proj_id="epsg3035",
        # Target geometry (regular, in meters, area preserving)
        projection=crs_3035.to_dict(),
        # Info on target grid: bounds and number of coords
        # with dx, dx spacing implicitly determined
        width=len(x),
        height=len(y),
        area_extent=area_extent,
    )

    # Target grid x, y cell center coords as xr.DataArrays
    x_da = xr.DataArray(x.astype("float64"), dims=("x",), name="x", attrs={"units": "m"})
    y_da = xr.DataArray(y.astype("float64"), dims=("y",), name="y", attrs={"units": "m"})
    return area, x_da, y_da


def _pyresample_resample_nearest_to_area(
    da: xr.DataArray,
    lat2d: xr.DataArray,
    lon2d: xr.DataArray,
    target_area: geometry.AreaDefinition,
    x: xr.DataArray,
    y: xr.DataArray,
    radius_km: float,
) -> xr.DataArray:

    # Checks on dims and shape correctness
    if lat2d.ndim != 2 or lon2d.ndim != 2:
        raise ValueError("lat2d/lon2d must be 2D")
    if lat2d.shape != lon2d.shape:
        raise ValueError("lat2d and lon2d must have the same shape")

    ydim, xdim = lat2d.dims
    if ydim not in da.dims or xdim not in da.dims:
        raise ValueError(f"da dims {da.dims} do not include spatial dims {(ydim, xdim)}")

    # Enforce that da is ordered (ydim, xdim) or (time, ydim, xdim) for correct slicing later
    lead_dims = [d for d in da.dims if d not in (ydim, xdim)]
    da_t = da.transpose(*lead_dims, ydim, xdim)
    print(f"After transpose: da_t.dims={da_t.dims}, da_t.shape={da_t.shape}, lead_dims={lead_dims}")

    # Construct a grid with lon/lat info per cell from lon/lat arrays of the source grid.
    # Used by pyresample to build a KD-tree and find nearest source point for each target point.
    # Using normalized longitudes to avoid 0/360 seam issues
    src_def = geometry.SwathDefinition(
        lons=_normalize_lon_180(lon2d.values),
        lats=lat2d.values,
    )

    # Wrapper for kd_tree resampling of a 2D array
    def _resample_2d(arr2d: np.ndarray) -> np.ndarray:
        return kd_tree.resample_nearest(
            src_def,
            arr2d,
            target_area,
            radius_of_influence=float(radius_km) * 1000.0,
            fill_value=np.nan,
        )

    # Extract data as numpy array
    data = da_t.values

    # 2D case: data along lat, lon
    if data.ndim == 2:
        print("Case: 2D (no time)")

        # Apply resampling obtaining target grid filled with resampled data
        res = _resample_2d(data)

        # Flip y (north<->south) due to pyresample interpreting y=0 as upmost row
        res = res[::-1, :]

        # Obtain resampled data as DataArray
        out = xr.DataArray(
            res,
            dims=("y", "x"),
            coords={"y": y, "x": x},
            name=da.name,
            attrs=da.attrs,
        )

        return out

    # 3D case: data along time, lat, lon
    if data.ndim == 3 and lead_dims == ["time"]:
        print("Case: 3D (time, Y, X)")

        # Apply resampling obtaining target grid filled with resampled data per time step
        res = np.stack([_resample_2d(data[i, :, :]) for i in range(data.shape[0])], axis=0)
        print(f"Stacked res.shape={res.shape} (time,height,width)")

        # Flip y (north<->south) due to pyresample interpreting y=0 as upper row
        res = res[:, ::-1, :]

        # Obtain resampled data as DataArray
        out = xr.DataArray(
            res,
            dims=("time", "y", "x"),
            coords={"time": da_t["time"], "y": y, "x": x},
            name=da.name,
            attrs=da.attrs,
        )

        return out

    raise ValueError(f"Unsupported dims for resampling: da.dims={da.dims}, lead_dims={lead_dims}")


# -----------------------------------------------------------------------------
# Preprocessors
# -----------------------------------------------------------------------------

def preprocess_solar_data(
    tech: TechnologyConfig,
    zarr_store: Path,
    grid_resolution_km: float = 30.0,
) -> None:

    # Open reference solar ds to obtain source grid
    ref_var = list(tech.variables.keys())[0]
    ref_dir = tech.raw_subdir / ref_var
    ref_file = next(p for p in ref_dir.iterdir() if p.is_file() and p.name.endswith(".grb.bz2"))
    ref_ds = _open_dataset(ref_file)

    # Project source grid to regular, area-preserving target grid in EPSG:3035
    target_area, x, y = _build_3035_target_grid_from_4326_source_grid(
        lat2d=ref_ds["latitude"],
        lon2d=ref_ds["longitude"],
        resolution_km=grid_resolution_km,
    )

    # Radius considered in search for nearest source cell
    radius_km = grid_resolution_km * 3.0

    # Loop over variables and their monthly files to regrid data
    for var in tech.variables:
        first_write = True
        var_dir = tech.raw_subdir / var

        for file_path in sorted(var_dir.iterdir()):
            if not file_path.is_file():
                continue

            # Ignore already decompressed files and .idx artefacts
            if file_path.name.endswith((".grb", ".idx")):
                continue

            # Obtain data for var and month
            ds = _open_dataset(file_path)

            # Fill target grid with resampled data
            out = _pyresample_resample_nearest_to_area(
                da=ds[var],
                lat2d=ds["latitude"],
                lon2d=ds["longitude"],
                target_area=target_area,
                x=x,
                y=y,
                radius_km=radius_km,
            )

            # Obtain resampled data as xarray Dataset
            out_ds = xr.Dataset({var: out}).chunk({"time": 24})

            # Attach CRS grid type and resolution info
            out_ds.attrs["crs"] = "EPSG:3035"
            out_ds.attrs["grid_resolution_m"] = float(grid_resolution_km) * 1000.0

            # Write to disk
            if first_write:
                out_ds.to_zarr(zarr_store, group=var, mode="w")
                first_write = False
            else:
                out_ds.to_zarr(zarr_store, group=var, mode="a", append_dim="time")


def preprocess_wind_data(
    tech: TechnologyConfig,
    zarr_store: Path,
    grid_resolution_km: float = 30.0,
) -> None:

    # Open reference wind file to obtain source grid
    ref_var = list(tech.variables.keys())[0]
    ref_dir = tech.raw_subdir / ref_var
    ref_file = next(p for p in ref_dir.iterdir() if p.is_file())
    ref_ds = _open_dataset(ref_file)

    if "RLAT" not in ref_ds or "RLON" not in ref_ds:
        raise ValueError("Wind dataset has no RLAT/RLON coordinates")

    # Project EPSG:4326 lon/lat source grid to EPSG:3035 area-preserving target grid in meters
    target_area, x, y = _build_3035_target_grid_from_4326_source_grid(
        lat2d=ref_ds["RLAT"],
        lon2d=ref_ds["RLON"],
        resolution_km=grid_resolution_km,
    )

    # Radius used to search for nearest source cell in resampling
    radius_km = grid_resolution_km * 3.0

    # Loop over variables and their monthly files to regrid data
    for var in tech.variables:
        first_write = True
        var_dir = tech.raw_subdir / var

        for file_path in sorted(var_dir.iterdir()):
            if not file_path.is_file():
                continue

            # Load data for var and month
            ds = _open_dataset(file_path)

            # Fill target grid with resampled data
            out = _pyresample_resample_nearest_to_area(
                da=ds["wind_speed"],
                lat2d=ds["RLAT"],
                lon2d=ds["RLON"],
                target_area=target_area,
                x=x,
                y=y,
                radius_km=radius_km,
            )

            # Obtain resampled data as xarray dataset
            out_ds = xr.Dataset({var: out}).chunk({"time": 24})

            # Attach CRS grid type and resolution info
            out_ds.attrs["crs"] = "EPSG:3035"
            out_ds.attrs["grid_resolution_m"] = float(grid_resolution_km) * 1000.0

            # Write to disk
            if first_write:
                out_ds.to_zarr(zarr_store, group=var, mode="w")
                first_write = False
            else:
                out_ds.to_zarr(zarr_store, group=var, mode="a", append_dim="time")


# -----------------------------------------------------------------------------
# Solar features
# -----------------------------------------------------------------------------

def build_solar_features(vars_path: Path, store_path: Path) -> None:

    ds_dir = vars_path
    ds_dif = xr.open_zarr(ds_dir, group="ASWDIFD_S")
    ds_dirr = xr.open_zarr(ds_dir, group="ASWDIR_S")

    # Feature GHI is sum of direct and diffuse irradiance
    ghi = ds_dif["ASWDIFD_S"] + ds_dirr["ASWDIR_S"]
    ghi = ghi.drop_vars("valid_time", errors="ignore")

    # Descriptive attrs
    ghi = ghi.assign_attrs(
        standard_name="surface_downwelling_shortwave_flux",
        long_name="Global Horizontal Irradiance",
        units="W m-2",
    )

    # Define ghi dataset coords
    coords = {"time": ghi["time"], "y": ghi["y"], "x": ghi["x"]}

    # Write to disk as zarr
    xr.Dataset(
        {"GHI": ghi},
        coords=coords,
        attrs={"description": "Global Horizontal Irradiance", "crs": "EPSG:3035"},
    ).chunk({"time": 24}).to_zarr(store_path, group="GHI", mode="w")

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


# --------------------------------------------------------------------------------
# Projected target grid helpers (regridding curvilinear source -> regular target)
# --------------------------------------------------------------------------------

def _normalize_lon_180(lon):
    return ((lon + 180) % 360) - 180


def _build_area_3035_from_latlon(
    lat2d: xr.DataArray,
    lon2d: xr.DataArray,
    resolution_km: float = 30.0,
) -> Tuple[geometry.AreaDefinition, xr.DataArray, xr.DataArray]:
    """
    Build a rectangular, regular target grid in EPSG:3035 with square cells.

    Returns:
      - pyresample AreaDefinition (target)
      - x centers (meters) as 1D DataArray
      - y centers (meters) as 1D DataArray

    Note:
      - No lat/lon target coordinates are created or stored.
      - Only x/y in meters define the grid.
    """
    dx = float(resolution_km) * 1000.0

    # Project source lat/lon to EPSG:3035 meters to get bounds
    to_3035 = Transformer.from_crs(crs_from="EPSG:4326", crs_to="EPSG:3035", always_xy=True)
    x_m, y_m = to_3035.transform(_normalize_lon_180(lon2d).values, lat2d.values)

    xmin = np.nanmin(x_m)
    xmax = np.nanmax(x_m)
    ymin = np.nanmin(y_m)
    ymax = np.nanmax(y_m)

    # Define target cell centers on a regular meter grid
    x = np.arange(xmin, xmax + dx, dx)
    y = np.arange(ymin, ymax + dx, dx)

    # AreaDefinition expects area extent as edges (llx, lly, urx, ury)
    area_extent = (
        float(x[0] - dx / 2),
        float(y[0] - dx / 2),
        float(x[-1] + dx / 2),
        float(y[-1] + dx / 2),
    )

    # Create a pyproj CRS reference object representing the EPSG:3035 system
    crs_3035 = CRS.from_epsg(3035)

    # Define grid as AreaDefinition regular raster object
    area = geometry.AreaDefinition(
        area_id="target_3035",
        description=f"EPSG:3035 regular grid {resolution_km:.1f}km",
        proj_id="epsg3035",
        projection=crs_3035.to_dict(),
        width=len(x),
        height=len(y),
        area_extent=area_extent,
    )

    # Sanity checks
    assert area.width == len(x)
    assert area.height == len(y)
    llx, lly, urx, ury = area.area_extent
    print("extent:", llx, lly, urx, ury)
    assert llx < urx
    assert lly < ury

    x_da = xr.DataArray(x.astype("float64"), dims=("x",), name="x", attrs={"units": "m"})
    y_da = xr.DataArray(y.astype("float64"), dims=("y",), name="y", attrs={"units": "m"})
    return area, x_da, y_da


def _print_resample_diagnostics(
    *,
    src_arr2d: np.ndarray,
    src_lat2d: np.ndarray,
    src_lon2d: np.ndarray,
    res_arr2d: np.ndarray,
    target_area,
    x_1d: np.ndarray,
    y_1d: np.ndarray,
    label: str,
):
    print(f"\n--- DIAGNOSTICS [{label}] ---")

    # What does target_area consider row0? (north or south)
    tlons, tlats = target_area.get_lonlats()  # shape (height,width)
    row0_lat = float(np.nanmean(tlats[0, :]))
    rowN_lat = float(np.nanmean(tlats[-1, :]))
    col0_lon = float(np.nanmean(tlons[:, 0]))
    colN_lon = float(np.nanmean(tlons[:, -1]))

    print(f"target_area: mean lat row0={row0_lat:.3f}, lastrow={rowN_lat:.3f}")
    print(f"target_area: mean lon col0={col0_lon:.3f}, lastcol={colN_lon:.3f}")
    print("target_area row0 is NORTH?" , row0_lat > rowN_lat)
    print("target_area col0 is WEST?"  , col0_lon < colN_lon)

    # What do y coords mean in lat/lon?
    inv = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)
    x_mid = float(x_1d[len(x_1d)//2])

    y0 = float(y_1d[0])
    yN = float(y_1d[-1])
    _, lat_y0 = inv.transform(x_mid, y0)
    _, lat_yN = inv.transform(x_mid, yN)
    print(f"Coords: y[0]={y0:.0f} -> lat≈{lat_y0:.3f}, y[-1]={yN:.0f} -> lat≈{lat_yN:.3f}")
    print("y increases NORTH?", lat_y0 < lat_yN)

    # Where is the max in SOURCE (lat/lon)?
    if np.isfinite(src_arr2d).any():
        si = np.nanargmax(src_arr2d)
        sy, sx = np.unravel_index(si, src_arr2d.shape)
        smax = float(src_arr2d[sy, sx])
        slat = float(src_lat2d[sy, sx])
        slon = float(src_lon2d[sy, sx])
        print(f"SOURCE max={smax:.3f} at (iy,ix)=({sy},{sx}) -> lat={slat:.3f}, lon={slon:.3f}")
    else:
        print("SOURCE: all NaN")

    # Where is the max in RESAMPLED (lat/lon)?
    if np.isfinite(res_arr2d).any():
        ri = np.nanargmax(res_arr2d)
        ry, rx = np.unravel_index(ri, res_arr2d.shape)
        rmax = float(res_arr2d[ry, rx])

        # True lat/lon of that cell according to target_area
        rlat_true = float(tlats[ry, rx])
        rlon_true = float(tlons[ry, rx])
        print(f"RESAMP max={rmax:.3f} at (iy,ix)=({ry},{rx}) -> TRUE lat={rlat_true:.3f}, lon={rlon_true:.3f}")

        lon_xy, lat_xy = inv.transform(float(x_1d[rx]), float(y_1d[ry]))
        print(f"RESAMP max mapped via (x,y) -> lat≈{lat_xy:.3f}, lon≈{lon_xy:.3f}")
    else:
        print("RESAMP: all NaN")

    # NaN fraction
    print(f"RESAMP NaN fraction: {float(np.isnan(res_arr2d).mean()):.3f}")
    print(f"--- END DIAGNOSTICS [{label}] ---\n")


def _pyresample_resample_nearest_to_area(
    da: xr.DataArray,
    lat2d: xr.DataArray,
    lon2d: xr.DataArray,
    target_area: geometry.AreaDefinition,
    x: xr.DataArray,
    y: xr.DataArray,
    radius_km: float,
) -> xr.DataArray:

    # Checks
    if lat2d.ndim != 2 or lon2d.ndim != 2:
        raise ValueError("lat2d/lon2d must be 2D")
    if lat2d.shape != lon2d.shape:
        raise ValueError("lat2d and lon2d must have the same shape")

    ydim, xdim = lat2d.dims
    if ydim not in da.dims or xdim not in da.dims:
        raise ValueError(f"da dims {da.dims} do not include spatial dims {(ydim, xdim)}")

    # Move spatial dims last
    lead_dims = [d for d in da.dims if d not in (ydim, xdim)]
    da_t = da.transpose(*lead_dims, ydim, xdim)
    print(f"After transpose: da_t.dims={da_t.dims}, da_t.shape={da_t.shape}, lead_dims={lead_dims}")

    src_def = geometry.SwathDefinition(lons=lon2d.values, lats=lat2d.values)

    def _resample_2d(arr2d: np.ndarray) -> np.ndarray:
        return kd_tree.resample_nearest(
            src_def,
            arr2d,
            target_area,
            radius_of_influence=float(radius_km) * 1000.0,
            fill_value=np.nan,
        )

    data = da_t.values

    if data.ndim == 2:
        print("Case: 2D (no time)")
        res = _resample_2d(data)

        _print_resample_diagnostics(
            src_arr2d=data,
            src_lat2d=lat2d.values,
            src_lon2d=lon2d.values,
            res_arr2d=res,
            target_area=target_area,
            x_1d=x.values,
            y_1d=y.values,
            label="BEFORE_FLIP",
        )

        # Flip y (north<->south) due to pyresample interpreting y=0 at the top
        res = res[::-1, :]

        _print_resample_diagnostics(
            src_arr2d=data,
            src_lat2d=lat2d.values,
            src_lon2d=lon2d.values,
            res_arr2d=res,
            target_area=target_area,
            x_1d=x.values,
            y_1d=y.values,
            label="AFTER_FLIP",
        )

        # Sanity checks
        assert res.shape == (len(y), len(x))
        assert res.shape == (target_area.height, target_area.width)

        out = xr.DataArray(
            res,
            dims=("y", "x"),
            coords={"y": y, "x": x},
            name=da.name,
            attrs=da.attrs,
        )

        return out

    if data.ndim == 3 and lead_dims == ["time"]:
        print("Case: 3D (time, Y, X)")
        res0 = _resample_2d(data[0, :, :])

        _print_resample_diagnostics(
            src_arr2d=data[0, :, :],
            src_lat2d=lat2d.values,
            src_lon2d=lon2d.values,
            res_arr2d=res0,
            target_area=target_area,
            x_1d=x.values,
            y_1d=y.values,
            label="T0_BEFORE_FLIP",
        )

        res = np.stack([_resample_2d(data[i, :, :]) for i in range(data.shape[0])], axis=0)
        print(f"Stacked res.shape={res.shape} (time,height,width)")

        # Flip y (north<->south) due to pyresample interpreting y=0 at the top
        res = res[:, ::-1, :]

        _print_resample_diagnostics(
            src_arr2d=data[0, :, :],
            src_lat2d=lat2d.values,
            src_lon2d=lon2d.values,
            res_arr2d=res[0, :, :],
            target_area=target_area,
            x_1d=x.values,
            y_1d=y.values,
            label="T0_AFTER_FLIP",
        )

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

    # Build target area (x/y) from a reference solar file (lat/lon source)
    ref_var = list(tech.variables.keys())[0]
    ref_dir = tech.raw_subdir / ref_var
    ref_file = next(p for p in ref_dir.iterdir() if p.is_file() and p.name.endswith(".grb.bz2"))
    ref_ds = _open_dataset(ref_file)

    target_area, x, y = _build_area_3035_from_latlon(
        lat2d=ref_ds["latitude"],
        lon2d=ref_ds["longitude"],
        resolution_km=grid_resolution_km,
    )

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

            # Regrid data to target grid
            out = _pyresample_resample_nearest_to_area(
                da=ds[var],
                lat2d=ds["latitude"],
                lon2d=ds["longitude"],
                target_area=target_area,
                x=x,
                y=y,
                radius_km=radius_km,
            )

            # Obtain data as an xarray Dataset
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

    # Build target area (x/y) from a reference wind file (RLAT/RLON source)
    ref_var = list(tech.variables.keys())[0]
    ref_dir = tech.raw_subdir / ref_var
    ref_file = next(p for p in ref_dir.iterdir() if p.is_file())
    ref_ds = _open_dataset(ref_file)

    if "RLAT" not in ref_ds or "RLON" not in ref_ds:
        raise ValueError("Wind dataset has no RLAT/RLON coordinates")

    target_area, x, y = _build_area_3035_from_latlon(
        lat2d=ref_ds["RLAT"],
        lon2d=ref_ds["RLON"],
        resolution_km=grid_resolution_km,
    )

    # Radius used to search for neighbours in regridding
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

            # Regrid data to the target grid
            out = _pyresample_resample_nearest_to_area(
                da=ds["wind_speed"],
                lat2d=ds["RLAT"],
                lon2d=ds["RLON"],
                target_area=target_area,
                x=x,
                y=y,
                radius_km=radius_km,
            )

            # Transform to xarray dataset
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

    ghi = ghi.assign_attrs(
        standard_name="surface_downwelling_shortwave_flux",
        long_name="Global Horizontal Irradiance",
        units="W m-2",
    )

    # Target is projected: x/y coords (no latitude/longitude)
    coords = {"time": ghi["time"], "y": ghi["y"], "x": ghi["x"]}

    xr.Dataset(
        {"GHI": ghi},
        coords=coords,
        attrs={"description": "Global Horizontal Irradiance", "crs": "EPSG:3035"},
    ).chunk({"time": 24}).to_zarr(store_path, group="GHI", mode="w")

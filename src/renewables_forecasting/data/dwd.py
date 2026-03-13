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

from renewables_forecasting.config.technologies import TechnologyConfig, WeatherVariableSource
from renewables_forecasting.config.paths import DWD_SOLAR_DATA_DIR_RAW, DWD_WIND_DATA_DIR_RAW
from renewables_forecasting.config.data_constants import GERMANY_LAT_MIN, GERMANY_LAT_MAX, GERMANY_LON_MIN, GERMANY_LON_MAX


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


def _save_regrid_data_to_zarr(
    da: xr.DataArray,
    store_path: Path,
    first_write: bool,
    grid_resolution_m,
    var: str
):
    # Obtain resampled data as xarray dataset
    out_ds = xr.Dataset({var: da}).chunk({"time": 24})

    # Attach CRS grid type and resolution info
    out_ds.attrs["crs"] = "EPSG:3035"
    out_ds.attrs["grid_resolution_m"] = float(grid_resolution_m)

    # Write to disk
    if first_write:
        out_ds.to_zarr(store_path, group=var, mode="w")
    else:
        out_ds.to_zarr(store_path, group=var, mode="a", append_dim="time")


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
    variables: Dict[str, WeatherVariableSource],
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


def _slice_germany_from_epsg_4326_grid(lat2d: xr.DataArray, lon2d: xr.DataArray):
    mask = (lat2d >= GERMANY_LAT_MIN) & (lat2d <= GERMANY_LAT_MAX) & (lon2d >= GERMANY_LON_MIN) & (lon2d <= GERMANY_LON_MAX)
    lat2d_germany = lat2d.where(mask)
    lon2d_germany = lon2d.where(mask)

    return lat2d_germany, lon2d_germany


def _build_3035_target_grid_from_4326_source_grid(
    lat2d: xr.DataArray,
    lon2d: xr.DataArray,
    cell_size_m: float = 30000.0,
) -> Tuple[xr.DataArray, xr.DataArray]:

    # Project the lat-, lon-coordinates of cell centers of a EPSG:4326 curvilinear source grid
    # to the EPSG:3035 coord ref system, obtaining cell center x-, y-coords being meters north- and eastward
    # from the EPSG:3035 origin for every source grid cell

    # EPSG:3035 is an equal-area projection (equal areas in meter space correspond to equal areas on earth's surface),
    # but does not guarantee equidistance of points. Equidistance between target grid cell centers is achieved through
    # target grid construction below

    # (x_ij, y_ij) being meters from EPSG:3035 origin,
    # lat_ij, lon_ij being EPSG:4326 (source grid) coords in degrees:
    # Build Transformer which calculates projection P such that:
    # (x_ij, y_ij) = P(lon_ij, lat_ij)

    to_3035 = Transformer.from_crs(crs_from="EPSG:4326", crs_to="EPSG:3035", always_xy=True)

    # Apply projection, obtaining:
    # x_m: 2D array of x-coords of the curvilinear grid cell centers in meters eastward from 3035 origin
    # y_m: 2D array of y-coords of the curvilinear grid cell centers in meters northward from 3035 origin
    x_m, y_m = to_3035.transform(_normalize_lon_180(lon2d).values, lat2d.values)

    # Get bounds
    xmin = np.nanmin(x_m)
    xmax = np.nanmax(x_m)
    ymin = np.nanmin(y_m)
    ymax = np.nanmax(y_m)

    # Construct equidistant target grid between bounds with desired spacing
    x = np.arange(xmin, xmax + cell_size_m, cell_size_m)  # todo: use linspace instead of arange?
    y = np.arange(ymin, ymax + cell_size_m, cell_size_m)

    # Save target grid (cell center x-, y-coords) as xr.DataArrays
    x_da = xr.DataArray(x.astype("float64"), dims=("x",), name="x", attrs={"units": "m"})
    y_da = xr.DataArray(y.astype("float64"), dims=("y",), name="y", attrs={"units": "m"})
    return x_da, y_da


def _build_and_save_epsg_3035_germany_target_grid(
    ref_ds: Path,
    store_path: Path,
    cell_size_m: float = 30000.0
) -> None:

    assert 1000 <= cell_size_m <= 100_000, (
        f"cell_size_m={cell_size_m} looks wrong - expected between 1km and 100km"
    )

    # Reference ds to retrieve cosmo lat-lon map
    ds = _open_dataset(ref_ds)
    lat2d = ds["latitude"]  # assumes solar ds
    lon2d = ds["longitude"]

    # Slice out Germany from map
    lat2d_germany, lon2d_germany = _slice_germany_from_epsg_4326_grid(lat2d, lon2d)

    # Project EPSG:4326 coord arrays over Germany to EPSG:3035 coord arrays
    x, y = _build_3035_target_grid_from_4326_source_grid(
        lat2d=lat2d_germany,
        lon2d=lon2d_germany,
        cell_size_m=cell_size_m,
    )

    # Save EPSG:3035 target map for later weather and plant capacity utilisation
    xr.Dataset(
        coords={"x": x, "y": y},
        attrs={
            "crs": "EPSG:3035",
            "grid_resolution_m": float(cell_size_m),
        },
    ).to_zarr(store_path, mode="w")


def load_target_grid(store_path: Path):

    ds = xr.open_zarr(store_path)
    x = ds['x']
    y = ds['y']

    return x, y


def _get_epsg_3035_area_obj(
        x: xr.DataArray,
        y: xr.DataArray,
        grid_resolution_m: float = 30000.0
):

    # Calculate outer grid edges (AreaDefinition expects this as area_extent)
    area_extent = (
        float(x[0] - grid_resolution_m / 2),
        float(y[0] - grid_resolution_m / 2),
        float(x[-1] + grid_resolution_m / 2),
        float(y[-1] + grid_resolution_m / 2),
    )

    # Create a pyproj coordinate reference system object representing the EPSG:3035 system
    crs_3035 = CRS.from_epsg(3035)

    # Define target grid as an AreaDefinition object with desired cell size
    # and coordinates of bounds and cells derived from EPSG:3035
    area = geometry.AreaDefinition(
        area_id="target_3035",
        description=f"EPSG:3035 regular grid {grid_resolution_m:.1f}m",
        proj_id="epsg3035",
        # Target geometry (in meters, area preserving)
        projection=crs_3035.to_dict(),
        # Target grid info: bound coordinates and number of coords. grid_resolution spacing is implicitly determined
        # -> width, height, area_extent determine regularity
        width=len(x),
        height=len(y),
        area_extent=area_extent,
    )

    return area


def _pyresample_resample_nearest_to_area(
    da: xr.DataArray,
    lat2d: xr.DataArray,
    lon2d: xr.DataArray,
    target_area: geometry.AreaDefinition,
    x: xr.DataArray,
    y: xr.DataArray,
    radius_of_influence_m: float,
) -> xr.DataArray:

    # Fill the regular meter-measured target grid with data resampled via nearest-neighbor resampling from the
    # curvilinear source grid

    # Checks on dims and shape correctness
    if lat2d.ndim != 2 or lon2d.ndim != 2:
        raise ValueError("lat2d/lon2d must be 2D")
    if lat2d.shape != lon2d.shape:
        raise ValueError("lat2d and lon2d must have the same shape")

    ydim, xdim = lat2d.dims
    if ydim not in da.dims or xdim not in da.dims:
        raise ValueError(f"da dims {da.dims} do not include spatial dims {(ydim, xdim)}")

    # Enforce that da is ordered (ydim, xdim) or (time, ydim, xdim) for correct slicing
    lead_dims = [d for d in da.dims if d not in (ydim, xdim)]
    da_t = da.transpose(*lead_dims, ydim, xdim)
    print(f"After transpose: da_t.dims={da_t.dims}, da_t.shape={da_t.shape}, lead_dims={lead_dims}")

    # Construct swath with lat-lon coords per cell from  lon-lat coord arrays of source grid.
    # Used by pyresample to build a KD-tree and find nearest source point for each target grid cell.
    src_def = geometry.SwathDefinition(
        lons=_normalize_lon_180(lon2d.values),  # normalized longitudes to avoid 0/360 seam issues
        lats=lat2d.values,
    )

    # Wrapper for kd_tree resampling of a 2D array
    def _resample_2d(arr2d: np.ndarray) -> np.ndarray:
        return kd_tree.resample_nearest(
            src_def,
            arr2d,
            target_area,
            radius_of_influence=float(radius_of_influence_m),
            fill_value=np.nan,
        )

    # Extract data as numpy array
    data = da_t.values

    if data.ndim == 3 and lead_dims == ["time"]:

        # Apply resampling obtaining target grid filled with resampled data per time step
        res = np.stack([_resample_2d(data[i, :, :]) for i in range(data.shape[0])], axis=0)
        print(f"Stacked res.shape={res.shape} (time,height,width)")

        # pyresample returns row 0 = north (top of area_extent).
        # Flip so row 0 = south, matching ascending y coordinate array
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

    raise ValueError(f"Unsupported dims for resampling: da.dims={da.dims}, lead_dims={lead_dims}. 3 dim needed.")


# -----------------------------------------------------------------------------
# Regrid weather data to regular meter grid
# -----------------------------------------------------------------------------

def regrid_cosmo_rea6_solar(
    tech: TechnologyConfig,
    data_zarr_store: Path,
    target_grid_zarr_store: Path,
    grid_resolution_m: float = 30000.0,
) -> None:

    # Load target grid
    x, y = load_target_grid(target_grid_zarr_store)
    area = _get_epsg_3035_area_obj(x, y, grid_resolution_m)

    # Radius considered in search for nearest source cell
    radius_of_influence_m = grid_resolution_m * 1.0

    # Loop over variables and their monthly files to regrid all solar data
    for var in tech.weather_variables:
        first_write = True
        var_dir = DWD_SOLAR_DATA_DIR_RAW / var

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
                target_area=area,
                x=x,
                y=y,
                radius_of_influence_m=radius_of_influence_m,
            )

            _save_regrid_data_to_zarr(
                da=out,
                store_path=data_zarr_store,
                var=var,
                first_write=first_write,
                grid_resolution_m=grid_resolution_m,
            )

            first_write = False


def regrid_cosmo_rea6_wind(
    tech: TechnologyConfig,
    data_zarr_store: Path,
    target_grid_zarr_store: Path,
    grid_resolution_m: float = 30000.0,
) -> None:

    # Get target grid
    x, y = load_target_grid(target_grid_zarr_store)
    area = _get_epsg_3035_area_obj(x, y, grid_resolution_m)

    # Radius used to search for nearest source cell in resampling. Adjust multiplier accordingly
    radius_of_influence_m = grid_resolution_m * 1.0

    # Loop over variables and their monthly files to regrid all wind data
    for var in tech.weather_variables:
        first_write = True
        var_dir = DWD_WIND_DATA_DIR_RAW / var

        for file_path in sorted(var_dir.iterdir()):

            # Ignore any dirs
            if not file_path.is_file():
                continue

            # Load data for var and month
            ds = _open_dataset(file_path)

            # Fill target grid with resampled data
            out = _pyresample_resample_nearest_to_area(
                da=ds["wind_speed"],  # for any of our variables, data is under 'wind_speed'
                lat2d=ds["RLAT"],
                lon2d=ds["RLON"],
                target_area=area,
                x=x,
                y=y,
                radius_of_influence_m=radius_of_influence_m,
            )

            _save_regrid_data_to_zarr(
                da=out,
                store_path=data_zarr_store,
                var=var,
                first_write=first_write,
                grid_resolution_m=grid_resolution_m,
            )

            first_write = False


# -----------------------------------------------------------------------------
# Solar features
# -----------------------------------------------------------------------------

def build_solar_feature_grid(vars_path: Path, store_path: Path) -> None:

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

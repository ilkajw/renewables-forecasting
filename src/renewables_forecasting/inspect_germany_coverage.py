import xarray as xr
import numpy as np
from pyproj import Transformer
from renewables_forecasting.config.paths import DATA_DIR


def germany_bbox_coverage_on_feature(da: xr.DataArray) -> float:

    # Germany bbox in lon/lat
    lon_min, lon_max = 5.9, 15.1
    lat_min, lat_max = 47.3, 55.1

    # Project bbox to EPSG:3035
    to3035 = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    xs, ys = [], []
    for lon in (lon_min, lon_max):
        for lat in (lat_min, lat_max):
            x, y = to3035.transform(lon, lat)
            xs.append(x); ys.append(y)
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    sub = da.sel(x=slice(x_min, x_max), y=slice(y_min, y_max))

    if "time" in sub.dims:
        sub = sub.isel(time=0)

    return float(np.isfinite(sub.values).mean())


wind_feature_store = DATA_DIR / "features/dwd/wind"
solar_feature_store = DATA_DIR / "features/dwd/solar"
ds = xr.open_zarr(solar_feature_store, group="GHI")
frac = germany_bbox_coverage_on_feature(ds["GHI"])
print("Germany coverage fraction:", frac)

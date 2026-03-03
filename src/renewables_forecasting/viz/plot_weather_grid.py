import xarray as xr

from renewables_forecasting.viz.weather_grid import plot_grid_heatmap
from renewables_forecasting.config.paths import DATA_DIR

#tech = "solar"
tech = "wind"
#var = "GHI"
var = "WS_100"
time = "2017-02-01T23:00"

zarr_path = DATA_DIR / f"features/dwd/{tech}/"

# Plot weather grids with location relative to center of Germany

plot_grid_heatmap(
    data=xr.open_zarr(zarr_path, group=var),
    var=var,
    time_point=time,
)



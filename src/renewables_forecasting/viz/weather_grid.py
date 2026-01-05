from pathlib import Path
import xarray as xr
from pyproj import Transformer

import matplotlib
matplotlib.use("TkAgg")  # or "QtAgg"

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def plot_grid_heatmap(
    data,
    var: str | None = None,
    *,
    time_point: str | None = None,
    time_slice: slice | None = None,
    time_reduce: str | None = None,
    cmap: str = "Reds",
    figsize=(8, 6),
):
    """
    Plot a 2D lat/lon heat map from gridded xarray data.

    Parameters
    ----------
    data
        xr.Dataset, xr.DataArray, or Path to zarr store
    var
        Variable name if data is a Dataset
    time_point
        Single timestamp (e.g. "2017-01-01T12:00")
    time_slice
        Time range, e.g. slice("2017-01-01", "2017-01-31")
    time_reduce
        Reduction over time: "mean", "sum", "max", "min"
    """

    # Normalize input -> DataArray
    if isinstance(data, Path):
        data = xr.open_zarr(data)

    if isinstance(data, xr.Dataset):
        if var is None:
            raise ValueError("`var` must be provided when data is a Dataset.")
        da = data[var]
    else:
        da = data

    # Time handling

    # Time point before time slice
    if time_point is not None:
        da = da.sel(time=time_point)

    elif time_reduce is not None:
        if time_slice is not None:
            # If with time reduce no time slice is given,
            # reduce over all time
            da = da.sel(time=time_slice)
        da = getattr(da, time_reduce)(dim="time")

    # Sanity check: must be 2D now
    if "time" in da.dims:
        raise ValueError(
            "Time dimension still present. "
            "Specify at least `time_point` or `time_reduce`."
        )

    # Plot
    plt.figure(figsize=figsize)

    to3035 = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    x0, y0 = to3035.transform(13.4050, 52.5200)  # Berlin lon/lat

    da_rel = da.assign_coords(
        x=(da.x - x0) / 1000,
        y=(da.y - y0) / 1000,
    )

    m = da_rel.plot.pcolormesh(x="x", y="y", cmap=cmap, robust=True)
    m.axes.set_xlabel("x (km from Berlin)")
    m.axes.set_ylabel("y (km from Berlin)")

    ax = m.axes
    ax.ticklabel_format(style="plain", axis="both", useOffset=False)
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax.set_aspect("equal", adjustable="box")

    time = time_point if time_point else time_slice if time_slice else "'all_time'"
    plt.title(f"{da.name} at time {time}")
    plt.tight_layout()
    plt.show()

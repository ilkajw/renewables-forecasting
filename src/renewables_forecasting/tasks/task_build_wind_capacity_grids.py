from datetime import date

from renewables_forecasting.config.paths import (
    MASTR_WIND_PLANTS_EFFECTIVE_START_WITH_COORDS_CSV,
    WIND_GRID_REFERENCE_DS_STORE,
    WIND_CAPACITY_GRIDS_ZARR_STORE
)
from renewables_forecasting.data.capacity_grids import build_capacity_grids


# ── Build daily grids of wind power capacity over Germany from coordinate-located plants,
# their effective active periods and capacity generation data ─────────────────────────────

def task_build_wind_capacity_grids(
        depends_on=MASTR_WIND_PLANTS_EFFECTIVE_START_WITH_COORDS_CSV,
        produces=WIND_CAPACITY_GRIDS_ZARR_STORE / ".complete"
):

    build_capacity_grids(
        ref_weather_ds=WIND_GRID_REFERENCE_DS_STORE,  # wind-specific
        plants_csv=MASTR_WIND_PLANTS_EFFECTIVE_START_WITH_COORDS_CSV,
        start_month=date(2015, 1, 1),
        end_month=date(2025, 12, 1),  # includes whole month of December 2025
        out_dir=WIND_CAPACITY_GRIDS_ZARR_STORE,
    )

    (WIND_CAPACITY_GRIDS_ZARR_STORE / ".complete").touch()

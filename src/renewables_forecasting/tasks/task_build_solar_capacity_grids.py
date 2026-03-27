from datetime import date

from renewables_forecasting.config.paths import (
    MASTR_SOLAR_PLANTS_EFFECTIVE_START_WITH_COORDS_CSV,
    SOLAR_GRID_REFERENCE_DS_STORE,
    SOLAR_CAPACITY_GRIDS_ZARR_STORE
)
from renewables_forecasting.data.capacity_grids import build_capacity_grids


# ── Build daily grids of solar power capacity over Germany from coordinate-located plants,
# their effective active periods and capacity generation data ─────────────────────────────

def task_build_solar_capacity_grids(
        depends_on=MASTR_SOLAR_PLANTS_EFFECTIVE_START_WITH_COORDS_CSV,
        produces=SOLAR_CAPACITY_GRIDS_ZARR_STORE / ".complete"
):

    build_capacity_grids(
        ref_weather_ds=SOLAR_GRID_REFERENCE_DS_STORE,
        plants_csv=MASTR_SOLAR_PLANTS_EFFECTIVE_START_WITH_COORDS_CSV,
        start_month=date(2015, 1, 1),
        end_month=date(2025, 12, 1),
        out_dir=SOLAR_CAPACITY_GRIDS_ZARR_STORE,
    )

    (SOLAR_CAPACITY_GRIDS_ZARR_STORE / ".complete").touch()

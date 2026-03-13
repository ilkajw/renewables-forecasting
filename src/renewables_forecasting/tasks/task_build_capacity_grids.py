from datetime import date

from renewables_forecasting.config.paths import (
    GEONAMES_POSTAL_CODE_DATA,
    MASTR_SOLAR_PLANTS_FILTERED_CSV,
    MASTR_SOLAR_PLANTS_FILTERED_WITH_COORDS_CSV,
    MASTR_SOLAR_PLANTS_FILTERED_WITH_COORDS_SQLITE,
    GRID_REFERENCE_DS_STORE,
    SOLAR_CAPACITY_GRIDS_ZARR_STORE
)
from renewables_forecasting.data.capacity_grids import assign_coords_to_plants, build_capacity_grids
from renewables_forecasting.data.mastr import csv_to_sql


def task_assign_coords_to_solar_plants(
        depends_on={
            "plants": MASTR_SOLAR_PLANTS_FILTERED_CSV,
            "plz": GEONAMES_POSTAL_CODE_DATA,
        },
        produces=MASTR_SOLAR_PLANTS_FILTERED_WITH_COORDS_CSV
):
    assign_coords_to_plants(
        plz_data_path=GEONAMES_POSTAL_CODE_DATA,
        plants_csv_path=MASTR_SOLAR_PLANTS_FILTERED_CSV,
        out_path=MASTR_SOLAR_PLANTS_FILTERED_WITH_COORDS_CSV,
        keep_existing_coords=False
    )


def task_create_solar_plant_db_with_coords(
        depends_on=MASTR_SOLAR_PLANTS_FILTERED_WITH_COORDS_CSV,
        produces=MASTR_SOLAR_PLANTS_FILTERED_WITH_COORDS_SQLITE
):
    csv_to_sql(
        csv_path=MASTR_SOLAR_PLANTS_FILTERED_WITH_COORDS_CSV,
        sql_path=MASTR_SOLAR_PLANTS_FILTERED_WITH_COORDS_SQLITE
    )


def task_build_solar_capacity_grids(
        depends_on={
            "csv": MASTR_SOLAR_PLANTS_FILTERED_WITH_COORDS_CSV,
            "db": MASTR_SOLAR_PLANTS_FILTERED_WITH_COORDS_SQLITE,
        },
        produces=SOLAR_CAPACITY_GRIDS_ZARR_STORE / ".complete"
):

    build_capacity_grids(
        ref_weather_ds=GRID_REFERENCE_DS_STORE,
        plants_db=MASTR_SOLAR_PLANTS_FILTERED_WITH_COORDS_SQLITE,
        plants_csv=MASTR_SOLAR_PLANTS_FILTERED_WITH_COORDS_CSV,
        start_month=date(2015, 1, 1),
        end_month=date(2025, 12, 1),
        out_dir=SOLAR_CAPACITY_GRIDS_ZARR_STORE,
    )

    (SOLAR_CAPACITY_GRIDS_ZARR_STORE / ".complete").touch()

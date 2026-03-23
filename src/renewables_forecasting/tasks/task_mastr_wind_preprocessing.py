from datetime import date

from renewables_forecasting.config.data_sources import MASTR_WIND_VARIABLES
from renewables_forecasting.config.paths import (
    MASTR_GESAMTDATENUEBERSICHT_PATH,
    MASTR_WIND_PLANTS_FILTERED_CSV,
    MASTR_WIND_PLANTS_FILTERED_SQLITE,
    MASTR_WIND_PLANTS_EFFECTIVE_START_DATE_CSV,
    MASTR_WIND_PLANTS_REJECTED_CSV,
    GEONAMES_POSTAL_CODE_DATA,
    MASTR_WIND_PLANTS_EFFECTIVE_START_WITH_COORDS_CSV
)
from renewables_forecasting.data.mastr import (
    filter_xmls_from_gesamtdatenuebersicht_to_csv,
    csv_to_sql,
    resolve_plant_commissioning_dates,
    assign_coords_to_plants
)


def task_filter_wind_plants_to_csv(
        depends_on=MASTR_GESAMTDATENUEBERSICHT_PATH,
        produces=MASTR_WIND_PLANTS_FILTERED_CSV
):

    filter_xmls_from_gesamtdatenuebersicht_to_csv(
        zip_path=MASTR_GESAMTDATENUEBERSICHT_PATH,
        naming_files="EinheitenWind",
        naming_units="EinheitWind",
        start=date(2015, 1, 1),
        end=date(2025, 12, 31),  # inclusive
        variables=MASTR_WIND_VARIABLES,
        exclude_filters={
            "EinheitBetriebsstatus": "31",  # filter out plants still 'InPlanung'
            # "WindAnLandOderAufSee": "4713"  # offshore plants
        },
        out_csv=MASTR_WIND_PLANTS_FILTERED_CSV
    )


def task_wind_csv_to_sqlite(depends_on=MASTR_WIND_PLANTS_FILTERED_CSV, produces=MASTR_WIND_PLANTS_FILTERED_SQLITE):
    csv_to_sql(
        csv_path=MASTR_WIND_PLANTS_FILTERED_CSV,
        sql_path=MASTR_WIND_PLANTS_FILTERED_SQLITE,
        name_table="einheiten_wind"
    )


# ── Assign effective start dates to all plants considering relocations ───────────────────────────────────────────
def task_wind_resolve_commissioning_dates(
        depends_on=MASTR_WIND_PLANTS_FILTERED_CSV,
        produces={
            "cleaned": MASTR_WIND_PLANTS_EFFECTIVE_START_DATE_CSV,
            "rejected": MASTR_WIND_PLANTS_REJECTED_CSV,
        }
):

    resolve_plant_commissioning_dates(
        plants_csv_path=MASTR_WIND_PLANTS_FILTERED_CSV,
        out_csv_path=MASTR_WIND_PLANTS_EFFECTIVE_START_DATE_CSV,
        rejected_csv_path=MASTR_WIND_PLANTS_REJECTED_CSV,
    )


# ── Assign coordinates to all plants based on their postal code ───────────────────────────────────────────────────
def task_assign_coords_to_wind_plants(
        depends_on={
            "plants": MASTR_WIND_PLANTS_EFFECTIVE_START_DATE_CSV,
            "plz": GEONAMES_POSTAL_CODE_DATA,
        },
        produces=MASTR_WIND_PLANTS_EFFECTIVE_START_WITH_COORDS_CSV
):
    assign_coords_to_plants(
        plz_data_path=GEONAMES_POSTAL_CODE_DATA,
        plants_csv_path=MASTR_WIND_PLANTS_EFFECTIVE_START_DATE_CSV,
        out_path=MASTR_WIND_PLANTS_EFFECTIVE_START_WITH_COORDS_CSV,
        keep_existing_coords=True
    )

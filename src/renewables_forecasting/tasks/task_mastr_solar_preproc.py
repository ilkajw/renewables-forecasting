from datetime import date

from renewables_forecasting.config.data_sources import MASTR_SOLAR_VARIABLES
from renewables_forecasting.config.paths import (
    MASTR_GESAMTDATENUEBERSICHT_PATH,
    MASTR_SOLAR_PLANTS_FILTERED_CSV,
    MASTR_SOLAR_PLANTS_EFFECTIVE_START_DATE_CSV,
    MASTR_SOLAR_PLANTS_REJECTED_CSV,
    GEONAMES_POSTAL_CODE_DATA,
    MASTR_SOLAR_PLANTS_EFFECTIVE_START_WITH_COORDS_CSV
)
from renewables_forecasting.data.mastr import (
    filter_xmls_from_gesamtdatenuebersicht_to_csv,
    resolve_plant_commissioning_dates,
    assign_coords_to_plants
)


# ── Filter solar plants active during the period of interest ──────────────────────────────────────
def task_filter_solar_plants_from_gesamtdatenuebersicht_to_csv(
        depends_on=MASTR_GESAMTDATENUEBERSICHT_PATH,
        produces=MASTR_SOLAR_PLANTS_FILTERED_CSV
):

    filter_xmls_from_gesamtdatenuebersicht_to_csv(
        start=date(2015, 1, 1),
        end=date(2025, 12, 31),  # inclusive
        zip_path=MASTR_GESAMTDATENUEBERSICHT_PATH,
        naming_files="EinheitenSolar",
        naming_units="EinheitSolar",
        variables=MASTR_SOLAR_VARIABLES,
        exclude_filters={
            "EinheitBetriebsstatus": "31"  # filter out plants still 'InPlanung'
        },
        out_csv=MASTR_SOLAR_PLANTS_FILTERED_CSV
    )


# ── Assign effective start dates to all plants considering relocations ───────────────────────────────────────────
def task_solar_resolve_commissioning_dates(
        depends_on=MASTR_SOLAR_PLANTS_FILTERED_CSV,
        produces={
            "cleaned": MASTR_SOLAR_PLANTS_EFFECTIVE_START_DATE_CSV,
            "rejected": MASTR_SOLAR_PLANTS_REJECTED_CSV,
        }
):

    resolve_plant_commissioning_dates(
        plants_csv_path=MASTR_SOLAR_PLANTS_FILTERED_CSV,
        out_csv_path=MASTR_SOLAR_PLANTS_EFFECTIVE_START_DATE_CSV,
        rejected_csv_path=MASTR_SOLAR_PLANTS_REJECTED_CSV,
    )


# ── Assign coordinates to all plants based on their postal code ───────────────────────────────────────────────────
def task_assign_coords_to_solar_plants(
        depends_on={
            "plants": MASTR_SOLAR_PLANTS_EFFECTIVE_START_DATE_CSV,
            "plz": GEONAMES_POSTAL_CODE_DATA,
        },
        produces=MASTR_SOLAR_PLANTS_EFFECTIVE_START_WITH_COORDS_CSV
):
    assign_coords_to_plants(
        plz_data_path=GEONAMES_POSTAL_CODE_DATA,
        plants_csv_path=MASTR_SOLAR_PLANTS_EFFECTIVE_START_DATE_CSV,
        out_path=MASTR_SOLAR_PLANTS_EFFECTIVE_START_WITH_COORDS_CSV,
        keep_existing_coords=False
    )
